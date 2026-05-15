package elevenlabs

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"time"

	"github.com/elloloop/llmrouter"
)

// sttErrorBodyLimit caps how many bytes of an upstream STT error body
// are surfaced in ErrUpstream.Body. Larger than the TTS limit because
// JSON error envelopes here are richer.
const sttErrorBodyLimit = 8 * 1024

// Transcribe implements llmrouter.Transcriber against the ElevenLabs
// /v1/speech-to-text endpoint via a multipart/form-data upload.
//
// In v0.3 streaming (req.Stream=true) is accepted but ignored — the
// ElevenLabs HTTP endpoint is non-streaming. A single Final
// TranscriptSegment is emitted; per-word timings are mapped from the
// response `words` array into TranscriptSegment.Words. Speaker labels
// from diarisation are not surfaced in TranscriptWord today (the
// llmrouter v0.3 schema does not model them).
//
// The Model defaults to "scribe_v1" when empty. When req.Raw contains a
// JSON object with a "diarize" key, that boolean is forwarded as the
// `diarize` multipart field.
func (p *Provider) Transcribe(ctx context.Context, req llmrouter.TranscribeRequest) (*llmrouter.TranscriptStream, error) {
	body, contentType, err := buildTranscriptionBody(req)
	if err != nil {
		return nil, err
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.cfg.BaseURL+"/v1/speech-to-text", body)
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", contentType)
	hreq.Header.Set("xi-api-key", p.cfg.APIKey)

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		snippet := readErrorBody(resp.Body, sttErrorBodyLimit)
		return nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: resp.StatusCode,
			Body:       snippet,
		}
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("elevenlabs: read transcription body: %w", err)
	}

	segment, err := decodeTranscriptionResponse(raw)
	if err != nil {
		return nil, err
	}

	stream, _, hooks := llmrouter.NewTranscriptStream(ctx)
	go func() {
		hooks.Send(segment)
		hooks.Finish(nil)
	}()
	return stream, nil
}

// buildTranscriptionBody assembles the multipart/form-data body for
// /v1/speech-to-text. Returns the body, content-type header value, and
// any encoding error.
func buildTranscriptionBody(req llmrouter.TranscribeRequest) (io.Reader, string, error) {
	if req.Audio == nil {
		return nil, "", errors.New("elevenlabs: transcribe: Audio reader required")
	}

	model := req.Model
	if model == "" {
		model = defaultSTTModel
	}

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)

	filename := req.Filename
	if filename == "" {
		filename = "audio" + extensionForFormat(req.AudioFormat)
	}

	fileWriter, err := mw.CreateFormFile("file", filename)
	if err != nil {
		return nil, "", err
	}
	if _, err := io.Copy(fileWriter, req.Audio); err != nil {
		return nil, "", fmt.Errorf("elevenlabs: copy audio: %w", err)
	}

	if err := mw.WriteField("model_id", model); err != nil {
		return nil, "", err
	}
	if req.Language != "" {
		if err := mw.WriteField("language_code", req.Language); err != nil {
			return nil, "", err
		}
	}
	if err := mw.WriteField("tag_audio_events", "true"); err != nil {
		return nil, "", err
	}
	if err := mw.WriteField("timestamps_granularity", "word"); err != nil {
		return nil, "", err
	}

	if diarize, ok := diarizeFromRaw(req.Raw); ok {
		if err := mw.WriteField("diarize", boolString(diarize)); err != nil {
			return nil, "", err
		}
	}

	if err := mw.Close(); err != nil {
		return nil, "", err
	}
	return &buf, mw.FormDataContentType(), nil
}

// diarizeFromRaw extracts the optional "diarize" boolean from the raw
// JSON request body. Returns false, false when not present or not a
// boolean.
func diarizeFromRaw(raw json.RawMessage) (bool, bool) {
	if len(raw) == 0 {
		return false, false
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return false, false
	}
	v, ok := m["diarize"]
	if !ok {
		return false, false
	}
	var b bool
	if err := json.Unmarshal(v, &b); err != nil {
		return false, false
	}
	return b, true
}

// boolString renders a Go bool as the lowercase string ElevenLabs
// expects in multipart fields.
func boolString(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

// extensionForFormat maps an audio MIME type to a conventional file
// extension for the multipart filename. Falls back to ".mp3" for
// unknown / empty values.
func extensionForFormat(format string) string {
	switch format {
	case "audio/mpeg", "audio/mp3", "mp3":
		return ".mp3"
	case "audio/wav", "audio/x-wav", "wav":
		return ".wav"
	case "audio/webm", "webm":
		return ".webm"
	case "audio/m4a", "audio/mp4", "audio/x-m4a", "m4a":
		return ".m4a"
	case "audio/flac", "audio/x-flac", "flac":
		return ".flac"
	case "audio/ogg", "ogg":
		return ".ogg"
	case "audio/opus", "opus":
		return ".opus"
	default:
		return ".mp3"
	}
}

// decodeTranscriptionResponse parses the upstream JSON into a single
// Final TranscriptSegment.
//
// ElevenLabs Scribe shape:
//
//	{
//	  "language_code": "en",
//	  "text": "hello world",
//	  "words": [
//	    {"text":"hello","type":"word","start":0.01,"end":0.30,"speaker_id":"speaker_1"},
//	    ...
//	  ]
//	}
//
// Word "type" of "spacing" / "audio_event" entries are filtered out:
// only entries with type=="word" (or missing type, for forward compat)
// contribute to TranscriptSegment.Words.
func decodeTranscriptionResponse(raw []byte) (llmrouter.TranscriptSegment, error) {
	var wire struct {
		LanguageCode string `json:"language_code"`
		Text         string `json:"text"`
		Words        []struct {
			Text  string  `json:"text"`
			Type  string  `json:"type"`
			Start float64 `json:"start"`
			End   float64 `json:"end"`
		} `json:"words"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return llmrouter.TranscriptSegment{}, fmt.Errorf("elevenlabs: decode transcription: %w", err)
	}

	seg := llmrouter.TranscriptSegment{
		Text:  wire.Text,
		Final: true,
		Raw:   json.RawMessage(raw),
	}

	if len(wire.Words) > 0 {
		words := make([]llmrouter.TranscriptWord, 0, len(wire.Words))
		for _, w := range wire.Words {
			if w.Type != "" && w.Type != "word" {
				continue
			}
			words = append(words, llmrouter.TranscriptWord{
				Word:  w.Text,
				Start: secondsToDuration(w.Start),
				End:   secondsToDuration(w.End),
			})
		}
		if len(words) > 0 {
			seg.Words = words
			seg.Start = words[0].Start
			seg.End = words[len(words)-1].End
		}
	}
	return seg, nil
}

// secondsToDuration converts float-seconds timestamps to time.Duration
// with nanosecond precision.
func secondsToDuration(seconds float64) time.Duration {
	return time.Duration(seconds * float64(time.Second))
}
