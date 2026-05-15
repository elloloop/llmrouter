package azureopenai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/elloloop/llmrouter"
)

// defaultSpeechFormat is the format used when SpeechRequest.Format is empty.
const defaultSpeechFormat = "mp3"

// defaultTranscriptionFormat is the format requested when the caller does
// not specify one — verbose_json gives us segment + word timings.
const defaultTranscriptionFormat = "verbose_json"

// speechStreamChunkSize is the byte budget per streamed audio chunk.
const speechStreamChunkSize = 8 * 1024

// Speak implements llmrouter.Speaker against an Azure OpenAI deployment's
// /audio/speech endpoint. URL is deployment-scoped with the api-version
// query parameter. Auth matches the chat path.
//
// When req.Stream is false the audio body arrives as a single chunk;
// otherwise it is forwarded in 8 KiB chunks.
func (p *Provider) Speak(ctx context.Context, req llmrouter.SpeechRequest) (*llmrouter.AudioStream, error) {
	body, err := buildSpeechRequestBody(req)
	if err != nil {
		return nil, err
	}

	endpoint := fmt.Sprintf("%s/openai/deployments/%s/audio/speech?api-version=%s",
		strings.TrimRight(p.cfg.BaseURL, "/"), p.deployment, p.apiVersion)

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	if err := p.applyAuth(ctx, hreq); err != nil {
		return nil, err
	}

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		snippet := readUpstreamErrorBody(resp.Body)
		resp.Body.Close()
		return nil, &llmrouter.ErrUpstream{
			Provider:   "azureopenai",
			StatusCode: resp.StatusCode,
			Body:       snippet,
		}
	}

	stream, sctx, hooks := llmrouter.NewAudioStream(ctx)
	stream.ContentType = resp.Header.Get("Content-Type")
	go pumpAudio(sctx, resp, req.Stream, hooks)
	return stream, nil
}

// buildSpeechRequestBody assembles the JSON body for /audio/speech.
func buildSpeechRequestBody(req llmrouter.SpeechRequest) ([]byte, error) {
	format := req.Format
	if format == "" {
		format = defaultSpeechFormat
	}

	var m map[string]json.RawMessage
	if len(req.Raw) > 0 {
		if err := json.Unmarshal(req.Raw, &m); err != nil {
			return nil, fmt.Errorf("azureopenai: invalid raw speech request: %w", err)
		}
	} else {
		m = map[string]json.RawMessage{}
		if req.Voice != "" {
			vb, _ := json.Marshal(req.Voice)
			m["voice"] = vb
		}
		if req.Speed != nil {
			sb, _ := json.Marshal(*req.Speed)
			m["speed"] = sb
		}
	}

	if req.Model != "" {
		mb, _ := json.Marshal(req.Model)
		m["model"] = mb
	}
	if req.Input != "" {
		ib, _ := json.Marshal(req.Input)
		m["input"] = ib
	}
	fb, _ := json.Marshal(format)
	m["response_format"] = fb
	return json.Marshal(m)
}

// pumpAudio reads the upstream audio body, forwards chunks, and always
// calls hooks.Finish exactly once.
func pumpAudio(ctx context.Context, resp *http.Response, streaming bool, hooks llmrouter.AudioProducerHooks) {
	defer resp.Body.Close()

	if !streaming {
		buf, err := io.ReadAll(resp.Body)
		if err != nil {
			hooks.Finish(fmt.Errorf("azureopenai: read audio body: %w", err))
			return
		}
		if !hooks.Send(llmrouter.AudioChunk{Data: buf, Raw: buf}) {
			hooks.Finish(ctx.Err())
			return
		}
		hooks.Finish(nil)
		return
	}

	buf := make([]byte, speechStreamChunkSize)
	for {
		select {
		case <-ctx.Done():
			hooks.Finish(ctx.Err())
			return
		default:
		}
		n, err := resp.Body.Read(buf)
		if n > 0 {
			chunk := make([]byte, n)
			copy(chunk, buf[:n])
			if !hooks.Send(llmrouter.AudioChunk{Data: chunk, Raw: chunk}) {
				hooks.Finish(ctx.Err())
				return
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				hooks.Finish(nil)
				return
			}
			hooks.Finish(fmt.Errorf("azureopenai: read audio stream: %w", err))
			return
		}
	}
}

// Transcribe implements llmrouter.Transcriber against an Azure OpenAI
// deployment's /audio/transcriptions endpoint. Streaming (req.Stream=true)
// is accepted but ignored in v0.3; a single Final TranscriptSegment is
// always emitted.
func (p *Provider) Transcribe(ctx context.Context, req llmrouter.TranscribeRequest) (*llmrouter.TranscriptStream, error) {
	body, contentType, err := buildTranscriptionBody(req)
	if err != nil {
		return nil, err
	}

	endpoint := fmt.Sprintf("%s/openai/deployments/%s/audio/transcriptions?api-version=%s",
		strings.TrimRight(p.cfg.BaseURL, "/"), p.deployment, p.apiVersion)

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, body)
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", contentType)
	if err := p.applyAuth(ctx, hreq); err != nil {
		return nil, err
	}

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		snippet := readUpstreamErrorBody(resp.Body)
		return nil, &llmrouter.ErrUpstream{
			Provider:   "azureopenai",
			StatusCode: resp.StatusCode,
			Body:       snippet,
		}
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("azureopenai: read transcription body: %w", err)
	}

	format := req.ResponseFormat
	if format == "" {
		format = defaultTranscriptionFormat
	}
	segment, err := decodeTranscriptionResponse(format, raw)
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
// /audio/transcriptions.
func buildTranscriptionBody(req llmrouter.TranscribeRequest) (io.Reader, string, error) {
	if req.Audio == nil {
		return nil, "", errors.New("azureopenai: transcribe: Audio reader required")
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
		return nil, "", fmt.Errorf("azureopenai: copy audio: %w", err)
	}

	if err := mw.WriteField("model", req.Model); err != nil {
		return nil, "", err
	}
	if req.Language != "" {
		if err := mw.WriteField("language", req.Language); err != nil {
			return nil, "", err
		}
	}
	if req.Prompt != "" {
		if err := mw.WriteField("prompt", req.Prompt); err != nil {
			return nil, "", err
		}
	}
	format := req.ResponseFormat
	if format == "" {
		format = defaultTranscriptionFormat
	}
	if err := mw.WriteField("response_format", format); err != nil {
		return nil, "", err
	}
	if req.Temperature != nil {
		if err := mw.WriteField("temperature", strconv.FormatFloat(*req.Temperature, 'f', -1, 64)); err != nil {
			return nil, "", err
		}
	}
	if err := mw.Close(); err != nil {
		return nil, "", err
	}
	return &buf, mw.FormDataContentType(), nil
}

// extensionForFormat maps an audio MIME type to a conventional file
// extension. Falls back to ".mp3".
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

// decodeTranscriptionResponse parses the upstream body into a single
// Final TranscriptSegment. Shape depends on response_format.
func decodeTranscriptionResponse(format string, raw []byte) (llmrouter.TranscriptSegment, error) {
	switch format {
	case "text", "srt", "vtt":
		return llmrouter.TranscriptSegment{
			Text:  string(bytes.TrimSpace(raw)),
			Final: true,
			Raw:   json.RawMessage(raw),
		}, nil
	}

	var wire struct {
		Text  string `json:"text"`
		Words []struct {
			Word  string  `json:"word"`
			Start float64 `json:"start"`
			End   float64 `json:"end"`
		} `json:"words"`
		Segments []struct {
			Text  string  `json:"text"`
			Start float64 `json:"start"`
			End   float64 `json:"end"`
		} `json:"segments"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return llmrouter.TranscriptSegment{}, fmt.Errorf("azureopenai: decode transcription: %w", err)
	}

	seg := llmrouter.TranscriptSegment{
		Text:  wire.Text,
		Final: true,
		Raw:   json.RawMessage(raw),
	}
	if len(wire.Segments) > 0 {
		seg.Start = secondsToDuration(wire.Segments[0].Start)
		seg.End = secondsToDuration(wire.Segments[len(wire.Segments)-1].End)
	}
	if len(wire.Words) > 0 {
		words := make([]llmrouter.TranscriptWord, 0, len(wire.Words))
		for _, w := range wire.Words {
			words = append(words, llmrouter.TranscriptWord{
				Word:  w.Word,
				Start: secondsToDuration(w.Start),
				End:   secondsToDuration(w.End),
			})
		}
		seg.Words = words
	}
	return seg, nil
}

// secondsToDuration converts float seconds to time.Duration.
func secondsToDuration(seconds float64) time.Duration {
	const nsPerSecond = 1_000_000_000.0
	return time.Duration(seconds * nsPerSecond)
}
