package deepgram

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/elloloop/llmrouter"
)

const (
	// defaultModel is the Deepgram model used when TranscribeRequest.Model
	// is empty. nova-3 is the current general-purpose default with the
	// best accuracy/latency trade-off.
	defaultModel = "nova-3"

	// defaultLanguage is the BCP-47 language hint used when
	// TranscribeRequest.Language is empty.
	defaultLanguage = "en"

	// defaultAudioContentType is the request Content-Type used when
	// TranscribeRequest.AudioFormat is empty. Deepgram accepts raw bytes
	// with the source MIME type as Content-Type.
	defaultAudioContentType = "audio/mpeg"

	// errorBodyCap bounds the bytes captured from an upstream error
	// response so a runaway HTML/JSON body never blows up an error
	// string. Matches the convention used in the anthropic provider.
	errorBodyCap = 8 * 1024
)

// rawQueryKeys are the extra Deepgram query parameters lifted from
// TranscribeRequest.Raw and forwarded on the request URL. Keep this list
// conservative — anything not listed here is ignored to avoid accidental
// passthrough of secrets or oversized payloads via query string.
var rawQueryKeys = []string{
	"diarize",
	"paragraphs",
	"summarize",
	"detect_language",
	"detect_topics",
	"detect_entities",
	"redact",
	"profanity_filter",
	"numerals",
	"measurements",
	"dictation",
	"keywords",
	"search",
	"replace",
	"tier",
	"version",
	"multichannel",
	"alternatives",
	"filler_words",
	"tag",
	"callback",
	"endpointing",
}

// Transcribe issues a batch /v1/listen request and returns a
// TranscriptStream that emits exactly one Final TranscriptSegment built
// from the upstream JSON response.
//
// req.Stream is accepted but ignored: WebSocket-based live transcription
// is deferred to a future version. req.Raw is merged into the query
// string for any of the keys in rawQueryKeys.
func (p *Provider) Transcribe(ctx context.Context, req llmrouter.TranscribeRequest) (*llmrouter.TranscriptStream, error) {
	if req.Audio == nil {
		return nil, errors.New("deepgram: transcribe: Audio reader required")
	}

	endpoint, err := buildListenURL(p.cfg.BaseURL, req)
	if err != nil {
		return nil, err
	}

	audioBytes, err := io.ReadAll(req.Audio)
	if err != nil {
		return nil, fmt.Errorf("deepgram: read audio: %w", err)
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		endpoint, bytes.NewReader(audioBytes))
	if err != nil {
		return nil, err
	}
	contentType := req.AudioFormat
	if contentType == "" {
		contentType = defaultAudioContentType
	}
	hreq.Header.Set("Content-Type", contentType)
	// Deepgram auth uses the literal token type "Token", NOT "Bearer".
	hreq.Header.Set("Authorization", "Token "+p.cfg.APIKey)

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("deepgram: http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		snippet := readUpstreamErrorBody(resp.Body)
		return nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: resp.StatusCode,
			Body:       snippet,
		}
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("deepgram: read response: %w", err)
	}

	segment, err := decodeListenResponse(raw)
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

// buildListenURL assembles the /v1/listen endpoint URL with all the
// query parameters Deepgram needs. Defaults are applied for model and
// language; punctuate/smart_format/utterances are always enabled to get
// the richest segment shape back.
func buildListenURL(baseURL string, req llmrouter.TranscribeRequest) (string, error) {
	u, err := url.Parse(baseURL + "/v1/listen")
	if err != nil {
		return "", fmt.Errorf("deepgram: parse base url: %w", err)
	}

	model := req.Model
	if model == "" {
		model = defaultModel
	}
	language := req.Language
	if language == "" {
		language = defaultLanguage
	}

	q := u.Query()
	q.Set("model", model)
	q.Set("language", language)
	q.Set("punctuate", "true")
	q.Set("smart_format", "true")
	q.Set("utterances", "true")

	mergeRawQuery(q, req.Raw)

	u.RawQuery = q.Encode()
	return u.String(), nil
}

// mergeRawQuery copies any of the allow-listed keys from req.Raw onto
// the outgoing query string. Values are serialised as their JSON scalar
// form: booleans/numbers/strings pass through naturally, anything else
// is skipped to avoid emitting "null" or "{...}" into the URL.
func mergeRawQuery(q url.Values, raw json.RawMessage) {
	if len(raw) == 0 {
		return
	}
	var src map[string]json.RawMessage
	if err := json.Unmarshal(raw, &src); err != nil {
		return
	}
	for _, key := range rawQueryKeys {
		v, ok := src[key]
		if !ok {
			continue
		}
		s, ok := scalarToString(v)
		if !ok {
			continue
		}
		q.Set(key, s)
	}
}

// scalarToString reduces a JSON scalar (bool, number, string) to its
// query-string representation. Objects, arrays, and null are rejected.
func scalarToString(raw json.RawMessage) (string, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return "", false
	}
	switch trimmed[0] {
	case '"':
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return "", false
		}
		return s, true
	case 't', 'f':
		var b bool
		if err := json.Unmarshal(trimmed, &b); err != nil {
			return "", false
		}
		if b {
			return "true", true
		}
		return "false", true
	case '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		// Preserve the original numeric formatting (no float rounding).
		return string(trimmed), true
	default:
		return "", false
	}
}

// readUpstreamErrorBody captures up to errorBodyCap bytes of an error
// response body for inclusion in *llmrouter.ErrUpstream.
func readUpstreamErrorBody(r io.Reader) string {
	b, _ := io.ReadAll(io.LimitReader(r, errorBodyCap))
	return string(b)
}

// deepgramWord is the JSON shape of one entry in
// results.channels[0].alternatives[0].words.
type deepgramWord struct {
	Word           string  `json:"word"`
	Start          float64 `json:"start"`
	End            float64 `json:"end"`
	Confidence     float64 `json:"confidence"`
	PunctuatedWord string  `json:"punctuated_word"`
}

// deepgramListenResponse is the subset of the /v1/listen JSON response
// the provider consumes.
type deepgramListenResponse struct {
	Metadata struct {
		Duration float64 `json:"duration"`
	} `json:"metadata"`
	Results struct {
		Channels []struct {
			Alternatives []struct {
				Transcript string         `json:"transcript"`
				Confidence float64        `json:"confidence"`
				Words      []deepgramWord `json:"words"`
			} `json:"alternatives"`
		} `json:"channels"`
	} `json:"results"`
}

// decodeListenResponse turns a Deepgram /v1/listen JSON body into a
// single Final TranscriptSegment. Missing channels/alternatives are
// tolerated and yield an empty segment.
func decodeListenResponse(raw []byte) (llmrouter.TranscriptSegment, error) {
	var wire deepgramListenResponse
	if err := json.Unmarshal(raw, &wire); err != nil {
		return llmrouter.TranscriptSegment{}, fmt.Errorf("deepgram: decode response: %w", err)
	}

	seg := llmrouter.TranscriptSegment{
		Final: true,
		Raw:   json.RawMessage(raw),
		End:   secondsToDuration(wire.Metadata.Duration),
	}
	if len(wire.Results.Channels) == 0 || len(wire.Results.Channels[0].Alternatives) == 0 {
		return seg, nil
	}
	alt := wire.Results.Channels[0].Alternatives[0]
	seg.Text = alt.Transcript
	seg.Confidence = float32(alt.Confidence)
	seg.Words = mapWords(alt.Words)
	return seg, nil
}

// mapWords converts Deepgram's per-word array into the library's
// TranscriptWord slice, preferring punctuated_word when populated so
// downstream callers see correctly capitalised/punctuated tokens.
func mapWords(in []deepgramWord) []llmrouter.TranscriptWord {
	if len(in) == 0 {
		return nil
	}
	out := make([]llmrouter.TranscriptWord, 0, len(in))
	for _, w := range in {
		token := w.PunctuatedWord
		if token == "" {
			token = w.Word
		}
		out = append(out, llmrouter.TranscriptWord{
			Word:       token,
			Start:      secondsToDuration(w.Start),
			End:        secondsToDuration(w.End),
			Confidence: float32(w.Confidence),
		})
	}
	return out
}

// secondsToDuration converts a float-seconds timestamp to time.Duration
// with nanosecond precision.
func secondsToDuration(seconds float64) time.Duration {
	const nsPerSecond = 1_000_000_000.0
	return time.Duration(seconds * nsPerSecond)
}
