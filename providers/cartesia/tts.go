package cartesia

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/elloloop/llmrouter"
)

// upstreamErrorCap caps the upstream error body included in ErrUpstream.
const upstreamErrorCap = 8 * 1024

// sseScannerBufferSize is the maximum SSE event size we'll tolerate.
// Cartesia base64 chunks can be large; 1 MiB is comfortable headroom.
const sseScannerBufferSize = 1 * 1024 * 1024

// ttsBytesPath is the Cartesia batch TTS endpoint.
const ttsBytesPath = "/tts/bytes"

// ttsSSEPath is the Cartesia Server-Sent Events streaming TTS endpoint.
const ttsSSEPath = "/tts/sse"

// Speak implements llmrouter.Speaker against Cartesia's TTS endpoints.
//
// When req.Stream is false the entire audio body is returned via a
// single AudioChunk from /tts/bytes. When req.Stream is true the body
// is decoded as Server-Sent Events from /tts/sse; each `chunk` event is
// base64-decoded and forwarded, and the stream terminates on the `done`
// event.
//
// Defaults: Model "sonic-2", Voice "79a125e8-cd45-4c13-8a67-188112f4dd22",
// Format "pcm".
func (p *Provider) Speak(ctx context.Context, req llmrouter.SpeechRequest) (*llmrouter.AudioStream, error) {
	body, err := buildSpeechRequestBody(req)
	if err != nil {
		return nil, err
	}

	path := ttsBytesPath
	acceptHeader := acceptForFormat(req.Format)
	if req.Stream {
		path = ttsSSEPath
		acceptHeader = "text/event-stream"
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.cfg.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", acceptHeader)
	hreq.Header.Set("X-API-Key", p.cfg.APIKey)
	hreq.Header.Set("Cartesia-Version", cartesiaVersion)

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("cartesia: http: %w", err)
	}
	if resp.StatusCode >= 400 {
		snippet := readUpstreamErrorBody(resp.Body)
		resp.Body.Close()
		return nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: resp.StatusCode,
			Body:       snippet,
		}
	}

	stream, sctx, hooks := llmrouter.NewAudioStream(ctx)
	if req.Stream {
		// SSE: the upstream Content-Type is text/event-stream, which is
		// not the audio MIME — derive from the request format instead.
		stream.ContentType = audioMIMEForFormat(req.Format)
		go pumpAudioSSE(sctx, resp, hooks)
	} else {
		stream.ContentType = resp.Header.Get("Content-Type")
		if stream.ContentType == "" {
			stream.ContentType = audioMIMEForFormat(req.Format)
		}
		go pumpAudioBatch(sctx, resp, hooks)
	}
	return stream, nil
}

// buildSpeechRequestBody assembles the JSON body for Cartesia TTS.
// req.Raw is honoured: when present it is taken as the base body and
// the typed fields (transcript, voice, model_id, output_format) are
// merged on top. When Raw is empty the body is built from the typed
// fields with documented defaults applied.
func buildSpeechRequestBody(req llmrouter.SpeechRequest) ([]byte, error) {
	voice := req.Voice
	if voice == "" {
		voice = defaultVoiceID
	}
	model := req.Model
	if model == "" {
		model = defaultTTSModel
	}

	var base map[string]json.RawMessage
	if len(req.Raw) > 0 {
		if err := json.Unmarshal(req.Raw, &base); err != nil {
			return nil, fmt.Errorf("cartesia: invalid raw speech request: %w", err)
		}
	} else {
		base = map[string]json.RawMessage{}
	}

	transcriptRaw, _ := json.Marshal(req.Input)
	base["transcript"] = transcriptRaw

	voiceRaw, _ := json.Marshal(map[string]any{
		"mode": "id",
		"id":   voice,
	})
	base["voice"] = voiceRaw

	modelRaw, _ := json.Marshal(model)
	base["model_id"] = modelRaw

	outputFormatRaw, _ := json.Marshal(outputFormatForFormat(req.Format))
	base["output_format"] = outputFormatRaw

	return json.Marshal(base)
}

// outputFormatForFormat maps SpeechRequest.Format onto Cartesia's
// output_format object. Unknown / empty values fall back to raw PCM
// (pcm_s16le @ 44.1 kHz) — Cartesia's documented default.
func outputFormatForFormat(format string) map[string]any {
	switch strings.ToLower(format) {
	case "mp3":
		return map[string]any{
			"container":   "mp3",
			"encoding":    "mp3",
			"sample_rate": 44100,
			"bit_rate":    128000,
		}
	case "wav":
		return map[string]any{
			"container":   "wav",
			"encoding":    "pcm_s16le",
			"sample_rate": 44100,
		}
	case "ulaw":
		return map[string]any{
			"container":   "raw",
			"encoding":    "pcm_mulaw",
			"sample_rate": 8000,
		}
	case "pcm", "":
		fallthrough
	default:
		return map[string]any{
			"container":   "raw",
			"encoding":    "pcm_s16le",
			"sample_rate": 44100,
		}
	}
}

// acceptForFormat returns the Accept header for the batch TTS request.
// Cartesia returns the raw audio bytes for /tts/bytes; the Accept hint
// lets the server pick a sensible Content-Type for the response.
func acceptForFormat(format string) string {
	switch strings.ToLower(format) {
	case "mp3":
		return "audio/mpeg"
	case "wav":
		return "audio/wav"
	case "ulaw":
		return "audio/basic"
	case "pcm", "":
		return "audio/pcm"
	default:
		return "audio/*"
	}
}

// audioMIMEForFormat returns the AudioStream.ContentType to use when the
// upstream doesn't provide one (streaming) or for /tts/bytes fallback.
func audioMIMEForFormat(format string) string {
	switch strings.ToLower(format) {
	case "mp3":
		return "audio/mpeg"
	case "wav":
		return "audio/wav"
	case "ulaw":
		return "audio/basic"
	case "pcm", "":
		return "audio/pcm"
	default:
		return "application/octet-stream"
	}
}

// pumpAudioBatch reads the full /tts/bytes response and emits it as one
// AudioChunk. Always calls hooks.Finish exactly once.
func pumpAudioBatch(ctx context.Context, resp *http.Response, hooks llmrouter.AudioProducerHooks) {
	defer resp.Body.Close()

	buf, err := io.ReadAll(resp.Body)
	if err != nil {
		hooks.Finish(fmt.Errorf("cartesia: read audio body: %w", err))
		return
	}
	if !hooks.Send(llmrouter.AudioChunk{Data: buf, Raw: buf}) {
		hooks.Finish(ctx.Err())
		return
	}
	hooks.Finish(nil)
}

// pumpAudioSSE reads the /tts/sse response, decodes each `chunk` event's
// base64 payload, and forwards an AudioChunk per chunk. Terminates on a
// `done` event or stream EOF. Always calls hooks.Finish exactly once.
func pumpAudioSSE(ctx context.Context, resp *http.Response, hooks llmrouter.AudioProducerHooks) {
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), sseScannerBufferSize)

	var dataLines []string
	flush := func() (done bool, err error) {
		if len(dataLines) == 0 {
			return false, nil
		}
		payload := strings.Join(dataLines, "\n")
		dataLines = dataLines[:0]
		return handleSSEEvent(ctx, payload, hooks)
	}

	for scanner.Scan() {
		if ctx.Err() != nil {
			hooks.Finish(ctx.Err())
			return
		}
		line := scanner.Text()
		if line == "" {
			done, err := flush()
			if err != nil {
				hooks.Finish(err)
				return
			}
			if done {
				hooks.Finish(nil)
				return
			}
			continue
		}
		switch {
		case strings.HasPrefix(line, "data: "):
			dataLines = append(dataLines, strings.TrimPrefix(line, "data: "))
		case strings.HasPrefix(line, "data:"):
			dataLines = append(dataLines, strings.TrimPrefix(line, "data:"))
		}
	}
	// Flush any trailing event that lacked a terminating blank line.
	if len(dataLines) > 0 {
		done, err := flush()
		if err != nil {
			hooks.Finish(err)
			return
		}
		if done {
			hooks.Finish(nil)
			return
		}
	}
	if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
		hooks.Finish(fmt.Errorf("cartesia: read stream: %w", err))
		return
	}
	hooks.Finish(nil)
}

// handleSSEEvent decodes a single SSE `data:` payload from Cartesia's
// /tts/sse endpoint. Returns done=true on a `done` envelope.
func handleSSEEvent(ctx context.Context, payload string, hooks llmrouter.AudioProducerHooks) (done bool, err error) {
	trimmed := strings.TrimSpace(payload)
	if trimmed == "" {
		return false, nil
	}

	var env struct {
		Type string `json:"type"`
		Data string `json:"data"`
	}
	if err := json.Unmarshal([]byte(trimmed), &env); err != nil {
		// Tolerate malformed events the same way other providers do —
		// drop them and keep reading.
		return false, nil
	}
	switch env.Type {
	case "chunk":
		if env.Data == "" {
			return false, nil
		}
		decoded, decErr := base64.StdEncoding.DecodeString(env.Data)
		if decErr != nil {
			return false, fmt.Errorf("cartesia: decode chunk: %w", decErr)
		}
		if !hooks.Send(llmrouter.AudioChunk{Data: decoded, Raw: []byte(payload)}) {
			return false, ctx.Err()
		}
		return false, nil
	case "done":
		return true, nil
	default:
		// "timestamps", "flush", or anything else — ignore.
		return false, nil
	}
}

// readUpstreamErrorBody reads up to upstreamErrorCap bytes from the
// upstream error body. The cap matches the convention used by the other
// providers (anthropic, openai, ...).
func readUpstreamErrorBody(body io.Reader) string {
	b, _ := io.ReadAll(io.LimitReader(body, upstreamErrorCap))
	return string(b)
}
