package elevenlabs

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/elloloop/llmrouter"
)

// ttsStreamChunkSize is the byte budget per streamed audio chunk.
const ttsStreamChunkSize = 8 * 1024

// errorBodySnippetLimit caps how many bytes of an upstream error body
// are surfaced in ErrUpstream.Body.
const errorBodySnippetLimit = 1024

// Speak implements llmrouter.Speaker against the ElevenLabs
// /v1/text-to-speech/<voice_id>(?:/stream) endpoint.
//
// When req.Stream is true the streaming variant is used and audio bytes
// are forwarded to the consumer in ttsStreamChunkSize-byte chunks. When
// false, the entire body arrives as a single AudioChunk.
//
// The Voice id defaults to "21m00Tcm4TlvDq8ikWAM" (Rachel) when empty.
// The Model defaults to "eleven_turbo_v2_5" when empty. The Format is
// mapped to ElevenLabs' `output_format` enum via elevenLabsFormat —
// notably WAV maps to PCM (ElevenLabs ships no native WAV container).
//
// req.Raw, when present, is merged OVER the typed body so callers can
// thread provider-specific extras such as `voice_settings`.
func (p *Provider) Speak(ctx context.Context, req llmrouter.SpeechRequest) (*llmrouter.AudioStream, error) {
	body, err := buildSpeechRequestBody(req)
	if err != nil {
		return nil, err
	}

	url := buildSpeechURL(p.cfg.BaseURL, req)
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", acceptForFormat(req.Format))
	hreq.Header.Set("xi-api-key", p.cfg.APIKey)

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		snippet := readErrorBody(resp.Body, errorBodySnippetLimit)
		resp.Body.Close()
		return nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: resp.StatusCode,
			Body:       snippet,
		}
	}

	stream, sctx, hooks := llmrouter.NewAudioStream(ctx)
	stream.ContentType = resp.Header.Get("Content-Type")
	go pumpAudio(sctx, resp, req.Stream, hooks)
	return stream, nil
}

// buildSpeechURL composes the TTS endpoint URL, picking the streaming
// variant when req.Stream is true.
func buildSpeechURL(baseURL string, req llmrouter.SpeechRequest) string {
	voice := req.Voice
	if voice == "" {
		voice = defaultVoiceID
	}
	if req.Stream {
		return baseURL + "/v1/text-to-speech/" + voice + "/stream"
	}
	return baseURL + "/v1/text-to-speech/" + voice
}

// buildSpeechRequestBody assembles the JSON body for /v1/text-to-speech.
// The typed Input / Model / Format / Voice fields are always overlaid on
// top of req.Raw (which may carry provider-specific extras such as
// `voice_settings`).
func buildSpeechRequestBody(req llmrouter.SpeechRequest) ([]byte, error) {
	model := req.Model
	if model == "" {
		model = defaultTTSModel
	}
	outputFormat := elevenLabsFormat(req.Format)

	m := map[string]json.RawMessage{}
	if len(req.Raw) > 0 {
		if err := json.Unmarshal(req.Raw, &m); err != nil {
			return nil, fmt.Errorf("elevenlabs: invalid raw speech request: %w", err)
		}
	}

	textBytes, _ := json.Marshal(req.Input)
	m["text"] = textBytes
	modelBytes, _ := json.Marshal(model)
	m["model_id"] = modelBytes
	formatBytes, _ := json.Marshal(outputFormat)
	m["output_format"] = formatBytes

	return json.Marshal(m)
}

// elevenLabsFormat maps SpeechRequest.Format to ElevenLabs'
// `output_format` enum. Defaults to "mp3_44100_128" for unknown / empty.
//
// Mapping:
//   - "mp3"   -> "mp3_44100_128" (highest standard MP3 bitrate)
//   - "opus"  -> "opus_48000_128"
//   - "pcm"   -> "pcm_44100"
//   - "wav"   -> "pcm_44100" (ElevenLabs ships no native WAV; PCM is closest)
//   - "ulaw"  -> "ulaw_8000"
//   - default -> "mp3_44100_128"
func elevenLabsFormat(format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "mp3", "":
		return "mp3_44100_128"
	case "opus":
		return "opus_48000_128"
	case "pcm":
		return "pcm_44100"
	case "wav":
		return "pcm_44100"
	case "ulaw":
		return "ulaw_8000"
	default:
		return "mp3_44100_128"
	}
}

// acceptForFormat picks an Accept header consistent with the requested
// output format. ElevenLabs honours the path / body format selection
// regardless, so this is informational.
func acceptForFormat(format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "opus":
		return "audio/opus"
	case "pcm", "wav":
		return "audio/pcm"
	case "ulaw":
		return "audio/basic"
	default:
		return "audio/mpeg"
	}
}

// pumpAudio reads the upstream audio body, forwards chunks, and always
// calls hooks.Finish exactly once.
func pumpAudio(ctx context.Context, resp *http.Response, streaming bool, hooks llmrouter.AudioProducerHooks) {
	defer resp.Body.Close()

	if !streaming {
		buf, err := io.ReadAll(resp.Body)
		if err != nil {
			hooks.Finish(fmt.Errorf("elevenlabs: read audio body: %w", err))
			return
		}
		if !hooks.Send(llmrouter.AudioChunk{Data: buf, Raw: buf}) {
			hooks.Finish(ctx.Err())
			return
		}
		hooks.Finish(nil)
		return
	}

	buf := make([]byte, ttsStreamChunkSize)
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
			hooks.Finish(fmt.Errorf("elevenlabs: read audio stream: %w", err))
			return
		}
	}
}

// readErrorBody reads up to `limit` bytes of an upstream error response
// for inclusion in ErrUpstream.Body.
func readErrorBody(r io.Reader, limit int) string {
	buf := make([]byte, limit)
	n, _ := io.ReadFull(io.LimitReader(r, int64(limit)), buf)
	return strings.TrimSpace(string(buf[:n]))
}
