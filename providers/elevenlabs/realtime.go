package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"

	"github.com/coder/websocket"
	"github.com/elloloop/llmrouter"
)

// defaultOptimizeStreamingLatency is the value sent on the WebSocket
// handshake's optimize_streaming_latency query parameter. Level 2 trades
// minor quality for ~75ms of head-of-stream latency reduction and is the
// recommended default for real-time agent use-cases.
const defaultOptimizeStreamingLatency = "2"

// defaultInactivityTimeoutSeconds is the value sent on the WebSocket
// handshake's inactivity_timeout query parameter. The server tears down
// the context after this many seconds without inbound frames.
const defaultInactivityTimeoutSeconds = "20"

// defaultStability is the BOS-frame voice_settings.stability value used
// when callers do not override via Raw.
const defaultStability = 0.5

// defaultSimilarityBoost is the BOS-frame voice_settings.similarity_boost
// value used when callers do not override via Raw.
const defaultSimilarityBoost = 0.8

// bosVoiceSettings carries the default voice_settings the BOS frame
// advertises. Callers that need different stability/similarity_boost
// values can post additional text frames or override via Raw downstream.
type bosVoiceSettings struct {
	Stability       float64 `json:"stability"`
	SimilarityBoost float64 `json:"similarity_boost"`
}

// bosFrame is the JSON shape ElevenLabs expects as the first frame on a
// /stream-input WebSocket. The xi_api_key field is omitted because the
// dial sets the equivalent header on the upgrade request.
type bosFrame struct {
	Text          string           `json:"text"`
	VoiceSettings bosVoiceSettings `json:"voice_settings"`
}

// textFrame is a mid-stream JSON frame carrying a chunk of input text.
type textFrame struct {
	Text string `json:"text"`
}

// serverFrame is the JSON shape ElevenLabs sends back over the WebSocket.
// Audio is base64-encoded PCM/MP3/opus bytes. IsFinal marks the last
// frame of the current TTS context.
type serverFrame struct {
	Audio   string `json:"audio"`
	IsFinal bool   `json:"isFinal"`
}

// RealtimeContext is the handle ElevenLabs callers use to stream
// additional text chunks into a live TTS context opened via
// Provider.SpeakRealtime. Methods are safe to call from a single
// producer goroutine; concurrent Appends are serialised internally.
type RealtimeContext struct {
	conn   *websocket.Conn
	cancel context.CancelFunc

	writeMu sync.Mutex

	done chan struct{} // closed once the pump goroutine returns
}

// Append streams another text chunk into the same TTS context. It is
// the caller's responsibility to ensure the chunk ends on a word
// boundary if low-latency synthesis quality matters.
func (rc *RealtimeContext) Append(ctx context.Context, text string) error {
	if text == "" {
		return nil
	}
	return rc.writeJSON(ctx, textFrame{Text: text})
}

// Finalize sends an EOS frame and waits for the server to flush the
// final audio. The associated AudioStream's Chunks channel will close
// shortly after Finalize returns.
func (rc *RealtimeContext) Finalize(ctx context.Context) error {
	if err := rc.writeJSON(ctx, textFrame{Text: ""}); err != nil {
		return err
	}
	select {
	case <-rc.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close tears down the connection without flushing. Idempotent.
func (rc *RealtimeContext) Close() error {
	if rc.cancel != nil {
		rc.cancel()
	}
	return rc.conn.Close(websocket.StatusNormalClosure, "client closed")
}

// writeJSON marshals v and writes a single text-type frame, serialising
// concurrent calls under writeMu.
func (rc *RealtimeContext) writeJSON(ctx context.Context, v any) error {
	buf, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("elevenlabs: marshal realtime frame: %w", err)
	}
	rc.writeMu.Lock()
	defer rc.writeMu.Unlock()
	if err := rc.conn.Write(ctx, websocket.MessageText, buf); err != nil {
		return fmt.Errorf("elevenlabs: write realtime frame: %w", err)
	}
	return nil
}

// SpeakRealtime opens a WebSocket TTS context against the ElevenLabs
// /v1/text-to-speech/<voice>/stream-input endpoint. Callers append text
// chunks via the returned RealtimeContext as they arrive (e.g. from an
// LLM streaming completion) and read audio bytes from the AudioStream
// concurrently. The two values are independent: read the stream from one
// goroutine while the producer goroutine appends text from another.
//
// Pass an empty SpeechRequest.Input when you intend to stream everything
// via Append. A non-empty Input is sent as the first text frame
// immediately after the BOS frame.
//
// The Voice id defaults to "21m00Tcm4TlvDq8ikWAM" (Rachel) when empty.
// The Model defaults to "eleven_turbo_v2_5" when empty. The Format is
// mapped to ElevenLabs' `output_format` enum via elevenLabsFormat.
func (p *Provider) SpeakRealtime(ctx context.Context, req llmrouter.SpeechRequest) (*llmrouter.AudioStream, *RealtimeContext, error) {
	dialURL := buildRealtimeURL(p.cfg.BaseURL, req)

	header := http.Header{}
	header.Set("xi-api-key", p.cfg.APIKey)

	conn, resp, err := websocket.Dial(ctx, dialURL, &websocket.DialOptions{
		HTTPClient: p.cfg.HTTP(),
		HTTPHeader: header,
	})
	if err != nil {
		if upErr := upstreamFromHandshake(resp, err); upErr != nil {
			return nil, nil, upErr
		}
		return nil, nil, fmt.Errorf("elevenlabs: dial realtime ws: %w", err)
	}
	// Generous read limit: a single base64 audio chunk can be a few hundred KB.
	conn.SetReadLimit(8 * 1024 * 1024)

	bos := bosFrame{
		Text: " ",
		VoiceSettings: bosVoiceSettings{
			Stability:       defaultStability,
			SimilarityBoost: defaultSimilarityBoost,
		},
	}
	bosBytes, _ := json.Marshal(bos)
	if err := conn.Write(ctx, websocket.MessageText, bosBytes); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "bos write failed")
		return nil, nil, fmt.Errorf("elevenlabs: send bos: %w", err)
	}

	if req.Input != "" {
		firstBytes, _ := json.Marshal(textFrame{Text: req.Input})
		if err := conn.Write(ctx, websocket.MessageText, firstBytes); err != nil {
			_ = conn.Close(websocket.StatusInternalError, "first frame write failed")
			return nil, nil, fmt.Errorf("elevenlabs: send first text frame: %w", err)
		}
	}

	stream, sctx, hooks := llmrouter.NewAudioStream(ctx)
	stream.ContentType = realtimeContentType(req.Format)

	rc := &RealtimeContext{
		conn: conn,
		done: make(chan struct{}),
	}
	pumpCtx, pumpCancel := context.WithCancel(sctx)
	rc.cancel = pumpCancel

	go pumpRealtime(pumpCtx, conn, hooks, rc.done)

	return stream, rc, nil
}

// buildRealtimeURL composes the wss:// URL for the /stream-input
// endpoint, attaching model_id, output_format, inactivity_timeout, and
// optimize_streaming_latency query parameters.
func buildRealtimeURL(baseURL string, req llmrouter.SpeechRequest) string {
	voice := req.Voice
	if voice == "" {
		voice = defaultVoiceID
	}
	model := req.Model
	if model == "" {
		model = defaultTTSModel
	}
	outputFormat := elevenLabsFormat(req.Format)

	wsBase := httpToWSScheme(baseURL)
	q := url.Values{}
	q.Set("model_id", model)
	q.Set("output_format", outputFormat)
	q.Set("inactivity_timeout", defaultInactivityTimeoutSeconds)
	q.Set("optimize_streaming_latency", defaultOptimizeStreamingLatency)

	return wsBase + "/v1/text-to-speech/" + voice + "/stream-input?" + q.Encode()
}

// httpToWSScheme rewrites http:// → ws:// and https:// → wss:// so that
// callers can keep using llmrouter.WithBaseURL for both REST and
// WebSocket endpoints. Any scheme outside that pair is returned
// unchanged (the dialer will surface a clear error).
func httpToWSScheme(baseURL string) string {
	switch {
	case strings.HasPrefix(baseURL, "https://"):
		return "wss://" + strings.TrimPrefix(baseURL, "https://")
	case strings.HasPrefix(baseURL, "http://"):
		return "ws://" + strings.TrimPrefix(baseURL, "http://")
	default:
		return baseURL
	}
}

// realtimeContentType maps SpeechRequest.Format to the MIME type the
// AudioStream advertises via ContentType. Distinct from the wire format
// negotiation in elevenLabsFormat — purely informational metadata.
func realtimeContentType(format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "opus":
		return "audio/opus"
	case "pcm", "wav":
		return "audio/pcm"
	case "ulaw":
		return "audio/basic"
	case "", "mp3":
		return "audio/mpeg"
	default:
		return "audio/mpeg"
	}
}

// upstreamFromHandshake produces an *ErrUpstream when the dial failed
// with an HTTP-level rejection (4xx/5xx). Returns nil when the dial
// failure was transport-level — the caller wraps that case separately.
func upstreamFromHandshake(resp *http.Response, dialErr error) *llmrouter.ErrUpstream {
	if resp == nil {
		return nil
	}
	if resp.StatusCode < 400 {
		return nil
	}
	body := ""
	if resp.Body != nil {
		buf, _ := io.ReadAll(io.LimitReader(resp.Body, errorBodySnippetLimit))
		body = strings.TrimSpace(string(buf))
		_ = resp.Body.Close()
	}
	if body == "" {
		body = strings.TrimSpace(dialErr.Error())
	}
	return &llmrouter.ErrUpstream{
		Provider:   providerName,
		StatusCode: resp.StatusCode,
		Body:       body,
	}
}

// pumpRealtime reads server frames, base64-decodes the audio payloads,
// emits AudioChunks, and always calls hooks.Finish exactly once. The
// done channel is closed on return so Finalize can synchronise with the
// final flush.
func pumpRealtime(ctx context.Context, conn *websocket.Conn, hooks llmrouter.AudioProducerHooks, done chan struct{}) {
	defer close(done)

	finishErr := error(nil)
	defer func() {
		hooks.Finish(finishErr)
		// Best-effort close — Finalize/Close may have already shut it down.
		_ = conn.Close(websocket.StatusNormalClosure, "stream finished")
	}()

	for {
		select {
		case <-ctx.Done():
			finishErr = ctx.Err()
			return
		default:
		}

		typ, data, err := conn.Read(ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, io.EOF) {
				return
			}
			// Normal closure is a clean end-of-stream signal.
			if websocket.CloseStatus(err) == websocket.StatusNormalClosure {
				return
			}
			finishErr = fmt.Errorf("elevenlabs: realtime read: %w", err)
			return
		}
		if typ != websocket.MessageText {
			continue
		}

		var frame serverFrame
		if jerr := json.Unmarshal(data, &frame); jerr != nil {
			finishErr = fmt.Errorf("elevenlabs: decode realtime frame: %w", jerr)
			return
		}

		if frame.Audio != "" {
			audio, derr := base64.StdEncoding.DecodeString(frame.Audio)
			if derr != nil {
				finishErr = fmt.Errorf("elevenlabs: base64 decode audio: %w", derr)
				return
			}
			raw := make([]byte, len(data))
			copy(raw, data)
			if !hooks.Send(llmrouter.AudioChunk{Data: audio, Raw: raw}) {
				finishErr = ctx.Err()
				return
			}
		}

		if frame.IsFinal {
			return
		}
	}
}
