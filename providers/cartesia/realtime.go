package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"strings"
	"sync"

	"github.com/coder/websocket"
	"github.com/elloloop/llmrouter"
	"github.com/google/uuid"
)

// realtimeWSPath is the Cartesia WebSocket TTS endpoint path.
const realtimeWSPath = "/tts/websocket"

// realtimeReadLimit caps a single inbound frame. Cartesia base64 audio
// chunks can be a few hundred KB; 8 MiB is comfortable headroom.
const realtimeReadLimit = 8 * 1024 * 1024

// realtimeServerFrame is the JSON envelope Cartesia sends back over the
// WebSocket. Data is a base64-encoded audio payload; Done marks the
// terminal frame for the active context.
type realtimeServerFrame struct {
	ContextID string `json:"context_id"`
	Done      bool   `json:"done"`
	Data      string `json:"data"`
}

// RealtimeContext is the handle Cartesia callers use to stream
// additional text into a live TTS context opened via Provider.SpeakRealtime.
// Append/Finalize are safe to call from a single producer goroutine;
// concurrent writes are serialised internally.
type RealtimeContext struct {
	conn      *websocket.Conn
	contextID string
	cancel    context.CancelFunc

	writeMu sync.Mutex

	closeOnce sync.Once
	closeErr  error
}

// Append streams another transcript fragment into the same TTS context.
// Empty strings are no-ops.
func (rc *RealtimeContext) Append(ctx context.Context, transcript string) error {
	if transcript == "" {
		return nil
	}
	frame := map[string]any{
		"context_id": rc.contextID,
		"transcript": transcript,
		"continue":   true,
	}
	return rc.writeJSON(ctx, frame)
}

// Finalize signals the end of input for the context by sending an empty
// transcript with continue:false. The associated AudioStream's Chunks
// channel closes shortly after the server flushes its final frame.
func (rc *RealtimeContext) Finalize(ctx context.Context) error {
	frame := map[string]any{
		"context_id": rc.contextID,
		"transcript": "",
		"continue":   false,
	}
	return rc.writeJSON(ctx, frame)
}

// Close tears down the connection without flushing. Idempotent — repeat
// calls return the same error from the first attempt.
func (rc *RealtimeContext) Close() error {
	rc.closeOnce.Do(func() {
		if rc.cancel != nil {
			rc.cancel()
		}
		rc.closeErr = rc.conn.Close(websocket.StatusNormalClosure, "")
	})
	return rc.closeErr
}

// writeJSON marshals v and writes a single text-type frame, serialising
// concurrent calls under writeMu.
func (rc *RealtimeContext) writeJSON(ctx context.Context, v any) error {
	buf, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("cartesia: marshal realtime frame: %w", err)
	}
	rc.writeMu.Lock()
	defer rc.writeMu.Unlock()
	if err := rc.conn.Write(ctx, websocket.MessageText, buf); err != nil {
		return fmt.Errorf("cartesia: write realtime frame: %w", err)
	}
	return nil
}

// SpeakRealtime opens a WebSocket TTS context against Cartesia's
// /tts/websocket endpoint. Callers append transcript chunks via the
// returned RealtimeContext as they arrive (e.g. from an LLM streaming
// completion) and read audio bytes from the AudioStream concurrently.
//
// The returned values are independent: drain the AudioStream from one
// goroutine while another calls Append/Finalize. Pass an empty
// SpeechRequest.Input when everything will arrive via Append; otherwise
// req.Input is sent as the first transcript frame.
//
// Defaults: Model "sonic-2", Voice defaultVoiceID, Format "pcm".
func (p *Provider) SpeakRealtime(ctx context.Context, req llmrouter.SpeechRequest) (*llmrouter.AudioStream, *RealtimeContext, error) {
	dialURL := buildRealtimeURL(p.cfg.BaseURL, p.cfg.APIKey)

	conn, _, err := websocket.Dial(ctx, dialURL, &websocket.DialOptions{
		HTTPClient: p.cfg.HTTP(),
	})
	if err != nil {
		return nil, nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: 0,
			Body:       err.Error(),
		}
	}
	conn.SetReadLimit(realtimeReadLimit)

	contextID := uuid.NewString()

	initial, err := buildInitialRealtimeFrame(req, contextID)
	if err != nil {
		_ = conn.Close(websocket.StatusInternalError, "build initial frame failed")
		return nil, nil, err
	}
	if err := conn.Write(ctx, websocket.MessageText, initial); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "initial write failed")
		return nil, nil, fmt.Errorf("cartesia: send initial realtime frame: %w", err)
	}

	stream, sctx, hooks := llmrouter.NewAudioStream(ctx)
	stream.ContentType = realtimeContentType(req.Format)

	pumpCtx, pumpCancel := context.WithCancel(sctx)
	rc := &RealtimeContext{
		conn:      conn,
		contextID: contextID,
		cancel:    pumpCancel,
	}

	go pumpRealtime(pumpCtx, conn, hooks)

	return stream, rc, nil
}

// buildRealtimeURL composes the wss:// URL for /tts/websocket, attaching
// api_key and cartesia_version query parameters.
func buildRealtimeURL(baseURL, apiKey string) string {
	wsBase := httpToWSScheme(baseURL)
	q := url.Values{}
	q.Set("api_key", apiKey)
	q.Set("cartesia_version", cartesiaVersion)
	return wsBase + realtimeWSPath + "?" + q.Encode()
}

// httpToWSScheme rewrites http(s):// → ws(s):// so callers can keep using
// llmrouter.WithBaseURL for both REST and WebSocket endpoints. Unknown
// schemes are returned unchanged (the dialer surfaces a clear error).
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

// buildInitialRealtimeFrame builds the first text frame sent over the
// realtime WebSocket. It reuses buildSpeechRequestBody to derive voice,
// model_id, and output_format, then overlays context_id, transcript, and
// continue:true.
func buildInitialRealtimeFrame(req llmrouter.SpeechRequest, contextID string) ([]byte, error) {
	body, err := buildSpeechRequestBody(req)
	if err != nil {
		return nil, err
	}
	var frame map[string]json.RawMessage
	if err := json.Unmarshal(body, &frame); err != nil {
		return nil, fmt.Errorf("cartesia: rebuild realtime frame: %w", err)
	}
	ctxRaw, _ := json.Marshal(contextID)
	frame["context_id"] = ctxRaw
	contRaw, _ := json.Marshal(true)
	frame["continue"] = contRaw
	// transcript already set by buildSpeechRequestBody from req.Input.
	return json.Marshal(frame)
}

// realtimeContentType maps SpeechRequest.Format to the MIME the
// AudioStream advertises via ContentType.
func realtimeContentType(format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "mp3":
		return "audio/mpeg"
	case "wav":
		return "audio/wav"
	case "ulaw":
		return "audio/basic"
	case "pcm", "":
		return "audio/pcm"
	default:
		return "audio/pcm"
	}
}

// pumpRealtime reads server frames, base64-decodes the audio payloads,
// emits AudioChunks, and always calls hooks.Finish exactly once. The
// connection is closed on return.
func pumpRealtime(ctx context.Context, conn *websocket.Conn, hooks llmrouter.AudioProducerHooks) {
	finishErr := error(nil)
	defer func() {
		hooks.Finish(finishErr)
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
			if websocket.CloseStatus(err) == websocket.StatusNormalClosure {
				return
			}
			finishErr = fmt.Errorf("cartesia: realtime read: %w", err)
			return
		}
		if typ != websocket.MessageText {
			continue
		}

		var frame realtimeServerFrame
		if jerr := json.Unmarshal(data, &frame); jerr != nil {
			finishErr = fmt.Errorf("cartesia: decode realtime frame: %w", jerr)
			return
		}

		if frame.Data != "" {
			audio, derr := base64.StdEncoding.DecodeString(frame.Data)
			if derr != nil {
				finishErr = fmt.Errorf("cartesia: base64 decode audio: %w", derr)
				return
			}
			raw := make([]byte, len(data))
			copy(raw, data)
			if !hooks.Send(llmrouter.AudioChunk{Data: audio, Raw: raw}) {
				finishErr = ctx.Err()
				return
			}
		}

		if frame.Done {
			return
		}
	}
}
