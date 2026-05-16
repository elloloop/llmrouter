package openairealtime

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

// sessionEventBuffer sizes the buffered Events channel. Realtime audio
// deltas arrive frequently; a small buffer cushions transient consumer
// lag without growing unbounded.
const sessionEventBuffer = 32

// realtimeReadLimit caps a single inbound frame at 8 MiB. Realtime
// audio.delta frames are base64-encoded PCM and can be several hundred
// KB when large chunks accumulate server-side.
const realtimeReadLimit = 8 * 1024 * 1024

// SessionConfig is the per-session configuration applied via an initial
// session.update event immediately after the WebSocket handshake. All
// fields are optional; Modalities defaults to ["text","audio"] and the
// audio formats default to "pcm16" when both Modalities include "audio"
// and the caller leaves them empty.
type SessionConfig struct {
	// Model identifier (e.g. "gpt-4o-realtime-preview"). Empty falls
	// back to defaultModel.
	Model string

	// Voice id (alloy/echo/fable/onyx/nova/shimmer/coral/verse).
	// Forwarded only when non-empty.
	Voice string

	// Instructions is the system-prompt equivalent applied to the
	// session. Forwarded only when non-empty.
	Instructions string

	// InputAudioFormat — pcm16 (default) / g711_ulaw / g711_alaw.
	// Forwarded only when non-empty.
	InputAudioFormat string

	// OutputAudioFormat — pcm16 (default) / g711_ulaw / g711_alaw.
	// Forwarded only when non-empty.
	OutputAudioFormat string

	// Modalities advertised to the server. ["text","audio"] is the
	// implicit default; pass ["text"] to disable audio output.
	Modalities []string

	// Temperature is the sampling temperature applied to generation.
	// Forwarded only when non-nil.
	Temperature *float64

	// Raw, when non-empty, is merged into the session.update payload
	// AFTER the typed fields. Use it for forward-compat fields the
	// typed API does not yet expose (e.g. turn_detection,
	// input_audio_transcription). Top-level keys in Raw override the
	// typed values.
	Raw json.RawMessage
}

// SessionEvent is a translated event from the upstream stream. Every
// inbound server frame is delivered as a SessionEvent so callers can
// introspect any event type, even ones the typed fields do not cover.
type SessionEvent struct {
	// Type is the upstream event type verbatim:
	// "session.created" | "session.updated" |
	// "response.text.delta" | "response.audio.delta" |
	// "response.audio.done" | "response.done" | "error" | ...
	Type string

	// Text is populated on response.text.delta with the delta string.
	Text string

	// AudioDelta is the base64-decoded audio bytes on
	// response.audio.delta. Empty for non-audio events.
	AudioDelta []byte

	// ResponseID is the upstream response_id when the event carries
	// one (response.* events). Empty otherwise.
	ResponseID string

	// Error is populated when Type == "error". The session terminates
	// shortly after this event is delivered.
	Error *llmrouter.ErrUpstream

	// Raw is the original upstream JSON frame, untouched.
	Raw json.RawMessage
}

// Session is a live WebSocket connection to a Realtime model.
//
// Events() is single-consumer — exactly one goroutine should range over
// it. The channel closes when the upstream terminates the stream, the
// context is cancelled, or Close is called. Err() returns the terminal
// error (nil on clean close) and may be called after the channel drains.
//
// Send* methods are safe for concurrent use; writes are serialised
// internally.
type Session struct {
	conn   *websocket.Conn
	cancel context.CancelFunc

	events chan SessionEvent
	errMu  chan struct{}
	err    error

	writeMu sync.Mutex

	closeOnce sync.Once
	closeErr  error
}

// Events returns the receive-only event channel. Closes after the
// pump goroutine finishes; Err() then returns the terminal error.
func (s *Session) Events() <-chan SessionEvent { return s.events }

// Err blocks until the pump finishes and returns the terminal error
// (nil on clean close). Safe to call multiple times — returns the same
// value.
func (s *Session) Err() error {
	<-s.errMu
	return s.err
}

// Close terminates the session. Idempotent — subsequent calls return
// the same error captured on the first invocation.
func (s *Session) Close() error {
	s.closeOnce.Do(func() {
		if s.cancel != nil {
			s.cancel()
		}
		s.closeErr = s.conn.Close(websocket.StatusNormalClosure, "client closed")
	})
	return s.closeErr
}

// SendText queues a user text message. It writes two frames in order:
//  1. conversation.item.create with role=user, content=[{type:input_text,...}]
//  2. response.create — triggers the model to generate a response.
//
// Use this for text-in / audio-out and text-in / text-out flows.
func (s *Session) SendText(ctx context.Context, text string) error {
	item := conversationItemCreate{
		Type: "conversation.item.create",
		Item: conversationItem{
			Type: "message",
			Role: "user",
			Content: []conversationContent{{
				Type: "input_text",
				Text: text,
			}},
		},
	}
	if err := s.writeJSON(ctx, item); err != nil {
		return err
	}
	return s.writeJSON(ctx, simpleEvent{Type: "response.create"})
}

// SendAudio appends raw audio bytes to the server-side input buffer.
// The bytes are base64-encoded on the wire per the OpenAI Realtime
// protocol. Audio in must match SessionConfig.InputAudioFormat.
// Follow up with Commit + CreateResponse to flush + generate.
func (s *Session) SendAudio(ctx context.Context, audio []byte) error {
	if len(audio) == 0 {
		return nil
	}
	encoded := base64.StdEncoding.EncodeToString(audio)
	return s.writeJSON(ctx, audioAppendEvent{
		Type:  "input_audio_buffer.append",
		Audio: encoded,
	})
}

// Commit finalizes the current input audio buffer. After this the
// server treats the accumulated bytes as a complete user turn.
func (s *Session) Commit(ctx context.Context) error {
	return s.writeJSON(ctx, simpleEvent{Type: "input_audio_buffer.commit"})
}

// CreateResponse asks the model to generate a response now. Used after
// Commit to drive audio-in / audio-out turns; SendText already emits
// this implicitly.
func (s *Session) CreateResponse(ctx context.Context) error {
	return s.writeJSON(ctx, simpleEvent{Type: "response.create"})
}

// UpdateSession changes session config mid-stream by emitting a fresh
// session.update event. Only the fields the caller populates are sent;
// the server merges them with the existing session state.
func (s *Session) UpdateSession(ctx context.Context, cfg SessionConfig) error {
	payload, err := buildSessionUpdate(cfg)
	if err != nil {
		return err
	}
	return s.writeRaw(ctx, payload)
}

// writeJSON marshals v and writes it as a single text frame.
func (s *Session) writeJSON(ctx context.Context, v any) error {
	buf, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("openairealtime: marshal event: %w", err)
	}
	return s.writeRaw(ctx, buf)
}

// writeRaw sends a pre-marshalled JSON frame.
func (s *Session) writeRaw(ctx context.Context, payload []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := s.conn.Write(ctx, websocket.MessageText, payload); err != nil {
		return fmt.Errorf("openairealtime: write event: %w", err)
	}
	return nil
}

// Connect opens a WebSocket session against the Realtime endpoint and
// sends the initial session.update derived from cfg. The returned
// Session begins emitting SessionEvents on Events() immediately.
func (p *Provider) Connect(ctx context.Context, cfg SessionConfig) (*Session, error) {
	model := cfg.Model
	if model == "" {
		model = defaultModel
	}

	dialURL := buildRealtimeURL(p.cfg.BaseURL, model)

	header := http.Header{}
	header.Set("Authorization", "Bearer "+p.cfg.APIKey)
	header.Set("OpenAI-Beta", betaHeader)

	conn, resp, err := websocket.Dial(ctx, dialURL, &websocket.DialOptions{
		HTTPClient: p.cfg.HTTP(),
		HTTPHeader: header,
		// Realtime requires a sub-protocol on the upgrade. coder/websocket
		// negotiates the default automatically when none is requested.
	})
	if err != nil {
		if upErr := upstreamFromHandshake(resp, err); upErr != nil {
			return nil, upErr
		}
		return nil, fmt.Errorf("openairealtime: dial realtime ws: %w", err)
	}
	conn.SetReadLimit(realtimeReadLimit)

	// Send initial session.update. If marshalling/writing fails we tear
	// down the just-opened connection so the caller is not left holding
	// a stale Session.
	initPayload, err := buildSessionUpdate(cfg)
	if err != nil {
		_ = conn.Close(websocket.StatusInternalError, "marshal session.update")
		return nil, err
	}
	if err := conn.Write(ctx, websocket.MessageText, initPayload); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "write session.update")
		return nil, fmt.Errorf("openairealtime: write initial session.update: %w", err)
	}

	pumpCtx, cancel := context.WithCancel(ctx)
	s := &Session{
		conn:   conn,
		cancel: cancel,
		events: make(chan SessionEvent, sessionEventBuffer),
		errMu:  make(chan struct{}),
	}

	go s.pump(pumpCtx)
	return s, nil
}

// pump reads server frames, translates each into a SessionEvent, and
// always calls finish exactly once before returning. The conn is best-
// effort closed on exit; Close may have already torn it down.
func (s *Session) pump(ctx context.Context) {
	var finishErr error
	defer func() {
		s.err = finishErr
		close(s.events)
		close(s.errMu)
		_ = s.conn.Close(websocket.StatusNormalClosure, "stream finished")
	}()

	for {
		select {
		case <-ctx.Done():
			finishErr = ctx.Err()
			return
		default:
		}

		typ, data, err := s.conn.Read(ctx)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, io.EOF) {
				return
			}
			if websocket.CloseStatus(err) == websocket.StatusNormalClosure {
				return
			}
			finishErr = fmt.Errorf("openairealtime: realtime read: %w", err)
			return
		}
		if typ != websocket.MessageText {
			continue
		}

		event, terminal, terr := translateFrame(data)
		if !s.deliver(ctx, event) {
			finishErr = ctx.Err()
			return
		}
		if terminal {
			finishErr = terr
			return
		}
	}
}

// deliver sends ev on the events channel, honouring ctx cancellation.
// Returns false if the context fired first — the caller must then exit
// with a context error.
func (s *Session) deliver(ctx context.Context, ev SessionEvent) bool {
	select {
	case s.events <- ev:
		return true
	case <-ctx.Done():
		return false
	}
}

// --- wire types ---------------------------------------------------------

// simpleEvent carries an event whose only field is `type`. Used for
// input_audio_buffer.commit and response.create.
type simpleEvent struct {
	Type string `json:"type"`
}

// audioAppendEvent is the input_audio_buffer.append wire shape.
type audioAppendEvent struct {
	Type  string `json:"type"`
	Audio string `json:"audio"`
}

// conversationItemCreate is the conversation.item.create wire shape.
type conversationItemCreate struct {
	Type string           `json:"type"`
	Item conversationItem `json:"item"`
}

// conversationItem describes a single conversation entry.
type conversationItem struct {
	Type    string                `json:"type"`
	Role    string                `json:"role"`
	Content []conversationContent `json:"content"`
}

// conversationContent is one content fragment within an item.
type conversationContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// sessionUpdateEnvelope is the typed shape of a session.update event.
// Optional fields use omitempty so the server treats them as untouched.
type sessionUpdateEnvelope struct {
	Type    string         `json:"type"`
	Session sessionPayload `json:"session"`
}

// sessionPayload is the inner "session" object of session.update.
type sessionPayload struct {
	Modalities        []string `json:"modalities,omitempty"`
	Voice             string   `json:"voice,omitempty"`
	Instructions      string   `json:"instructions,omitempty"`
	InputAudioFormat  string   `json:"input_audio_format,omitempty"`
	OutputAudioFormat string   `json:"output_audio_format,omitempty"`
	Temperature       *float64 `json:"temperature,omitempty"`
}

// serverFrame is the minimum shape needed to dispatch inbound events.
// All other fields are accessed via the raw json.RawMessage on the
// SessionEvent.
type serverFrame struct {
	Type       string          `json:"type"`
	Delta      string          `json:"delta"`
	ResponseID string          `json:"response_id"`
	Error      *serverErrField `json:"error"`
}

// serverErrField is the inner `error` object of an upstream error event.
type serverErrField struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

// --- helpers ------------------------------------------------------------

// buildSessionUpdate composes the JSON payload for a session.update
// event. When cfg.Raw is non-empty, its top-level keys are merged into
// the inner session object after the typed fields, so Raw wins on
// conflict.
func buildSessionUpdate(cfg SessionConfig) ([]byte, error) {
	env := sessionUpdateEnvelope{
		Type: "session.update",
		Session: sessionPayload{
			Modalities:        cfg.Modalities,
			Voice:             cfg.Voice,
			Instructions:      cfg.Instructions,
			InputAudioFormat:  cfg.InputAudioFormat,
			OutputAudioFormat: cfg.OutputAudioFormat,
			Temperature:       cfg.Temperature,
		},
	}
	if len(cfg.Raw) == 0 {
		buf, err := json.Marshal(env)
		if err != nil {
			return nil, fmt.Errorf("openairealtime: marshal session.update: %w", err)
		}
		return buf, nil
	}

	// Merge Raw into the session object — Raw keys override typed keys.
	merged := make(map[string]any)
	typedBuf, err := json.Marshal(env.Session)
	if err != nil {
		return nil, fmt.Errorf("openairealtime: marshal typed session payload: %w", err)
	}
	if err := json.Unmarshal(typedBuf, &merged); err != nil {
		return nil, fmt.Errorf("openairealtime: re-decode typed session payload: %w", err)
	}
	var overlay map[string]any
	if err := json.Unmarshal(cfg.Raw, &overlay); err != nil {
		return nil, fmt.Errorf("openairealtime: decode raw session payload: %w", err)
	}
	for k, v := range overlay {
		merged[k] = v
	}
	out := map[string]any{
		"type":    "session.update",
		"session": merged,
	}
	buf, err := json.Marshal(out)
	if err != nil {
		return nil, fmt.Errorf("openairealtime: marshal merged session.update: %w", err)
	}
	return buf, nil
}

// buildRealtimeURL composes the wss:// URL for the /realtime endpoint
// with the model query parameter attached. The caller's BaseURL may be
// http(s):// or ws(s):// — both normalise to ws(s)://.
func buildRealtimeURL(baseURL, model string) string {
	wsBase := normaliseWSScheme(strings.TrimRight(baseURL, "/"))
	q := url.Values{}
	q.Set("model", model)
	return wsBase + "/realtime?" + q.Encode()
}

// normaliseWSScheme rewrites http:// → ws:// and https:// → wss:// so
// that callers can keep using llmrouter.WithBaseURL for both REST and
// WebSocket endpoints. ws:// and wss:// are returned unchanged.
func normaliseWSScheme(baseURL string) string {
	switch {
	case strings.HasPrefix(baseURL, "https://"):
		return "wss://" + strings.TrimPrefix(baseURL, "https://")
	case strings.HasPrefix(baseURL, "http://"):
		return "ws://" + strings.TrimPrefix(baseURL, "http://")
	default:
		return baseURL
	}
}

// upstreamFromHandshake produces an *ErrUpstream when the dial failed
// with an HTTP-level rejection (>=400). Returns nil when the dial
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
	if body == "" && dialErr != nil {
		body = strings.TrimSpace(dialErr.Error())
	}
	return &llmrouter.ErrUpstream{
		Provider:   providerName,
		StatusCode: resp.StatusCode,
		Body:       body,
	}
}

// translateFrame decodes one inbound text frame into a SessionEvent.
// The second return value is true when this event is terminal (i.e.
// the pump must finish after delivering it); the third return value
// is the terminal error to surface via Err().
//
// Decode failures are surfaced as a terminal error rather than as an
// in-band SessionEvent — a malformed upstream frame indicates a bug or
// a protocol mismatch, and continuing to read makes diagnosis harder.
func translateFrame(data []byte) (SessionEvent, bool, error) {
	rawCopy := make(json.RawMessage, len(data))
	copy(rawCopy, data)

	var frame serverFrame
	if err := json.Unmarshal(data, &frame); err != nil {
		// Surface the malformed frame as a terminal "raw" event so the
		// caller still sees the bytes that broke us.
		return SessionEvent{Type: "raw", Raw: rawCopy},
			true,
			fmt.Errorf("openairealtime: decode server frame: %w", err)
	}

	switch frame.Type {
	case "response.text.delta":
		return SessionEvent{
			Type:       frame.Type,
			Text:       frame.Delta,
			ResponseID: frame.ResponseID,
			Raw:        rawCopy,
		}, false, nil

	case "response.audio.delta":
		audio, derr := base64.StdEncoding.DecodeString(frame.Delta)
		if derr != nil {
			return SessionEvent{Type: frame.Type, Raw: rawCopy},
				true,
				fmt.Errorf("openairealtime: base64 decode audio delta: %w", derr)
		}
		return SessionEvent{
			Type:       frame.Type,
			AudioDelta: audio,
			ResponseID: frame.ResponseID,
			Raw:        rawCopy,
		}, false, nil

	case "response.done":
		return SessionEvent{
			Type:       frame.Type,
			ResponseID: frame.ResponseID,
			Raw:        rawCopy,
		}, false, nil

	case "error":
		msg := ""
		if frame.Error != nil {
			msg = frame.Error.Message
			if msg == "" {
				msg = frame.Error.Code
			}
			if msg == "" {
				msg = frame.Error.Type
			}
		}
		if msg == "" {
			msg = string(rawCopy)
		}
		upErr := &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: 0,
			Body:       msg,
		}
		return SessionEvent{
			Type:  "error",
			Error: upErr,
			Raw:   rawCopy,
		}, true, upErr

	default:
		// Pass-through for every other event type so callers can
		// introspect raw state (session.created, response.created,
		// response.audio.done, rate_limits.updated, ...).
		return SessionEvent{
			Type:       frame.Type,
			ResponseID: frame.ResponseID,
			Raw:        rawCopy,
		}, false, nil
	}
}
