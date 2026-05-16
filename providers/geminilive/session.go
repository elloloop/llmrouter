package geminilive

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
	"time"

	"github.com/coder/websocket"
	"github.com/elloloop/llmrouter"
)

// sessionEventBuffer sizes the buffered Events channel. Live audio
// deltas arrive frequently; a small buffer cushions transient consumer
// lag without growing unbounded.
const sessionEventBuffer = 32

// liveReadLimit caps a single inbound frame at 8 MiB. Live audio
// inlineData frames are base64-encoded PCM and can be several hundred
// KB when large chunks accumulate server-side.
const liveReadLimit = 8 * 1024 * 1024

// setupAckTimeout is how long Connect waits for the server's
// setupComplete frame before giving up and tearing down the dial.
const setupAckTimeout = 5 * time.Second

// liveAPIPath is the WebSocket path for the bidirectional Live
// endpoint. The API key is attached as a query parameter (Google's
// choice — there is no auth header).
const liveAPIPath = "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

// defaultInputAudioMime is the wire mime/rate Gemini Live expects for
// incremental microphone PCM frames sent via realtime_input.
const defaultInputAudioMime = "audio/pcm;rate=16000"

// Event Type values delivered on Session.Events(). Documented as
// package-level constants so callers can switch on them without
// re-typing magic strings.
const (
	EventTypeSetupComplete = "setup.complete"
	EventTypeServerText    = "server.text"
	EventTypeServerAudio   = "server.audio"
	EventTypeServerTool    = "server.tool_call"
	EventTypeTurnComplete  = "server.turn_complete"
	EventTypeError         = "error"
	EventTypeRaw           = "raw"
)

// SessionConfig is the per-session configuration applied via the
// initial `setup` message sent immediately after the WebSocket
// handshake. Field names align with openairealtime.SessionConfig where
// the semantics overlap.
type SessionConfig struct {
	// Model identifier (e.g. "models/gemini-2.0-flash-exp"). Empty
	// falls back to defaultModel.
	Model string

	// Voice id (Aoede, Charon, Fenrir, Kore, Puck). Forwarded only
	// when non-empty.
	Voice string

	// Instructions is the system-prompt equivalent applied to the
	// session (maps to system_instruction). Forwarded only when
	// non-empty.
	Instructions string

	// OutputAudioFormat — informational; Gemini Live always returns
	// PCM16 at 24 kHz. Reserved for future parity with
	// openairealtime.SessionConfig.OutputAudioFormat.
	OutputAudioFormat string

	// Modalities advertised to the server (response_modalities).
	// ["AUDIO"] is the implicit default at the server. Common values:
	// ["AUDIO"], ["TEXT"], or both.
	Modalities []string

	// Temperature is the sampling temperature applied to generation.
	// Forwarded only when non-nil.
	Temperature *float64

	// TopP is the nucleus-sampling probability mass. Forwarded only
	// when non-nil.
	TopP *float64

	// Tools advertised to the model for function calling. Mirror the
	// root llmrouter.Tool shape. Live sessions support tool calls.
	Tools []llmrouter.Tool

	// Raw, when non-empty, is merged into the inner setup object
	// AFTER the typed fields. Use it for forward-compat fields the
	// typed API does not yet expose. Top-level keys in Raw override
	// the typed values.
	Raw json.RawMessage
}

// SessionEvent is a translated event from the upstream stream. Every
// inbound server frame is delivered as a SessionEvent so callers can
// introspect any event type, even ones the typed fields do not cover.
type SessionEvent struct {
	// Type is the translated event type. One of the EventType*
	// constants above, or a derived top-level key for pass-through
	// frames.
	Type string

	// Text is populated on server.text with the model's text part.
	Text string

	// AudioDelta is the base64-decoded audio bytes on server.audio.
	// Empty for non-audio events.
	AudioDelta []byte

	// AudioMime is the upstream mime type (e.g. "audio/pcm;rate=24000")
	// associated with AudioDelta. Empty for non-audio events.
	AudioMime string

	// ToolCallID is set on server.tool_call. Echo it back via
	// Session.SendToolResult.
	ToolCallID string

	// ToolName is set on server.tool_call.
	ToolName string

	// ToolArgs is the raw JSON args object on server.tool_call.
	// Gemini sends function-call args as an object — preserved here
	// verbatim so callers can unmarshal into typed structs.
	ToolArgs json.RawMessage

	// Error is populated when Type == "error". The session terminates
	// shortly after this event is delivered.
	Error *llmrouter.ErrUpstream

	// Raw is the original upstream JSON frame, untouched.
	Raw json.RawMessage
}

// Session is a live WebSocket connection to a Gemini Live model.
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

// SendText sends a user turn with text content and marks the turn
// complete via the client_content envelope.
func (s *Session) SendText(ctx context.Context, text string) error {
	payload := clientContentEnvelope{
		ClientContent: clientContent{
			Turns: []turn{{
				Role:  "user",
				Parts: []part{{Text: text}},
			}},
			TurnComplete: true,
		},
	}
	return s.writeJSON(ctx, payload)
}

// SendAudio appends a chunk of raw PCM audio bytes to the realtime
// input buffer. Send raw PCM bytes (16kHz mono PCM_S16LE is the
// recommended input format). Empty buffers are a no-op.
func (s *Session) SendAudio(ctx context.Context, audio []byte) error {
	if len(audio) == 0 {
		return nil
	}
	payload := realtimeInputEnvelope{
		RealtimeInput: realtimeInput{
			MediaChunks: []mediaChunk{{
				MimeType: defaultInputAudioMime,
				Data:     base64.StdEncoding.EncodeToString(audio),
			}},
		},
	}
	return s.writeJSON(ctx, payload)
}

// SendToolResult delivers a function-call response back to the model.
// Pair toolCallID and name with the values from a prior
// server.tool_call event; response carries the JSON result object.
func (s *Session) SendToolResult(ctx context.Context, toolCallID, name string, response json.RawMessage) error {
	payload := toolResponseEnvelope{
		ToolResponse: toolResponse{
			FunctionResponses: []functionResponse{{
				ID:       toolCallID,
				Name:     name,
				Response: response,
			}},
		},
	}
	return s.writeJSON(ctx, payload)
}

// writeJSON marshals v and writes it as a single text frame.
func (s *Session) writeJSON(ctx context.Context, v any) error {
	buf, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("geminilive: marshal event: %w", err)
	}
	return s.writeRaw(ctx, buf)
}

// writeRaw sends a pre-marshalled JSON frame.
func (s *Session) writeRaw(ctx context.Context, payload []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := s.conn.Write(ctx, websocket.MessageText, payload); err != nil {
		return fmt.Errorf("geminilive: write event: %w", err)
	}
	return nil
}

// Connect opens a WebSocket session against the Live endpoint, sends
// the initial setup message derived from cfg, and waits for the
// server's setupComplete acknowledgement before returning. The returned
// Session emits setup.complete as its first event, then begins emitting
// translated server frames on Events().
func (p *Provider) Connect(ctx context.Context, cfg SessionConfig) (*Session, error) {
	dialURL := buildLiveURL(p.cfg.BaseURL, p.cfg.APIKey)

	conn, resp, err := websocket.Dial(ctx, dialURL, &websocket.DialOptions{
		HTTPClient: p.cfg.HTTP(),
	})
	if err != nil {
		if upErr := upstreamFromHandshake(resp, err); upErr != nil {
			return nil, upErr
		}
		return nil, fmt.Errorf("geminilive: dial live ws: %w", err)
	}
	conn.SetReadLimit(liveReadLimit)

	// Send the initial setup message. On any failure tear down the
	// just-opened connection so the caller is not left holding a
	// stale Session.
	setupPayload, err := buildSetup(cfg)
	if err != nil {
		_ = conn.Close(websocket.StatusInternalError, "marshal setup")
		return nil, err
	}
	if err := conn.Write(ctx, websocket.MessageText, setupPayload); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "write setup")
		return nil, fmt.Errorf("geminilive: write initial setup: %w", err)
	}

	// Wait for setupComplete (bounded by setupAckTimeout). Without a
	// successful setup the session is unusable.
	ackCtx, ackCancel := context.WithTimeout(ctx, setupAckTimeout)
	ackRaw, ackErr := readSetupAck(ackCtx, conn)
	ackCancel()
	if ackErr != nil {
		_ = conn.Close(websocket.StatusProtocolError, "no setup ack")
		return nil, ackErr
	}

	pumpCtx, cancel := context.WithCancel(ctx)
	s := &Session{
		conn:   conn,
		cancel: cancel,
		events: make(chan SessionEvent, sessionEventBuffer),
		errMu:  make(chan struct{}),
	}

	// Surface the setupComplete frame as the first SessionEvent so
	// callers see a deterministic start signal before any model output.
	s.events <- SessionEvent{Type: EventTypeSetupComplete, Raw: ackRaw}

	go s.pump(pumpCtx)
	return s, nil
}

// readSetupAck blocks until the server emits a setupComplete frame or
// the context fires. Any other frame received before the ack is
// rejected as a protocol violation.
func readSetupAck(ctx context.Context, conn *websocket.Conn) (json.RawMessage, error) {
	typ, data, err := conn.Read(ctx)
	if err != nil {
		return nil, fmt.Errorf("geminilive: read setup ack: %w", err)
	}
	if typ != websocket.MessageText {
		return nil, fmt.Errorf("geminilive: setup ack frame type = %v, want text", typ)
	}
	var probe struct {
		SetupComplete json.RawMessage `json:"setupComplete"`
		Error         *serverErrField `json:"error"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, fmt.Errorf("geminilive: decode setup ack: %w", err)
	}
	if probe.Error != nil {
		return nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: 0,
			Body:       probe.Error.Message,
		}
	}
	if probe.SetupComplete == nil {
		return nil, fmt.Errorf("geminilive: expected setupComplete frame, got %s", truncate(string(data), 256))
	}
	rawCopy := make(json.RawMessage, len(data))
	copy(rawCopy, data)
	return rawCopy, nil
}

// pump reads server frames, translates each into one or more
// SessionEvents, and always calls finish exactly once before returning.
// The conn is best-effort closed on exit; Close may have already torn
// it down.
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
			finishErr = fmt.Errorf("geminilive: live read: %w", err)
			return
		}
		if typ != websocket.MessageText {
			continue
		}

		events, terminal, terr := translateFrame(data)
		for _, ev := range events {
			if !s.deliver(ctx, ev) {
				finishErr = ctx.Err()
				return
			}
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

// setupEnvelope wraps the initial `setup` payload sent right after the
// upgrade. The server replies with {"setupComplete":{}} when it is
// happy with the configuration.
type setupEnvelope struct {
	Setup setupPayload `json:"setup"`
}

// setupPayload is the inner setup object. Fields use omitempty so
// untouched values stay absent from the wire frame.
type setupPayload struct {
	Model             string            `json:"model"`
	GenerationConfig  *generationConfig `json:"generation_config,omitempty"`
	SystemInstruction *systemInstr      `json:"system_instruction,omitempty"`
	Tools             []geminiTool      `json:"tools,omitempty"`
}

// generationConfig mirrors the Gemini Live shape with the fields this
// package actively translates. Anything else can be merged in via
// SessionConfig.Raw.
type generationConfig struct {
	ResponseModalities []string     `json:"response_modalities,omitempty"`
	SpeechConfig       *speechConfig `json:"speech_config,omitempty"`
	Temperature        *float64      `json:"temperature,omitempty"`
	TopP               *float64      `json:"top_p,omitempty"`
}

// speechConfig nests the voice selection.
type speechConfig struct {
	VoiceConfig *voiceConfig `json:"voice_config,omitempty"`
}

// voiceConfig nests the prebuilt voice id.
type voiceConfig struct {
	PrebuiltVoiceConfig *prebuiltVoiceConfig `json:"prebuilt_voice_config,omitempty"`
}

// prebuiltVoiceConfig carries the voice name.
type prebuiltVoiceConfig struct {
	VoiceName string `json:"voice_name,omitempty"`
}

// systemInstr models the system_instruction object as
// {parts:[{text:"..."}]}.
type systemInstr struct {
	Parts []part `json:"parts"`
}

// geminiTool is the function-declarations shape Gemini Live expects:
// {function_declarations:[{name,description,parameters}]}.
type geminiTool struct {
	FunctionDeclarations []functionDeclaration `json:"function_declarations,omitempty"`
}

// functionDeclaration describes a single callable function.
type functionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// clientContentEnvelope wraps a user-content turn.
type clientContentEnvelope struct {
	ClientContent clientContent `json:"client_content"`
}

// clientContent is the inner shape of a client_content frame.
type clientContent struct {
	Turns        []turn `json:"turns"`
	TurnComplete bool   `json:"turn_complete"`
}

// turn is one role+parts entry in a client_content.turns array.
type turn struct {
	Role  string `json:"role"`
	Parts []part `json:"parts"`
}

// part is a single content fragment. Only one of Text/InlineData is
// populated per part.
type part struct {
	Text       string      `json:"text,omitempty"`
	InlineData *inlineData `json:"inlineData,omitempty"`
}

// inlineData is the base64 audio (or other binary) shape Gemini uses
// in serverContent.modelTurn.parts.
type inlineData struct {
	MimeType string `json:"mimeType,omitempty"`
	Data     string `json:"data,omitempty"`
}

// realtimeInputEnvelope wraps incremental media chunks.
type realtimeInputEnvelope struct {
	RealtimeInput realtimeInput `json:"realtime_input"`
}

// realtimeInput is the inner shape of a realtime_input frame.
type realtimeInput struct {
	MediaChunks []mediaChunk `json:"media_chunks"`
}

// mediaChunk is one base64-encoded chunk of audio (or other media).
type mediaChunk struct {
	MimeType string `json:"mime_type"`
	Data     string `json:"data"`
}

// toolResponseEnvelope wraps the response to a model-issued tool call.
type toolResponseEnvelope struct {
	ToolResponse toolResponse `json:"tool_response"`
}

// toolResponse is the inner shape of a tool_response frame.
type toolResponse struct {
	FunctionResponses []functionResponse `json:"function_responses"`
}

// functionResponse mirrors the request shape Gemini emits for tool
// calls — id is required, name is helpful, response is the result.
type functionResponse struct {
	ID       string          `json:"id"`
	Name     string          `json:"name,omitempty"`
	Response json.RawMessage `json:"response,omitempty"`
}

// serverFrame is the discriminated-union shape of every inbound frame.
// Only the top-level key relevant to the message will be populated.
type serverFrame struct {
	SetupComplete json.RawMessage     `json:"setupComplete"`
	ServerContent *serverContentField `json:"serverContent"`
	ToolCall      *toolCallField      `json:"toolCall"`
	Error         *serverErrField     `json:"error"`
}

// serverContentField is the body of a serverContent frame.
type serverContentField struct {
	ModelTurn    *modelTurn `json:"modelTurn"`
	TurnComplete bool       `json:"turnComplete"`
}

// modelTurn is the assistant turn body inside serverContent.
type modelTurn struct {
	Role  string         `json:"role"`
	Parts []serverPart   `json:"parts"`
}

// serverPart is a content fragment in a modelTurn. Either Text or
// InlineData will be set per fragment.
type serverPart struct {
	Text       string      `json:"text"`
	InlineData *inlineData `json:"inlineData"`
}

// toolCallField is the body of a toolCall frame.
type toolCallField struct {
	FunctionCalls []functionCall `json:"functionCalls"`
}

// functionCall is one declared call inside a toolCall frame.
type functionCall struct {
	ID   string          `json:"id"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

// serverErrField is the inner `error` object of an upstream error.
type serverErrField struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

// --- helpers ------------------------------------------------------------

// buildSetup composes the JSON payload for the initial setup message.
// When cfg.Raw is non-empty, its top-level keys are merged into the
// inner setup object after the typed fields, so Raw wins on conflict.
func buildSetup(cfg SessionConfig) ([]byte, error) {
	model := cfg.Model
	if model == "" {
		model = defaultModel
	}

	env := setupEnvelope{
		Setup: setupPayload{
			Model:             model,
			GenerationConfig:  buildGenerationConfig(cfg),
			SystemInstruction: buildSystemInstruction(cfg.Instructions),
			Tools:             flattenTools(cfg.Tools),
		},
	}

	if len(cfg.Raw) == 0 {
		buf, err := json.Marshal(env)
		if err != nil {
			return nil, fmt.Errorf("geminilive: marshal setup: %w", err)
		}
		return buf, nil
	}

	// Merge Raw into the setup object — Raw keys override typed keys.
	merged := make(map[string]any)
	typedBuf, err := json.Marshal(env.Setup)
	if err != nil {
		return nil, fmt.Errorf("geminilive: marshal typed setup payload: %w", err)
	}
	if err := json.Unmarshal(typedBuf, &merged); err != nil {
		return nil, fmt.Errorf("geminilive: re-decode typed setup payload: %w", err)
	}
	var overlay map[string]any
	if err := json.Unmarshal(cfg.Raw, &overlay); err != nil {
		return nil, fmt.Errorf("geminilive: decode raw setup payload: %w", err)
	}
	for k, v := range overlay {
		merged[k] = v
	}
	out := map[string]any{
		"setup": merged,
	}
	buf, err := json.Marshal(out)
	if err != nil {
		return nil, fmt.Errorf("geminilive: marshal merged setup: %w", err)
	}
	return buf, nil
}

// buildGenerationConfig assembles the generation_config object from
// the typed SessionConfig fields. Returns nil when every field is
// untouched so the outer omitempty kicks in.
func buildGenerationConfig(cfg SessionConfig) *generationConfig {
	gc := &generationConfig{
		ResponseModalities: cfg.Modalities,
		Temperature:        cfg.Temperature,
		TopP:               cfg.TopP,
	}
	if cfg.Voice != "" {
		gc.SpeechConfig = &speechConfig{
			VoiceConfig: &voiceConfig{
				PrebuiltVoiceConfig: &prebuiltVoiceConfig{VoiceName: cfg.Voice},
			},
		}
	}
	if len(gc.ResponseModalities) == 0 && gc.SpeechConfig == nil && gc.Temperature == nil && gc.TopP == nil {
		return nil
	}
	return gc
}

// buildSystemInstruction wraps a plain string as the
// {parts:[{text:"..."}]} shape Gemini Live expects. Empty input yields
// nil so the outer omitempty drops the field entirely.
func buildSystemInstruction(text string) *systemInstr {
	if text == "" {
		return nil
	}
	return &systemInstr{Parts: []part{{Text: text}}}
}

// flattenTools converts root llmrouter.Tool values into the
// function_declarations array Gemini Live expects. Returns nil when
// tools is empty so the caller's omitempty kicks in.
//
// All llmrouter tools are collapsed into a single tool entry whose
// function_declarations holds every call — this matches the most
// common Gemini Live usage pattern.
func flattenTools(tools []llmrouter.Tool) []geminiTool {
	if len(tools) == 0 {
		return nil
	}
	decls := make([]functionDeclaration, 0, len(tools))
	for _, t := range tools {
		decls = append(decls, functionDeclaration{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		})
	}
	return []geminiTool{{FunctionDeclarations: decls}}
}

// buildLiveURL composes the wss:// URL for the BidiGenerateContent
// endpoint with the API key attached as a query parameter. The
// caller's BaseURL may be http(s):// or ws(s):// — both normalise to
// ws(s)://.
func buildLiveURL(baseURL, apiKey string) string {
	wsBase := normaliseWSScheme(strings.TrimRight(baseURL, "/"))
	q := url.Values{}
	q.Set("key", apiKey)
	return wsBase + liveAPIPath + "?" + q.Encode()
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

// translateFrame decodes one inbound text frame into one or more
// SessionEvents. A serverContent frame with N parts produces N events
// (text/audio per part) plus an optional turn-complete event when the
// turnComplete flag is set. A toolCall frame with N calls produces N
// events.
//
// The second return value is true when this frame is terminal (i.e.
// the pump must finish after delivering it); the third return value
// is the terminal error to surface via Err().
//
// Decode failures are surfaced as a terminal error rather than as an
// in-band SessionEvent — a malformed upstream frame indicates a bug
// or a protocol mismatch, and continuing to read makes diagnosis
// harder.
func translateFrame(data []byte) ([]SessionEvent, bool, error) {
	rawCopy := make(json.RawMessage, len(data))
	copy(rawCopy, data)

	var frame serverFrame
	if err := json.Unmarshal(data, &frame); err != nil {
		return []SessionEvent{{Type: EventTypeRaw, Raw: rawCopy}},
			true,
			fmt.Errorf("geminilive: decode server frame: %w", err)
	}

	switch {
	case frame.Error != nil:
		msg := frame.Error.Message
		if msg == "" {
			msg = frame.Error.Status
		}
		if msg == "" {
			msg = string(rawCopy)
		}
		upErr := &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: 0,
			Body:       msg,
		}
		return []SessionEvent{{
			Type:  EventTypeError,
			Error: upErr,
			Raw:   rawCopy,
		}}, true, upErr

	case frame.SetupComplete != nil:
		// A second setupComplete is unusual — pass through so callers
		// can observe it.
		return []SessionEvent{{Type: EventTypeSetupComplete, Raw: rawCopy}}, false, nil

	case frame.ServerContent != nil:
		out := translateServerContent(frame.ServerContent, rawCopy)
		return out, false, nil

	case frame.ToolCall != nil:
		out := translateToolCall(frame.ToolCall, rawCopy)
		return out, false, nil

	default:
		// Pass-through for every other event so callers can introspect
		// raw state. Derive Type from the first top-level key for some
		// usability.
		typ := derivePassThroughType(data)
		return []SessionEvent{{Type: typ, Raw: rawCopy}}, false, nil
	}
}

// translateServerContent fans a serverContent frame out to one event
// per modelTurn.part plus an optional turn-complete event when the
// frame carries turnComplete=true. Every emitted event carries the
// full rawCopy so callers can re-inspect the original frame.
func translateServerContent(sc *serverContentField, rawCopy json.RawMessage) []SessionEvent {
	var events []SessionEvent
	if sc.ModelTurn != nil {
		for _, p := range sc.ModelTurn.Parts {
			if p.InlineData != nil {
				decoded, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
				if err != nil {
					// Skip undecodable audio rather than terminate the
					// session — an isolated bad chunk should not kill
					// the whole conversation. Surface the raw payload.
					events = append(events, SessionEvent{
						Type:      EventTypeServerAudio,
						AudioMime: p.InlineData.MimeType,
						Raw:       rawCopy,
					})
					continue
				}
				events = append(events, SessionEvent{
					Type:       EventTypeServerAudio,
					AudioDelta: decoded,
					AudioMime:  p.InlineData.MimeType,
					Raw:        rawCopy,
				})
				continue
			}
			if p.Text != "" {
				events = append(events, SessionEvent{
					Type: EventTypeServerText,
					Text: p.Text,
					Raw:  rawCopy,
				})
			}
		}
	}
	if sc.TurnComplete {
		events = append(events, SessionEvent{
			Type: EventTypeTurnComplete,
			Raw:  rawCopy,
		})
	}
	return events
}

// translateToolCall fans a toolCall frame out to one server.tool_call
// event per declared functionCall.
func translateToolCall(tc *toolCallField, rawCopy json.RawMessage) []SessionEvent {
	out := make([]SessionEvent, 0, len(tc.FunctionCalls))
	for _, fc := range tc.FunctionCalls {
		out = append(out, SessionEvent{
			Type:       EventTypeServerTool,
			ToolCallID: fc.ID,
			ToolName:   fc.Name,
			ToolArgs:   fc.Args,
			Raw:        rawCopy,
		})
	}
	return out
}

// derivePassThroughType picks a human-readable Type for an inbound
// frame we do not natively recognise. We use the first top-level JSON
// key as a sensible label, defaulting to EventTypeRaw if the payload
// is not a JSON object.
func derivePassThroughType(data []byte) string {
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(data, &probe); err != nil {
		return EventTypeRaw
	}
	for k := range probe {
		return k
	}
	return EventTypeRaw
}

// truncate caps a debug string at n bytes for inclusion in error
// messages. Avoids leaking entire upstream payloads into logs.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
