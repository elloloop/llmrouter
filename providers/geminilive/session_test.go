package geminilive_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/geminilive"
)

// scriptedFrame is one server frame the fake will emit after the
// initial setup is observed. After: optional pause inserted before
// emission.
type scriptedFrame struct {
	payload string
	after   time.Duration
}

// serverOpts configures the fake Live server.
type serverOpts struct {
	// rejectHandshake, when non-zero, returns this HTTP status during
	// the websocket upgrade instead of accepting.
	rejectHandshake int
	// rejectBody is the response body for a rejected handshake.
	rejectBody string
	// skipSetupAck, when true, makes the server accept the upgrade and
	// consume the setup message but never reply with setupComplete.
	// Used to exercise the Connect timeout path.
	skipSetupAck bool
	// holdOpen keeps the connection alive after the script completes
	// instead of closing it, so client-driven Close paths can be
	// exercised.
	holdOpen bool
}

// serverCapture records everything the fake server observed during a
// single session.
type serverCapture struct {
	mu             sync.Mutex
	upgradeHeaders http.Header
	upgradeURL     string
	textFrames     []string
}

func (c *serverCapture) addFrame(s string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.textFrames = append(c.textFrames, s)
}

func (c *serverCapture) frames() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]string, len(c.textFrames))
	copy(out, c.textFrames)
	return out
}

func (c *serverCapture) headers() http.Header {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.upgradeHeaders.Clone()
}

func (c *serverCapture) url() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.upgradeURL
}

// defaultSetupAck is what the fake server replies with after observing
// the client's setup frame. Mirrors the real Gemini Live ack shape.
const defaultSetupAck = `{"setupComplete":{}}`

// newFakeServer returns a fake Gemini Live websocket server, a capture
// struct, a Provider pointed at it, and a done channel that closes
// when the handler returns.
//
// The handler reads the initial setup frame, replies with
// setupComplete, then emits each scripted frame in order. It always
// also drains any further inbound frames into capture.textFrames.
func newFakeServer(t *testing.T, opts serverOpts, script []scriptedFrame) (*geminilive.Provider, *serverCapture, *httptest.Server, <-chan struct{}) {
	t.Helper()
	cap := &serverCapture{}
	done := make(chan struct{})

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.mu.Lock()
		cap.upgradeHeaders = r.Header.Clone()
		cap.upgradeURL = r.URL.String()
		cap.mu.Unlock()

		if opts.rejectHandshake != 0 {
			w.WriteHeader(opts.rejectHandshake)
			if opts.rejectBody != "" {
				_, _ = w.Write([]byte(opts.rejectBody))
			}
			close(done)
			return
		}

		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			InsecureSkipVerify: true,
		})
		if err != nil {
			t.Logf("accept: %v", err)
			close(done)
			return
		}
		conn.SetReadLimit(-1)
		ctx := r.Context()

		// First: consume the client's setup frame synchronously so we
		// can ack before the client times out.
		typ, payload, err := conn.Read(ctx)
		if err != nil {
			_ = conn.Close(websocket.StatusInternalError, "no setup")
			close(done)
			return
		}
		if typ == websocket.MessageText {
			cap.addFrame(string(payload))
		}

		if !opts.skipSetupAck {
			if err := conn.Write(ctx, websocket.MessageText, []byte(defaultSetupAck)); err != nil {
				_ = conn.Close(websocket.StatusInternalError, "write ack")
				close(done)
				return
			}
		}

		// Reader goroutine — captures everything else the client sends.
		readerDone := make(chan struct{})
		go func() {
			defer close(readerDone)
			for {
				typ, payload, err := conn.Read(ctx)
				if err != nil {
					return
				}
				if typ == websocket.MessageText {
					cap.addFrame(string(payload))
				}
			}
		}()

		// Send scripted frames.
		for _, f := range script {
			if f.after > 0 {
				select {
				case <-time.After(f.after):
				case <-ctx.Done():
					_ = conn.Close(websocket.StatusNormalClosure, "ctx done")
					<-readerDone
					close(done)
					return
				}
			}
			if err := conn.Write(ctx, websocket.MessageText, []byte(f.payload)); err != nil {
				_ = conn.Close(websocket.StatusInternalError, "write")
				<-readerDone
				close(done)
				return
			}
		}

		if opts.holdOpen {
			select {
			case <-ctx.Done():
			case <-readerDone:
			}
		}

		_ = conn.Close(websocket.StatusNormalClosure, "script complete")
		<-readerDone
		close(done)
	})

	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)

	p, err := geminilive.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p, cap, srv, done
}

// drainEvents reads until the channel closes or timeout fires.
func drainEvents(t *testing.T, sess *geminilive.Session, timeout time.Duration) []geminilive.SessionEvent {
	t.Helper()
	var out []geminilive.SessionEvent
	deadline := time.After(timeout)
	for {
		select {
		case ev, ok := <-sess.Events():
			if !ok {
				return out
			}
			out = append(out, ev)
		case <-deadline:
			t.Fatalf("drainEvents: timeout after %d events", len(out))
		}
	}
}

// waitForFrames polls cap.frames() until at least n frames are present
// or the deadline fires. Returns the snapshot.
func waitForFrames(t *testing.T, cap *serverCapture, n int, timeout time.Duration) []string {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		f := cap.frames()
		if len(f) >= n {
			return f
		}
		if time.Now().After(deadline) {
			t.Fatalf("waitForFrames: only got %d/%d frames; latest=%v", len(f), n, f)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// ---------------------------------------------------------------------------
// Connect — happy path + setup translation
// ---------------------------------------------------------------------------

func TestConnect_HappyPath(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{
		Voice:        "Aoede",
		Instructions: "you are a test bot",
	})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	// First event must be setup.complete.
	select {
	case ev := <-sess.Events():
		if ev.Type != geminilive.EventTypeSetupComplete {
			t.Errorf("first event type = %q, want setup.complete", ev.Type)
		}
		if len(ev.Raw) == 0 {
			t.Error("Raw payload missing on setup.complete")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no setup.complete event delivered")
	}

	// Confirm setup was the first frame the server saw.
	frames := waitForFrames(t, cap, 1, 2*time.Second)
	var first map[string]any
	if err := json.Unmarshal([]byte(frames[0]), &first); err != nil {
		t.Fatalf("decode first frame: %v", err)
	}
	setup, ok := first["setup"].(map[string]any)
	if !ok {
		t.Fatalf("first frame missing setup object: %v", first)
	}
	if setup["model"] != "models/gemini-2.0-flash-exp" {
		t.Errorf("setup.model = %v, want models/gemini-2.0-flash-exp", setup["model"])
	}
	sysInstr, _ := setup["system_instruction"].(map[string]any)
	if sysInstr == nil {
		t.Fatal("system_instruction missing")
	}
	parts, _ := sysInstr["parts"].([]any)
	if len(parts) != 1 {
		t.Fatalf("system_instruction.parts len = %d, want 1", len(parts))
	}
	p0, _ := parts[0].(map[string]any)
	if p0["text"] != "you are a test bot" {
		t.Errorf("system_instruction.parts[0].text mismatch: %v", p0["text"])
	}

	gc, _ := setup["generation_config"].(map[string]any)
	if gc == nil {
		t.Fatal("generation_config missing — Voice should populate it")
	}
	sc, _ := gc["speech_config"].(map[string]any)
	vc, _ := sc["voice_config"].(map[string]any)
	pvc, _ := vc["prebuilt_voice_config"].(map[string]any)
	if pvc["voice_name"] != "Aoede" {
		t.Errorf("voice_name = %v, want Aoede", pvc["voice_name"])
	}
}

func TestConnect_APIKeyInQueryStringNotHeader(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	// API key must be in the query string, NOT in any header.
	if !strings.Contains(cap.url(), "key=test-key") {
		t.Errorf("upgrade URL %q missing key= query param", cap.url())
	}
	hdrs := cap.headers()
	if got := hdrs.Get("Authorization"); got != "" {
		t.Errorf("Authorization header unexpectedly set: %q", got)
	}
	if got := hdrs.Get("X-Goog-Api-Key"); got != "" {
		t.Errorf("X-Goog-Api-Key header unexpectedly set: %q", got)
	}
}

func TestConnect_LiveEndpointPath(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	u := cap.url()
	if !strings.HasPrefix(u, "/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent") {
		t.Errorf("upgrade URL %q does not start with bidi path", u)
	}
}

func TestConnect_HandshakeRejection(t *testing.T) {
	p, _, _, done := newFakeServer(t, serverOpts{
		rejectHandshake: http.StatusUnauthorized,
		rejectBody:      `{"error":{"message":"bad key"}}`,
	}, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var upErr *llmrouter.ErrUpstream
	if !errors.As(err, &upErr) {
		t.Fatalf("err = %v, want *ErrUpstream", err)
	}
	if upErr.StatusCode != http.StatusUnauthorized {
		t.Errorf("StatusCode = %d, want 401", upErr.StatusCode)
	}
	if upErr.Provider != "geminilive" {
		t.Errorf("Provider = %q, want geminilive", upErr.Provider)
	}
	if !strings.Contains(upErr.Body, "bad key") {
		t.Errorf("Body = %q, want substring 'bad key'", upErr.Body)
	}
	<-done
}

func TestConnect_HandshakeForbidden(t *testing.T) {
	p, _, _, done := newFakeServer(t, serverOpts{
		rejectHandshake: http.StatusForbidden,
		rejectBody:      `denied`,
	}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err == nil {
		t.Fatal("expected error")
	}
	var upErr *llmrouter.ErrUpstream
	if !errors.As(err, &upErr) {
		t.Fatalf("want *ErrUpstream, got %T: %v", err, err)
	}
	if upErr.StatusCode != http.StatusForbidden {
		t.Errorf("StatusCode = %d, want 403", upErr.StatusCode)
	}
	<-done
}

func TestConnect_DefaultModelApplied(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	frames := waitForFrames(t, cap, 1, 2*time.Second)
	if !strings.Contains(frames[0], `"model":"models/gemini-2.0-flash-exp"`) {
		t.Errorf("default model missing from setup: %s", frames[0])
	}
}

func TestConnect_CustomModelApplied(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{Model: "models/gemini-2.5-pro"})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	frames := waitForFrames(t, cap, 1, 2*time.Second)
	if !strings.Contains(frames[0], `"model":"models/gemini-2.5-pro"`) {
		t.Errorf("custom model missing from setup: %s", frames[0])
	}
}

func TestConnect_ModalitiesEmittedInGenerationConfig(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{
		Modalities: []string{"AUDIO", "TEXT"},
	})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()
	frames := waitForFrames(t, cap, 1, 2*time.Second)
	var env map[string]any
	_ = json.Unmarshal([]byte(frames[0]), &env)
	setup, _ := env["setup"].(map[string]any)
	gc, _ := setup["generation_config"].(map[string]any)
	mods, _ := gc["response_modalities"].([]any)
	if len(mods) != 2 || mods[0] != "AUDIO" || mods[1] != "TEXT" {
		t.Errorf("response_modalities = %v, want [AUDIO TEXT]", mods)
	}
}

func TestConnect_TemperatureAndTopPEmitted(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	temp := 0.4
	topP := 0.9
	sess, err := p.Connect(ctx, geminilive.SessionConfig{
		Temperature: &temp,
		TopP:        &topP,
	})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()
	frames := waitForFrames(t, cap, 1, 2*time.Second)
	var env map[string]any
	_ = json.Unmarshal([]byte(frames[0]), &env)
	setup, _ := env["setup"].(map[string]any)
	gc, _ := setup["generation_config"].(map[string]any)
	if gc["temperature"].(float64) != 0.4 {
		t.Errorf("temperature = %v, want 0.4", gc["temperature"])
	}
	if gc["top_p"].(float64) != 0.9 {
		t.Errorf("top_p = %v, want 0.9", gc["top_p"])
	}
}

func TestConnect_GenerationConfigOmittedWhenAllZero(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()
	frames := waitForFrames(t, cap, 1, 2*time.Second)
	var env map[string]any
	_ = json.Unmarshal([]byte(frames[0]), &env)
	setup, _ := env["setup"].(map[string]any)
	if _, present := setup["generation_config"]; present {
		t.Errorf("generation_config should be omitted when no fields set: %v", setup["generation_config"])
	}
	if _, present := setup["system_instruction"]; present {
		t.Errorf("system_instruction should be omitted when empty")
	}
	if _, present := setup["tools"]; present {
		t.Errorf("tools should be omitted when empty")
	}
}

func TestConnect_InstructionsOnlyPopulatesSystemInstruction(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{Instructions: "hi"})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()
	frames := waitForFrames(t, cap, 1, 2*time.Second)
	if !strings.Contains(frames[0], `"system_instruction":{"parts":[{"text":"hi"}]}`) {
		t.Errorf("system_instruction misformatted: %s", frames[0])
	}
}

func TestConnect_VoiceOnlyPopulatesSpeechConfig(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{Voice: "Puck"})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()
	frames := waitForFrames(t, cap, 1, 2*time.Second)
	if !strings.Contains(frames[0], `"voice_name":"Puck"`) {
		t.Errorf("voice_name missing or wrong: %s", frames[0])
	}
}

func TestConnect_SetupAckTimeoutFails(t *testing.T) {
	p, _, _, done := newFakeServer(t, serverOpts{skipSetupAck: true, holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()

	_, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err == nil {
		t.Fatal("expected setup-ack timeout error, got nil")
	}
	if !strings.Contains(err.Error(), "geminilive") {
		t.Errorf("err = %v, want geminilive-prefixed error", err)
	}
	cancel()
	<-done
}

func TestConnect_SetupErrorBeforeAck(t *testing.T) {
	// Server replies with an error envelope instead of setupComplete.
	p, _, _, done := newFakeServer(t, serverOpts{skipSetupAck: true}, []scriptedFrame{
		{payload: `{"error":{"code":401,"message":"missing api key","status":"UNAUTHENTICATED"}}`},
	})
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	_, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err == nil {
		t.Fatal("expected error from setup-time error frame")
	}
	var upErr *llmrouter.ErrUpstream
	if !errors.As(err, &upErr) {
		t.Fatalf("err = %v, want *ErrUpstream", err)
	}
	if !strings.Contains(upErr.Body, "missing api key") {
		t.Errorf("Body = %q", upErr.Body)
	}
	<-done
}

// ---------------------------------------------------------------------------
// SendText / SendAudio / SendToolResult
// ---------------------------------------------------------------------------

func TestSendText_EmitsClientContent(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	if err := sess.SendText(ctx, "hello"); err != nil {
		t.Fatalf("SendText: %v", err)
	}

	frames := waitForFrames(t, cap, 2, 2*time.Second)
	var second map[string]any
	if err := json.Unmarshal([]byte(frames[1]), &second); err != nil {
		t.Fatalf("decode second frame: %v", err)
	}
	cc, _ := second["client_content"].(map[string]any)
	if cc == nil {
		t.Fatalf("client_content missing: %v", second)
	}
	if cc["turn_complete"] != true {
		t.Errorf("turn_complete = %v, want true", cc["turn_complete"])
	}
	turns, _ := cc["turns"].([]any)
	if len(turns) != 1 {
		t.Fatalf("turns len = %d, want 1", len(turns))
	}
	t0, _ := turns[0].(map[string]any)
	if t0["role"] != "user" {
		t.Errorf("turns[0].role = %v, want user", t0["role"])
	}
	parts, _ := t0["parts"].([]any)
	if len(parts) != 1 {
		t.Fatalf("turns[0].parts len = %d, want 1", len(parts))
	}
	p0, _ := parts[0].(map[string]any)
	if p0["text"] != "hello" {
		t.Errorf("text = %v, want hello", p0["text"])
	}
}

func TestSendAudio_Base64EncodesPayloadWithMime(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	raw := []byte{0x00, 0x01, 0x02, 0xFF, 0x7F, 0x80}
	if err := sess.SendAudio(ctx, raw); err != nil {
		t.Fatalf("SendAudio: %v", err)
	}
	frames := waitForFrames(t, cap, 2, 2*time.Second)

	var frame map[string]any
	if err := json.Unmarshal([]byte(frames[1]), &frame); err != nil {
		t.Fatalf("decode: %v", err)
	}
	ri, _ := frame["realtime_input"].(map[string]any)
	if ri == nil {
		t.Fatalf("realtime_input missing: %v", frame)
	}
	chunks, _ := ri["media_chunks"].([]any)
	if len(chunks) != 1 {
		t.Fatalf("media_chunks len = %d, want 1", len(chunks))
	}
	c0, _ := chunks[0].(map[string]any)
	if c0["mime_type"] != "audio/pcm;rate=16000" {
		t.Errorf("mime_type = %v, want audio/pcm;rate=16000", c0["mime_type"])
	}
	want := base64.StdEncoding.EncodeToString(raw)
	if c0["data"] != want {
		t.Errorf("data = %v, want %v", c0["data"], want)
	}
}

func TestSendAudio_EmptyIsNoop(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	if err := sess.SendAudio(ctx, nil); err != nil {
		t.Fatalf("SendAudio(nil): %v", err)
	}
	// Give the pump a chance to do nothing.
	time.Sleep(50 * time.Millisecond)
	if got := len(cap.frames()); got != 1 {
		t.Errorf("frame count = %d, want 1 (only setup)", got)
	}
}

func TestSendToolResult_EmitsToolResponse(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	if err := sess.SendToolResult(ctx, "fc_1", "get_weather", json.RawMessage(`{"weather":"sunny"}`)); err != nil {
		t.Fatalf("SendToolResult: %v", err)
	}
	frames := waitForFrames(t, cap, 2, 2*time.Second)
	var second map[string]any
	if err := json.Unmarshal([]byte(frames[1]), &second); err != nil {
		t.Fatalf("decode: %v", err)
	}
	tr, _ := second["tool_response"].(map[string]any)
	if tr == nil {
		t.Fatalf("tool_response missing: %v", second)
	}
	resps, _ := tr["function_responses"].([]any)
	if len(resps) != 1 {
		t.Fatalf("function_responses len = %d, want 1", len(resps))
	}
	r0, _ := resps[0].(map[string]any)
	if r0["id"] != "fc_1" {
		t.Errorf("id = %v, want fc_1", r0["id"])
	}
	if r0["name"] != "get_weather" {
		t.Errorf("name = %v, want get_weather", r0["name"])
	}
	resp, _ := r0["response"].(map[string]any)
	if resp["weather"] != "sunny" {
		t.Errorf("response.weather = %v, want sunny", resp["weather"])
	}
}

func TestSendToolResult_VariedPayloads(t *testing.T) {
	cases := []struct {
		name string
		id   string
		fn   string
		body string
	}{
		{"object", "c1", "f", `{"ok":true}`},
		{"array", "c2", "f", `[1,2,3]`},
		{"null", "c3", "f", `null`},
		{"string-json", "c4", "f", `"hello"`},
		{"long-id", "c_" + strings.Repeat("x", 100), "f", `{}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			sess, err := p.Connect(ctx, geminilive.SessionConfig{})
			if err != nil {
				t.Fatalf("Connect: %v", err)
			}
			defer func() {
				_ = sess.Close()
				<-done
			}()
			if err := sess.SendToolResult(ctx, tc.id, tc.fn, json.RawMessage(tc.body)); err != nil {
				t.Fatalf("SendToolResult: %v", err)
			}
			frames := waitForFrames(t, cap, 2, 2*time.Second)
			if !strings.Contains(frames[1], `"id":"`+tc.id+`"`) {
				t.Errorf("frame missing id %q: %s", tc.id, frames[1])
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Pump translation
// ---------------------------------------------------------------------------

func TestPump_TranslatesServerTextPart(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"serverContent":{"modelTurn":{"role":"model","parts":[{"text":"hello "}]},"turnComplete":false}}`},
		{payload: `{"serverContent":{"modelTurn":{"role":"model","parts":[{"text":"world"}]},"turnComplete":false}}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{Modalities: []string{"TEXT"}})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var assembled strings.Builder
	for _, ev := range events {
		if ev.Type == geminilive.EventTypeServerText {
			assembled.WriteString(ev.Text)
		}
	}
	if assembled.String() != "hello world" {
		t.Errorf("assembled = %q, want %q", assembled.String(), "hello world")
	}
}

func TestPump_TranslatesServerAudioPart(t *testing.T) {
	want := []byte{0x10, 0x20, 0x30, 0x40, 0x50}
	encoded := base64.StdEncoding.EncodeToString(want)
	payload := `{"serverContent":{"modelTurn":{"role":"model","parts":[{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"` + encoded + `"}}]}}}`
	p, _, _, done := newFakeServer(t, serverOpts{}, []scriptedFrame{{payload: payload}})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var found bool
	for _, ev := range events {
		if ev.Type == geminilive.EventTypeServerAudio {
			found = true
			if string(ev.AudioDelta) != string(want) {
				t.Errorf("audio bytes = %v, want %v", ev.AudioDelta, want)
			}
			if ev.AudioMime != "audio/pcm;rate=24000" {
				t.Errorf("AudioMime = %q, want audio/pcm;rate=24000", ev.AudioMime)
			}
		}
	}
	if !found {
		t.Error("no server.audio event delivered")
	}
}

func TestPump_AudioMimePreserved(t *testing.T) {
	for _, mime := range []string{
		"audio/pcm;rate=24000",
		"audio/pcm;rate=16000",
		"audio/wav",
		"audio/mpeg",
		"audio/ogg",
		"audio/flac",
	} {
		t.Run(mime, func(t *testing.T) {
			encoded := base64.StdEncoding.EncodeToString([]byte{0x01, 0x02})
			payload := `{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"mimeType":"` + mime + `","data":"` + encoded + `"}}]}}}`
			p, _, _, done := newFakeServer(t, serverOpts{}, []scriptedFrame{{payload: payload}})
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			sess, err := p.Connect(ctx, geminilive.SessionConfig{})
			if err != nil {
				t.Fatalf("Connect: %v", err)
			}
			events := drainEvents(t, sess, 3*time.Second)
			<-done

			var got string
			for _, ev := range events {
				if ev.Type == geminilive.EventTypeServerAudio {
					got = ev.AudioMime
				}
			}
			if got != mime {
				t.Errorf("AudioMime = %q, want %q", got, mime)
			}
		})
	}
}

func TestPump_MultipleAudioChunksInOneFrameEmitMultipleEvents(t *testing.T) {
	a := base64.StdEncoding.EncodeToString([]byte{0xAA, 0xBB})
	b := base64.StdEncoding.EncodeToString([]byte{0xCC, 0xDD})
	payload := `{"serverContent":{"modelTurn":{"parts":[` +
		`{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"` + a + `"}},` +
		`{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"` + b + `"}}` +
		`]}}}`
	p, _, _, done := newFakeServer(t, serverOpts{}, []scriptedFrame{{payload: payload}})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var audioEvents []geminilive.SessionEvent
	for _, ev := range events {
		if ev.Type == geminilive.EventTypeServerAudio {
			audioEvents = append(audioEvents, ev)
		}
	}
	if len(audioEvents) != 2 {
		t.Fatalf("audio event count = %d, want 2", len(audioEvents))
	}
	if string(audioEvents[0].AudioDelta) != string([]byte{0xAA, 0xBB}) {
		t.Errorf("audio[0] bytes = %v, want [AA BB]", audioEvents[0].AudioDelta)
	}
	if string(audioEvents[1].AudioDelta) != string([]byte{0xCC, 0xDD}) {
		t.Errorf("audio[1] bytes = %v, want [CC DD]", audioEvents[1].AudioDelta)
	}
}

func TestPump_TranslatesMixedTextAndAudioInOneFrame(t *testing.T) {
	audio := base64.StdEncoding.EncodeToString([]byte{0xDE, 0xAD})
	payload := `{"serverContent":{"modelTurn":{"parts":[` +
		`{"text":"hi"},` +
		`{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"` + audio + `"}}` +
		`]}}}`
	p, _, _, done := newFakeServer(t, serverOpts{}, []scriptedFrame{{payload: payload}})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var sawText, sawAudio bool
	for _, ev := range events {
		if ev.Type == geminilive.EventTypeServerText && ev.Text == "hi" {
			sawText = true
		}
		if ev.Type == geminilive.EventTypeServerAudio && string(ev.AudioDelta) == string([]byte{0xDE, 0xAD}) {
			sawAudio = true
		}
	}
	if !sawText {
		t.Error("missing server.text event")
	}
	if !sawAudio {
		t.Error("missing server.audio event")
	}
}

func TestPump_TranslatesTurnComplete(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"serverContent":{"modelTurn":{"parts":[{"text":"done"}]},"turnComplete":true}}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var sawText, sawTurnComplete bool
	for _, ev := range events {
		if ev.Type == geminilive.EventTypeServerText {
			sawText = true
		}
		if ev.Type == geminilive.EventTypeTurnComplete {
			sawTurnComplete = true
		}
	}
	if !sawText {
		t.Error("missing server.text event")
	}
	if !sawTurnComplete {
		t.Error("missing server.turn_complete event")
	}
}

func TestPump_TurnCompleteOnlyFrame(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"serverContent":{"turnComplete":true}}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var sawTC bool
	for _, ev := range events {
		if ev.Type == geminilive.EventTypeTurnComplete {
			sawTC = true
		}
	}
	if !sawTC {
		t.Error("missing turn_complete event")
	}
}

func TestPump_TranslatesToolCall(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"toolCall":{"functionCalls":[{"id":"fc_1","name":"get_weather","args":{"city":"NYC"}}]}}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var found bool
	for _, ev := range events {
		if ev.Type == geminilive.EventTypeServerTool {
			found = true
			if ev.ToolCallID != "fc_1" {
				t.Errorf("ToolCallID = %q, want fc_1", ev.ToolCallID)
			}
			if ev.ToolName != "get_weather" {
				t.Errorf("ToolName = %q, want get_weather", ev.ToolName)
			}
			var args map[string]any
			if err := json.Unmarshal(ev.ToolArgs, &args); err != nil {
				t.Fatalf("decode args: %v", err)
			}
			if args["city"] != "NYC" {
				t.Errorf("args.city = %v, want NYC", args["city"])
			}
		}
	}
	if !found {
		t.Fatal("no server.tool_call event delivered")
	}
}

func TestPump_TranslatesMultipleToolCallsInOneFrame(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"toolCall":{"functionCalls":[` +
			`{"id":"fc_1","name":"a","args":{}},` +
			`{"id":"fc_2","name":"b","args":{"x":1}},` +
			`{"id":"fc_3","name":"c","args":null}` +
			`]}}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var ids []string
	for _, ev := range events {
		if ev.Type == geminilive.EventTypeServerTool {
			ids = append(ids, ev.ToolCallID)
		}
	}
	if len(ids) != 3 {
		t.Fatalf("got %d tool_call events, want 3 (ids=%v)", len(ids), ids)
	}
	for i, want := range []string{"fc_1", "fc_2", "fc_3"} {
		if ids[i] != want {
			t.Errorf("ids[%d] = %q, want %q", i, ids[i], want)
		}
	}
}

func TestPump_RawPreservedOnEveryEvent(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"serverContent":{"modelTurn":{"parts":[{"text":"x"}]}}}`},
		{payload: `{"toolCall":{"functionCalls":[{"id":"c","name":"n","args":{}}]}}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	for _, ev := range events {
		if len(ev.Raw) == 0 {
			t.Errorf("event %q missing Raw", ev.Type)
		}
	}
}

func TestPump_ErrorFrameTerminatesSession(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"error":{"code":429,"message":"rate limited","status":"RESOURCE_EXHAUSTED"}}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{holdOpen: true}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	terminal := sess.Err()
	_ = sess.Close()
	<-done

	if len(events) == 0 {
		t.Fatal("no events delivered")
	}
	last := events[len(events)-1]
	if last.Type != geminilive.EventTypeError {
		t.Errorf("last event type = %q, want error", last.Type)
	}
	if last.Error == nil {
		t.Fatal("error event missing ErrUpstream")
	}
	if last.Error.Provider != "geminilive" {
		t.Errorf("Provider = %q", last.Error.Provider)
	}
	if last.Error.StatusCode != 0 {
		t.Errorf("StatusCode = %d, want 0 (mid-stream)", last.Error.StatusCode)
	}
	if !strings.Contains(last.Error.Body, "rate limited") {
		t.Errorf("Body = %q", last.Error.Body)
	}
	var upErr *llmrouter.ErrUpstream
	if !errors.As(terminal, &upErr) {
		t.Fatalf("Err() = %v, want *ErrUpstream", terminal)
	}
}

func TestPump_PassesThroughUnknownFrame(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"someUnknownEvent":{"foo":"bar"}}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var sawPassThrough bool
	for _, ev := range events {
		if ev.Type == "someUnknownEvent" {
			sawPassThrough = true
			if len(ev.Raw) == 0 {
				t.Error("Raw missing on pass-through event")
			}
		}
	}
	if !sawPassThrough {
		t.Errorf("did not see pass-through event with derived type; events=%v", typesOf(events))
	}
}

func TestPump_SecondSetupCompleteIsPassedThrough(t *testing.T) {
	// Real servers shouldn't emit a second setupComplete, but if one
	// arrives it should not crash the pump.
	script := []scriptedFrame{
		{payload: `{"setupComplete":{}}`},
		{payload: `{"serverContent":{"turnComplete":true}}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var setupCount int
	for _, ev := range events {
		if ev.Type == geminilive.EventTypeSetupComplete {
			setupCount++
		}
	}
	if setupCount != 2 {
		t.Errorf("setup.complete event count = %d, want 2 (initial + duplicate)", setupCount)
	}
}

// ---------------------------------------------------------------------------
// Lifecycle: Close + context cancel
// ---------------------------------------------------------------------------

func TestClose_Idempotent(t *testing.T) {
	p, _, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	err1 := sess.Close()
	err2 := sess.Close()
	err3 := sess.Close()
	if err1 != err2 || err2 != err3 {
		t.Errorf("Close returned different errors: %v / %v / %v", err1, err2, err3)
	}
	if err := sess.Err(); err != nil && !errors.Is(err, context.Canceled) {
		t.Logf("Err() after close = %v", err)
	}
	<-done
}

func TestContextCancel_ClosesConnection(t *testing.T) {
	p, _, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)

	ctx, cancel := context.WithCancel(context.Background())
	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	cancel()

	select {
	case _, ok := <-sess.Events():
		if ok {
			drainEvents(t, sess, 2*time.Second)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Events channel did not close after context cancel")
	}
	terminal := sess.Err()
	if terminal == nil {
		t.Error("Err() = nil, want context.Canceled")
	}
	_ = sess.Close()
	<-done
}

func TestErr_ReturnsSameValueAcrossCalls(t *testing.T) {
	p, _, _, done := newFakeServer(t, serverOpts{}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	_ = drainEvents(t, sess, 3*time.Second)
	<-done

	e1 := sess.Err()
	e2 := sess.Err()
	if e1 != e2 {
		t.Errorf("Err() not stable: %v / %v", e1, e2)
	}
	_ = sess.Close()
}

// ---------------------------------------------------------------------------
// Raw merge + Tools
// ---------------------------------------------------------------------------

func TestSessionConfig_RawMergesAndOverrides(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{
		Voice: "Aoede",
		Raw:   json.RawMessage(`{"model":"models/override","input_audio_transcription":{}}`),
	})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	frames := waitForFrames(t, cap, 1, 2*time.Second)
	var env map[string]any
	if err := json.Unmarshal([]byte(frames[0]), &env); err != nil {
		t.Fatalf("decode: %v", err)
	}
	setup, _ := env["setup"].(map[string]any)
	if setup["model"] != "models/override" {
		t.Errorf("model = %v, want models/override (Raw must override)", setup["model"])
	}
	if _, present := setup["input_audio_transcription"]; !present {
		t.Errorf("input_audio_transcription missing — Raw merge failed: %v", setup)
	}
}

func TestSessionConfig_RawInvalidJSONErrors(t *testing.T) {
	p, _, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err := p.Connect(ctx, geminilive.SessionConfig{
		Raw: json.RawMessage(`{not json`),
	})
	if err == nil {
		t.Fatal("expected error for invalid Raw JSON, got nil")
	}
	// The server may have already accepted the upgrade; close to make
	// the done channel fire.
	_ = done
}

func TestSessionConfig_ToolsEmittedAsFunctionDeclarations(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{
		Tools: []llmrouter.Tool{
			{
				Type: "function",
				Function: llmrouter.ToolFunction{
					Name:        "get_weather",
					Description: "Get weather",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	frames := waitForFrames(t, cap, 1, 2*time.Second)
	var env map[string]any
	_ = json.Unmarshal([]byte(frames[0]), &env)
	setup, _ := env["setup"].(map[string]any)
	tools, _ := setup["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools len = %d, want 1", len(tools))
	}
	tool0, _ := tools[0].(map[string]any)
	decls, _ := tool0["function_declarations"].([]any)
	if len(decls) != 1 {
		t.Fatalf("function_declarations len = %d, want 1", len(decls))
	}
	d0, _ := decls[0].(map[string]any)
	if d0["name"] != "get_weather" {
		t.Errorf("decl.name = %v", d0["name"])
	}
	if d0["description"] != "Get weather" {
		t.Errorf("decl.description = %v", d0["description"])
	}
	params, _ := d0["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Errorf("decl.parameters.type = %v", params["type"])
	}
}

func TestSessionConfig_MultipleToolsCollapsedIntoOneEntry(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{
		Tools: []llmrouter.Tool{
			{Type: "function", Function: llmrouter.ToolFunction{Name: "a"}},
			{Type: "function", Function: llmrouter.ToolFunction{Name: "b"}},
			{Type: "function", Function: llmrouter.ToolFunction{Name: "c"}},
		},
	})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	frames := waitForFrames(t, cap, 1, 2*time.Second)
	var env map[string]any
	_ = json.Unmarshal([]byte(frames[0]), &env)
	setup, _ := env["setup"].(map[string]any)
	tools, _ := setup["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools len = %d, want 1 (collapsed)", len(tools))
	}
	tool0, _ := tools[0].(map[string]any)
	decls, _ := tool0["function_declarations"].([]any)
	if len(decls) != 3 {
		t.Fatalf("function_declarations len = %d, want 3", len(decls))
	}
	for i, want := range []string{"a", "b", "c"} {
		d, _ := decls[i].(map[string]any)
		if d["name"] != want {
			t.Errorf("decl[%d].name = %v, want %v", i, d["name"], want)
		}
	}
}

func TestSessionConfig_ToolWithoutDescriptionOrParams(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{
		Tools: []llmrouter.Tool{
			{Type: "function", Function: llmrouter.ToolFunction{Name: "ping"}},
		},
	})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	frames := waitForFrames(t, cap, 1, 2*time.Second)
	if strings.Contains(frames[0], `"description"`) {
		t.Errorf("description should be omitted when empty: %s", frames[0])
	}
	if strings.Contains(frames[0], `"parameters"`) {
		t.Errorf("parameters should be omitted when nil: %s", frames[0])
	}
}

// ---------------------------------------------------------------------------
// URL building
// ---------------------------------------------------------------------------

func TestConnect_BaseURLWithTrailingSlashNormalised(t *testing.T) {
	// Capture the URL via a fake server, then construct a provider that
	// adds a trailing slash on top.
	p, cap, srv, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	_ = p
	custom, err := geminilive.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL+"/"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	sess, err := custom.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()
	u := cap.url()
	if strings.Contains(u, "//ws/") {
		t.Errorf("upgrade URL has double slash: %q", u)
	}
}

func TestConnect_WSSBaseURLAccepted(t *testing.T) {
	// httptest is http://; ensure that wss:// scheme on the BaseURL
	// option is also accepted by WithBaseURL parsing (we can't actually
	// dial wss:// against httptest, so we only check construction).
	_, err := geminilive.New(
		llmrouter.WithAPIKey("k"),
		llmrouter.WithBaseURL("wss://generativelanguage.googleapis.com"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------

func TestSession_ConcurrentSendsAreSerialised(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, geminilive.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	const n = 20
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			_ = sess.SendAudio(ctx, []byte{0x01, 0x02, 0x03})
		}()
	}
	wg.Wait()
	frames := waitForFrames(t, cap, n+1, 3*time.Second)
	if len(frames) < n+1 {
		t.Errorf("frame count = %d, want at least %d", len(frames), n+1)
	}
	// Verify every realtime_input frame parses cleanly — proves writes
	// did not corrupt each other.
	for _, f := range frames[1:] {
		var probe map[string]any
		if err := json.Unmarshal([]byte(f), &probe); err != nil {
			t.Errorf("corrupted frame: %v: %s", err, f)
		}
	}
}

// ---------------------------------------------------------------------------
// Additional voice and event coverage
// ---------------------------------------------------------------------------

func TestConnect_AllPrebuiltVoicesAccepted(t *testing.T) {
	for _, voice := range []string{"Aoede", "Charon", "Fenrir", "Kore", "Puck"} {
		t.Run(voice, func(t *testing.T) {
			p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			sess, err := p.Connect(ctx, geminilive.SessionConfig{Voice: voice})
			if err != nil {
				t.Fatalf("Connect: %v", err)
			}
			defer func() {
				_ = sess.Close()
				<-done
			}()
			frames := waitForFrames(t, cap, 1, 2*time.Second)
			if !strings.Contains(frames[0], `"voice_name":"`+voice+`"`) {
				t.Errorf("voice_name=%q missing in %s", voice, frames[0])
			}
		})
	}
}

func TestSessionEvent_ConstantsExposed(t *testing.T) {
	cases := []struct {
		got, want string
	}{
		{geminilive.EventTypeSetupComplete, "setup.complete"},
		{geminilive.EventTypeServerText, "server.text"},
		{geminilive.EventTypeServerAudio, "server.audio"},
		{geminilive.EventTypeServerTool, "server.tool_call"},
		{geminilive.EventTypeTurnComplete, "server.turn_complete"},
		{geminilive.EventTypeError, "error"},
		{geminilive.EventTypeRaw, "raw"},
	}
	for _, tc := range cases {
		t.Run(tc.want, func(t *testing.T) {
			if tc.got != tc.want {
				t.Errorf("constant = %q, want %q", tc.got, tc.want)
			}
		})
	}
}

func TestSendText_VariedInputs(t *testing.T) {
	cases := []struct {
		name string
		text string
	}{
		{"ascii", "hello"},
		{"unicode", "こんにちは"},
		{"emoji", "hi 👋"},
		{"empty", ""},
		{"long", strings.Repeat("x", 4096)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			sess, err := p.Connect(ctx, geminilive.SessionConfig{})
			if err != nil {
				t.Fatalf("Connect: %v", err)
			}
			defer func() {
				_ = sess.Close()
				<-done
			}()
			if err := sess.SendText(ctx, tc.text); err != nil {
				t.Fatalf("SendText: %v", err)
			}
			frames := waitForFrames(t, cap, 2, 2*time.Second)
			var env map[string]any
			_ = json.Unmarshal([]byte(frames[1]), &env)
			cc, _ := env["client_content"].(map[string]any)
			turns, _ := cc["turns"].([]any)
			t0, _ := turns[0].(map[string]any)
			parts, _ := t0["parts"].([]any)
			p0, _ := parts[0].(map[string]any)
			gotText, _ := p0["text"].(string)
			if tc.text == "" {
				// Empty text uses omitempty — the text field may be absent.
				if gotText != "" {
					t.Errorf("text = %q, want empty", gotText)
				}
			} else if gotText != tc.text {
				t.Errorf("text = %q, want %q", gotText, tc.text)
			}
		})
	}
}

func TestSendAudio_VariedChunkSizes(t *testing.T) {
	cases := []struct {
		name string
		size int
	}{
		{"tiny-4", 4},
		{"small-128", 128},
		{"med-1024", 1024},
		{"large-8192", 8192},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			sess, err := p.Connect(ctx, geminilive.SessionConfig{})
			if err != nil {
				t.Fatalf("Connect: %v", err)
			}
			defer func() {
				_ = sess.Close()
				<-done
			}()
			buf := make([]byte, tc.size)
			for i := range buf {
				buf[i] = byte(i % 256)
			}
			if err := sess.SendAudio(ctx, buf); err != nil {
				t.Fatalf("SendAudio: %v", err)
			}
			frames := waitForFrames(t, cap, 2, 2*time.Second)
			want := base64.StdEncoding.EncodeToString(buf)
			if !strings.Contains(frames[1], want) {
				t.Errorf("frame missing expected base64 for %d-byte buffer", tc.size)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func typesOf(events []geminilive.SessionEvent) []string {
	out := make([]string, 0, len(events))
	for _, ev := range events {
		out = append(out, ev.Type)
	}
	return out
}
