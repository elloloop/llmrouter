package openairealtime_test

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
	"github.com/elloloop/llmrouter/providers/openairealtime"
)

// scriptedFrame is one server frame the fake will emit after the initial
// session.update is observed. After: optional pause inserted before
// emission.
type scriptedFrame struct {
	payload string
	after   time.Duration
}

// serverOpts configures the fake Realtime server.
type serverOpts struct {
	// rejectHandshake, when non-zero, returns this HTTP status during
	// the websocket upgrade instead of accepting.
	rejectHandshake int
	// rejectBody is the response body for a rejected handshake.
	rejectBody string
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

// newFakeServer returns a fake OpenAI Realtime websocket server, a
// capture struct, a Provider pointed at it, and a done channel that
// closes when the handler returns.
//
// The handler scripts `script` frames after observing the initial
// session.update from the client. It always also drains any further
// inbound frames into capture.textFrames.
func newFakeServer(t *testing.T, opts serverOpts, script []scriptedFrame) (*openairealtime.Provider, *serverCapture, *httptest.Server, <-chan struct{}) {
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

		// Reader goroutine — captures everything the client sends.
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
			// Block until either the request context fires (test shutdown)
			// or the reader detects the client closed (Session.Close()
			// sends a close frame; the reader exits when its Read returns).
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

	p, err := openairealtime.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p, cap, srv, done
}

// drainEvents reads until the channel closes or timeout fires.
func drainEvents(t *testing.T, sess *openairealtime.Session, timeout time.Duration) []openairealtime.SessionEvent {
	t.Helper()
	var out []openairealtime.SessionEvent
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

func TestConnect_HappyPath(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"type":"session.created","session":{"id":"sess_1"}}`},
	}
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, script)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{
		Voice:        "alloy",
		Instructions: "you are a test bot",
	})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	select {
	case ev := <-sess.Events():
		if ev.Type != "session.created" {
			t.Errorf("first event type = %q, want session.created", ev.Type)
		}
		if len(ev.Raw) == 0 {
			t.Error("Raw payload missing on session.created")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("no session.created event delivered")
	}

	// Confirm session.update was the first frame the server saw.
	frames := waitForFrames(t, cap, 1, 2*time.Second)
	if len(frames) < 1 {
		t.Fatal("no client frames captured")
	}
	var first map[string]any
	if err := json.Unmarshal([]byte(frames[0]), &first); err != nil {
		t.Fatalf("decode first frame: %v", err)
	}
	if first["type"] != "session.update" {
		t.Errorf("first frame type = %v, want session.update", first["type"])
	}
	innerSession, _ := first["session"].(map[string]any)
	if innerSession["voice"] != "alloy" {
		t.Errorf("session.voice = %v, want alloy", innerSession["voice"])
	}
	if innerSession["instructions"] != "you are a test bot" {
		t.Errorf("session.instructions mismatch: got %v", innerSession["instructions"])
	}
}

func TestConnect_AssertsAuthAndBetaHeaders(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	// Server may have already captured headers by the time Connect returns.
	hdrs := cap.headers()
	if got := hdrs.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q, want Bearer test-key", got)
	}
	if got := hdrs.Get("OpenAI-Beta"); got != "realtime=v1" {
		t.Errorf("OpenAI-Beta = %q, want realtime=v1", got)
	}
}

func TestConnect_PutsModelInQuery(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{Model: "gpt-4o-realtime-preview-2024-12-17"})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	u := cap.url()
	if !strings.Contains(u, "model=gpt-4o-realtime-preview-2024-12-17") {
		t.Errorf("upgrade URL %q missing model query", u)
	}
	if !strings.HasPrefix(u, "/realtime") {
		t.Errorf("upgrade URL %q does not start with /realtime", u)
	}
}

func TestConnect_DefaultModelApplied(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	u := cap.url()
	if !strings.Contains(u, "model=gpt-4o-realtime-preview") {
		t.Errorf("upgrade URL %q missing default model", u)
	}
}

func TestConnect_HandshakeRejection(t *testing.T) {
	p, _, _, done := newFakeServer(t, serverOpts{
		rejectHandshake: http.StatusUnauthorized,
		rejectBody:      `{"error":{"message":"bad key"}}`,
	}, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_, err := p.Connect(ctx, openairealtime.SessionConfig{})
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
	if upErr.Provider != "openairealtime" {
		t.Errorf("Provider = %q, want openairealtime", upErr.Provider)
	}
	if !strings.Contains(upErr.Body, "bad key") {
		t.Errorf("Body = %q, want substring 'bad key'", upErr.Body)
	}
	<-done
}

func TestSendText_EmitsItemThenResponseCreate(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
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

	// Expect 3 frames total: session.update + conversation.item.create + response.create.
	frames := waitForFrames(t, cap, 3, 2*time.Second)
	if len(frames) < 3 {
		t.Fatalf("expected 3 frames, got %d: %v", len(frames), frames)
	}

	var second, third map[string]any
	if err := json.Unmarshal([]byte(frames[1]), &second); err != nil {
		t.Fatalf("decode second: %v", err)
	}
	if err := json.Unmarshal([]byte(frames[2]), &third); err != nil {
		t.Fatalf("decode third: %v", err)
	}
	if second["type"] != "conversation.item.create" {
		t.Errorf("second frame type = %v, want conversation.item.create", second["type"])
	}
	if third["type"] != "response.create" {
		t.Errorf("third frame type = %v, want response.create", third["type"])
	}

	item, _ := second["item"].(map[string]any)
	if item["role"] != "user" {
		t.Errorf("item.role = %v, want user", item["role"])
	}
	content, _ := item["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("item.content len = %d, want 1", len(content))
	}
	c0, _ := content[0].(map[string]any)
	if c0["type"] != "input_text" {
		t.Errorf("content[0].type = %v, want input_text", c0["type"])
	}
	if c0["text"] != "hello" {
		t.Errorf("content[0].text = %v, want hello", c0["text"])
	}
}

func TestSendAudio_Base64EncodesPayload(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
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
	if frame["type"] != "input_audio_buffer.append" {
		t.Errorf("type = %v, want input_audio_buffer.append", frame["type"])
	}
	encoded, _ := frame["audio"].(string)
	want := base64.StdEncoding.EncodeToString(raw)
	if encoded != want {
		t.Errorf("audio = %q, want %q", encoded, want)
	}
}

func TestSendAudio_EmptyIsNoop(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
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
		t.Errorf("frame count = %d, want 1 (only session.update)", got)
	}
}

func TestCommitAndCreateResponse(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	if err := sess.Commit(ctx); err != nil {
		t.Fatalf("Commit: %v", err)
	}
	if err := sess.CreateResponse(ctx); err != nil {
		t.Fatalf("CreateResponse: %v", err)
	}
	frames := waitForFrames(t, cap, 3, 2*time.Second)
	var commit, resp map[string]any
	_ = json.Unmarshal([]byte(frames[1]), &commit)
	_ = json.Unmarshal([]byte(frames[2]), &resp)
	if commit["type"] != "input_audio_buffer.commit" {
		t.Errorf("commit.type = %v", commit["type"])
	}
	if resp["type"] != "response.create" {
		t.Errorf("response.type = %v", resp["type"])
	}
}

func TestUpdateSession_SendsSessionUpdate(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{Voice: "alloy"})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	temp := 0.7
	if err := sess.UpdateSession(ctx, openairealtime.SessionConfig{
		Voice:       "verse",
		Temperature: &temp,
	}); err != nil {
		t.Fatalf("UpdateSession: %v", err)
	}
	frames := waitForFrames(t, cap, 2, 2*time.Second)
	var update map[string]any
	if err := json.Unmarshal([]byte(frames[1]), &update); err != nil {
		t.Fatalf("decode update: %v", err)
	}
	if update["type"] != "session.update" {
		t.Errorf("update.type = %v, want session.update", update["type"])
	}
	inner, _ := update["session"].(map[string]any)
	if inner["voice"] != "verse" {
		t.Errorf("session.voice = %v, want verse", inner["voice"])
	}
	if inner["temperature"].(float64) != 0.7 {
		t.Errorf("session.temperature = %v, want 0.7", inner["temperature"])
	}
}

func TestPump_TranslatesTextDelta(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"type":"response.created","response":{"id":"resp_1"}}`},
		{payload: `{"type":"response.text.delta","response_id":"resp_1","delta":"hel"}`},
		{payload: `{"type":"response.text.delta","response_id":"resp_1","delta":"lo"}`},
		{payload: `{"type":"response.done","response":{"id":"resp_1"},"response_id":"resp_1"}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{Modalities: []string{"text"}})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var text strings.Builder
	var sawDone bool
	for _, ev := range events {
		switch ev.Type {
		case "response.text.delta":
			text.WriteString(ev.Text)
			if ev.ResponseID != "resp_1" {
				t.Errorf("ResponseID = %q, want resp_1", ev.ResponseID)
			}
		case "response.done":
			sawDone = true
		}
	}
	if text.String() != "hello" {
		t.Errorf("assembled text = %q, want hello", text.String())
	}
	if !sawDone {
		t.Error("no response.done event")
	}
}

func TestPump_TranslatesAudioDelta(t *testing.T) {
	want := []byte{0x10, 0x20, 0x30, 0x40, 0x50}
	encoded := base64.StdEncoding.EncodeToString(want)
	script := []scriptedFrame{
		{payload: `{"type":"response.audio.delta","response_id":"r1","delta":"` + encoded + `"}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var foundAudio bool
	for _, ev := range events {
		if ev.Type == "response.audio.delta" {
			foundAudio = true
			if string(ev.AudioDelta) != string(want) {
				t.Errorf("audio bytes = %v, want %v", ev.AudioDelta, want)
			}
			if ev.ResponseID != "r1" {
				t.Errorf("ResponseID = %q, want r1", ev.ResponseID)
			}
		}
	}
	if !foundAudio {
		t.Error("no response.audio.delta event delivered")
	}
}

func TestPump_PassesThroughUnknownEventTypes(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"type":"rate_limits.updated","rate_limits":[]}`},
		{payload: `{"type":"response.audio.done","response_id":"r1"}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	types := map[string]bool{}
	for _, ev := range events {
		types[ev.Type] = true
		if len(ev.Raw) == 0 {
			t.Errorf("event %q missing Raw", ev.Type)
		}
	}
	for _, want := range []string{"rate_limits.updated", "response.audio.done"} {
		if !types[want] {
			t.Errorf("missing pass-through event %q (got %v)", want, types)
		}
	}
}

func TestPump_ErrorEventTerminatesSession(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"type":"error","error":{"type":"invalid_request_error","code":"missing_voice","message":"voice not allowed"}}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{holdOpen: true}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
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
	if last.Type != "error" {
		t.Errorf("last event type = %q, want error", last.Type)
	}
	if last.Error == nil {
		t.Fatal("error event missing ErrUpstream")
	}
	if last.Error.Provider != "openairealtime" {
		t.Errorf("Provider = %q", last.Error.Provider)
	}
	if last.Error.StatusCode != 0 {
		t.Errorf("StatusCode = %d, want 0 (mid-stream)", last.Error.StatusCode)
	}
	if !strings.Contains(last.Error.Body, "voice not allowed") {
		t.Errorf("Body = %q", last.Error.Body)
	}

	var upErr *llmrouter.ErrUpstream
	if !errors.As(terminal, &upErr) {
		t.Fatalf("Err() = %v, want *ErrUpstream", terminal)
	}
}

func TestClose_Idempotent(t *testing.T) {
	p, _, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	err1 := sess.Close()
	err2 := sess.Close()
	err3 := sess.Close()
	if err1 != err2 || err2 != err3 {
		t.Errorf("Close returned different errors across calls: %v / %v / %v", err1, err2, err3)
	}
	// Pump must finish so Err() returns.
	if err := sess.Err(); err != nil && !errors.Is(err, context.Canceled) {
		// Some race conditions surface a wrapped read error here; both
		// are valid as long as the session has fully drained.
		t.Logf("Err() after close = %v", err)
	}
	<-done
}

func TestContextCancel_ClosesConnection(t *testing.T) {
	p, _, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)

	ctx, cancel := context.WithCancel(context.Background())
	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	cancel()

	// Events channel must close once the pump observes ctx.Done().
	select {
	case _, ok := <-sess.Events():
		if ok {
			// Drain any in-flight events; eventually the channel closes.
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

func TestSessionConfig_RawMergesAndOverrides(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{
		Voice: "alloy",
		Raw:   json.RawMessage(`{"voice":"verse","turn_detection":{"type":"server_vad"}}`),
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
	inner, _ := env["session"].(map[string]any)
	if inner["voice"] != "verse" {
		t.Errorf("voice = %v, want verse (Raw must override)", inner["voice"])
	}
	td, ok := inner["turn_detection"].(map[string]any)
	if !ok {
		t.Fatalf("turn_detection missing or wrong type: %v", inner["turn_detection"])
	}
	if td["type"] != "server_vad" {
		t.Errorf("turn_detection.type = %v", td["type"])
	}
}

// ---------------------------------------------------------------------------
// Tools + ToolChoice on session.update
// ---------------------------------------------------------------------------

func TestSessionConfig_ToolsEmittedAsFlatArray(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{
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
	if err := json.Unmarshal([]byte(frames[0]), &env); err != nil {
		t.Fatalf("decode: %v", err)
	}
	inner, _ := env["session"].(map[string]any)
	tools, ok := inner["tools"].([]any)
	if !ok {
		t.Fatalf("tools missing or wrong type: %v", inner["tools"])
	}
	if len(tools) != 1 {
		t.Fatalf("tools len = %d, want 1", len(tools))
	}
	tool0, _ := tools[0].(map[string]any)
	if tool0["type"] != "function" {
		t.Errorf("tools[0].type = %v, want function", tool0["type"])
	}
	if tool0["name"] != "get_weather" {
		t.Errorf("tools[0].name = %v, want get_weather", tool0["name"])
	}
	if tool0["description"] != "Get weather" {
		t.Errorf("tools[0].description = %v", tool0["description"])
	}
	if _, present := tool0["function"]; present {
		t.Errorf("tools[0] must be flat — no nested 'function' wrapper; got %v", tool0)
	}
	params, _ := tool0["parameters"].(map[string]any)
	if params["type"] != "object" {
		t.Errorf("tools[0].parameters not nested object: %v", tool0["parameters"])
	}
}

func TestSessionConfig_ToolsOmittedWhenEmpty(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
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
	inner, _ := env["session"].(map[string]any)
	if _, present := inner["tools"]; present {
		t.Errorf("tools should be omitted when none: %v", inner["tools"])
	}
	if _, present := inner["tool_choice"]; present {
		t.Errorf("tool_choice should be omitted when nil: %v", inner["tool_choice"])
	}
}

func TestSessionConfig_MultipleToolsFlattened(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{
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
	inner, _ := env["session"].(map[string]any)
	tools, _ := inner["tools"].([]any)
	if len(tools) != 3 {
		t.Fatalf("tools len = %d, want 3", len(tools))
	}
	for i, want := range []string{"a", "b", "c"} {
		ti, _ := tools[i].(map[string]any)
		if ti["name"] != want {
			t.Errorf("tools[%d].name = %v, want %v", i, ti["name"], want)
		}
	}
}

func TestSessionConfig_ToolDescriptionAndParamsOmitemptyHonoured(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{
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

func TestSessionConfig_ToolChoiceAutoMarshals(t *testing.T) {
	cases := []struct {
		name string
		tc   *llmrouter.ToolChoice
		want string
	}{
		{"auto", &llmrouter.ToolChoice{Mode: "auto"}, `"tool_choice":"auto"`},
		{"none", &llmrouter.ToolChoice{Mode: "none"}, `"tool_choice":"none"`},
		{"required", &llmrouter.ToolChoice{Mode: "required"}, `"tool_choice":"required"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			sess, err := p.Connect(ctx, openairealtime.SessionConfig{
				Tools: []llmrouter.Tool{
					{Type: "function", Function: llmrouter.ToolFunction{Name: "f"}},
				},
				ToolChoice: tc.tc,
			})
			if err != nil {
				t.Fatalf("Connect: %v", err)
			}
			defer func() {
				_ = sess.Close()
				<-done
			}()

			frames := waitForFrames(t, cap, 1, 2*time.Second)
			if !strings.Contains(frames[0], tc.want) {
				t.Fatalf("want %q in frame: %s", tc.want, frames[0])
			}
		})
	}
}

func TestSessionConfig_ToolChoiceSpecificMarshals(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{
		Tools: []llmrouter.Tool{
			{Type: "function", Function: llmrouter.ToolFunction{Name: "get_weather"}},
		},
		ToolChoice: &llmrouter.ToolChoice{Mode: "specific", Function: "get_weather"},
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
	inner, _ := env["session"].(map[string]any)
	tc, ok := inner["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice not an object: %v", inner["tool_choice"])
	}
	if tc["type"] != "function" {
		t.Errorf("tool_choice.type = %v, want function", tc["type"])
	}
	fn, _ := tc["function"].(map[string]any)
	if fn["name"] != "get_weather" {
		t.Errorf("tool_choice.function.name = %v, want get_weather", fn["name"])
	}
}

// ---------------------------------------------------------------------------
// translateFrame for tool-call streaming
// ---------------------------------------------------------------------------

func TestPump_TranslatesFunctionCallArgumentsDelta(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"type":"response.function_call_arguments.delta","response_id":"resp_1","call_id":"call_abc","name":"get_weather","delta":"{\"loc"}`},
		{payload: `{"type":"response.function_call_arguments.delta","response_id":"resp_1","call_id":"call_abc","name":"get_weather","delta":"\":\"NYC"}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var deltaCount int
	var assembled strings.Builder
	for _, ev := range events {
		if ev.Type == "response.function_call_arguments.delta" {
			deltaCount++
			if ev.ToolCallID != "call_abc" {
				t.Errorf("ToolCallID = %q, want call_abc", ev.ToolCallID)
			}
			if ev.ToolName != "get_weather" {
				t.Errorf("ToolName = %q, want get_weather", ev.ToolName)
			}
			if ev.ResponseID != "resp_1" {
				t.Errorf("ResponseID = %q, want resp_1", ev.ResponseID)
			}
			assembled.WriteString(ev.ToolArgumentsDelta)
			if ev.ToolArguments != "" {
				t.Errorf("ToolArguments should be empty on delta event: %q", ev.ToolArguments)
			}
			if len(ev.Raw) == 0 {
				t.Error("Raw missing on delta event")
			}
		}
	}
	if deltaCount != 2 {
		t.Errorf("delta count = %d, want 2", deltaCount)
	}
	if assembled.String() != `{"loc":"NYC` {
		t.Errorf("assembled args = %q, want %q", assembled.String(), `{"loc":"NYC`)
	}
}

func TestPump_TranslatesFunctionCallArgumentsDone(t *testing.T) {
	script := []scriptedFrame{
		{payload: `{"type":"response.function_call_arguments.done","response_id":"resp_1","call_id":"call_xyz","name":"get_time","arguments":"{\"tz\":\"UTC\"}"}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var found bool
	for _, ev := range events {
		if ev.Type == "response.function_call_arguments.done" {
			found = true
			if ev.ToolCallID != "call_xyz" {
				t.Errorf("ToolCallID = %q, want call_xyz", ev.ToolCallID)
			}
			if ev.ToolName != "get_time" {
				t.Errorf("ToolName = %q, want get_time", ev.ToolName)
			}
			if ev.ToolArguments != `{"tz":"UTC"}` {
				t.Errorf("ToolArguments = %q", ev.ToolArguments)
			}
			if ev.ToolArgumentsDelta != "" {
				t.Errorf("ToolArgumentsDelta should be empty on done event: %q", ev.ToolArgumentsDelta)
			}
			if ev.ResponseID != "resp_1" {
				t.Errorf("ResponseID = %q, want resp_1", ev.ResponseID)
			}
			if len(ev.Raw) == 0 {
				t.Error("Raw missing on done event")
			}
		}
	}
	if !found {
		t.Fatal("done event not delivered")
	}
}

func TestPump_FunctionCallEventTypePassesThrough(t *testing.T) {
	// Confirm the Type string is preserved verbatim so consumers can
	// switch on the upstream event name.
	script := []scriptedFrame{
		{payload: `{"type":"response.function_call_arguments.delta","call_id":"c1","delta":"x"}`},
		{payload: `{"type":"response.function_call_arguments.done","call_id":"c1","arguments":"x"}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	gotTypes := map[string]bool{}
	for _, ev := range events {
		gotTypes[ev.Type] = true
	}
	for _, want := range []string{
		"response.function_call_arguments.delta",
		"response.function_call_arguments.done",
	} {
		if !gotTypes[want] {
			t.Errorf("missing %q in delivered types: %v", want, gotTypes)
		}
	}
}

// ---------------------------------------------------------------------------
// SendToolResult
// ---------------------------------------------------------------------------

func TestSendToolResult_EmitsItemThenResponseCreate(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		_ = sess.Close()
		<-done
	}()

	if err := sess.SendToolResult(ctx, "call_abc", `{"weather":"sunny"}`); err != nil {
		t.Fatalf("SendToolResult: %v", err)
	}

	// 3 frames: session.update + conversation.item.create + response.create
	frames := waitForFrames(t, cap, 3, 2*time.Second)
	if len(frames) < 3 {
		t.Fatalf("expected 3 frames, got %d: %v", len(frames), frames)
	}

	var second, third map[string]any
	if err := json.Unmarshal([]byte(frames[1]), &second); err != nil {
		t.Fatalf("decode second: %v", err)
	}
	if err := json.Unmarshal([]byte(frames[2]), &third); err != nil {
		t.Fatalf("decode third: %v", err)
	}
	if second["type"] != "conversation.item.create" {
		t.Errorf("second frame type = %v, want conversation.item.create", second["type"])
	}
	if third["type"] != "response.create" {
		t.Errorf("third frame type = %v, want response.create", third["type"])
	}

	item, _ := second["item"].(map[string]any)
	if item["type"] != "function_call_output" {
		t.Errorf("item.type = %v, want function_call_output", item["type"])
	}
	if item["call_id"] != "call_abc" {
		t.Errorf("item.call_id = %v, want call_abc", item["call_id"])
	}
	if item["output"] != `{"weather":"sunny"}` {
		t.Errorf("item.output = %v", item["output"])
	}
	// Role and content should NOT be present on function_call_output items.
	if _, present := item["role"]; present {
		t.Errorf("item.role must be omitted on function_call_output: %v", item["role"])
	}
	if _, present := item["content"]; present {
		t.Errorf("item.content must be omitted on function_call_output: %v", item["content"])
	}
}

func TestPump_FunctionCallArgumentsDoneEmptyArgs(t *testing.T) {
	// A done event with empty arguments string is valid — the model
	// declared an arg-less tool call.
	script := []scriptedFrame{
		{payload: `{"type":"response.function_call_arguments.done","call_id":"c1","name":"ping","arguments":""}`},
	}
	p, _, _, done := newFakeServer(t, serverOpts{}, script)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	events := drainEvents(t, sess, 3*time.Second)
	<-done

	var found bool
	for _, ev := range events {
		if ev.Type == "response.function_call_arguments.done" {
			found = true
			if ev.ToolCallID != "c1" {
				t.Errorf("ToolCallID = %q", ev.ToolCallID)
			}
			if ev.ToolName != "ping" {
				t.Errorf("ToolName = %q", ev.ToolName)
			}
			if ev.ToolArguments != "" {
				t.Errorf("ToolArguments should be empty: %q", ev.ToolArguments)
			}
		}
	}
	if !found {
		t.Fatal("done event not delivered")
	}
}

func TestSendToolResult_VariedPayloads(t *testing.T) {
	cases := []struct {
		name       string
		toolCallID string
		output     string
	}{
		{"empty-output", "call_1", ""},
		{"json-output", "call_2", `{"ok":true}`},
		{"plain-text", "call_3", "sunny"},
		{"unicode", "call_4", "結果: ok"},
		{"long-id", "call_" + strings.Repeat("x", 100), "ok"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
			if err != nil {
				t.Fatalf("Connect: %v", err)
			}
			defer func() {
				_ = sess.Close()
				<-done
			}()

			if err := sess.SendToolResult(ctx, tc.toolCallID, tc.output); err != nil {
				t.Fatalf("SendToolResult: %v", err)
			}

			frames := waitForFrames(t, cap, 3, 2*time.Second)
			var second map[string]any
			_ = json.Unmarshal([]byte(frames[1]), &second)
			item, _ := second["item"].(map[string]any)
			if item["call_id"] != tc.toolCallID {
				t.Errorf("call_id = %v, want %v", item["call_id"], tc.toolCallID)
			}
			// Empty output should be omitted by the omitempty tag.
			if tc.output == "" {
				if _, present := item["output"]; present {
					t.Errorf("empty output should be omitted; got %v", item["output"])
				}
			} else {
				if item["output"] != tc.output {
					t.Errorf("output = %v, want %v", item["output"], tc.output)
				}
			}
		})
	}
}

func TestSessionConfig_DefaultsModalitiesOmitted(t *testing.T) {
	p, cap, _, done := newFakeServer(t, serverOpts{holdOpen: true}, nil)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	sess, err := p.Connect(ctx, openairealtime.SessionConfig{})
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
	inner, _ := env["session"].(map[string]any)
	if _, present := inner["modalities"]; present {
		t.Errorf("modalities should be omitted when caller did not set; got %v", inner["modalities"])
	}
	if _, present := inner["voice"]; present {
		t.Errorf("voice should be omitted when empty; got %v", inner["voice"])
	}
	if _, present := inner["temperature"]; present {
		t.Errorf("temperature should be omitted when nil")
	}
}
