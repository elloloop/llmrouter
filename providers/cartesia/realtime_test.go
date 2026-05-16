package cartesia

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
)

// realtimeCapture records what the fake WebSocket server saw across
// the lifetime of a single connection.
type realtimeCapture struct {
	mu sync.Mutex

	queryAPIKey  string
	queryVersion string

	frames []map[string]any
}

func (rc *realtimeCapture) recordFrame(buf []byte) {
	var frame map[string]any
	if err := json.Unmarshal(buf, &frame); err != nil {
		return
	}
	rc.mu.Lock()
	defer rc.mu.Unlock()
	rc.frames = append(rc.frames, frame)
}

func (rc *realtimeCapture) snapshot() []map[string]any {
	rc.mu.Lock()
	defer rc.mu.Unlock()
	out := make([]map[string]any, len(rc.frames))
	copy(out, rc.frames)
	return out
}

// fakeRealtimeServer is a configurable in-process WebSocket TTS server.
// audioChunks are emitted as base64-encoded data frames; the last frame
// also carries done:true (or a separate done frame is sent if
// trailingDone is true).
type fakeRealtimeServer struct {
	audioChunks       [][]byte
	trailingDone      bool
	expectedFrameCount int
	// waitForFrames blocks the server's emit loop until this many frames
	// have arrived from the client. 0 means start emitting immediately.
	waitForFrames int
	// onConnect, if non-nil, is invoked after a successful accept.
	onConnect func(conn *websocket.Conn)
}

func newFakeRealtimeServer(t *testing.T, cfg fakeRealtimeServer, cap *realtimeCapture) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.mu.Lock()
		cap.queryAPIKey = r.URL.Query().Get("api_key")
		cap.queryVersion = r.URL.Query().Get("cartesia_version")
		cap.mu.Unlock()

		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{InsecureSkipVerify: true})
		if err != nil {
			return
		}
		defer conn.Close(websocket.StatusNormalClosure, "bye")
		conn.SetReadLimit(8 * 1024 * 1024)

		readCtx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()

		// Read the initial frame.
		typ, data, err := conn.Read(readCtx)
		if err != nil || typ != websocket.MessageText {
			return
		}
		cap.recordFrame(data)

		if cfg.onConnect != nil {
			cfg.onConnect(conn)
		}

		// Drain any additional client frames concurrently so reads from
		// Append/Finalize are observed.
		done := make(chan struct{})
		go func() {
			defer close(done)
			for {
				ftyp, fdata, ferr := conn.Read(readCtx)
				if ferr != nil {
					return
				}
				if ftyp != websocket.MessageText {
					continue
				}
				cap.recordFrame(fdata)
			}
		}()

		// Optionally wait until N total frames have arrived.
		if cfg.waitForFrames > 0 {
			deadline := time.Now().Add(2 * time.Second)
			for time.Now().Before(deadline) {
				cap.mu.Lock()
				count := len(cap.frames)
				cap.mu.Unlock()
				if count >= cfg.waitForFrames {
					break
				}
				time.Sleep(10 * time.Millisecond)
			}
		}

		// Emit audio chunks.
		for i, payload := range cfg.audioChunks {
			isLast := i == len(cfg.audioChunks)-1
			frame := map[string]any{
				"context_id": "test-ctx",
				"data":       base64.StdEncoding.EncodeToString(payload),
				"done":       isLast && !cfg.trailingDone,
			}
			buf, _ := json.Marshal(frame)
			if err := conn.Write(readCtx, websocket.MessageText, buf); err != nil {
				return
			}
		}
		if cfg.trailingDone {
			doneFrame := map[string]any{"context_id": "test-ctx", "done": true}
			buf, _ := json.Marshal(doneFrame)
			_ = conn.Write(readCtx, websocket.MessageText, buf)
		}
		// Give the client a moment to drain before the cancel fires.
		select {
		case <-done:
		case <-time.After(500 * time.Millisecond):
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// newFakeRealtimeErrServer rejects the WebSocket handshake with the given
// HTTP status code.
func newFakeRealtimeErrServer(t *testing.T, status int) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		_, _ = w.Write([]byte(`{"error":"bad"}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func drainAudio(t *testing.T, stream *llmrouter.AudioStream) ([]llmrouter.AudioChunk, error) {
	t.Helper()
	var chunks []llmrouter.AudioChunk
	for c := range stream.Chunks() {
		chunks = append(chunks, c)
	}
	return chunks, stream.Err()
}

func TestSpeakRealtime_HappyPath(t *testing.T) {
	payloads := [][]byte{[]byte("frame-1"), []byte("frame-2"), []byte("frame-3")}
	cap := &realtimeCapture{}
	srv := newFakeRealtimeServer(t, fakeRealtimeServer{audioChunks: payloads}, cap)
	p := newTestProvider(t, srv.URL)

	stream, rc, err := p.SpeakRealtime(context.Background(), llmrouter.SpeechRequest{
		Input:  "hello world",
		Format: "pcm",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer rc.Close()

	chunks, derr := drainAudio(t, stream)
	if derr != nil {
		t.Fatalf("drain err: %v", derr)
	}
	if len(chunks) != 3 {
		t.Fatalf("want 3 chunks, got %d", len(chunks))
	}
	for i, c := range chunks {
		if string(c.Data) != string(payloads[i]) {
			t.Fatalf("chunk %d data: got %q want %q", i, c.Data, payloads[i])
		}
		if len(c.Raw) == 0 {
			t.Fatalf("chunk %d: Raw must carry original payload", i)
		}
	}
}

func TestSpeakRealtime_QueryParams(t *testing.T) {
	cap := &realtimeCapture{}
	srv := newFakeRealtimeServer(t, fakeRealtimeServer{
		audioChunks: [][]byte{[]byte("x")},
	}, cap)
	p := newTestProvider(t, srv.URL)

	stream, rc, err := p.SpeakRealtime(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer rc.Close()
	_, _ = drainAudio(t, stream)

	cap.mu.Lock()
	defer cap.mu.Unlock()
	if cap.queryAPIKey != "test-key" {
		t.Fatalf("api_key query param: got %q want test-key", cap.queryAPIKey)
	}
	if cap.queryVersion != cartesiaVersion {
		t.Fatalf("cartesia_version query param: got %q want %q", cap.queryVersion, cartesiaVersion)
	}
}

func TestSpeakRealtime_InitialFrame(t *testing.T) {
	cap := &realtimeCapture{}
	srv := newFakeRealtimeServer(t, fakeRealtimeServer{
		audioChunks: [][]byte{[]byte("x")},
	}, cap)
	p := newTestProvider(t, srv.URL)

	stream, rc, err := p.SpeakRealtime(context.Background(), llmrouter.SpeechRequest{Input: "the input"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer rc.Close()
	_, _ = drainAudio(t, stream)

	frames := cap.snapshot()
	if len(frames) < 1 {
		t.Fatalf("expected at least one frame, got %d", len(frames))
	}
	first := frames[0]

	t.Run("context_id non-empty", func(t *testing.T) {
		cid, _ := first["context_id"].(string)
		if cid == "" {
			t.Fatalf("context_id missing or empty: %v", first["context_id"])
		}
	})
	t.Run("continue is true", func(t *testing.T) {
		if cont, _ := first["continue"].(bool); !cont {
			t.Fatalf("continue: got %v want true", first["continue"])
		}
	})
	t.Run("model_id defaults to sonic-2", func(t *testing.T) {
		if first["model_id"] != "sonic-2" {
			t.Fatalf("model_id: got %v want sonic-2", first["model_id"])
		}
	})
	t.Run("transcript matches input", func(t *testing.T) {
		if first["transcript"] != "the input" {
			t.Fatalf("transcript: got %v want 'the input'", first["transcript"])
		}
	})
	t.Run("voice is an object with mode/id", func(t *testing.T) {
		voice, _ := first["voice"].(map[string]any)
		if voice == nil {
			t.Fatalf("voice missing: %v", first["voice"])
		}
		if voice["mode"] != "id" || voice["id"] != defaultVoiceID {
			t.Fatalf("voice: got %v want {mode:id, id:%s}", voice, defaultVoiceID)
		}
	})
}

func TestSpeakRealtime_ContentType(t *testing.T) {
	cases := []struct {
		format string
		want   string
	}{
		{"mp3", "audio/mpeg"},
		{"wav", "audio/wav"},
		{"pcm", "audio/pcm"},
		{"ulaw", "audio/basic"},
		{"", "audio/pcm"},
		{"unknown", "audio/pcm"},
	}
	for _, tc := range cases {
		t.Run(tc.format, func(t *testing.T) {
			cap := &realtimeCapture{}
			srv := newFakeRealtimeServer(t, fakeRealtimeServer{
				audioChunks: [][]byte{[]byte("x")},
			}, cap)
			p := newTestProvider(t, srv.URL)

			stream, rc, err := p.SpeakRealtime(context.Background(), llmrouter.SpeechRequest{
				Input:  "hi",
				Format: tc.format,
			})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			defer rc.Close()
			if stream.ContentType != tc.want {
				t.Fatalf("ContentType: got %q want %q", stream.ContentType, tc.want)
			}
			_, _ = drainAudio(t, stream)
		})
	}
}

func TestSpeakRealtime_EmptyInput(t *testing.T) {
	cap := &realtimeCapture{}
	srv := newFakeRealtimeServer(t, fakeRealtimeServer{
		audioChunks: [][]byte{[]byte("late")},
		waitForFrames: 2, // wait for initial + Append before emitting
	}, cap)
	p := newTestProvider(t, srv.URL)

	stream, rc, err := p.SpeakRealtime(context.Background(), llmrouter.SpeechRequest{Input: ""})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer rc.Close()

	// Server saw initial frame; no chunks until Append.
	if err := rc.Append(context.Background(), "now say this"); err != nil {
		t.Fatalf("append: %v", err)
	}

	chunks, derr := drainAudio(t, stream)
	if derr != nil {
		t.Fatalf("drain err: %v", derr)
	}
	if len(chunks) != 1 {
		t.Fatalf("want 1 chunk, got %d", len(chunks))
	}

	frames := cap.snapshot()
	if len(frames) < 1 {
		t.Fatalf("expected at least the initial frame, got %d", len(frames))
	}
	if frames[0]["transcript"] != "" {
		t.Fatalf("initial transcript should be empty, got %v", frames[0]["transcript"])
	}
}

func TestSpeakRealtime_Append(t *testing.T) {
	cap := &realtimeCapture{}
	srv := newFakeRealtimeServer(t, fakeRealtimeServer{
		audioChunks:   [][]byte{[]byte("x")},
		waitForFrames: 2, // initial + Append
	}, cap)
	p := newTestProvider(t, srv.URL)

	stream, rc, err := p.SpeakRealtime(context.Background(), llmrouter.SpeechRequest{Input: "first"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer rc.Close()

	if err := rc.Append(context.Background(), "second"); err != nil {
		t.Fatalf("append: %v", err)
	}
	_, _ = drainAudio(t, stream)

	frames := cap.snapshot()
	if len(frames) < 2 {
		t.Fatalf("expected at least 2 frames (initial + append), got %d", len(frames))
	}
	appendFrame := frames[1]
	if appendFrame["transcript"] != "second" {
		t.Fatalf("append transcript: got %v want 'second'", appendFrame["transcript"])
	}
	if cont, _ := appendFrame["continue"].(bool); !cont {
		t.Fatalf("append continue: want true, got %v", appendFrame["continue"])
	}
	if appendFrame["context_id"] != frames[0]["context_id"] {
		t.Fatalf("append context_id mismatch: %v vs %v", appendFrame["context_id"], frames[0]["context_id"])
	}
}

func TestSpeakRealtime_Finalize(t *testing.T) {
	cap := &realtimeCapture{}
	srv := newFakeRealtimeServer(t, fakeRealtimeServer{
		audioChunks:   [][]byte{[]byte("x")},
		waitForFrames: 2, // initial + finalize
	}, cap)
	p := newTestProvider(t, srv.URL)

	stream, rc, err := p.SpeakRealtime(context.Background(), llmrouter.SpeechRequest{Input: "all"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer rc.Close()

	if err := rc.Finalize(context.Background()); err != nil {
		t.Fatalf("finalize: %v", err)
	}
	_, _ = drainAudio(t, stream)

	frames := cap.snapshot()
	if len(frames) < 2 {
		t.Fatalf("expected at least 2 frames (initial + finalize), got %d", len(frames))
	}
	fin := frames[1]
	if fin["transcript"] != "" {
		t.Fatalf("finalize transcript: got %v want ''", fin["transcript"])
	}
	if cont, ok := fin["continue"].(bool); !ok || cont {
		t.Fatalf("finalize continue: want false, got %v", fin["continue"])
	}
	if fin["context_id"] != frames[0]["context_id"] {
		t.Fatalf("finalize context_id mismatch: %v vs %v", fin["context_id"], frames[0]["context_id"])
	}
}

func TestSpeakRealtime_ContextCancel(t *testing.T) {
	// Server that holds the connection open and never emits anything.
	cap := &realtimeCapture{}
	srv := newFakeRealtimeServer(t, fakeRealtimeServer{
		audioChunks: [][]byte{},
		onConnect: func(conn *websocket.Conn) {
			// Park the server side; the client cancel should still
			// tear the connection down.
			time.Sleep(2 * time.Second)
		},
	}, cap)
	p := newTestProvider(t, srv.URL)

	ctx, cancel := context.WithCancel(context.Background())
	stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{Input: "hi"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Cancel mid-stream and confirm the pump exits.
	cancel()

	done := make(chan struct{})
	go func() {
		_, _ = drainAudio(t, stream)
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("stream did not terminate after cancel")
	}
	_ = rc.Close()
}

func TestSpeakRealtime_HandshakeError(t *testing.T) {
	t.Run("4xx returns ErrUpstream with StatusCode 0", func(t *testing.T) {
		srv := newFakeRealtimeErrServer(t, http.StatusUnauthorized)
		p := newTestProvider(t, srv.URL)
		_, _, err := p.SpeakRealtime(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
		if err == nil {
			t.Fatal("expected error")
		}
		var upErr *llmrouter.ErrUpstream
		if !errors.As(err, &upErr) {
			t.Fatalf("expected *ErrUpstream, got %T: %v", err, err)
		}
		if upErr.Provider != "cartesia" {
			t.Fatalf("provider: got %q want cartesia", upErr.Provider)
		}
		if upErr.StatusCode != 0 {
			t.Fatalf("status: got %d want 0", upErr.StatusCode)
		}
		if upErr.Body == "" {
			t.Fatalf("body should carry handshake error message")
		}
	})

	t.Run("bad voice / 400 returns ErrUpstream", func(t *testing.T) {
		srv := newFakeRealtimeErrServer(t, http.StatusBadRequest)
		p := newTestProvider(t, srv.URL)
		_, _, err := p.SpeakRealtime(context.Background(), llmrouter.SpeechRequest{
			Input: "hi",
			Voice: "bogus",
		})
		if err == nil {
			t.Fatal("expected error")
		}
		var upErr *llmrouter.ErrUpstream
		if !errors.As(err, &upErr) {
			t.Fatalf("expected *ErrUpstream, got %T", err)
		}
		if upErr.Provider != "cartesia" {
			t.Fatalf("provider: got %q want cartesia", upErr.Provider)
		}
	})
}

func TestSpeakRealtime_CloseIdempotent(t *testing.T) {
	cap := &realtimeCapture{}
	srv := newFakeRealtimeServer(t, fakeRealtimeServer{
		audioChunks: [][]byte{[]byte("x")},
	}, cap)
	p := newTestProvider(t, srv.URL)

	stream, rc, err := p.SpeakRealtime(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, _ = drainAudio(t, stream)

	first := rc.Close()
	second := rc.Close()
	third := rc.Close()
	if !errorsEqual(first, second) || !errorsEqual(second, third) {
		t.Fatalf("close should be idempotent: %v / %v / %v", first, second, third)
	}
}

func TestSpeakRealtime_ContextIDPropagation(t *testing.T) {
	cap := &realtimeCapture{}
	srv := newFakeRealtimeServer(t, fakeRealtimeServer{
		audioChunks:   [][]byte{[]byte("x")},
		waitForFrames: 3, // initial + append + finalize
	}, cap)
	p := newTestProvider(t, srv.URL)

	stream, rc, err := p.SpeakRealtime(context.Background(), llmrouter.SpeechRequest{Input: "one"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer rc.Close()

	if err := rc.Append(context.Background(), "two"); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := rc.Finalize(context.Background()); err != nil {
		t.Fatalf("finalize: %v", err)
	}
	_, _ = drainAudio(t, stream)

	frames := cap.snapshot()
	if len(frames) < 3 {
		t.Fatalf("expected 3 frames, got %d", len(frames))
	}
	cid := frames[0]["context_id"]
	if cid == nil || cid == "" {
		t.Fatalf("initial context_id missing")
	}
	if frames[1]["context_id"] != cid {
		t.Fatalf("append context_id: got %v want %v", frames[1]["context_id"], cid)
	}
	if frames[2]["context_id"] != cid {
		t.Fatalf("finalize context_id: got %v want %v", frames[2]["context_id"], cid)
	}
	// And it must match the RealtimeContext.contextID.
	if cidStr, _ := cid.(string); cidStr != rc.contextID {
		t.Fatalf("wire context_id %q != rc.contextID %q", cidStr, rc.contextID)
	}
}

func TestSpeakRealtime_AppendEmptyIsNoop(t *testing.T) {
	cap := &realtimeCapture{}
	srv := newFakeRealtimeServer(t, fakeRealtimeServer{
		audioChunks: [][]byte{[]byte("x")},
	}, cap)
	p := newTestProvider(t, srv.URL)

	stream, rc, err := p.SpeakRealtime(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer rc.Close()

	if err := rc.Append(context.Background(), ""); err != nil {
		t.Fatalf("append empty: %v", err)
	}
	_, _ = drainAudio(t, stream)

	frames := cap.snapshot()
	for i, f := range frames {
		if i == 0 {
			continue
		}
		if f["transcript"] == "" {
			t.Fatalf("empty append should not produce a frame, found %v", f)
		}
	}
}

func TestBuildRealtimeURL(t *testing.T) {
	cases := []struct {
		name    string
		baseURL string
		wantPrefix string
	}{
		{"https → wss", "https://api.cartesia.ai", "wss://api.cartesia.ai/tts/websocket?"},
		{"http → ws", "http://localhost:9000", "ws://localhost:9000/tts/websocket?"},
		{"unknown left alone", "ws://example", "ws://example/tts/websocket?"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := buildRealtimeURL(tc.baseURL, "K")
			if !strings.HasPrefix(got, tc.wantPrefix) {
				t.Fatalf("got %q, want prefix %q", got, tc.wantPrefix)
			}
			if !strings.Contains(got, "api_key=K") {
				t.Fatalf("missing api_key in %q", got)
			}
			if !strings.Contains(got, "cartesia_version="+cartesiaVersion) {
				t.Fatalf("missing cartesia_version in %q", got)
			}
		})
	}
}

// errorsEqual treats two errors as equal when both are nil or both
// stringify to the same value. Sufficient for the close-idempotence
// check above.
func errorsEqual(a, b error) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return a.Error() == b.Error()
}
