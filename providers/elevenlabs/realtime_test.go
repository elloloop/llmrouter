package elevenlabs_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/elevenlabs"
)

// capturedRealtime records what the fake server saw on the WS upgrade
// and during the in-band frame exchange.
type capturedRealtime struct {
	mu         sync.Mutex
	apiKey     string
	authHeader string
	path       string
	query      url.Values
	bos        map[string]any
	textFrames []string
	gotEOS     bool
	connClosed chan struct{}
}

// fakeRealtimeOpts tunes the behaviour of the fake server.
type fakeRealtimeOpts struct {
	// audioChunks are the bytes the server base64-encodes into successive
	// frames. The last frame carries isFinal=true.
	audioChunks [][]byte
	// rejectStatus, when non-zero, causes the HTTP upgrade to be rejected
	// with this status before the WS handshake completes.
	rejectStatus int
	// holdAfterEOS, when true, keeps the connection open after EOS so the
	// test can exercise client-side Close / cancel paths.
	holdAfterEOS bool
	// suppressFinal, when true, never sets isFinal=true and after sending
	// chunks blocks indefinitely on the connection. Useful for cancel
	// tests where the client must terminate the stream.
	suppressFinal bool
}

// newFakeRealtimeServer stands up an httptest server that emulates the
// /v1/text-to-speech/<voice>/stream-input websocket endpoint.
func newFakeRealtimeServer(t *testing.T, opts fakeRealtimeOpts) (*httptest.Server, *capturedRealtime) {
	t.Helper()
	cap := &capturedRealtime{connClosed: make(chan struct{})}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.mu.Lock()
		cap.apiKey = r.Header.Get("xi-api-key")
		cap.authHeader = r.Header.Get("Authorization")
		cap.path = r.URL.Path
		cap.query = r.URL.Query()
		cap.mu.Unlock()

		if opts.rejectStatus != 0 {
			w.WriteHeader(opts.rejectStatus)
			_, _ = w.Write([]byte(`{"detail":"forbidden"}`))
			return
		}

		conn, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			InsecureSkipVerify: true,
		})
		if err != nil {
			t.Logf("fake server: accept: %v", err)
			return
		}
		defer func() {
			_ = conn.Close(websocket.StatusNormalClosure, "server done")
			close(cap.connClosed)
		}()
		conn.SetReadLimit(8 * 1024 * 1024)

		ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()

		// 1) Read BOS.
		_, bosBytes, err := conn.Read(ctx)
		if err != nil {
			t.Logf("fake server: read bos: %v", err)
			return
		}
		var bos map[string]any
		_ = json.Unmarshal(bosBytes, &bos)
		cap.mu.Lock()
		cap.bos = bos
		cap.mu.Unlock()

		chunks := opts.audioChunks
		if len(chunks) == 0 {
			chunks = [][]byte{
				[]byte("chunk-one"),
				[]byte("chunk-two"),
				[]byte("chunk-three"),
			}
		}

		// In suppressFinal mode the server streams audio immediately
		// without waiting for the EOS so the client can be cancelled
		// mid-stream. In normal mode the server waits for EOS before
		// flushing audio.
		if opts.suppressFinal {
			// Start a reader goroutine to capture any text frames.
			readerDone := make(chan struct{})
			go func() {
				defer close(readerDone)
				for {
					_, raw, err := conn.Read(ctx)
					if err != nil {
						return
					}
					var tf struct {
						Text string `json:"text"`
					}
					if jerr := json.Unmarshal(raw, &tf); jerr != nil {
						continue
					}
					if tf.Text == "" {
						cap.mu.Lock()
						cap.gotEOS = true
						cap.mu.Unlock()
						continue
					}
					cap.mu.Lock()
					cap.textFrames = append(cap.textFrames, tf.Text)
					cap.mu.Unlock()
				}
			}()
			for _, c := range chunks {
				frame := map[string]any{
					"audio":   base64.StdEncoding.EncodeToString(c),
					"isFinal": false,
				}
				fb, _ := json.Marshal(frame)
				if err := conn.Write(ctx, websocket.MessageText, fb); err != nil {
					<-readerDone
					return
				}
			}
			<-readerDone
			return
		}

		// 2) Read text frames until EOS (frame whose text is "").
		for {
			_, raw, err := conn.Read(ctx)
			if err != nil {
				return
			}
			var tf struct {
				Text string `json:"text"`
			}
			if jerr := json.Unmarshal(raw, &tf); jerr != nil {
				continue
			}
			if tf.Text == "" {
				cap.mu.Lock()
				cap.gotEOS = true
				cap.mu.Unlock()
				break
			}
			cap.mu.Lock()
			cap.textFrames = append(cap.textFrames, tf.Text)
			cap.mu.Unlock()
		}

		// 3) Emit audio frames.
		for i, c := range chunks {
			frame := map[string]any{
				"audio":   base64.StdEncoding.EncodeToString(c),
				"isFinal": i == len(chunks)-1,
			}
			fb, _ := json.Marshal(frame)
			if err := conn.Write(ctx, websocket.MessageText, fb); err != nil {
				return
			}
		}

		if opts.holdAfterEOS {
			// Drain until peer closes / cancels.
			for {
				if _, _, err := conn.Read(ctx); err != nil {
					return
				}
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv, cap
}

// newTestProvider builds a Provider pointing at the fake server.
func newTestProvider(t *testing.T, baseURL string) *elevenlabs.Provider {
	t.Helper()
	p, err := elevenlabs.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(baseURL),
	)
	if err != nil {
		t.Fatalf("elevenlabs.New: %v", err)
	}
	return p
}

// collectAudio drains the stream and returns the assembled bytes.
func collectAudio(t *testing.T, stream *llmrouter.AudioStream) [][]byte {
	t.Helper()
	var out [][]byte
	timeout := time.After(5 * time.Second)
	for {
		select {
		case <-timeout:
			t.Fatalf("timeout draining audio stream")
		case c, ok := <-stream.Chunks():
			if !ok {
				return out
			}
			cp := make([]byte, len(c.Data))
			copy(cp, c.Data)
			out = append(out, cp)
		}
	}
}

func TestSpeakRealtime_HappyPath_SingleInput(t *testing.T) {
	srv, cap := newFakeRealtimeServer(t, fakeRealtimeOpts{
		audioChunks: [][]byte{
			[]byte("alpha"),
			[]byte("beta"),
			[]byte("gamma"),
		},
	})
	p := newTestProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{
		Voice: "voice-x",
		Model: "model-x",
		Input: "hello world",
	})
	if err != nil {
		t.Fatalf("SpeakRealtime: %v", err)
	}
	if err := rc.Finalize(ctx); err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	chunks := collectAudio(t, stream)
	if err := stream.Err(); err != nil {
		t.Fatalf("stream err: %v", err)
	}
	_ = rc.Close()

	if len(chunks) != 3 {
		t.Fatalf("got %d chunks want 3", len(chunks))
	}
	expected := []string{"alpha", "beta", "gamma"}
	for i, want := range expected {
		if string(chunks[i]) != want {
			t.Errorf("chunk %d: got %q want %q", i, chunks[i], want)
		}
	}

	cap.mu.Lock()
	defer cap.mu.Unlock()
	if cap.apiKey != "test-key" {
		t.Errorf("xi-api-key: got %q", cap.apiKey)
	}
	if cap.path != "/v1/text-to-speech/voice-x/stream-input" {
		t.Errorf("path: got %q", cap.path)
	}
	if cap.bos["text"] != " " {
		t.Errorf("BOS text: got %v", cap.bos["text"])
	}
	if vs, ok := cap.bos["voice_settings"].(map[string]any); !ok {
		t.Errorf("BOS voice_settings missing/wrong type: %T", cap.bos["voice_settings"])
	} else {
		if vs["stability"].(float64) != 0.5 {
			t.Errorf("stability: %v", vs["stability"])
		}
		if vs["similarity_boost"].(float64) != 0.8 {
			t.Errorf("similarity_boost: %v", vs["similarity_boost"])
		}
	}
	if len(cap.textFrames) != 1 || cap.textFrames[0] != "hello world" {
		t.Errorf("text frames: %#v", cap.textFrames)
	}
	if !cap.gotEOS {
		t.Errorf("did not see EOS")
	}
}

func TestSpeakRealtime_MultiTurn_AppendThenFinalize(t *testing.T) {
	srv, cap := newFakeRealtimeServer(t, fakeRealtimeOpts{})
	p := newTestProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{
		Voice: "v",
		Model: "m",
		Input: "",
	})
	if err != nil {
		t.Fatalf("SpeakRealtime: %v", err)
	}
	if err := rc.Append(ctx, "hello "); err != nil {
		t.Fatalf("Append 1: %v", err)
	}
	if err := rc.Append(ctx, "world."); err != nil {
		t.Fatalf("Append 2: %v", err)
	}
	if err := rc.Finalize(ctx); err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	chunks := collectAudio(t, stream)
	if len(chunks) != 3 {
		t.Fatalf("got %d chunks want 3", len(chunks))
	}
	_ = rc.Close()

	cap.mu.Lock()
	defer cap.mu.Unlock()
	if got := strings.Join(cap.textFrames, "|"); got != "hello |world." {
		t.Errorf("text frames joined: %q", got)
	}
	if !cap.gotEOS {
		t.Errorf("EOS missing")
	}
}

func TestSpeakRealtime_ContentTypeByFormat(t *testing.T) {
	cases := []struct {
		format string
		want   string
	}{
		{"mp3", "audio/mpeg"},
		{"", "audio/mpeg"},
		{"opus", "audio/opus"},
		{"pcm", "audio/pcm"},
		{"wav", "audio/pcm"}, // realtimeContentType: wav→audio/pcm (deviation from prompt note)
		{"ulaw", "audio/basic"},
		{"unknown-x", "audio/mpeg"},
	}
	for _, c := range cases {
		c := c
		t.Run("format="+c.format, func(t *testing.T) {
			srv, _ := newFakeRealtimeServer(t, fakeRealtimeOpts{})
			p := newTestProvider(t, srv.URL)
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{
				Voice:  "v",
				Format: c.format,
			})
			if err != nil {
				t.Fatalf("SpeakRealtime: %v", err)
			}
			if stream.ContentType != c.want {
				t.Errorf("ContentType=%q want %q", stream.ContentType, c.want)
			}
			_ = rc.Finalize(ctx)
			collectAudio(t, stream)
			_ = rc.Close()
		})
	}
}

func TestSpeakRealtime_DefaultVoiceApplied(t *testing.T) {
	srv, cap := newFakeRealtimeServer(t, fakeRealtimeOpts{})
	p := newTestProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{})
	if err != nil {
		t.Fatalf("SpeakRealtime: %v", err)
	}
	_ = rc.Finalize(ctx)
	collectAudio(t, stream)
	_ = rc.Close()

	cap.mu.Lock()
	defer cap.mu.Unlock()
	if cap.path != "/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM/stream-input" {
		t.Errorf("default voice path: got %q", cap.path)
	}
}

func TestSpeakRealtime_DefaultModelApplied(t *testing.T) {
	srv, cap := newFakeRealtimeServer(t, fakeRealtimeOpts{})
	p := newTestProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{Voice: "v"})
	if err != nil {
		t.Fatalf("SpeakRealtime: %v", err)
	}
	_ = rc.Finalize(ctx)
	collectAudio(t, stream)
	_ = rc.Close()

	cap.mu.Lock()
	defer cap.mu.Unlock()
	if got := cap.query.Get("model_id"); got != "eleven_turbo_v2_5" {
		t.Errorf("model_id: got %q", got)
	}
}

func TestSpeakRealtime_QueryParams(t *testing.T) {
	cases := []struct {
		format string
		want   string
	}{
		{"mp3", "mp3_44100_128"},
		{"opus", "opus_48000_128"},
		{"pcm", "pcm_44100"},
		{"wav", "pcm_44100"},
		{"ulaw", "ulaw_8000"},
		{"", "mp3_44100_128"},
		{"weird", "mp3_44100_128"},
	}
	for _, c := range cases {
		c := c
		t.Run("output_format/"+c.format, func(t *testing.T) {
			srv, cap := newFakeRealtimeServer(t, fakeRealtimeOpts{})
			p := newTestProvider(t, srv.URL)

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{
				Voice:  "v",
				Format: c.format,
			})
			if err != nil {
				t.Fatalf("SpeakRealtime: %v", err)
			}
			_ = rc.Finalize(ctx)
			collectAudio(t, stream)
			_ = rc.Close()

			cap.mu.Lock()
			defer cap.mu.Unlock()
			if got := cap.query.Get("output_format"); got != c.want {
				t.Errorf("output_format: got %q want %q", got, c.want)
			}
			if got := cap.query.Get("optimize_streaming_latency"); got != "2" {
				t.Errorf("optimize_streaming_latency: got %q", got)
			}
			if got := cap.query.Get("inactivity_timeout"); got != "20" {
				t.Errorf("inactivity_timeout: got %q", got)
			}
		})
	}
}

func TestSpeakRealtime_AuthHeader_NotInQueryOrAuthorization(t *testing.T) {
	srv, cap := newFakeRealtimeServer(t, fakeRealtimeOpts{})
	p := newTestProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{Voice: "v"})
	if err != nil {
		t.Fatalf("SpeakRealtime: %v", err)
	}
	_ = rc.Finalize(ctx)
	collectAudio(t, stream)
	_ = rc.Close()

	cap.mu.Lock()
	defer cap.mu.Unlock()
	if cap.apiKey != "test-key" {
		t.Errorf("xi-api-key header missing/wrong: %q", cap.apiKey)
	}
	if cap.authHeader != "" {
		t.Errorf("Authorization header should be empty, got: %q", cap.authHeader)
	}
	if cap.query.Get("xi-api-key") != "" || cap.query.Get("xi_api_key") != "" {
		t.Errorf("api key leaked into query: %v", cap.query)
	}
}

func TestSpeakRealtime_ContextCancelMidStream(t *testing.T) {
	// suppressFinal=true ensures the pump never sees isFinal and so the
	// only way the stream terminates is via context cancellation.
	srv, cap := newFakeRealtimeServer(t, fakeRealtimeOpts{
		suppressFinal: true,
		audioChunks: [][]byte{
			[]byte("one"),
		},
	})
	p := newTestProvider(t, srv.URL)

	ctx, cancel := context.WithCancel(context.Background())

	stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{
		Voice: "v",
		Input: "hi",
	})
	if err != nil {
		cancel()
		t.Fatalf("SpeakRealtime: %v", err)
	}

	// Wait for at least one chunk to arrive then cancel.
	select {
	case <-stream.Chunks():
	case <-time.After(2 * time.Second):
		cancel()
		t.Fatalf("timeout waiting for first chunk")
	}
	cancel()

	// Drain chunks until the channel closes.
drain:
	for {
		select {
		case _, ok := <-stream.Chunks():
			if !ok {
				break drain
			}
		case <-time.After(3 * time.Second):
			t.Fatalf("timeout waiting for stream close after cancel")
		}
	}
	// The pump treats context.Canceled as a clean end-of-stream and
	// returns nil from Finish — assert only that Err is reachable
	// (does not block) and that the channel did close.
	_ = stream.Err()
	_ = rc.Close()

	select {
	case <-cap.connClosed:
	case <-time.After(2 * time.Second):
		t.Errorf("server side did not observe connection close after client cancel")
	}
}

func TestSpeakRealtime_HandshakeRejected_ReturnsErrUpstream(t *testing.T) {
	srv, _ := newFakeRealtimeServer(t, fakeRealtimeOpts{rejectStatus: 403})
	p := newTestProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, _, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{Voice: "v"})
	if err == nil {
		t.Fatalf("expected error on 403 handshake")
	}
	var up *llmrouter.ErrUpstream
	if !errors.As(err, &up) {
		t.Fatalf("expected *ErrUpstream, got %T: %v", err, err)
	}
	if up.Provider != "elevenlabs" {
		t.Errorf("Provider: got %q", up.Provider)
	}
	if up.StatusCode != 403 {
		t.Errorf("StatusCode: got %d want 403", up.StatusCode)
	}
	if !strings.Contains(up.Body, "forbidden") {
		t.Errorf("Body: %q", up.Body)
	}
}

func TestSpeakRealtime_CloseIsIdempotent(t *testing.T) {
	srv, _ := newFakeRealtimeServer(t, fakeRealtimeOpts{})
	p := newTestProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{Voice: "v"})
	if err != nil {
		t.Fatalf("SpeakRealtime: %v", err)
	}
	_ = rc.Finalize(ctx)
	collectAudio(t, stream)

	// First close: may return nil or "already closed" depending on
	// whether the pump goroutine raced ahead. We only assert that it
	// does not panic.
	_ = rc.Close()
	// Second close must also not panic and must not block.
	doneCh := make(chan struct{})
	go func() {
		_ = rc.Close()
		close(doneCh)
	}()
	select {
	case <-doneCh:
	case <-time.After(2 * time.Second):
		t.Fatalf("second Close blocked")
	}
}

func TestSpeakRealtime_AppendAfterClose_ReturnsError(t *testing.T) {
	srv, _ := newFakeRealtimeServer(t, fakeRealtimeOpts{suppressFinal: true})
	p := newTestProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{Voice: "v"})
	if err != nil {
		t.Fatalf("SpeakRealtime: %v", err)
	}
	_ = rc.Close()
	// Drain so test does not leak the pump goroutine.
	go func() {
		for range stream.Chunks() {
		}
	}()

	if err := rc.Append(ctx, "after close"); err == nil {
		t.Errorf("expected error appending after Close")
	}
}

func TestSpeakRealtime_AppendAfterFinalize_NoCrash(t *testing.T) {
	// The pump has likely returned by the time Finalize returns; any
	// subsequent Append should either return an error or be a no-op,
	// but must not panic.
	srv, _ := newFakeRealtimeServer(t, fakeRealtimeOpts{})
	p := newTestProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{Voice: "v"})
	if err != nil {
		t.Fatalf("SpeakRealtime: %v", err)
	}
	if err := rc.Finalize(ctx); err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	collectAudio(t, stream)

	// Append("") is documented as a no-op (returns nil without writing).
	if err := rc.Append(ctx, ""); err != nil {
		t.Errorf("Append(\"\") after Finalize: got %v want nil", err)
	}
	// Append non-empty: do not assert success vs error — just no panic.
	_ = rc.Append(ctx, "tail")
	_ = rc.Close()
}

func TestSpeakRealtime_Append_EmptyStringIsNoop(t *testing.T) {
	srv, cap := newFakeRealtimeServer(t, fakeRealtimeOpts{})
	p := newTestProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{Voice: "v"})
	if err != nil {
		t.Fatalf("SpeakRealtime: %v", err)
	}
	if err := rc.Append(ctx, ""); err != nil {
		t.Errorf("Append(\"\"): got err %v want nil", err)
	}
	if err := rc.Append(ctx, "real chunk"); err != nil {
		t.Errorf("Append non-empty: %v", err)
	}
	_ = rc.Finalize(ctx)
	collectAudio(t, stream)
	_ = rc.Close()

	cap.mu.Lock()
	defer cap.mu.Unlock()
	if len(cap.textFrames) != 1 || cap.textFrames[0] != "real chunk" {
		t.Errorf("expected exactly one non-empty text frame, got: %#v", cap.textFrames)
	}
}

func TestSpeakRealtime_RawChunkDataMatches(t *testing.T) {
	srv, _ := newFakeRealtimeServer(t, fakeRealtimeOpts{
		audioChunks: [][]byte{
			{0x00, 0x01, 0x02, 0x03, 0xff, 0xfe, 0xfd},
			{0xaa, 0xbb, 0xcc},
			{0xde, 0xad, 0xbe, 0xef},
		},
	})
	p := newTestProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, rc, err := p.SpeakRealtime(ctx, llmrouter.SpeechRequest{
		Voice: "v",
		Input: "ping",
	})
	if err != nil {
		t.Fatalf("SpeakRealtime: %v", err)
	}
	if err := rc.Finalize(ctx); err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	chunks := collectAudio(t, stream)
	_ = rc.Close()

	want := [][]byte{
		{0x00, 0x01, 0x02, 0x03, 0xff, 0xfe, 0xfd},
		{0xaa, 0xbb, 0xcc},
		{0xde, 0xad, 0xbe, 0xef},
	}
	if len(chunks) != len(want) {
		t.Fatalf("chunk count: got %d want %d", len(chunks), len(want))
	}
	for i := range want {
		if string(chunks[i]) != string(want[i]) {
			t.Errorf("chunk %d: got %x want %x", i, chunks[i], want[i])
		}
	}
}
