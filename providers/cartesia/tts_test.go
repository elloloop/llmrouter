package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
)

// captured holds what the fake server saw on a request — used by tests
// to assert headers, body shape, and the path that was hit.
type captured struct {
	mu      sync.Mutex
	path    string
	headers http.Header
	body    map[string]any
}

func (c *captured) snapshot() captured {
	c.mu.Lock()
	defer c.mu.Unlock()
	return captured{path: c.path, headers: c.headers.Clone(), body: c.body}
}

// newFakeBatchServer returns a test server that serves /tts/bytes with
// the supplied binary audio payload and Content-Type.
func newFakeBatchServer(t *testing.T, audio []byte, contentType string, cap *captured) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		cap.mu.Lock()
		cap.path = r.URL.Path
		cap.headers = r.Header.Clone()
		cap.body = body
		cap.mu.Unlock()

		if contentType != "" {
			w.Header().Set("Content-Type", contentType)
		} else {
			// Prevent Go's auto Content-Type sniffing so tests can
			// exercise the "no upstream content-type" fallback path.
			w.Header()["Content-Type"] = nil
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(audio)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// newFakeSSEServer returns a test server that serves /tts/sse, emitting
// the supplied SSE events verbatim.
func newFakeSSEServer(t *testing.T, events []string, cap *captured) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		cap.mu.Lock()
		cap.path = r.URL.Path
		cap.headers = r.Header.Clone()
		cap.body = body
		cap.mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		for _, ev := range events {
			_, _ = io.WriteString(w, ev)
			if flusher != nil {
				flusher.Flush()
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// newFakeErrServer returns a test server that responds with the given
// status code and body.
func newFakeErrServer(t *testing.T, status int, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// drainStream collects all AudioChunks and the terminal error.
func drainStream(t *testing.T, stream *llmrouter.AudioStream) ([]llmrouter.AudioChunk, error) {
	t.Helper()
	var chunks []llmrouter.AudioChunk
	for c := range stream.Chunks() {
		chunks = append(chunks, c)
	}
	return chunks, stream.Err()
}

func newTestProvider(t *testing.T, baseURL string) *Provider {
	t.Helper()
	p, err := New(llmrouter.WithAPIKey("test-key"), llmrouter.WithBaseURL(baseURL))
	if err != nil {
		t.Fatalf("new provider: %v", err)
	}
	return p
}

func TestSpeak_Batch(t *testing.T) {
	t.Run("returns single chunk with audio bytes", func(t *testing.T) {
		audio := []byte("\x00\x01\x02\x03binary-audio")
		cap := &captured{}
		srv := newFakeBatchServer(t, audio, "audio/mpeg", cap)
		p := newTestProvider(t, srv.URL)

		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
			Input:  "hello world",
			Format: "mp3",
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		chunks, err := drainStream(t, stream)
		if err != nil {
			t.Fatalf("stream err: %v", err)
		}
		if len(chunks) != 1 {
			t.Fatalf("want 1 chunk, got %d", len(chunks))
		}
		if string(chunks[0].Data) != string(audio) {
			t.Fatalf("chunk data mismatch: got %q want %q", chunks[0].Data, audio)
		}
		if stream.ContentType != "audio/mpeg" {
			t.Fatalf("content-type from response header expected, got %q", stream.ContentType)
		}
	})

	t.Run("hits /tts/bytes for non-streaming", func(t *testing.T) {
		cap := &captured{}
		srv := newFakeBatchServer(t, []byte("x"), "audio/pcm", cap)
		p := newTestProvider(t, srv.URL)

		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainStream(t, stream)
		if got := cap.snapshot().path; got != "/tts/bytes" {
			t.Fatalf("expected /tts/bytes, got %q", got)
		}
	})

	t.Run("falls back to format-derived content type when header missing", func(t *testing.T) {
		cap := &captured{}
		srv := newFakeBatchServer(t, []byte("x"), "", cap)
		p := newTestProvider(t, srv.URL)

		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi", Format: "wav"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainStream(t, stream)
		if stream.ContentType != "audio/wav" {
			t.Fatalf("expected audio/wav, got %q", stream.ContentType)
		}
	})
}

func TestSpeak_Streaming(t *testing.T) {
	encode := func(s string) string { return base64.StdEncoding.EncodeToString([]byte(s)) }
	t.Run("emits chunks and finishes on done", func(t *testing.T) {
		events := []string{
			"data: " + fmt.Sprintf(`{"type":"chunk","data":"%s","step":0}`, encode("frame-1")) + "\n\n",
			"data: " + fmt.Sprintf(`{"type":"chunk","data":"%s","step":1}`, encode("frame-2")) + "\n\n",
			"data: " + fmt.Sprintf(`{"type":"chunk","data":"%s","step":2}`, encode("frame-3")) + "\n\n",
			"data: " + `{"type":"done"}` + "\n\n",
		}
		cap := &captured{}
		srv := newFakeSSEServer(t, events, cap)
		p := newTestProvider(t, srv.URL)

		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
			Input:  "hi",
			Stream: true,
			Format: "pcm",
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		chunks, err := drainStream(t, stream)
		if err != nil {
			t.Fatalf("stream err: %v", err)
		}
		if len(chunks) != 3 {
			t.Fatalf("want 3 chunks, got %d", len(chunks))
		}
		want := []string{"frame-1", "frame-2", "frame-3"}
		for i, c := range chunks {
			if string(c.Data) != want[i] {
				t.Fatalf("chunk %d: got %q want %q", i, c.Data, want[i])
			}
			if len(c.Raw) == 0 {
				t.Fatalf("chunk %d: Raw should be the original SSE payload", i)
			}
		}
	})

	t.Run("hits /tts/sse for streaming", func(t *testing.T) {
		events := []string{"data: " + `{"type":"done"}` + "\n\n"}
		cap := &captured{}
		srv := newFakeSSEServer(t, events, cap)
		p := newTestProvider(t, srv.URL)

		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi", Stream: true})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainStream(t, stream)
		if got := cap.snapshot().path; got != "/tts/sse" {
			t.Fatalf("expected /tts/sse, got %q", got)
		}
	})

	t.Run("derives content type from request format", func(t *testing.T) {
		events := []string{
			"data: " + fmt.Sprintf(`{"type":"chunk","data":"%s"}`, encode("x")) + "\n\n",
			"data: " + `{"type":"done"}` + "\n\n",
		}
		cap := &captured{}
		srv := newFakeSSEServer(t, events, cap)
		p := newTestProvider(t, srv.URL)

		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
			Input:  "hi",
			Stream: true,
			Format: "ulaw",
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainStream(t, stream)
		if stream.ContentType != "audio/basic" {
			t.Fatalf("expected audio/basic for ulaw, got %q", stream.ContentType)
		}
	})

	t.Run("accept header is text/event-stream", func(t *testing.T) {
		events := []string{"data: " + `{"type":"done"}` + "\n\n"}
		cap := &captured{}
		srv := newFakeSSEServer(t, events, cap)
		p := newTestProvider(t, srv.URL)
		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi", Stream: true})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainStream(t, stream)
		if got := cap.snapshot().headers.Get("Accept"); got != "text/event-stream" {
			t.Fatalf("expected Accept text/event-stream, got %q", got)
		}
	})

	t.Run("ignores unknown event types", func(t *testing.T) {
		events := []string{
			"data: " + `{"type":"flush"}` + "\n\n",
			"data: " + fmt.Sprintf(`{"type":"chunk","data":"%s"}`, encode("only-one")) + "\n\n",
			"data: " + `{"type":"timestamps","words":[]}` + "\n\n",
			"data: " + `{"type":"done"}` + "\n\n",
		}
		cap := &captured{}
		srv := newFakeSSEServer(t, events, cap)
		p := newTestProvider(t, srv.URL)

		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi", Stream: true})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		chunks, err := drainStream(t, stream)
		if err != nil {
			t.Fatalf("stream err: %v", err)
		}
		if len(chunks) != 1 {
			t.Fatalf("want 1 chunk, got %d", len(chunks))
		}
	})

	t.Run("tolerates malformed event json", func(t *testing.T) {
		events := []string{
			"data: " + `{not json` + "\n\n",
			"data: " + fmt.Sprintf(`{"type":"chunk","data":"%s"}`, encode("ok")) + "\n\n",
			"data: " + `{"type":"done"}` + "\n\n",
		}
		cap := &captured{}
		srv := newFakeSSEServer(t, events, cap)
		p := newTestProvider(t, srv.URL)

		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi", Stream: true})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		chunks, err := drainStream(t, stream)
		if err != nil {
			t.Fatalf("stream err: %v", err)
		}
		if len(chunks) != 1 {
			t.Fatalf("want 1 chunk, got %d", len(chunks))
		}
	})

	t.Run("surfaces base64 decode error", func(t *testing.T) {
		events := []string{
			"data: " + `{"type":"chunk","data":"!!!!not-base64"}` + "\n\n",
			"data: " + `{"type":"done"}` + "\n\n",
		}
		cap := &captured{}
		srv := newFakeSSEServer(t, events, cap)
		p := newTestProvider(t, srv.URL)

		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi", Stream: true})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, err = drainStream(t, stream)
		if err == nil {
			t.Fatal("expected error from invalid base64")
		}
	})
}

func TestSpeak_Defaults(t *testing.T) {
	t.Run("default voice applied when empty", func(t *testing.T) {
		cap := &captured{}
		srv := newFakeBatchServer(t, []byte("x"), "audio/pcm", cap)
		p := newTestProvider(t, srv.URL)
		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainStream(t, stream)
		snap := cap.snapshot()
		voice, _ := snap.body["voice"].(map[string]any)
		if voice == nil {
			t.Fatalf("voice missing from body: %+v", snap.body)
		}
		if got := voice["id"]; got != defaultVoiceID {
			t.Fatalf("expected default voice id %q, got %v", defaultVoiceID, got)
		}
		if got := voice["mode"]; got != "id" {
			t.Fatalf("expected voice mode 'id', got %v", got)
		}
	})

	t.Run("default model applied when empty", func(t *testing.T) {
		cap := &captured{}
		srv := newFakeBatchServer(t, []byte("x"), "audio/pcm", cap)
		p := newTestProvider(t, srv.URL)
		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainStream(t, stream)
		snap := cap.snapshot()
		if got := snap.body["model_id"]; got != "sonic-2" {
			t.Fatalf("expected model_id sonic-2, got %v", got)
		}
	})

	t.Run("explicit model overrides default", func(t *testing.T) {
		cap := &captured{}
		srv := newFakeBatchServer(t, []byte("x"), "audio/pcm", cap)
		p := newTestProvider(t, srv.URL)
		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
			Input: "hi",
			Model: "sonic-turbo",
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainStream(t, stream)
		if got := cap.snapshot().body["model_id"]; got != "sonic-turbo" {
			t.Fatalf("expected model_id sonic-turbo, got %v", got)
		}
	})

	t.Run("explicit voice overrides default", func(t *testing.T) {
		cap := &captured{}
		srv := newFakeBatchServer(t, []byte("x"), "audio/pcm", cap)
		p := newTestProvider(t, srv.URL)
		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
			Input: "hi",
			Voice: "custom-voice",
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainStream(t, stream)
		voice, _ := cap.snapshot().body["voice"].(map[string]any)
		if voice == nil || voice["id"] != "custom-voice" {
			t.Fatalf("expected voice id custom-voice, got %v", voice)
		}
	})

	t.Run("transcript matches input", func(t *testing.T) {
		cap := &captured{}
		srv := newFakeBatchServer(t, []byte("x"), "audio/pcm", cap)
		p := newTestProvider(t, srv.URL)
		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "say this"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainStream(t, stream)
		if got := cap.snapshot().body["transcript"]; got != "say this" {
			t.Fatalf("expected transcript 'say this', got %v", got)
		}
	})
}

func TestSpeak_FormatMapping(t *testing.T) {
	cases := []struct {
		name       string
		format     string
		container  string
		encoding   string
		sample     float64
		hasBitRate bool
	}{
		{"mp3", "mp3", "mp3", "mp3", 44100, true},
		{"wav", "wav", "wav", "pcm_s16le", 44100, false},
		{"pcm", "pcm", "raw", "pcm_s16le", 44100, false},
		{"ulaw", "ulaw", "raw", "pcm_mulaw", 8000, false},
		{"default empty -> pcm raw", "", "raw", "pcm_s16le", 44100, false},
		{"unknown falls back to pcm", "ogg", "raw", "pcm_s16le", 44100, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cap := &captured{}
			srv := newFakeBatchServer(t, []byte("x"), "audio/pcm", cap)
			p := newTestProvider(t, srv.URL)
			stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
				Input:  "hi",
				Format: tc.format,
			})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			_, _ = drainStream(t, stream)
			of, _ := cap.snapshot().body["output_format"].(map[string]any)
			if of == nil {
				t.Fatalf("output_format missing")
			}
			if of["container"] != tc.container {
				t.Fatalf("container: got %v want %s", of["container"], tc.container)
			}
			if of["encoding"] != tc.encoding {
				t.Fatalf("encoding: got %v want %s", of["encoding"], tc.encoding)
			}
			if got, _ := of["sample_rate"].(float64); got != tc.sample {
				t.Fatalf("sample_rate: got %v want %v", of["sample_rate"], tc.sample)
			}
			_, hasBR := of["bit_rate"]
			if hasBR != tc.hasBitRate {
				t.Fatalf("bit_rate presence: got %v want %v", hasBR, tc.hasBitRate)
			}
		})
	}
}

func TestSpeak_Headers(t *testing.T) {
	cap := &captured{}
	srv := newFakeBatchServer(t, []byte("x"), "audio/pcm", cap)
	p := newTestProvider(t, srv.URL)
	stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, _ = drainStream(t, stream)
	h := cap.snapshot().headers

	t.Run("X-API-Key set", func(t *testing.T) {
		if got := h.Get("X-Api-Key"); got != "test-key" {
			t.Fatalf("expected X-API-Key=test-key, got %q", got)
		}
	})
	t.Run("Cartesia-Version set", func(t *testing.T) {
		if got := h.Get("Cartesia-Version"); got != "2024-11-13" {
			t.Fatalf("expected Cartesia-Version=2024-11-13, got %q", got)
		}
	})
	t.Run("Content-Type set", func(t *testing.T) {
		if got := h.Get("Content-Type"); got != "application/json" {
			t.Fatalf("expected Content-Type application/json, got %q", got)
		}
	})
	t.Run("Accept set for batch", func(t *testing.T) {
		if got := h.Get("Accept"); got == "" {
			t.Fatal("Accept header must be set on batch requests")
		}
	})
}

func TestSpeak_UpstreamError(t *testing.T) {
	t.Run("4xx returns ErrUpstream", func(t *testing.T) {
		srv := newFakeErrServer(t, http.StatusUnauthorized, `{"error":"bad key"}`)
		p := newTestProvider(t, srv.URL)
		_, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
		if err == nil {
			t.Fatal("expected error")
		}
		var upErr *llmrouter.ErrUpstream
		if !errorsAs(err, &upErr) {
			t.Fatalf("expected *ErrUpstream, got %T: %v", err, err)
		}
		if upErr.Provider != "cartesia" {
			t.Fatalf("expected provider 'cartesia', got %q", upErr.Provider)
		}
		if upErr.StatusCode != http.StatusUnauthorized {
			t.Fatalf("expected status 401, got %d", upErr.StatusCode)
		}
		if !strings.Contains(upErr.Body, "bad key") {
			t.Fatalf("expected body to contain 'bad key', got %q", upErr.Body)
		}
	})

	t.Run("5xx returns ErrUpstream", func(t *testing.T) {
		srv := newFakeErrServer(t, http.StatusInternalServerError, "boom")
		p := newTestProvider(t, srv.URL)
		_, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
		var upErr *llmrouter.ErrUpstream
		if !errorsAs(err, &upErr) {
			t.Fatalf("expected *ErrUpstream, got %T", err)
		}
		if upErr.StatusCode != http.StatusInternalServerError {
			t.Fatalf("expected 500, got %d", upErr.StatusCode)
		}
	})

	t.Run("body capped at 8 KiB", func(t *testing.T) {
		large := strings.Repeat("a", 16*1024)
		srv := newFakeErrServer(t, http.StatusBadRequest, large)
		p := newTestProvider(t, srv.URL)
		_, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
		var upErr *llmrouter.ErrUpstream
		if !errorsAs(err, &upErr) {
			t.Fatalf("expected *ErrUpstream, got %T", err)
		}
		if len(upErr.Body) > 8*1024 {
			t.Fatalf("body should be capped at 8 KiB, got %d", len(upErr.Body))
		}
	})
}

func TestSpeak_RawPassthrough(t *testing.T) {
	t.Run("raw fields merged into body", func(t *testing.T) {
		cap := &captured{}
		srv := newFakeBatchServer(t, []byte("x"), "audio/pcm", cap)
		p := newTestProvider(t, srv.URL)

		raw := json.RawMessage(`{"language":"en","emotion":["happy"]}`)
		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
			Input: "hi",
			Raw:   raw,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainStream(t, stream)
		body := cap.snapshot().body
		if body["language"] != "en" {
			t.Fatalf("expected language 'en' from raw, got %v", body["language"])
		}
		emotion, _ := body["emotion"].([]any)
		if len(emotion) != 1 || emotion[0] != "happy" {
			t.Fatalf("expected emotion ['happy'], got %v", body["emotion"])
		}
		// typed fields still applied
		if body["transcript"] != "hi" {
			t.Fatalf("transcript should override / supplement raw, got %v", body["transcript"])
		}
		if body["model_id"] != "sonic-2" {
			t.Fatalf("model_id default should still apply, got %v", body["model_id"])
		}
	})

	t.Run("typed fields override raw", func(t *testing.T) {
		cap := &captured{}
		srv := newFakeBatchServer(t, []byte("x"), "audio/pcm", cap)
		p := newTestProvider(t, srv.URL)

		raw := json.RawMessage(`{"model_id":"old-model","transcript":"wrong"}`)
		stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
			Input: "right",
			Model: "sonic-2",
			Raw:   raw,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainStream(t, stream)
		body := cap.snapshot().body
		if body["model_id"] != "sonic-2" {
			t.Fatalf("typed model should override raw, got %v", body["model_id"])
		}
		if body["transcript"] != "right" {
			t.Fatalf("typed input should override raw transcript, got %v", body["transcript"])
		}
	})

	t.Run("invalid raw returns error", func(t *testing.T) {
		cap := &captured{}
		srv := newFakeBatchServer(t, []byte("x"), "audio/pcm", cap)
		p := newTestProvider(t, srv.URL)
		_, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
			Input: "hi",
			Raw:   json.RawMessage(`{not json`),
		})
		if err == nil {
			t.Fatal("expected error for invalid raw")
		}
	})
}

func TestSpeak_ContextCancel(t *testing.T) {
	t.Run("cancelled context terminates batch stream", func(t *testing.T) {
		// server that takes a while to respond
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("audio"))
		}))
		t.Cleanup(srv.Close)
		p := newTestProvider(t, srv.URL)

		ctx, cancel := context.WithCancel(context.Background())
		stream, err := p.Speak(ctx, llmrouter.SpeechRequest{Input: "hi"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		cancel()
		// drain — should finish (stream might be small enough to succeed).
		done := make(chan struct{})
		go func() {
			_, _ = drainStream(t, stream)
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatal("stream did not terminate after cancel")
		}
	})
}

// errorsAs wraps errors.As to avoid importing errors directly in only
// one test, keeping the test file compact. Returns true on match.
func errorsAs(err error, target any) bool {
	type unwrapper interface{ Unwrap() error }
	for err != nil {
		switch v := target.(type) {
		case **llmrouter.ErrUpstream:
			if u, ok := err.(*llmrouter.ErrUpstream); ok {
				*v = u
				return true
			}
		}
		if u, ok := err.(unwrapper); ok {
			err = u.Unwrap()
			continue
		}
		return false
	}
	return false
}
