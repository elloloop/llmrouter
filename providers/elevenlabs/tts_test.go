package elevenlabs_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/elevenlabs"
)

// capturedTTSRequest holds the bits of the inbound request a test
// wants to inspect after the handler has returned.
type capturedTTSRequest struct {
	method      string
	path        string
	apiKey      string
	contentType string
	accept      string
	body        map[string]json.RawMessage
}

// fakeTTSServer stands up an httptest server that records the request
// and writes the given status + body + content-type. When streaming is
// true the body is flushed in two halves to exercise chunked reads.
func fakeTTSServer(t *testing.T, status int, contentType string, body []byte, streaming bool) (*httptest.Server, *capturedTTSRequest) {
	t.Helper()
	cap := &capturedTTSRequest{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.method = r.Method
		cap.path = r.URL.Path
		cap.apiKey = r.Header.Get("xi-api-key")
		cap.contentType = r.Header.Get("Content-Type")
		cap.accept = r.Header.Get("Accept")
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &cap.body)

		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(status)
		if status >= 400 {
			_, _ = w.Write(body)
			return
		}
		if streaming {
			half := len(body) / 2
			_, _ = w.Write(body[:half])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			_, _ = w.Write(body[half:])
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			return
		}
		_, _ = w.Write(body)
	}))
	return srv, cap
}

func newTTSProvider(t *testing.T, srvURL string) *elevenlabs.Provider {
	t.Helper()
	p, err := elevenlabs.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srvURL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func drainAudio(t *testing.T, stream *llmrouter.AudioStream) [][]byte {
	t.Helper()
	var chunks [][]byte
	for c := range stream.Chunks() {
		chunks = append(chunks, c.Data)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err = %v, want nil", err)
	}
	return chunks
}

func TestSpeak_NonStreaming_EmitsSingleChunk(t *testing.T) {
	audio := []byte("\xFF\xFBfake-mp3-bytes")
	srv, _ := fakeTTSServer(t, http.StatusOK, "audio/mpeg", audio, false)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := p.Speak(ctx, llmrouter.SpeechRequest{
		Input: "hello",
	})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	chunks := drainAudio(t, stream)
	if len(chunks) != 1 {
		t.Fatalf("chunk count = %d, want 1", len(chunks))
	}
	if string(chunks[0]) != string(audio) {
		t.Errorf("chunk[0] = %q, want %q", chunks[0], audio)
	}
}

func TestSpeak_NonStreaming_ContentTypePropagated(t *testing.T) {
	srv, _ := fakeTTSServer(t, http.StatusOK, "audio/mpeg", []byte("x"), false)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)

	stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Input: "hi",
	})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	if stream.ContentType != "audio/mpeg" {
		t.Errorf("ContentType = %q, want audio/mpeg", stream.ContentType)
	}
	drainAudio(t, stream)
}

func TestSpeak_Streaming_MultipleChunks(t *testing.T) {
	body := make([]byte, ttsStreamPayloadSize)
	for i := range body {
		body[i] = byte(i % 256)
	}
	srv, _ := fakeTTSServer(t, http.StatusOK, "audio/mpeg", body, true)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)

	stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Input:  "stream please",
		Stream: true,
	})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	chunks := drainAudio(t, stream)
	if len(chunks) < 2 {
		t.Fatalf("chunk count = %d, want >= 2 for streaming body", len(chunks))
	}
	// Concatenation must equal the original body.
	var got []byte
	for _, c := range chunks {
		got = append(got, c...)
	}
	if len(got) != len(body) {
		t.Fatalf("total bytes = %d, want %d", len(got), len(body))
	}
	for i := range got {
		if got[i] != body[i] {
			t.Fatalf("byte %d = %d, want %d", i, got[i], body[i])
		}
	}
}

// ttsStreamPayloadSize is just over two stream-chunk windows so we can
// assert multi-chunk delivery without relying on Flush timing.
const ttsStreamPayloadSize = 8*1024*2 + 100

func TestSpeak_StreamingHitsStreamPath(t *testing.T) {
	srv, cap := fakeTTSServer(t, http.StatusOK, "audio/mpeg", []byte("x"), false)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)

	stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Input:  "x",
		Voice:  "voice-abc",
		Stream: true,
	})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	drainAudio(t, stream)
	if cap.path != "/v1/text-to-speech/voice-abc/stream" {
		t.Errorf("path = %q, want /v1/text-to-speech/voice-abc/stream", cap.path)
	}
}

func TestSpeak_NonStreamingHitsBasePath(t *testing.T) {
	srv, cap := fakeTTSServer(t, http.StatusOK, "audio/mpeg", []byte("x"), false)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)

	stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Input: "x",
		Voice: "voice-abc",
	})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	drainAudio(t, stream)
	if cap.path != "/v1/text-to-speech/voice-abc" {
		t.Errorf("path = %q, want /v1/text-to-speech/voice-abc", cap.path)
	}
}

func TestSpeak_VoiceDefaultAppliedWhenEmpty(t *testing.T) {
	srv, cap := fakeTTSServer(t, http.StatusOK, "audio/mpeg", []byte("x"), false)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)
	stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	drainAudio(t, stream)
	if !strings.HasSuffix(cap.path, "/21m00Tcm4TlvDq8ikWAM") {
		t.Errorf("path = %q, want default Rachel voice id suffix", cap.path)
	}
}

func TestSpeak_DefaultModelApplied(t *testing.T) {
	srv, cap := fakeTTSServer(t, http.StatusOK, "audio/mpeg", []byte("x"), false)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)
	stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "hi"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	drainAudio(t, stream)

	var got string
	if err := json.Unmarshal(cap.body["model_id"], &got); err != nil {
		t.Fatalf("model_id field: %v", err)
	}
	if got != "eleven_turbo_v2_5" {
		t.Errorf("model_id = %q, want eleven_turbo_v2_5", got)
	}
}

func TestSpeak_RequestBodyShape(t *testing.T) {
	srv, cap := fakeTTSServer(t, http.StatusOK, "audio/mpeg", []byte("x"), false)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)
	stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Input: "hello world",
		Model: "eleven_multilingual_v2",
	})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	drainAudio(t, stream)

	var text, model, format string
	_ = json.Unmarshal(cap.body["text"], &text)
	_ = json.Unmarshal(cap.body["model_id"], &model)
	_ = json.Unmarshal(cap.body["output_format"], &format)
	if text != "hello world" {
		t.Errorf("text = %q, want hello world", text)
	}
	if model != "eleven_multilingual_v2" {
		t.Errorf("model_id = %q, want eleven_multilingual_v2", model)
	}
	if format != "mp3_44100_128" {
		t.Errorf("output_format = %q, want mp3_44100_128", format)
	}
}

func TestSpeak_AuthHeaderUsesXIAPIKey(t *testing.T) {
	srv, cap := fakeTTSServer(t, http.StatusOK, "audio/mpeg", []byte("x"), false)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)
	stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "x"})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	drainAudio(t, stream)
	if cap.apiKey != "test-key" {
		t.Errorf("xi-api-key = %q, want test-key", cap.apiKey)
	}
}

func TestSpeak_FormatMapping(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"", "mp3_44100_128"},
		{"mp3", "mp3_44100_128"},
		{"MP3", "mp3_44100_128"},
		{"opus", "opus_48000_128"},
		{"pcm", "pcm_44100"},
		{"wav", "pcm_44100"},
		{"ulaw", "ulaw_8000"},
		{"flac", "mp3_44100_128"}, // unknown -> default
		{"aac", "mp3_44100_128"},  // unknown -> default
	}
	for _, c := range cases {
		t.Run("format="+c.in, func(t *testing.T) {
			srv, cap := fakeTTSServer(t, http.StatusOK, "audio/mpeg", []byte("x"), false)
			defer srv.Close()

			p := newTTSProvider(t, srv.URL)
			stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
				Input:  "x",
				Format: c.in,
			})
			if err != nil {
				t.Fatalf("Speak: %v", err)
			}
			drainAudio(t, stream)

			var got string
			_ = json.Unmarshal(cap.body["output_format"], &got)
			if got != c.want {
				t.Errorf("format %q -> output_format %q, want %q", c.in, got, c.want)
			}
		})
	}
}

func TestSpeak_RawPreservesVoiceSettings(t *testing.T) {
	srv, cap := fakeTTSServer(t, http.StatusOK, "audio/mpeg", []byte("x"), false)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)
	raw := json.RawMessage(`{"voice_settings":{"stability":0.4,"similarity_boost":0.9}}`)
	stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Input: "x",
		Raw:   raw,
	})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	drainAudio(t, stream)

	vs, ok := cap.body["voice_settings"]
	if !ok {
		t.Fatal("voice_settings missing from body")
	}
	var got struct {
		Stability       float64 `json:"stability"`
		SimilarityBoost float64 `json:"similarity_boost"`
	}
	if err := json.Unmarshal(vs, &got); err != nil {
		t.Fatalf("voice_settings: %v", err)
	}
	if got.Stability != 0.4 || got.SimilarityBoost != 0.9 {
		t.Errorf("voice_settings = %+v, want {0.4,0.9}", got)
	}
}

func TestSpeak_TypedFieldsOverrideRaw(t *testing.T) {
	srv, cap := fakeTTSServer(t, http.StatusOK, "audio/mpeg", []byte("x"), false)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)
	raw := json.RawMessage(`{"text":"raw-text","model_id":"raw-model"}`)
	stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
		Input: "typed-input",
		Model: "typed-model",
		Raw:   raw,
	})
	if err != nil {
		t.Fatalf("Speak: %v", err)
	}
	drainAudio(t, stream)

	var text, model string
	_ = json.Unmarshal(cap.body["text"], &text)
	_ = json.Unmarshal(cap.body["model_id"], &model)
	if text != "typed-input" {
		t.Errorf("text = %q, want typed-input", text)
	}
	if model != "typed-model" {
		t.Errorf("model_id = %q, want typed-model", model)
	}
}

func TestSpeak_4xxReturnsErrUpstream(t *testing.T) {
	srv, _ := fakeTTSServer(t, http.StatusUnauthorized, "application/json",
		[]byte(`{"detail":{"status":"invalid_api_key","message":"bad key"}}`), false)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)
	_, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "x"})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var upErr *llmrouter.ErrUpstream
	if !errors.As(err, &upErr) {
		t.Fatalf("err = %T, want *ErrUpstream", err)
	}
	if upErr.Provider != "elevenlabs" {
		t.Errorf("Provider = %q, want elevenlabs", upErr.Provider)
	}
	if upErr.StatusCode != http.StatusUnauthorized {
		t.Errorf("StatusCode = %d, want 401", upErr.StatusCode)
	}
	if !strings.Contains(upErr.Body, "bad key") {
		t.Errorf("Body = %q, want substring 'bad key'", upErr.Body)
	}
}

func TestSpeak_5xxReturnsErrUpstream(t *testing.T) {
	srv, _ := fakeTTSServer(t, http.StatusBadGateway, "text/plain", []byte("upstream down"), false)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)
	_, err := p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "x"})
	var upErr *llmrouter.ErrUpstream
	if !errors.As(err, &upErr) {
		t.Fatalf("err = %v, want ErrUpstream", err)
	}
	if upErr.StatusCode != http.StatusBadGateway {
		t.Errorf("StatusCode = %d, want 502", upErr.StatusCode)
	}
}

func TestSpeak_ContextCancelledBeforeRequest(t *testing.T) {
	srv, _ := fakeTTSServer(t, http.StatusOK, "audio/mpeg", []byte("x"), false)
	defer srv.Close()

	p := newTTSProvider(t, srv.URL)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Speak(ctx, llmrouter.SpeechRequest{Input: "x"}); err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
}

func TestSpeak_AcceptHeaderMatchesFormat(t *testing.T) {
	cases := []struct {
		format string
		accept string
	}{
		{"", "audio/mpeg"},
		{"mp3", "audio/mpeg"},
		{"opus", "audio/opus"},
		{"pcm", "audio/pcm"},
		{"wav", "audio/pcm"},
		{"ulaw", "audio/basic"},
	}
	for _, c := range cases {
		t.Run("format="+c.format, func(t *testing.T) {
			srv, cap := fakeTTSServer(t, http.StatusOK, "audio/mpeg", []byte("x"), false)
			defer srv.Close()

			p := newTTSProvider(t, srv.URL)
			stream, err := p.Speak(context.Background(), llmrouter.SpeechRequest{
				Input:  "x",
				Format: c.format,
			})
			if err != nil {
				t.Fatalf("Speak: %v", err)
			}
			drainAudio(t, stream)
			if cap.accept != c.accept {
				t.Errorf("Accept = %q, want %q", cap.accept, c.accept)
			}
		})
	}
}

func TestSpeak_HTTPClientErrorBubbles(t *testing.T) {
	// Bad URL forces a transport-level error; must NOT be wrapped as ErrUpstream.
	p, err := elevenlabs.New(
		llmrouter.WithAPIKey("k"),
		llmrouter.WithBaseURL("http://127.0.0.1:1"),
		llmrouter.WithTimeout(500*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Speak(context.Background(), llmrouter.SpeechRequest{Input: "x"})
	if err == nil {
		t.Fatal("expected transport error, got nil")
	}
	var upErr *llmrouter.ErrUpstream
	if errors.As(err, &upErr) {
		t.Fatalf("transport error wrongly wrapped as ErrUpstream: %v", err)
	}
}
