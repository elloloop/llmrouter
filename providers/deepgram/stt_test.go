package deepgram

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
)

// sampleResponse mirrors a realistic Deepgram /v1/listen JSON payload
// with word-level timing, confidence, and a smart-formatted
// punctuated_word that differs from word.
const sampleResponse = `{
  "metadata": {
    "transaction_key": "k",
    "request_id": "req-1",
    "sha256": "abc",
    "duration": 1.25,
    "channels": 1
  },
  "results": {
    "channels": [{
      "alternatives": [{
        "transcript": "hello world",
        "confidence": 0.98,
        "words": [
          {"word": "hello", "start": 0.01, "end": 0.5,  "confidence": 0.99, "punctuated_word": "Hello"},
          {"word": "world", "start": 0.55, "end": 1.0,  "confidence": 0.97, "punctuated_word": "world."}
        ]
      }]
    }],
    "utterances": [
      {"start": 0.0, "end": 1.0, "transcript": "hello world"}
    ]
  }
}`

// newTestServer returns an httptest server that runs the supplied
// handler and a Provider configured to point at it. Tests use this to
// observe the exact request shape produced by Transcribe.
func newTestServer(t *testing.T, handler http.HandlerFunc) (*Provider, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	p, err := New(
		llmrouter.WithAPIKey("dg-test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("unexpected error building provider: %v", err)
	}
	return p, srv
}

// drainSegments reads every segment off a TranscriptStream and returns
// them along with the terminal error. Bounded by t.Deadline via the
// underlying channel close.
func drainSegments(t *testing.T, stream *llmrouter.TranscriptStream) ([]llmrouter.TranscriptSegment, error) {
	t.Helper()
	var got []llmrouter.TranscriptSegment
	for seg := range stream.Segments() {
		got = append(got, seg)
	}
	return got, stream.Err()
}

// happyHandler responds with the canned sampleResponse and records the
// last received request for assertions.
func happyHandler(t *testing.T, captured **http.Request, capturedBody *[]byte) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		*captured = r
		*capturedBody = body
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(sampleResponse))
	}
}

func TestTranscribe(t *testing.T) {
	t.Run("posts_to_v1_listen", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("audio-bytes"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if _, err := drainSegments(t, stream); err != nil {
			t.Fatalf("stream err: %v", err)
		}
		if req.URL.Path != "/v1/listen" {
			t.Fatalf("expected path /v1/listen, got %q", req.URL.Path)
		}
		if req.Method != http.MethodPost {
			t.Fatalf("expected POST, got %s", req.Method)
		}
	})

	t.Run("default_model_applied", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		if got := req.URL.Query().Get("model"); got != "nova-3" {
			t.Fatalf("expected default model nova-3, got %q", got)
		}
	})

	t.Run("default_language_applied", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		if got := req.URL.Query().Get("language"); got != "en" {
			t.Fatalf("expected default language en, got %q", got)
		}
	})

	t.Run("custom_model_and_language_forwarded", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:    strings.NewReader("a"),
			Model:    "nova-2-medical",
			Language: "fr",
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		if got := req.URL.Query().Get("model"); got != "nova-2-medical" {
			t.Fatalf("expected model nova-2-medical, got %q", got)
		}
		if got := req.URL.Query().Get("language"); got != "fr" {
			t.Fatalf("expected language fr, got %q", got)
		}
	})

	t.Run("fixed_query_flags_set", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		q := req.URL.Query()
		for key, want := range map[string]string{
			"punctuate":    "true",
			"smart_format": "true",
			"utterances":   "true",
		} {
			if got := q.Get(key); got != want {
				t.Fatalf("expected %s=%q, got %q", key, want, got)
			}
		}
	})

	t.Run("auth_header_uses_token_prefix_not_bearer", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		got := req.Header.Get("Authorization")
		if got != "Token dg-test-key" {
			t.Fatalf("expected Authorization=%q, got %q", "Token dg-test-key", got)
		}
		if strings.HasPrefix(got, "Bearer ") {
			t.Fatalf("Authorization must not use Bearer prefix, got %q", got)
		}
	})

	t.Run("content_type_defaults_to_audio_mpeg", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		if got := req.Header.Get("Content-Type"); got != "audio/mpeg" {
			t.Fatalf("expected Content-Type audio/mpeg, got %q", got)
		}
	})

	t.Run("content_type_forwards_audio_format", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:       strings.NewReader("a"),
			AudioFormat: "audio/webm",
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		if got := req.Header.Get("Content-Type"); got != "audio/webm" {
			t.Fatalf("expected Content-Type audio/webm, got %q", got)
		}
	})

	t.Run("body_is_raw_audio_bytes", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		payload := "raw-audio-bytes-here"
		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader(payload),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		if string(body) != payload {
			t.Fatalf("body mismatch: want %q got %q", payload, string(body))
		}
	})

	t.Run("returns_final_segment_with_text", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, err := drainSegments(t, stream)
		if err != nil {
			t.Fatalf("stream err: %v", err)
		}
		if len(segs) != 1 {
			t.Fatalf("expected 1 segment, got %d", len(segs))
		}
		if !segs[0].Final {
			t.Fatal("expected Final=true")
		}
		if segs[0].Text != "hello world" {
			t.Fatalf("expected text 'hello world', got %q", segs[0].Text)
		}
	})

	t.Run("confidence_propagates", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, _ := drainSegments(t, stream)
		if len(segs) != 1 {
			t.Fatalf("expected 1 segment, got %d", len(segs))
		}
		// float32(0.98) — compare with tolerance to avoid binary precision flake.
		if diff := segs[0].Confidence - 0.98; diff > 0.001 || diff < -0.001 {
			t.Fatalf("expected confidence ~0.98, got %v", segs[0].Confidence)
		}
	})

	t.Run("duration_maps_to_end", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, _ := drainSegments(t, stream)
		if len(segs) != 1 {
			t.Fatalf("expected 1 segment, got %d", len(segs))
		}
		wantEnd := 1250 * time.Millisecond
		if segs[0].End != wantEnd {
			t.Fatalf("expected End=%v, got %v", wantEnd, segs[0].End)
		}
		if segs[0].Start != 0 {
			t.Fatalf("expected Start=0, got %v", segs[0].Start)
		}
	})

	t.Run("words_use_punctuated_word", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, _ := drainSegments(t, stream)
		if len(segs) != 1 || len(segs[0].Words) != 2 {
			t.Fatalf("expected 1 segment with 2 words, got %+v", segs)
		}
		if segs[0].Words[0].Word != "Hello" {
			t.Fatalf("expected punctuated word 'Hello', got %q", segs[0].Words[0].Word)
		}
		if segs[0].Words[1].Word != "world." {
			t.Fatalf("expected punctuated word 'world.', got %q", segs[0].Words[1].Word)
		}
	})

	t.Run("words_fallback_to_word_when_punctuated_missing", func(t *testing.T) {
		body := `{
		  "metadata": {"duration": 0.5},
		  "results": {"channels": [{"alternatives": [{
		    "transcript": "hi",
		    "confidence": 0.9,
		    "words": [{"word": "hi", "start": 0, "end": 0.5, "confidence": 0.9}]
		  }]}]}
		}`
		p, _ := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(body))
		})

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, _ := drainSegments(t, stream)
		if len(segs) != 1 || len(segs[0].Words) != 1 {
			t.Fatalf("expected 1 segment with 1 word, got %+v", segs)
		}
		if segs[0].Words[0].Word != "hi" {
			t.Fatalf("expected fallback word 'hi', got %q", segs[0].Words[0].Word)
		}
	})

	t.Run("word_timings_in_duration", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, _ := drainSegments(t, stream)
		if len(segs) != 1 || len(segs[0].Words) != 2 {
			t.Fatalf("expected 2 words, got %+v", segs)
		}
		// 0.01s = 10ms; 0.5s = 500ms.
		if segs[0].Words[0].Start != 10*time.Millisecond {
			t.Fatalf("expected first word start=10ms, got %v", segs[0].Words[0].Start)
		}
		if segs[0].Words[0].End != 500*time.Millisecond {
			t.Fatalf("expected first word end=500ms, got %v", segs[0].Words[0].End)
		}
	})

	t.Run("raw_diarize_passthrough", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
			Raw:   json.RawMessage(`{"diarize": true}`),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		if got := req.URL.Query().Get("diarize"); got != "true" {
			t.Fatalf("expected diarize=true, got %q", got)
		}
	})

	t.Run("raw_paragraphs_summarize_passthrough", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
			Raw:   json.RawMessage(`{"paragraphs": true, "summarize": "v2"}`),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		q := req.URL.Query()
		if got := q.Get("paragraphs"); got != "true" {
			t.Fatalf("expected paragraphs=true, got %q", got)
		}
		if got := q.Get("summarize"); got != "v2" {
			t.Fatalf("expected summarize=v2, got %q", got)
		}
	})

	t.Run("raw_unknown_key_ignored", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
			Raw:   json.RawMessage(`{"not_an_allowed_key": "x"}`),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		if req.URL.Query().Has("not_an_allowed_key") {
			t.Fatalf("disallowed key leaked into query: %q", req.URL.RawQuery)
		}
	})

	t.Run("raw_object_value_skipped", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
			Raw:   json.RawMessage(`{"keywords": {"nested": true}}`),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, _ = drainSegments(t, stream)
		if req.URL.Query().Has("keywords") {
			t.Fatalf("non-scalar value leaked into query: %q", req.URL.RawQuery)
		}
	})

	t.Run("missing_audio_reader_errors", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("dg-test"))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		_, err = p.Transcribe(context.Background(), llmrouter.TranscribeRequest{})
		if err == nil {
			t.Fatal("expected error when Audio is nil")
		}
	})

	t.Run("4xx_returns_err_upstream", func(t *testing.T) {
		p, _ := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":"bad audio"}`))
		})

		_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err == nil {
			t.Fatal("expected error from 400")
		}
		var upstream *llmrouter.ErrUpstream
		if !errors.As(err, &upstream) {
			t.Fatalf("expected *llmrouter.ErrUpstream, got %T", err)
		}
		if upstream.Provider != "deepgram" {
			t.Fatalf("expected provider deepgram, got %q", upstream.Provider)
		}
		if upstream.StatusCode != http.StatusBadRequest {
			t.Fatalf("expected status 400, got %d", upstream.StatusCode)
		}
		if !strings.Contains(upstream.Body, "bad audio") {
			t.Fatalf("expected body to contain upstream message, got %q", upstream.Body)
		}
	})

	t.Run("5xx_returns_err_upstream", func(t *testing.T) {
		p, _ := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`server boom`))
		})

		_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		var upstream *llmrouter.ErrUpstream
		if !errors.As(err, &upstream) {
			t.Fatalf("expected *llmrouter.ErrUpstream, got %T (%v)", err, err)
		}
		if upstream.StatusCode != http.StatusInternalServerError {
			t.Fatalf("expected 500, got %d", upstream.StatusCode)
		}
	})

	t.Run("stream_false_uses_batch_http_path", func(t *testing.T) {
		// Sanity check that req.Stream=false continues to take the
		// batch HTTP path (no behaviour change after the streaming
		// refactor introduced req.Stream=true routing).
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio:  strings.NewReader("a"),
			Stream: false,
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, err := drainSegments(t, stream)
		if err != nil {
			t.Fatalf("stream err: %v", err)
		}
		if req == nil {
			t.Fatal("expected batch HTTP path to be hit when Stream=false")
		}
		if req.URL.Path != "/v1/listen" || req.Method != http.MethodPost {
			t.Fatalf("expected POST /v1/listen, got %s %s", req.Method, req.URL.Path)
		}
		if len(segs) != 1 {
			t.Fatalf("expected 1 segment from batch path, got %d", len(segs))
		}
	})

	t.Run("raw_segment_preserved", func(t *testing.T) {
		var req *http.Request
		var body []byte
		p, _ := newTestServer(t, happyHandler(t, &req, &body))

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, _ := drainSegments(t, stream)
		if len(segs) != 1 || len(segs[0].Raw) == 0 {
			t.Fatalf("expected Raw to be populated, got %+v", segs)
		}
		// Quick sanity check that Raw decodes back to JSON.
		var anything any
		if err := json.Unmarshal(segs[0].Raw, &anything); err != nil {
			t.Fatalf("segment Raw is not valid JSON: %v", err)
		}
	})

	t.Run("empty_channels_yields_empty_segment", func(t *testing.T) {
		body := `{"metadata":{"duration":0.0},"results":{"channels":[]}}`
		p, _ := newTestServer(t, func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(body))
		})

		stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
			Audio: strings.NewReader("a"),
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		segs, err := drainSegments(t, stream)
		if err != nil {
			t.Fatalf("stream err: %v", err)
		}
		if len(segs) != 1 || segs[0].Text != "" || !segs[0].Final {
			t.Fatalf("expected one empty Final segment, got %+v", segs)
		}
	})
}
