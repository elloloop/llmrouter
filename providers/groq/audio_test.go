package groq_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/groq"
)

// fakeTranscriptionServer returns an httptest server that mimics Groq's
// OpenAI-compatible /audio/transcriptions endpoint. The captured request
// is exposed via the returned *http.Request pointer for header / path
// assertions.
func fakeTranscriptionServer(t *testing.T, status int, body string) (*httptest.Server, *capturedRequest) {
	t.Helper()
	cap := &capturedRequest{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.method = r.Method
		cap.path = r.URL.Path
		cap.authHeader = r.Header.Get("Authorization")
		cap.contentType = r.Header.Get("Content-Type")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		fmt.Fprint(w, body)
	}))
	return srv, cap
}

type capturedRequest struct {
	method      string
	path        string
	authHeader  string
	contentType string
}

func newProviderForServer(t *testing.T, srvURL string) *groq.Provider {
	t.Helper()
	p, err := groq.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srvURL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func sampleAudioReader() io.Reader {
	return bytes.NewReader([]byte("RIFFfake-wav-bytes"))
}

func TestTranscribe_SuccessReturnsTextSegment(t *testing.T) {
	srv, _ := fakeTranscriptionServer(t, http.StatusOK, `{"text":"hello world"}`)
	defer srv.Close()

	p := newProviderForServer(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model:       "whisper-large-v3",
		Audio:       sampleAudioReader(),
		AudioFormat: "audio/wav",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}

	var segments []llmrouter.TranscriptSegment
	for seg := range stream.Segments() {
		segments = append(segments, seg)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err = %v, want nil", err)
	}
	if len(segments) != 1 {
		t.Fatalf("segment count = %d, want 1", len(segments))
	}
	if segments[0].Text != "hello world" {
		t.Errorf("Text = %q, want %q", segments[0].Text, "hello world")
	}
	if !segments[0].Final {
		t.Errorf("Final = false, want true")
	}
}

func TestTranscribe_HitsTranscriptionsPath(t *testing.T) {
	srv, cap := fakeTranscriptionServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newProviderForServer(t, srv.URL)

	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model: "whisper-large-v3",
		Audio: sampleAudioReader(),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if cap.path != "/audio/transcriptions" {
		t.Errorf("path = %q, want /audio/transcriptions", cap.path)
	}
	if cap.method != http.MethodPost {
		t.Errorf("method = %q, want POST", cap.method)
	}
}

func TestTranscribe_AuthorizationBearerPropagated(t *testing.T) {
	srv, cap := fakeTranscriptionServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newProviderForServer(t, srv.URL)

	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model: "whisper-large-v3",
		Audio: sampleAudioReader(),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if cap.authHeader != "Bearer test-key" {
		t.Errorf("Authorization = %q, want Bearer test-key", cap.authHeader)
	}
}

func TestTranscribe_ContentTypeIsMultipart(t *testing.T) {
	srv, cap := fakeTranscriptionServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newProviderForServer(t, srv.URL)

	stream, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model: "whisper-large-v3",
		Audio: sampleAudioReader(),
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	for range stream.Segments() {
	}
	if !strings.HasPrefix(cap.contentType, "multipart/form-data") {
		t.Errorf("Content-Type = %q, want multipart/form-data prefix", cap.contentType)
	}
}

func TestTranscribe_UpstreamError_RewritesProviderToGroq(t *testing.T) {
	srv, _ := fakeTranscriptionServer(t, http.StatusUnauthorized, `{"error":{"message":"bad key"}}`)
	defer srv.Close()

	p := newProviderForServer(t, srv.URL)

	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model: "whisper-large-v3",
		Audio: sampleAudioReader(),
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var upErr *llmrouter.ErrUpstream
	if !errors.As(err, &upErr) {
		t.Fatalf("error = %T %v, want *llmrouter.ErrUpstream", err, err)
	}
	if upErr.Provider != "groq" {
		t.Errorf("Provider = %q, want groq (must not leak openai)", upErr.Provider)
	}
	if upErr.Provider == "openai" {
		t.Errorf("Provider = openai, must be rewritten to groq")
	}
	if upErr.StatusCode != http.StatusUnauthorized {
		t.Errorf("StatusCode = %d, want 401", upErr.StatusCode)
	}
	if !strings.Contains(upErr.Body, "bad key") {
		t.Errorf("Body = %q, want substring 'bad key'", upErr.Body)
	}
}

func TestTranscribe_UpstreamError_5xxRewritesProvider(t *testing.T) {
	srv, _ := fakeTranscriptionServer(t, http.StatusBadGateway, "upstream down")
	defer srv.Close()

	p := newProviderForServer(t, srv.URL)

	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model: "whisper-large-v3",
		Audio: sampleAudioReader(),
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var upErr *llmrouter.ErrUpstream
	if !errors.As(err, &upErr) {
		t.Fatalf("error = %T, want *llmrouter.ErrUpstream", err)
	}
	if upErr.Provider != "groq" {
		t.Errorf("Provider = %q, want groq", upErr.Provider)
	}
	if upErr.StatusCode != http.StatusBadGateway {
		t.Errorf("StatusCode = %d, want 502", upErr.StatusCode)
	}
}

func TestTranscribe_TooManyRequestsRewritesProvider(t *testing.T) {
	srv, _ := fakeTranscriptionServer(t, http.StatusTooManyRequests, "rate limited")
	defer srv.Close()

	p := newProviderForServer(t, srv.URL)

	_, err := p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model: "whisper-large-v3",
		Audio: sampleAudioReader(),
	})
	var upErr *llmrouter.ErrUpstream
	if !errors.As(err, &upErr) {
		t.Fatalf("error = %v, want ErrUpstream", err)
	}
	if upErr.Provider != "groq" {
		t.Errorf("Provider = %q, want groq", upErr.Provider)
	}
}

func TestTranscribe_DefaultBaseURL_TargetsGroqHost(t *testing.T) {
	// Without WithBaseURL, the inner provider must target Groq's host.
	// We use a stub transport that fails with the request URL embedded so
	// we can assert the destination without making a real network call.
	p, err := groq.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithHTTPClient(&http.Client{
			Transport: &errTransport{},
			Timeout:   1 * time.Second,
		}),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model: "whisper-large-v3",
		Audio: sampleAudioReader(),
	})
	if err == nil {
		t.Fatal("expected transport error from stub")
	}
	if !strings.Contains(err.Error(), "https://api.groq.com/openai/v1/audio/transcriptions") {
		t.Errorf("error = %v, want URL containing Groq default transcriptions endpoint", err)
	}
}

func TestTranscribe_NonUpstreamErrorPassesThrough(t *testing.T) {
	// Point at an invalid URL so the HTTP client fails before any
	// response — the error must not be wrapped as *ErrUpstream.
	p, err := groq.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL("http://127.0.0.1:1"),
		llmrouter.WithTimeout(500*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.Transcribe(context.Background(), llmrouter.TranscribeRequest{
		Model: "whisper-large-v3",
		Audio: sampleAudioReader(),
	})
	if err == nil {
		t.Fatal("expected transport error, got nil")
	}
	var upErr *llmrouter.ErrUpstream
	if errors.As(err, &upErr) {
		t.Fatalf("transport error wrongly wrapped as ErrUpstream: %v", err)
	}
}

func TestTranscribe_ContextCancelledBeforeRequest(t *testing.T) {
	srv, _ := fakeTranscriptionServer(t, http.StatusOK, `{"text":"x"}`)
	defer srv.Close()

	p := newProviderForServer(t, srv.URL)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Transcribe(ctx, llmrouter.TranscribeRequest{
		Model: "whisper-large-v3",
		Audio: sampleAudioReader(),
	}); err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
}
