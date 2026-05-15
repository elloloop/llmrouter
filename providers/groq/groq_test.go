package groq_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/groq"
)

// fakeSSEServer returns an httptest server that emits the given payloads
// as SSE events followed by a [DONE] sentinel. It also captures the
// inbound request for header/path assertions in the test.
func fakeSSEServer(t *testing.T, payloads []string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("auth header = %q, want Bearer test-key", got)
		}
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q, want /chat/completions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		for _, p := range payloads {
			fmt.Fprintf(w, "data: %s\n\n", p)
			if flusher != nil {
				flusher.Flush()
			}
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}))
}

// errorStatusServer returns a server that responds with the given status
// and body on every request.
func errorStatusServer(t *testing.T, status int, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(status)
		fmt.Fprint(w, body)
	}))
}

func TestDefaultBaseURL_PointsAtGroq(t *testing.T) {
	if groq.DefaultBaseURL != "https://api.groq.com/openai/v1" {
		t.Errorf("DefaultBaseURL = %q, want https://api.groq.com/openai/v1", groq.DefaultBaseURL)
	}
}

func TestNew_RequiresAPIKey(t *testing.T) {
	if _, err := groq.New(); err == nil {
		t.Fatal("expected error for missing api key")
	}
}

func TestNew_RejectsEmptyAPIKey(t *testing.T) {
	if _, err := groq.New(llmrouter.WithAPIKey("   ")); err == nil {
		t.Fatal("expected error for blank api key")
	}
}

func TestNew_SucceedsWithAPIKey(t *testing.T) {
	p, err := groq.New(llmrouter.WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p == nil {
		t.Fatal("New returned nil provider")
	}
}

func TestNew_NilOptionsAreIgnored(t *testing.T) {
	if _, err := groq.New(llmrouter.WithAPIKey("test-key"), nil); err != nil {
		t.Fatalf("New with nil option: %v", err)
	}
}

func TestName_ReturnsGroq(t *testing.T) {
	p, err := groq.New(llmrouter.WithAPIKey("test-key"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := p.Name(); got != "groq" {
		t.Errorf("Name = %q, want groq", got)
	}
}

func TestCompletionStream_UsesUserSuppliedBaseURL(t *testing.T) {
	srv := fakeSSEServer(t, []string{
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
	})
	defer srv.Close()

	p, err := groq.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "llama-3.1-70b",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err = %v, want nil", err)
	}
}

func TestCompletionStream_ForwardsChunksAndCompletes(t *testing.T) {
	payloads := []string{
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"llama-3.1-70b","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"llama-3.1-70b","choices":[{"index":0,"delta":{"content":" world"}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"llama-3.1-70b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
	}
	srv := fakeSSEServer(t, payloads)
	defer srv.Close()

	p, err := groq.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "llama-3.1-70b",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}

	var (
		gotContent string
		gotFinish  string
		gotUsage   *llmrouter.Usage
		gotCount   int
	)
	for chunk := range stream.Chunks() {
		gotCount++
		for _, c := range chunk.Choices {
			gotContent += c.Delta.Content
			if c.FinishReason != "" {
				gotFinish = c.FinishReason
			}
		}
		if chunk.Usage != nil {
			gotUsage = chunk.Usage
		}
		if len(chunk.Raw) == 0 {
			t.Errorf("chunk.Raw empty, expected wire bytes preserved")
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err = %v, want nil", err)
	}
	if gotCount != 3 {
		t.Errorf("chunk count = %d, want 3", gotCount)
	}
	if gotContent != "Hello world" {
		t.Errorf("content = %q, want %q", gotContent, "Hello world")
	}
	if gotFinish != "stop" {
		t.Errorf("finish_reason = %q, want stop", gotFinish)
	}
	if gotUsage == nil || gotUsage.TotalTokens != 5 {
		t.Errorf("usage = %+v, want TotalTokens=5", gotUsage)
	}
}

func TestCompletionStream_UpstreamError_RewritesProviderToGroq(t *testing.T) {
	srv := errorStatusServer(t, http.StatusUnauthorized, `{"error":{"message":"bad key"}}`)
	defer srv.Close()

	p, err := groq.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "llama-3.1-70b",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var upErr *llmrouter.ErrUpstream
	if !errors.As(err, &upErr) {
		t.Fatalf("error = %T %v, want *llmrouter.ErrUpstream", err, err)
	}
	if upErr.Provider != "groq" {
		t.Errorf("Provider = %q, want groq", upErr.Provider)
	}
	if upErr.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", upErr.StatusCode)
	}
	if !strings.Contains(upErr.Body, "bad key") {
		t.Errorf("body = %q, want substring 'bad key'", upErr.Body)
	}
}

func TestCompletionStream_UpstreamError_5xxRewritesProvider(t *testing.T) {
	srv := errorStatusServer(t, http.StatusBadGateway, "upstream down")
	defer srv.Close()

	p, err := groq.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "llama-3.1-70b",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
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
		t.Errorf("status = %d, want 502", upErr.StatusCode)
	}
}

func TestCompletionStream_NonUpstreamErrorPassesThrough(t *testing.T) {
	// Point at an invalid URL so the HTTP client fails before any
	// response is parsed; the error must not be wrapped as *ErrUpstream.
	p, err := groq.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL("http://127.0.0.1:1"),
		llmrouter.WithTimeout(500*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "llama-3.1-70b",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("expected transport error, got nil")
	}
	var upErr *llmrouter.ErrUpstream
	if errors.As(err, &upErr) {
		t.Fatalf("transport error wrongly wrapped as ErrUpstream: %v", err)
	}
}

func TestCompletionStream_ContextCancelledBeforeRequest(t *testing.T) {
	srv := fakeSSEServer(t, []string{
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
	})
	defer srv.Close()

	p, err := groq.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "m",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	}); err == nil {
		t.Fatal("expected error from cancelled context, got nil")
	}
}

func TestCompletionStream_PreservesRawWireBytes(t *testing.T) {
	payload := `{"id":"raw-1","object":"chat.completion.chunk","created":1,"model":"llama","choices":[{"index":0,"delta":{"content":"x"}}]}`
	srv := fakeSSEServer(t, []string{payload})
	defer srv.Close()

	p, err := groq.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "llama",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	var sawRaw bool
	for chunk := range stream.Chunks() {
		if string(chunk.Raw) == payload {
			sawRaw = true
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err = %v", err)
	}
	if !sawRaw {
		t.Errorf("expected to see raw wire payload preserved")
	}
}

func TestCompletionStream_DefaultBaseURL_AppliedWhenAbsent(t *testing.T) {
	// We cannot reach the real Groq host from tests, but we can verify
	// that without WithBaseURL the inner provider attempts a request
	// targeting the Groq host (DNS / dial failure is fine — what we
	// assert is that no config error fires and the error is a transport
	// error, not an llmrouter.ErrInvalidConfig).
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
	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "llama-3.1-70b",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("expected transport error from stub")
	}
	if !strings.Contains(err.Error(), "https://api.groq.com/openai/v1/chat/completions") {
		t.Errorf("error = %v, want URL containing Groq default", err)
	}
}

func TestCompletionStream_UserBaseURLOverridesDefault(t *testing.T) {
	srv := fakeSSEServer(t, []string{
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
	})
	defer srv.Close()

	// The fake server asserts that the auth header is "Bearer test-key"
	// and the path is "/chat/completions" — the request must land here,
	// not at api.groq.com.
	p, err := groq.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "m",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err = %v", err)
	}
}

func TestCompletionStream_UpstreamErrorIs(t *testing.T) {
	srv := errorStatusServer(t, http.StatusTooManyRequests, "rate limited")
	defer srv.Close()

	p, err := groq.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "m",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	var upErr *llmrouter.ErrUpstream
	if !errors.As(err, &upErr) {
		t.Fatalf("error = %v, want ErrUpstream", err)
	}
	if upErr.Provider != "groq" {
		t.Errorf("Provider = %q, want groq", upErr.Provider)
	}
}

func TestCompletionStream_ErrorMessageMentionsGroq(t *testing.T) {
	srv := errorStatusServer(t, http.StatusUnauthorized, "nope")
	defer srv.Close()

	p, err := groq.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "m",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil || !strings.Contains(err.Error(), "groq") {
		t.Fatalf("error = %v, want substring 'groq'", err)
	}
	if strings.Contains(err.Error(), "openai upstream") {
		t.Errorf("error = %v, must not advertise 'openai upstream' to caller", err)
	}
}

// errTransport is a stub http.RoundTripper that returns the request URL
// in the error so we can assert which host the inner provider targeted.
type errTransport struct{}

func (errTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return nil, fmt.Errorf("stub transport refused %s", r.URL.String())
}
