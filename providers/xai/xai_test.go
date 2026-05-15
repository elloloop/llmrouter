package xai_test

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
	"github.com/elloloop/llmrouter/providers/xai"
)

// fakeSSEServer returns an httptest server that emits the given payloads
// as SSE events followed by a [DONE] sentinel. It also asserts the
// incoming Authorization header and request path.
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

func errorServer(status int, body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		fmt.Fprint(w, body)
	}))
}

func TestNew(t *testing.T) {
	t.Run("succeeds with api key", func(t *testing.T) {
		p, err := xai.New(llmrouter.WithAPIKey("test-key"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if p == nil {
			t.Fatal("provider is nil")
		}
	})

	t.Run("returns error when api key missing", func(t *testing.T) {
		_, err := xai.New()
		if err == nil {
			t.Fatal("expected error for missing api key")
		}
	})

	t.Run("returns error when api key empty", func(t *testing.T) {
		_, err := xai.New(llmrouter.WithAPIKey(""))
		if err == nil {
			t.Fatal("expected error for empty api key")
		}
	})

	t.Run("default base URL is xAI", func(t *testing.T) {
		if xai.DefaultBaseURL != "https://api.x.ai/v1" {
			t.Errorf("DefaultBaseURL = %q, want https://api.x.ai/v1", xai.DefaultBaseURL)
		}
	})

	t.Run("user supplied WithBaseURL overrides default", func(t *testing.T) {
		srv := fakeSSEServer(t, []string{
			`{"id":"x1","object":"chat.completion.chunk","created":1,"model":"grok-2","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`,
		})
		defer srv.Close()

		p, err := xai.New(
			llmrouter.WithAPIKey("test-key"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
			Model:    "grok-2",
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
	})

	t.Run("nil option is tolerated", func(t *testing.T) {
		_, err := xai.New(llmrouter.WithAPIKey("test-key"), nil)
		if err != nil {
			t.Fatalf("New with nil option: %v", err)
		}
	})
}

func TestName(t *testing.T) {
	t.Run("returns xai", func(t *testing.T) {
		p, err := xai.New(llmrouter.WithAPIKey("test-key"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if got := p.Name(); got != "xai" {
			t.Errorf("Name = %q, want xai", got)
		}
	})

	t.Run("name is stable across calls", func(t *testing.T) {
		p, err := xai.New(llmrouter.WithAPIKey("test-key"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		first := p.Name()
		second := p.Name()
		if first != second {
			t.Errorf("Name not stable: %q vs %q", first, second)
		}
	})
}

func TestCompletionStream(t *testing.T) {
	t.Run("forwards chunks end to end", func(t *testing.T) {
		payloads := []string{
			`{"id":"x1","object":"chat.completion.chunk","created":1,"model":"grok-2","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
			`{"id":"x1","object":"chat.completion.chunk","created":1,"model":"grok-2","choices":[{"index":0,"delta":{"content":" world"}}]}`,
			`{"id":"x1","object":"chat.completion.chunk","created":1,"model":"grok-2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		}
		srv := fakeSSEServer(t, payloads)
		defer srv.Close()

		p, err := xai.New(
			llmrouter.WithAPIKey("test-key"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
			Model:    "grok-2",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}

		var content, finish string
		var usage *llmrouter.Usage
		var count int
		for chunk := range stream.Chunks() {
			count++
			for _, c := range chunk.Choices {
				content += c.Delta.Content
				if c.FinishReason != "" {
					finish = c.FinishReason
				}
			}
			if chunk.Usage != nil {
				usage = chunk.Usage
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatalf("stream.Err = %v", err)
		}
		if count != 3 {
			t.Errorf("count = %d, want 3", count)
		}
		if content != "Hello world" {
			t.Errorf("content = %q, want %q", content, "Hello world")
		}
		if finish != "stop" {
			t.Errorf("finish = %q, want stop", finish)
		}
		if usage == nil || usage.TotalTokens != 5 {
			t.Errorf("usage = %+v, want TotalTokens=5", usage)
		}
	})

	t.Run("preserves raw wire bytes", func(t *testing.T) {
		srv := fakeSSEServer(t, []string{
			`{"id":"x1","object":"chat.completion.chunk","created":1,"model":"grok-2","choices":[{"index":0,"delta":{"content":"r"},"finish_reason":"stop"}]}`,
		})
		defer srv.Close()

		p, err := xai.New(
			llmrouter.WithAPIKey("test-key"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
			Model:    "grok-2",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}
		seenRaw := false
		for chunk := range stream.Chunks() {
			if len(chunk.Raw) > 0 {
				seenRaw = true
			}
		}
		if !seenRaw {
			t.Error("expected at least one chunk to carry Raw bytes")
		}
	})

	t.Run("rewrites 401 ErrUpstream to xai", func(t *testing.T) {
		srv := errorServer(http.StatusUnauthorized, `{"error":{"message":"bad key"}}`)
		defer srv.Close()

		p, err := xai.New(
			llmrouter.WithAPIKey("test-key"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
			Model:    "grok-2",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %T, want *llmrouter.ErrUpstream", err)
		}
		if ue.Provider != "xai" {
			t.Errorf("Provider = %q, want xai", ue.Provider)
		}
		if ue.StatusCode != http.StatusUnauthorized {
			t.Errorf("StatusCode = %d, want 401", ue.StatusCode)
		}
		if !strings.Contains(ue.Body, "bad key") {
			t.Errorf("Body = %q, want substring 'bad key'", ue.Body)
		}
	})

	t.Run("rewrites 429 ErrUpstream to xai", func(t *testing.T) {
		srv := errorServer(http.StatusTooManyRequests, `rate limited`)
		defer srv.Close()
		p, err := xai.New(
			llmrouter.WithAPIKey("test-key"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
			Model:    "grok-2",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %T, want *llmrouter.ErrUpstream", err)
		}
		if ue.Provider != "xai" {
			t.Errorf("Provider = %q, want xai", ue.Provider)
		}
		if ue.StatusCode != http.StatusTooManyRequests {
			t.Errorf("StatusCode = %d, want 429", ue.StatusCode)
		}
	})

	t.Run("rewrites 500 ErrUpstream to xai", func(t *testing.T) {
		srv := errorServer(http.StatusInternalServerError, `boom`)
		defer srv.Close()
		p, err := xai.New(
			llmrouter.WithAPIKey("test-key"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
			Model:    "grok-2",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %T, want *llmrouter.ErrUpstream", err)
		}
		if ue.Provider != "xai" {
			t.Errorf("Provider = %q, want xai", ue.Provider)
		}
	})

	t.Run("error message contains provider name", func(t *testing.T) {
		srv := errorServer(http.StatusBadRequest, `nope`)
		defer srv.Close()
		p, err := xai.New(
			llmrouter.WithAPIKey("test-key"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
			Model:    "grok-2",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err == nil || !strings.Contains(err.Error(), "xai") {
			t.Errorf("err = %v, want substring 'xai'", err)
		}
	})

	t.Run("context cancellation surfaces", func(t *testing.T) {
		// Server hangs without writing anything; cancelling ctx should
		// terminate the stream.
		block := make(chan struct{})
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			<-block
		}))
		defer srv.Close()
		defer close(block)

		p, err := xai.New(
			llmrouter.WithAPIKey("test-key"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
			Model:    "grok-2",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}
		cancel()
		// Drain the stream — it must terminate.
		done := make(chan struct{})
		go func() {
			for range stream.Chunks() {
			}
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatal("stream did not terminate on ctx cancel")
		}
	})

	t.Run("handles single chunk stream", func(t *testing.T) {
		srv := fakeSSEServer(t, []string{
			`{"id":"x1","object":"chat.completion.chunk","created":1,"model":"grok-2","choices":[{"index":0,"delta":{"content":"only"},"finish_reason":"stop"}]}`,
		})
		defer srv.Close()
		p, err := xai.New(
			llmrouter.WithAPIKey("test-key"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
			Model:    "grok-2",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}
		var content string
		for chunk := range stream.Chunks() {
			for _, c := range chunk.Choices {
				content += c.Delta.Content
			}
		}
		if content != "only" {
			t.Errorf("content = %q, want only", content)
		}
	})

	t.Run("handles empty stream", func(t *testing.T) {
		srv := fakeSSEServer(t, []string{})
		defer srv.Close()
		p, err := xai.New(
			llmrouter.WithAPIKey("test-key"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
			Model:    "grok-2",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}
		count := 0
		for range stream.Chunks() {
			count++
		}
		if count != 0 {
			t.Errorf("count = %d, want 0", count)
		}
		if err := stream.Err(); err != nil {
			t.Errorf("stream.Err = %v, want nil", err)
		}
	})
}

func TestErrUpstreamPassthrough(t *testing.T) {
	t.Run("nil error passes through", func(t *testing.T) {
		// Indirectly: a happy stream returns nil from stream.Err().
		srv := fakeSSEServer(t, []string{
			`{"id":"x1","object":"chat.completion.chunk","created":1,"model":"grok-2","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`,
		})
		defer srv.Close()
		p, err := xai.New(
			llmrouter.WithAPIKey("test-key"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
			Model:    "grok-2",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}
		for range stream.Chunks() {
		}
		if err := stream.Err(); err != nil {
			t.Errorf("stream.Err = %v, want nil", err)
		}
	})

	t.Run("transport error not classified as ErrUpstream", func(t *testing.T) {
		// Point at a closed server to force a transport failure.
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
		srv.Close()
		p, err := xai.New(
			llmrouter.WithAPIKey("test-key"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
			Model:    "grok-2",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		var ue *llmrouter.ErrUpstream
		if errors.As(err, &ue) {
			t.Errorf("transport error wrongly classified as ErrUpstream: %v", ue)
		}
	})
}
