package openai_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/openai"
)

// fakeSSEServer returns an httptest server that emits the given payloads as
// SSE events followed by a [DONE] sentinel.
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

func TestCompletionStream_ForwardsChunksAndCompletes(t *testing.T) {
	payloads := []string{
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":" world"}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
	}
	srv := fakeSSEServer(t, payloads)
	defer srv.Close()

	p, err := openai.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p.Name() != "openai" {
		t.Fatalf("Name = %q, want openai", p.Name())
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "gpt-4o-mini",
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

func TestCompletionStream_UpstreamError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":{"message":"bad key"}}`)
	}))
	defer srv.Close()

	p, err := openai.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "gpt-4o-mini",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var upErr *llmrouter.ErrUpstream
	if !errorsAs(err, &upErr) {
		t.Fatalf("error = %T %v, want *llmrouter.ErrUpstream", err, err)
	}
	if upErr.StatusCode != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", upErr.StatusCode)
	}
	if !strings.Contains(upErr.Body, "bad key") {
		t.Errorf("body = %q, want substring 'bad key'", upErr.Body)
	}
}

func TestNew_RequiresAPIKey(t *testing.T) {
	_, err := openai.New()
	if err == nil {
		t.Fatal("expected error for missing api key")
	}
}

// errorsAs is a tiny shim to avoid importing errors just for one assertion.
func errorsAs(err error, target **llmrouter.ErrUpstream) bool {
	for e := err; e != nil; {
		if t, ok := e.(*llmrouter.ErrUpstream); ok {
			*target = t
			return true
		}
		type unwrapper interface{ Unwrap() error }
		u, ok := e.(unwrapper)
		if !ok {
			return false
		}
		e = u.Unwrap()
	}
	return false
}
