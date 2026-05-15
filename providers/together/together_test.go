package together_test

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
	"github.com/elloloop/llmrouter/providers/together"
)

// fakeSSEServer returns an httptest server that emits the given payloads
// as SSE events followed by a [DONE] sentinel. It also records the path
// and Authorization header observed on the inbound request.
func fakeSSEServer(t *testing.T, payloads []string) (*httptest.Server, *recorded) {
	t.Helper()
	rec := &recorded{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.path = r.URL.Path
		rec.auth = r.Header.Get("Authorization")
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
	return srv, rec
}

type recorded struct {
	path string
	auth string
}

func sampleChatRequest() llmrouter.ChatRequest {
	return llmrouter.ChatRequest{
		Model:    "meta-llama/Llama-3-70b-chat-hf",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	}
}

func TestTogetherProvider(t *testing.T) {
	t.Run("DefaultBaseURLConstant", func(t *testing.T) {
		if together.DefaultBaseURL != "https://api.together.xyz/v1" {
			t.Errorf("DefaultBaseURL = %q, want https://api.together.xyz/v1", together.DefaultBaseURL)
		}
	})

	t.Run("NameReturnsTogether", func(t *testing.T) {
		p, err := together.New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if got := p.Name(); got != "together" {
			t.Errorf("Name = %q, want together", got)
		}
	})

	t.Run("NewWithoutAPIKeyFails", func(t *testing.T) {
		_, err := together.New()
		if err == nil {
			t.Fatal("expected error for missing api key")
		}
	})

	t.Run("NewWithWhitespaceAPIKeyFails", func(t *testing.T) {
		_, err := together.New(llmrouter.WithAPIKey("   "))
		if err == nil {
			t.Fatal("expected error for whitespace api key")
		}
	})

	t.Run("NewIgnoresNilOption", func(t *testing.T) {
		_, err := together.New(nil, llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
	})

	t.Run("NewWithEmptyBaseURLFails", func(t *testing.T) {
		_, err := together.New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(""))
		if err == nil {
			t.Fatal("expected error for empty base url")
		}
	})

	t.Run("NewAcceptsUserBaseURLOverride", func(t *testing.T) {
		srv, _ := fakeSSEServer(t, nil)
		defer srv.Close()
		p, err := together.New(
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if p == nil {
			t.Fatal("nil provider")
		}
	})

	t.Run("CompletionStreamForwardsContent", func(t *testing.T) {
		payloads := []string{
			`{"id":"a","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
			`{"id":"a","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":" together"}}]}`,
		}
		srv, _ := fakeSSEServer(t, payloads)
		defer srv.Close()

		p, err := together.New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		stream, err := p.CompletionStream(ctx, sampleChatRequest())
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}
		var content string
		for chunk := range stream.Chunks() {
			for _, c := range chunk.Choices {
				content += c.Delta.Content
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatalf("stream.Err: %v", err)
		}
		if content != "Hello together" {
			t.Errorf("content = %q, want %q", content, "Hello together")
		}
	})

	t.Run("CompletionStreamPreservesRawBytes", func(t *testing.T) {
		payload := `{"id":"a","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"x"}}]}`
		srv, _ := fakeSSEServer(t, []string{payload})
		defer srv.Close()

		p, _ := together.New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
		stream, err := p.CompletionStream(context.Background(), sampleChatRequest())
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}
		var sawRaw bool
		for chunk := range stream.Chunks() {
			if len(chunk.Raw) > 0 {
				sawRaw = true
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatalf("stream.Err: %v", err)
		}
		if !sawRaw {
			t.Error("expected chunk.Raw to be populated")
		}
	})

	t.Run("CompletionStreamSurfacesUsage", func(t *testing.T) {
		payloads := []string{
			`{"id":"a","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":6,"total_tokens":10}}`,
		}
		srv, _ := fakeSSEServer(t, payloads)
		defer srv.Close()

		p, _ := together.New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
		stream, err := p.CompletionStream(context.Background(), sampleChatRequest())
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}
		var usage *llmrouter.Usage
		for chunk := range stream.Chunks() {
			if chunk.Usage != nil {
				usage = chunk.Usage
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatalf("stream.Err: %v", err)
		}
		if usage == nil || usage.TotalTokens != 10 {
			t.Errorf("usage = %+v, want TotalTokens=10", usage)
		}
	})

	t.Run("UpstreamErrorRewritesProviderName", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
			fmt.Fprint(w, `{"error":"nope"}`)
		}))
		defer srv.Close()

		p, _ := together.New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
		_, err := p.CompletionStream(context.Background(), sampleChatRequest())
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("error = %T %v, want *llmrouter.ErrUpstream", err, err)
		}
		if ue.Provider != "together" {
			t.Errorf("Provider = %q, want together", ue.Provider)
		}
	})

	t.Run("UpstreamErrorPropagatesStatusCode", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprint(w, `rate limited`)
		}))
		defer srv.Close()

		p, _ := together.New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
		_, err := p.CompletionStream(context.Background(), sampleChatRequest())
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("error = %T %v", err, err)
		}
		if ue.StatusCode != http.StatusTooManyRequests {
			t.Errorf("StatusCode = %d, want 429", ue.StatusCode)
		}
	})

	t.Run("UpstreamErrorPropagatesBody", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, `missing model field`)
		}))
		defer srv.Close()

		p, _ := together.New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
		_, err := p.CompletionStream(context.Background(), sampleChatRequest())
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("error = %T %v", err, err)
		}
		if !strings.Contains(ue.Body, "missing model field") {
			t.Errorf("Body = %q, want to contain 'missing model field'", ue.Body)
		}
	})

	t.Run("UpstreamErrorMessageIncludesTogether", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer srv.Close()

		p, _ := together.New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
		_, err := p.CompletionStream(context.Background(), sampleChatRequest())
		if err == nil || !strings.Contains(err.Error(), "together upstream") {
			t.Errorf("err = %v, want to contain 'together upstream'", err)
		}
	})

	t.Run("AuthorizationHeaderSent", func(t *testing.T) {
		srv, rec := fakeSSEServer(t, nil)
		defer srv.Close()

		p, _ := together.New(llmrouter.WithAPIKey("secret-key"), llmrouter.WithBaseURL(srv.URL))
		stream, err := p.CompletionStream(context.Background(), sampleChatRequest())
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}
		for range stream.Chunks() {
		}
		_ = stream.Err()
		if rec.auth != "Bearer secret-key" {
			t.Errorf("Authorization = %q, want Bearer secret-key", rec.auth)
		}
	})

	t.Run("RequestHitsChatCompletionsPath", func(t *testing.T) {
		srv, rec := fakeSSEServer(t, nil)
		defer srv.Close()

		p, _ := together.New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
		stream, err := p.CompletionStream(context.Background(), sampleChatRequest())
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}
		for range stream.Chunks() {
		}
		_ = stream.Err()
		if rec.path != "/chat/completions" {
			t.Errorf("path = %q, want /chat/completions", rec.path)
		}
	})

	t.Run("ContextCancellationAbortsStream", func(t *testing.T) {
		// Server that streams forever; we'll cancel the context.
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher, _ := w.(http.Flusher)
			for i := 0; i < 1000; i++ {
				select {
				case <-r.Context().Done():
					return
				default:
				}
				fmt.Fprintf(w, "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"tick\"}}]}\n\n")
				if flusher != nil {
					flusher.Flush()
				}
				time.Sleep(20 * time.Millisecond)
			}
		}))
		defer srv.Close()

		p, _ := together.New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
		ctx, cancel := context.WithCancel(context.Background())
		stream, err := p.CompletionStream(ctx, sampleChatRequest())
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}
		// Read one chunk then cancel.
		<-stream.Chunks()
		cancel()
		// Drain — must not hang.
		done := make(chan struct{})
		go func() {
			for range stream.Chunks() {
			}
			_ = stream.Err()
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Fatal("stream did not close after cancel")
		}
	})

	t.Run("NonUpstreamErrorPassThrough", func(t *testing.T) {
		// Point at a closed server to force a transport error (not an
		// *llmrouter.ErrUpstream). Verify the error is surfaced
		// unchanged.
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
		closedURL := srv.URL
		srv.Close()

		p, _ := together.New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(closedURL))
		_, err := p.CompletionStream(context.Background(), sampleChatRequest())
		if err == nil {
			t.Fatal("expected transport error, got nil")
		}
		var ue *llmrouter.ErrUpstream
		if errors.As(err, &ue) {
			t.Fatalf("err = %v, did not expect *llmrouter.ErrUpstream", err)
		}
	})

	t.Run("ProviderImplementsLLMRouterProvider", func(t *testing.T) {
		var _ llmrouter.Provider = (*together.Provider)(nil)
	})
}
