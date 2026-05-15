package azureopenai_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/azureopenai"
)

const (
	testKey        = "test-key"
	testDeployment = "my-deploy"
	testAPIVersion = "2024-10-21"
)

// fakeSSEServer returns an httptest server that emits payloads as SSE
// events and runs an optional request inspector for header/path assertions.
func fakeSSEServer(t *testing.T, payloads []string, inspect func(r *http.Request)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if inspect != nil {
			inspect(r)
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

func newAPIKeyProvider(t *testing.T, baseURL string, extraOpts ...llmrouter.Option) *azureopenai.Provider {
	t.Helper()
	opts := []llmrouter.Option{
		llmrouter.WithAPIKey(testKey),
		llmrouter.WithBaseURL(baseURL),
		azureopenai.WithDeployment(testDeployment),
		azureopenai.WithAPIVersion(testAPIVersion),
	}
	opts = append(opts, extraOpts...)
	p, err := azureopenai.New(opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// ---------------------------------------------------------------------
// New() — config validation table.
// ---------------------------------------------------------------------

func TestNew_ConfigValidation(t *testing.T) {
	dummyAAD := azureopenai.AADTokenSource(func(ctx context.Context) (string, error) {
		return "tok", nil
	})

	cases := []struct {
		name      string
		opts      []llmrouter.Option
		wantErrIs error
		wantOK    bool
	}{
		{
			name: "happy_path_apikey",
			opts: []llmrouter.Option{
				llmrouter.WithAPIKey("k"),
				llmrouter.WithBaseURL("https://r.openai.azure.com"),
				azureopenai.WithDeployment("d"),
				azureopenai.WithAPIVersion("2024-10-21"),
			},
			wantOK: true,
		},
		{
			name: "happy_path_aad",
			opts: []llmrouter.Option{
				azureopenai.WithAADToken(dummyAAD),
				llmrouter.WithBaseURL("https://r.openai.azure.com"),
				azureopenai.WithDeployment("d"),
				azureopenai.WithAPIVersion("2024-10-21"),
			},
			wantOK: true,
		},
		{
			name: "missing_apikey_and_aad",
			opts: []llmrouter.Option{
				llmrouter.WithBaseURL("https://r.openai.azure.com"),
				azureopenai.WithDeployment("d"),
				azureopenai.WithAPIVersion("2024-10-21"),
			},
			wantErrIs: llmrouter.ErrInvalidConfig,
		},
		{
			name: "both_apikey_and_aad",
			opts: []llmrouter.Option{
				llmrouter.WithAPIKey("k"),
				azureopenai.WithAADToken(dummyAAD),
				llmrouter.WithBaseURL("https://r.openai.azure.com"),
				azureopenai.WithDeployment("d"),
				azureopenai.WithAPIVersion("2024-10-21"),
			},
			wantErrIs: llmrouter.ErrInvalidConfig,
		},
		{
			name: "missing_baseurl",
			opts: []llmrouter.Option{
				llmrouter.WithAPIKey("k"),
				azureopenai.WithDeployment("d"),
				azureopenai.WithAPIVersion("2024-10-21"),
			},
			wantErrIs: llmrouter.ErrInvalidConfig,
		},
		{
			name: "missing_deployment",
			opts: []llmrouter.Option{
				llmrouter.WithAPIKey("k"),
				llmrouter.WithBaseURL("https://r.openai.azure.com"),
				azureopenai.WithAPIVersion("2024-10-21"),
			},
			wantErrIs: llmrouter.ErrInvalidConfig,
		},
		{
			name: "missing_apiversion",
			opts: []llmrouter.Option{
				llmrouter.WithAPIKey("k"),
				llmrouter.WithBaseURL("https://r.openai.azure.com"),
				azureopenai.WithDeployment("d"),
			},
			wantErrIs: llmrouter.ErrInvalidConfig,
		},
		{
			name: "missing_all_extras",
			opts: []llmrouter.Option{
				llmrouter.WithAPIKey("k"),
				llmrouter.WithBaseURL("https://r.openai.azure.com"),
			},
			wantErrIs: llmrouter.ErrInvalidConfig,
		},
		{
			name: "empty_deployment",
			opts: []llmrouter.Option{
				llmrouter.WithAPIKey("k"),
				llmrouter.WithBaseURL("https://r.openai.azure.com"),
				azureopenai.WithDeployment("   "),
				azureopenai.WithAPIVersion("2024-10-21"),
			},
			// WithDeployment returns the error directly (not wrapped in ErrInvalidConfig).
		},
		{
			name: "empty_apiversion",
			opts: []llmrouter.Option{
				llmrouter.WithAPIKey("k"),
				llmrouter.WithBaseURL("https://r.openai.azure.com"),
				azureopenai.WithDeployment("d"),
				azureopenai.WithAPIVersion(""),
			},
		},
		{
			name: "nil_aad_source",
			opts: []llmrouter.Option{
				azureopenai.WithAADToken(nil),
				llmrouter.WithBaseURL("https://r.openai.azure.com"),
				azureopenai.WithDeployment("d"),
				azureopenai.WithAPIVersion("2024-10-21"),
			},
		},
		{
			name: "empty_apikey",
			opts: []llmrouter.Option{
				llmrouter.WithAPIKey(""),
				llmrouter.WithBaseURL("https://r.openai.azure.com"),
				azureopenai.WithDeployment("d"),
				azureopenai.WithAPIVersion("2024-10-21"),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := azureopenai.New(tc.opts...)
			if tc.wantOK {
				if err != nil {
					t.Fatalf("New: unexpected err: %v", err)
				}
				if p == nil {
					t.Fatalf("New: nil provider")
				}
				if p.Name() != "azureopenai" {
					t.Errorf("Name = %q, want azureopenai", p.Name())
				}
				return
			}
			if err == nil {
				t.Fatalf("New: expected error, got nil")
			}
			if tc.wantErrIs != nil && !errors.Is(err, tc.wantErrIs) {
				t.Errorf("err = %v, want errors.Is(_, %v)", err, tc.wantErrIs)
			}
		})
	}
}

// ---------------------------------------------------------------------
// End-to-end happy path (API key).
// ---------------------------------------------------------------------

func TestCompletionStream_HappyPath_APIKey(t *testing.T) {
	payloads := []string{
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"}}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
	}

	var pathSeen, apiKeySeen, authSeen, apiVersionSeen string
	srv := fakeSSEServer(t, payloads, func(r *http.Request) {
		pathSeen = r.URL.Path
		apiKeySeen = r.Header.Get("api-key")
		authSeen = r.Header.Get("Authorization")
		apiVersionSeen = r.URL.Query().Get("api-version")
	})
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "gpt-4o",
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

	t.Run("chunk_count", func(t *testing.T) {
		if gotCount != 3 {
			t.Errorf("count = %d, want 3", gotCount)
		}
	})
	t.Run("content_concatenated", func(t *testing.T) {
		if gotContent != "Hello world" {
			t.Errorf("content = %q, want %q", gotContent, "Hello world")
		}
	})
	t.Run("finish_reason", func(t *testing.T) {
		if gotFinish != "stop" {
			t.Errorf("finish = %q, want stop", gotFinish)
		}
	})
	t.Run("usage_populated", func(t *testing.T) {
		if gotUsage == nil {
			t.Fatalf("usage nil, want populated")
		}
		if gotUsage.TotalTokens != 5 || gotUsage.PromptTokens != 3 || gotUsage.CompletionTokens != 2 {
			t.Errorf("usage = %+v, want {3,2,5}", gotUsage)
		}
	})
	t.Run("url_path", func(t *testing.T) {
		want := "/openai/deployments/" + testDeployment + "/chat/completions"
		if pathSeen != want {
			t.Errorf("path = %q, want %q", pathSeen, want)
		}
	})
	t.Run("api_version_query", func(t *testing.T) {
		if apiVersionSeen != testAPIVersion {
			t.Errorf("api-version = %q, want %q", apiVersionSeen, testAPIVersion)
		}
	})
	t.Run("api_key_header_set", func(t *testing.T) {
		if apiKeySeen != testKey {
			t.Errorf("api-key = %q, want %q", apiKeySeen, testKey)
		}
	})
	t.Run("no_authorization_header_with_apikey_auth", func(t *testing.T) {
		if authSeen != "" {
			t.Errorf("Authorization = %q, want empty when using api-key auth", authSeen)
		}
	})
}

// ---------------------------------------------------------------------
// End-to-end happy path (AAD).
// ---------------------------------------------------------------------

func TestCompletionStream_HappyPath_AAD(t *testing.T) {
	payloads := []string{
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`,
	}

	var calls int32
	source := azureopenai.AADTokenSource(func(ctx context.Context) (string, error) {
		atomic.AddInt32(&calls, 1)
		return "aad-token-v1", nil
	})

	var authSeen, apiKeySeen string
	srv := fakeSSEServer(t, payloads, func(r *http.Request) {
		authSeen = r.Header.Get("Authorization")
		apiKeySeen = r.Header.Get("api-key")
	})
	defer srv.Close()

	p, err := azureopenai.New(
		azureopenai.WithAADToken(source),
		llmrouter.WithBaseURL(srv.URL),
		azureopenai.WithDeployment(testDeployment),
		azureopenai.WithAPIVersion(testAPIVersion),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "gpt-4o",
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

	t.Run("authorization_header_bearer", func(t *testing.T) {
		want := "Bearer aad-token-v1"
		if authSeen != want {
			t.Errorf("Authorization = %q, want %q", authSeen, want)
		}
	})
	t.Run("no_api_key_header_with_aad_auth", func(t *testing.T) {
		if apiKeySeen != "" {
			t.Errorf("api-key = %q, want empty when using AAD auth", apiKeySeen)
		}
	})
	t.Run("aad_source_called_once", func(t *testing.T) {
		if got := atomic.LoadInt32(&calls); got != 1 {
			t.Errorf("aad source calls = %d, want 1", got)
		}
	})

	// Second request: AAD source should be called again (refreshed per request).
	stream2, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "gpt-4o",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi again")},
	})
	if err != nil {
		t.Fatalf("CompletionStream (2): %v", err)
	}
	for range stream2.Chunks() {
	}
	t.Run("aad_source_called_per_request", func(t *testing.T) {
		if got := atomic.LoadInt32(&calls); got != 2 {
			t.Errorf("aad source calls after 2 requests = %d, want 2", got)
		}
	})
}

func TestCompletionStream_AADSourceError(t *testing.T) {
	source := azureopenai.AADTokenSource(func(ctx context.Context) (string, error) {
		return "", errors.New("identity unavailable")
	})

	// We don't actually need a server — the failure happens before HTTP dispatch.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("server should not be hit when AAD source fails")
	}))
	defer srv.Close()

	p, err := azureopenai.New(
		azureopenai.WithAADToken(source),
		llmrouter.WithBaseURL(srv.URL),
		azureopenai.WithDeployment(testDeployment),
		azureopenai.WithAPIVersion(testAPIVersion),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "gpt-4o",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatalf("expected error from AAD source failure, got nil")
	}
	if !strings.Contains(err.Error(), "identity unavailable") {
		t.Errorf("err = %v, expected to wrap source error", err)
	}
}

func TestCompletionStream_AADSourceEmptyToken(t *testing.T) {
	source := azureopenai.AADTokenSource(func(ctx context.Context) (string, error) {
		return "   ", nil
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("server should not be hit when AAD token empty")
	}))
	defer srv.Close()

	p, err := azureopenai.New(
		azureopenai.WithAADToken(source),
		llmrouter.WithBaseURL(srv.URL),
		azureopenai.WithDeployment(testDeployment),
		azureopenai.WithAPIVersion(testAPIVersion),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "gpt-4o",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatalf("expected error from empty token, got nil")
	}
}

// ---------------------------------------------------------------------
// Upstream HTTP errors table.
// ---------------------------------------------------------------------

func TestCompletionStream_UpstreamErrors(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
	}{
		{"unauthorized_401", http.StatusUnauthorized, `{"error":"unauthorized"}`},
		{"forbidden_403", http.StatusForbidden, `{"error":"forbidden"}`},
		{"rate_limit_429", http.StatusTooManyRequests, `{"error":"slow down"}`},
		{"server_error_500", http.StatusInternalServerError, `oops`},
		{"bad_request_400", http.StatusBadRequest, `{"error":"bad"}`},
		{"not_found_404", http.StatusNotFound, `{"error":"no deployment"}`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer srv.Close()

			p := newAPIKeyProvider(t, srv.URL)
			_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
				Model:    "gpt-4o",
				Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
			})
			if err == nil {
				t.Fatalf("expected ErrUpstream, got nil")
			}
			var ue *llmrouter.ErrUpstream
			if !errors.As(err, &ue) {
				t.Fatalf("err = %v, want *ErrUpstream", err)
			}
			if ue.StatusCode != tc.status {
				t.Errorf("StatusCode = %d, want %d", ue.StatusCode, tc.status)
			}
			if ue.Provider != "azureopenai" {
				t.Errorf("Provider = %q, want azureopenai", ue.Provider)
			}
			if !strings.Contains(ue.Body, strings.TrimSpace(tc.body)) {
				t.Errorf("Body = %q, want substring %q", ue.Body, tc.body)
			}
		})
	}
}

func TestCompletionStream_UpstreamErrorBodyCappedAt1KiB(t *testing.T) {
	big := strings.Repeat("A", 4096)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, big)
	}))
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "gpt-4o",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	var ue *llmrouter.ErrUpstream
	if !errors.As(err, &ue) {
		t.Fatalf("err = %v, want *ErrUpstream", err)
	}
	if len(ue.Body) > 1024 {
		t.Errorf("body length = %d, want <= 1024", len(ue.Body))
	}
}

// ---------------------------------------------------------------------
// SSE edge cases.
// ---------------------------------------------------------------------

func TestCompletionStream_MalformedChunkIsSkipped(t *testing.T) {
	payloads := []string{
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"A"}}]}`,
		`{not-valid-json`,
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"B"},"finish_reason":"stop"}]}`,
	}
	srv := fakeSSEServer(t, payloads, nil)
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "gpt-4o",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	var content string
	var count int
	for chunk := range stream.Chunks() {
		count++
		for _, c := range chunk.Choices {
			content += c.Delta.Content
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err = %v", err)
	}
	if count != 2 {
		t.Errorf("count = %d, want 2 (malformed skipped)", count)
	}
	if content != "AB" {
		t.Errorf("content = %q, want AB", content)
	}
}

func TestCompletionStream_ContextCancellation(t *testing.T) {
	// Server emits one chunk then closes — exercises clean producer exit.
	srv := fakeSSEServer(t, []string{
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`,
	}, nil)
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "gpt-4o",
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
}

// ---------------------------------------------------------------------
// Request body assembly.
// ---------------------------------------------------------------------

func TestCompletionStream_BodyForcesStreamAndIncludeUsage(t *testing.T) {
	captured := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- body
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "gpt-4o",
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
	body := <-captured
	if !strings.Contains(string(body), `"stream":true`) {
		t.Errorf("body missing stream:true: %s", body)
	}
	if !strings.Contains(string(body), `"include_usage":true`) {
		t.Errorf("body missing include_usage:true: %s", body)
	}
	if !strings.Contains(string(body), `"model":"gpt-4o"`) {
		t.Errorf("body missing model:gpt-4o: %s", body)
	}
}

func TestCompletionStream_RawPassthrough(t *testing.T) {
	captured := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- body
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	raw := []byte(`{"messages":[{"role":"user","content":"raw hi"}],"custom_field":"keep_me"}`)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Raw: raw,
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	body := <-captured

	t.Run("custom_field_preserved", func(t *testing.T) {
		if !strings.Contains(string(body), `"custom_field":"keep_me"`) {
			t.Errorf("custom field dropped: %s", body)
		}
	})
	t.Run("stream_forced_true", func(t *testing.T) {
		if !strings.Contains(string(body), `"stream":true`) {
			t.Errorf("stream:true missing: %s", body)
		}
	})
	t.Run("include_usage_forced", func(t *testing.T) {
		if !strings.Contains(string(body), `"include_usage":true`) {
			t.Errorf("include_usage missing: %s", body)
		}
	})
}

func TestCompletionStream_RawInvalidJSON(t *testing.T) {
	p := newAPIKeyProvider(t, "https://example.invalid")
	_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Raw: []byte(`{not-json`),
	})
	if err == nil {
		t.Fatalf("expected error for invalid raw JSON")
	}
	if !strings.Contains(err.Error(), "invalid raw request") {
		t.Errorf("err = %v, want invalid raw request", err)
	}
}

func TestCompletionStream_ModelOverlayWhenEmpty(t *testing.T) {
	captured := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- body
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	// Raw with a model, ChatRequest.Model empty — Raw's model should remain.
	raw := []byte(`{"model":"original-model","messages":[{"role":"user","content":"hi"}]}`)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{Raw: raw})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	body := <-captured
	if !strings.Contains(string(body), `"model":"original-model"`) {
		t.Errorf("expected raw model preserved, got: %s", body)
	}
}

func TestCompletionStream_ModelOverlayWhenSet(t *testing.T) {
	captured := make(chan []byte, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		captured <- body
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	raw := []byte(`{"model":"original-model","messages":[{"role":"user","content":"hi"}]}`)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model: "override-model",
		Raw:   raw,
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	body := <-captured
	if !strings.Contains(string(body), `"model":"override-model"`) {
		t.Errorf("expected model overridden, got: %s", body)
	}
}

// ---------------------------------------------------------------------
// HTTP transport errors.
// ---------------------------------------------------------------------

func TestCompletionStream_HTTPErrorReturned(t *testing.T) {
	p := newAPIKeyProvider(t, "http://127.0.0.1:1")
	_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "gpt-4o",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatalf("expected transport error, got nil")
	}
}

// ---------------------------------------------------------------------
// Misc.
// ---------------------------------------------------------------------

func TestProvider_Name(t *testing.T) {
	p := newAPIKeyProvider(t, "https://example.invalid")
	if p.Name() != "azureopenai" {
		t.Errorf("Name = %q, want azureopenai", p.Name())
	}
}

func TestCompletionStream_PathContainsTrailingSlashHandled(t *testing.T) {
	var seenPath, seenAPIVersion string
	srv := fakeSSEServer(t, nil, func(r *http.Request) {
		seenPath = r.URL.Path
		seenAPIVersion = r.URL.Query().Get("api-version")
	})
	defer srv.Close()

	// Even if user passes a trailing slash, WithBaseURL trims it; check still passes.
	p := newAPIKeyProvider(t, srv.URL+"/")
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "gpt-4o",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	want := "/openai/deployments/" + testDeployment + "/chat/completions"
	if seenPath != want {
		t.Errorf("path = %q, want %q", seenPath, want)
	}
	if seenAPIVersion != testAPIVersion {
		t.Errorf("api-version = %q, want %q", seenAPIVersion, testAPIVersion)
	}
}
