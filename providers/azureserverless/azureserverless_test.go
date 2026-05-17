package azureserverless_test

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
	"github.com/elloloop/llmrouter/providers/azureserverless"
)

const (
	testKey     = "test-key"
	testBaseURL = "https://llama-eus.eastus.models.ai.azure.com"
)

// fakeSSEServer returns an httptest server that emits payloads as SSE
// events and runs an optional request inspector for header/path/body
// assertions.
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

// newAPIKeyProvider builds a provider configured against the given
// base URL using api-key auth — the default for most tests.
func newAPIKeyProvider(t *testing.T, baseURL string, extraOpts ...llmrouter.Option) *azureserverless.Provider {
	t.Helper()
	opts := []llmrouter.Option{
		llmrouter.WithAPIKey(testKey),
		llmrouter.WithBaseURL(baseURL),
	}
	opts = append(opts, extraOpts...)
	p, err := azureserverless.New(opts...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// ---------------------------------------------------------------------
// New() — config validation table.
// ---------------------------------------------------------------------

func TestNew_ConfigValidation(t *testing.T) {
	dummyAAD := azureserverless.AADTokenSource(func(ctx context.Context) (string, error) {
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
				llmrouter.WithBaseURL(testBaseURL),
			},
			wantOK: true,
		},
		{
			name: "happy_path_aad",
			opts: []llmrouter.Option{
				azureserverless.WithAADToken(dummyAAD),
				llmrouter.WithBaseURL(testBaseURL),
			},
			wantOK: true,
		},
		{
			name: "happy_path_hub_scoped_baseurl",
			opts: []llmrouter.Option{
				llmrouter.WithAPIKey("k"),
				llmrouter.WithBaseURL("https://my-hub.eastus.inference.ai.azure.com"),
			},
			wantOK: true,
		},
		{
			name: "missing_apikey_and_aad",
			opts: []llmrouter.Option{
				llmrouter.WithBaseURL(testBaseURL),
			},
			wantErrIs: llmrouter.ErrInvalidConfig,
		},
		{
			name: "both_apikey_and_aad",
			opts: []llmrouter.Option{
				llmrouter.WithAPIKey("k"),
				azureserverless.WithAADToken(dummyAAD),
				llmrouter.WithBaseURL(testBaseURL),
			},
			wantErrIs: llmrouter.ErrInvalidConfig,
		},
		{
			name: "missing_baseurl_apikey",
			opts: []llmrouter.Option{
				llmrouter.WithAPIKey("k"),
			},
			wantErrIs: llmrouter.ErrInvalidConfig,
		},
		{
			name: "missing_baseurl_aad",
			opts: []llmrouter.Option{
				azureserverless.WithAADToken(dummyAAD),
			},
			wantErrIs: llmrouter.ErrInvalidConfig,
		},
		{
			name: "nil_aad_source_rejected_by_option",
			opts: []llmrouter.Option{
				azureserverless.WithAADToken(nil),
				llmrouter.WithBaseURL(testBaseURL),
			},
		},
		{
			name: "empty_apikey_rejected_by_option",
			opts: []llmrouter.Option{
				llmrouter.WithAPIKey(""),
				llmrouter.WithBaseURL(testBaseURL),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := azureserverless.New(tc.opts...)
			if tc.wantOK {
				if err != nil {
					t.Fatalf("New: unexpected err: %v", err)
				}
				if p == nil {
					t.Fatalf("New: nil provider")
				}
				if p.Name() != "azureserverless" {
					t.Errorf("Name = %q, want azureserverless", p.Name())
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
// URL construction.
// ---------------------------------------------------------------------

func TestCompletionStream_URLPathAndQuery(t *testing.T) {
	var seenPath, seenRawQuery string
	srv := fakeSSEServer(t, nil, func(r *http.Request) {
		seenPath = r.URL.Path
		seenRawQuery = r.URL.RawQuery
	})
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "Meta-Llama-3-70B-Instruct",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}

	t.Run("default_path", func(t *testing.T) {
		if seenPath != "/v1/chat/completions" {
			t.Errorf("path = %q, want /v1/chat/completions", seenPath)
		}
	})
	t.Run("no_query_string", func(t *testing.T) {
		if seenRawQuery != "" {
			t.Errorf("raw query = %q, want empty (Foundry Serverless does not use api-version)", seenRawQuery)
		}
	})
	t.Run("no_api_version_query_specifically", func(t *testing.T) {
		if strings.Contains(seenRawQuery, "api-version") {
			t.Errorf("query contains api-version: %q", seenRawQuery)
		}
	})
}

func TestCompletionStream_BaseURLTrailingSlashTrimmed(t *testing.T) {
	var seenPath string
	srv := fakeSSEServer(t, nil, func(r *http.Request) {
		seenPath = r.URL.Path
	})
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL+"/")
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "m",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	if seenPath != "/v1/chat/completions" {
		t.Errorf("path = %q, want /v1/chat/completions", seenPath)
	}
}

func TestCompletionStream_HubScopedEndpointWorks(t *testing.T) {
	// Hub-scoped endpoints serve multiple models — only difference vs
	// deployment-scoped is the hostname; we just assert the path is the
	// same and the request reaches the server.
	var seenPath string
	srv := fakeSSEServer(t, nil, func(r *http.Request) {
		seenPath = r.URL.Path
	})
	defer srv.Close()
	p := newAPIKeyProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "any-hub-deployment",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	if seenPath != "/v1/chat/completions" {
		t.Errorf("path = %q, want /v1/chat/completions", seenPath)
	}
}

// ---------------------------------------------------------------------
// Auth headers.
// ---------------------------------------------------------------------

func TestCompletionStream_APIKeyHeaders(t *testing.T) {
	var apiKeySeen, authSeen, betaSeen string
	srv := fakeSSEServer(t, nil, func(r *http.Request) {
		apiKeySeen = r.Header.Get("api-key")
		authSeen = r.Header.Get("Authorization")
		betaSeen = r.Header.Get("OpenAI-Beta")
	})
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "m",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}

	t.Run("api_key_header_set", func(t *testing.T) {
		if apiKeySeen != testKey {
			t.Errorf("api-key = %q, want %q", apiKeySeen, testKey)
		}
	})
	t.Run("authorization_absent", func(t *testing.T) {
		if authSeen != "" {
			t.Errorf("Authorization = %q, want empty when using api-key auth", authSeen)
		}
	})
	t.Run("openai_beta_absent", func(t *testing.T) {
		if betaSeen != "" {
			t.Errorf("OpenAI-Beta = %q, want empty", betaSeen)
		}
	})
}

func TestCompletionStream_AADHeaders(t *testing.T) {
	var calls int32
	source := azureserverless.AADTokenSource(func(ctx context.Context) (string, error) {
		atomic.AddInt32(&calls, 1)
		return "aad-token-v1", nil
	})

	var authSeen, apiKeySeen string
	srv := fakeSSEServer(t, nil, func(r *http.Request) {
		authSeen = r.Header.Get("Authorization")
		apiKeySeen = r.Header.Get("api-key")
	})
	defer srv.Close()

	p, err := azureserverless.New(
		azureserverless.WithAADToken(source),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "m",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}

	t.Run("authorization_bearer", func(t *testing.T) {
		want := "Bearer aad-token-v1"
		if authSeen != want {
			t.Errorf("Authorization = %q, want %q", authSeen, want)
		}
	})
	t.Run("api_key_absent", func(t *testing.T) {
		if apiKeySeen != "" {
			t.Errorf("api-key = %q, want empty when using AAD", apiKeySeen)
		}
	})
	t.Run("aad_source_called_once", func(t *testing.T) {
		if got := atomic.LoadInt32(&calls); got != 1 {
			t.Errorf("aad calls after 1 request = %d, want 1", got)
		}
	})

	stream2, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "m",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "again")},
	})
	if err != nil {
		t.Fatalf("CompletionStream (2): %v", err)
	}
	for range stream2.Chunks() {
	}
	t.Run("aad_source_called_per_request", func(t *testing.T) {
		if got := atomic.LoadInt32(&calls); got != 2 {
			t.Errorf("aad calls after 2 requests = %d, want 2", got)
		}
	})
}

func TestCompletionStream_AADSourceError(t *testing.T) {
	source := azureserverless.AADTokenSource(func(ctx context.Context) (string, error) {
		return "", errors.New("identity unavailable")
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("server should not be hit when AAD source fails")
	}))
	defer srv.Close()

	p, err := azureserverless.New(
		azureserverless.WithAADToken(source),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "m",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatalf("expected error from AAD source failure, got nil")
	}
	if !strings.Contains(err.Error(), "identity unavailable") {
		t.Errorf("err = %v, want to wrap source error", err)
	}
}

func TestCompletionStream_AADSourceEmptyToken(t *testing.T) {
	source := azureserverless.AADTokenSource(func(ctx context.Context) (string, error) {
		return "   ", nil
	})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("server should not be hit when AAD token empty")
	}))
	defer srv.Close()

	p, err := azureserverless.New(
		azureserverless.WithAADToken(source),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "m",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatalf("expected error from empty token, got nil")
	}
	if !strings.Contains(err.Error(), "empty token") {
		t.Errorf("err = %v, want empty-token error", err)
	}
}

// ---------------------------------------------------------------------
// End-to-end happy path.
// ---------------------------------------------------------------------

func TestCompletionStream_HappyPath(t *testing.T) {
	payloads := []string{
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"llama","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"llama","choices":[{"index":0,"delta":{"content":" world"}}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"llama","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
	}
	srv := fakeSSEServer(t, payloads, nil)
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "llama",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}

	var (
		content string
		finish  string
		usage   *llmrouter.Usage
		count   int
		rawLens []int
	)
	for chunk := range stream.Chunks() {
		count++
		rawLens = append(rawLens, len(chunk.Raw))
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

	t.Run("chunk_count", func(t *testing.T) {
		if count != 3 {
			t.Errorf("count = %d, want 3", count)
		}
	})
	t.Run("content_concatenated", func(t *testing.T) {
		if content != "Hello world" {
			t.Errorf("content = %q, want Hello world", content)
		}
	})
	t.Run("finish_reason", func(t *testing.T) {
		if finish != "stop" {
			t.Errorf("finish = %q, want stop", finish)
		}
	})
	t.Run("usage_populated", func(t *testing.T) {
		if usage == nil {
			t.Fatalf("usage nil")
		}
		if usage.TotalTokens != 5 || usage.PromptTokens != 3 || usage.CompletionTokens != 2 {
			t.Errorf("usage = %+v, want {3,2,5}", usage)
		}
	})
	t.Run("raw_present_on_each_chunk", func(t *testing.T) {
		for i, n := range rawLens {
			if n == 0 {
				t.Errorf("chunk %d Raw empty", i)
			}
		}
	})
}

func TestCompletionStream_ChunkRawByteEqualToPayload(t *testing.T) {
	payload := `{"id":"x","object":"chat.completion.chunk","created":1,"model":"llama","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`
	srv := fakeSSEServer(t, []string{payload}, nil)
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "llama",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	var got string
	for chunk := range stream.Chunks() {
		got = string(chunk.Raw)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err = %v", err)
	}
	if got != payload {
		t.Errorf("chunk.Raw = %q, want byte-equal to %q", got, payload)
	}
}

func TestCompletionStream_MultilineDataLines(t *testing.T) {
	// SSE spec: multi-line data: events join with "\n". Some upstreams
	// split JSON across two data: lines.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		// Two lines that concatenate (with newline) into valid JSON.
		fmt.Fprint(w, "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\n")
		fmt.Fprint(w, "data: \"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
		if f != nil {
			f.Flush()
		}
	}))
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "m",
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
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err = %v", err)
	}
	if content != "hi" {
		t.Errorf("content = %q, want hi", content)
	}
}

func TestCompletionStream_DataLineWithoutSpace(t *testing.T) {
	// Some upstreams emit `data:<payload>` without the space after the
	// colon. Both forms must work.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		fmt.Fprint(w, "data:{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n")
		fmt.Fprint(w, "data:[DONE]\n\n")
		if f != nil {
			f.Flush()
		}
	}))
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "m",
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
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err = %v", err)
	}
	if content != "ok" {
		t.Errorf("content = %q, want ok", content)
	}
}

func TestCompletionStream_DoneTerminatesCleanly(t *testing.T) {
	srv := fakeSSEServer(t, []string{
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`,
	}, nil)
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
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
		t.Errorf("stream.Err = %v, want nil after [DONE]", err)
	}
}

func TestCompletionStream_MalformedChunkSkipped(t *testing.T) {
	payloads := []string{
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"A"}}]}`,
		`{not-valid-json`,
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"B"},"finish_reason":"stop"}]}`,
	}
	srv := fakeSSEServer(t, payloads, nil)
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "m",
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

func TestCompletionStream_MidStreamErrorEnvelope(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"A\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"error\":{\"message\":\"quota exceeded\",\"type\":\"insufficient_quota\"}}\n\n")
		if f != nil {
			f.Flush()
		}
	}))
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "m",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	err = stream.Err()
	var ue *llmrouter.ErrUpstream
	if !errors.As(err, &ue) {
		t.Fatalf("stream.Err = %v, want *ErrUpstream", err)
	}
	t.Run("provider_name", func(t *testing.T) {
		if ue.Provider != "azureserverless" {
			t.Errorf("Provider = %q, want azureserverless", ue.Provider)
		}
	})
	t.Run("status_code_zero", func(t *testing.T) {
		if ue.StatusCode != 0 {
			t.Errorf("StatusCode = %d, want 0 for mid-stream error", ue.StatusCode)
		}
	})
	t.Run("body_contains_message", func(t *testing.T) {
		if !strings.Contains(ue.Body, "quota exceeded") {
			t.Errorf("Body = %q, want substring 'quota exceeded'", ue.Body)
		}
	})
}

func TestCompletionStream_ContextCancellation(t *testing.T) {
	srv := fakeSSEServer(t, []string{
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`,
	}, nil)
	defer srv.Close()

	p := newAPIKeyProvider(t, srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "m",
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

	temp := 0.5
	topP := 0.9
	p := newAPIKeyProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:       "llama",
		Messages:    []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		Temperature: &temp,
		TopP:        &topP,
		Stop:        []string{"END"},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	body := <-captured

	t.Run("stream_true", func(t *testing.T) {
		if !strings.Contains(string(body), `"stream":true`) {
			t.Errorf("body missing stream:true: %s", body)
		}
	})
	t.Run("include_usage_true", func(t *testing.T) {
		if !strings.Contains(string(body), `"include_usage":true`) {
			t.Errorf("body missing include_usage:true: %s", body)
		}
	})
	t.Run("model_present", func(t *testing.T) {
		if !strings.Contains(string(body), `"model":"llama"`) {
			t.Errorf("body missing model: %s", body)
		}
	})
	t.Run("temperature_forwarded", func(t *testing.T) {
		if !strings.Contains(string(body), `"temperature":0.5`) {
			t.Errorf("body missing temperature: %s", body)
		}
	})
	t.Run("top_p_forwarded", func(t *testing.T) {
		if !strings.Contains(string(body), `"top_p":0.9`) {
			t.Errorf("body missing top_p: %s", body)
		}
	})
	t.Run("stop_forwarded", func(t *testing.T) {
		if !strings.Contains(string(body), `"stop":["END"]`) {
			t.Errorf("body missing stop: %s", body)
		}
	})
}

func TestCompletionStream_RawPassthroughPreservesUnmodeledFields(t *testing.T) {
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
	raw := []byte(`{"model":"llama","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"x"}}],"response_format":{"type":"json_object"},"logprobs":true}`)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{Raw: raw})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	body := <-captured

	t.Run("tools_preserved", func(t *testing.T) {
		if !strings.Contains(string(body), `"tools":[`) {
			t.Errorf("tools dropped: %s", body)
		}
	})
	t.Run("response_format_preserved", func(t *testing.T) {
		if !strings.Contains(string(body), `"response_format":{"type":"json_object"}`) {
			t.Errorf("response_format dropped: %s", body)
		}
	})
	t.Run("logprobs_preserved", func(t *testing.T) {
		if !strings.Contains(string(body), `"logprobs":true`) {
			t.Errorf("logprobs dropped: %s", body)
		}
	})
	t.Run("stream_forced", func(t *testing.T) {
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

func TestCompletionStream_RawModelOverlay(t *testing.T) {
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
	raw := []byte(`{"model":"original","messages":[{"role":"user","content":"hi"}]}`)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model: "override",
		Raw:   raw,
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	body := <-captured
	if !strings.Contains(string(body), `"model":"override"`) {
		t.Errorf("expected model overridden, got: %s", body)
	}
}

func TestCompletionStream_RawModelPreservedWhenReqModelEmpty(t *testing.T) {
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
	raw := []byte(`{"model":"original","messages":[{"role":"user","content":"hi"}]}`)
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{Raw: raw})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	body := <-captured
	if !strings.Contains(string(body), `"model":"original"`) {
		t.Errorf("expected raw model preserved, got: %s", body)
	}
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

// ---------------------------------------------------------------------
// Upstream HTTP errors.
// ---------------------------------------------------------------------

func TestCompletionStream_UpstreamErrors(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
	}{
		{"bad_request_400", http.StatusBadRequest, `{"error":"bad"}`},
		{"unauthorized_401", http.StatusUnauthorized, `{"error":"unauthorized"}`},
		{"forbidden_403", http.StatusForbidden, `{"error":"forbidden"}`},
		{"not_found_404", http.StatusNotFound, `{"error":"no deployment"}`},
		{"rate_limit_429", http.StatusTooManyRequests, `{"error":"slow down"}`},
		{"server_error_500", http.StatusInternalServerError, `oops`},
		{"bad_gateway_502", http.StatusBadGateway, `upstream down`},
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
				Model:    "m",
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
			if ue.Provider != "azureserverless" {
				t.Errorf("Provider = %q, want azureserverless", ue.Provider)
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
		Model:    "m",
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

func TestCompletionStream_NetworkErrorNotWrappedAsUpstream(t *testing.T) {
	// Address that should immediately fail to connect.
	p := newAPIKeyProvider(t, "http://127.0.0.1:1")
	_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "m",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatalf("expected transport error, got nil")
	}
	var ue *llmrouter.ErrUpstream
	if errors.As(err, &ue) {
		t.Errorf("network error wrapped as *ErrUpstream: %v", err)
	}
}

// ---------------------------------------------------------------------
// Misc.
// ---------------------------------------------------------------------

func TestProvider_Name(t *testing.T) {
	p := newAPIKeyProvider(t, "https://example.invalid")
	if p.Name() != "azureserverless" {
		t.Errorf("Name = %q, want azureserverless", p.Name())
	}
}
