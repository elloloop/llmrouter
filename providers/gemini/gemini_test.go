package gemini

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

// geminiSSEFixture is a complete :streamGenerateContent SSE response
// covering a role primer, two text deltas, and a terminal finish frame
// with usageMetadata.
const geminiSSEFixture = `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]}}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":11,"candidatesTokenCount":5,"totalTokenCount":16}}

`

// newFixtureServer returns an httptest server that asserts the standard
// gemini auth/path contract and replies with the supplied SSE body.
func newFixtureServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get(apiKeyHeader); got == "" {
			t.Errorf("missing %s header", apiKeyHeader)
		}
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("unexpected Authorization header: %q", got)
		}
		if !strings.Contains(r.URL.Path, ":streamGenerateContent") {
			t.Errorf("path %q missing :streamGenerateContent", r.URL.Path)
		}
		if got := r.URL.Query().Get("alt"); got != "sse" {
			t.Errorf("alt = %q, want sse", got)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, body)
	}))
}

// recordingServer captures the last received request body/url for
// downstream assertions.
type recordingServer struct {
	srv     *httptest.Server
	body    []byte
	urlPath string
	headers http.Header
}

func newRecordingServer(t *testing.T, sseBody string) *recordingServer {
	t.Helper()
	rs := &recordingServer{}
	rs.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		rs.body = b
		rs.urlPath = r.URL.RequestURI()
		rs.headers = r.Header.Clone()
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, sseBody)
	}))
	return rs
}

func (r *recordingServer) close() { r.srv.Close() }

// runSimpleStream drives one CompletionStream call against srv and
// returns the collected chunks.
func runSimpleStream(t *testing.T, srvURL string, req llmrouter.ChatRequest) ([]llmrouter.Chunk, error) {
	t.Helper()
	p, err := New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srvURL),
	)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.CompletionStream(ctx, req)
	if err != nil {
		return nil, err
	}
	var chunks []llmrouter.Chunk
	for c := range stream.Chunks() {
		chunks = append(chunks, c)
	}
	return chunks, stream.Err()
}

// ---------------------------------------------------------------------
// New / config
// ---------------------------------------------------------------------

func TestNew(t *testing.T) {
	t.Run("requires api key", func(t *testing.T) {
		_, err := New()
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !errors.Is(err, llmrouter.ErrInvalidConfig) {
			t.Errorf("err = %v, want ErrInvalidConfig", err)
		}
	})

	t.Run("rejects empty api key", func(t *testing.T) {
		_, err := New(llmrouter.WithAPIKey(""))
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})

	t.Run("defaults base url", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if p.cfg.BaseURL != defaultBaseURL {
			t.Errorf("BaseURL = %q, want %q", p.cfg.BaseURL, defaultBaseURL)
		}
	})

	t.Run("honours WithBaseURL override", func(t *testing.T) {
		p, err := New(
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL("https://example.com/v1beta"),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if p.cfg.BaseURL != "https://example.com/v1beta" {
			t.Errorf("BaseURL = %q", p.cfg.BaseURL)
		}
	})

	t.Run("name is gemini", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if p.Name() != "gemini" {
			t.Errorf("Name() = %q", p.Name())
		}
	})

	t.Run("rejects nil http client", func(t *testing.T) {
		_, err := New(llmrouter.WithAPIKey("k"), llmrouter.WithHTTPClient(nil))
		if err == nil {
			t.Fatal("expected error")
		}
	})
}

// ---------------------------------------------------------------------
// Wire contract: headers, URL, method
// ---------------------------------------------------------------------

func TestRequestWire(t *testing.T) {
	t.Run("sends x-goog-api-key header", func(t *testing.T) {
		rs := newRecordingServer(t, geminiSSEFixture)
		defer rs.close()
		_, err := runSimpleStream(t, rs.srv.URL, llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := rs.headers.Get(apiKeyHeader); got != "test-key" {
			t.Errorf("%s = %q, want test-key", apiKeyHeader, got)
		}
	})

	t.Run("does not send Authorization header", func(t *testing.T) {
		rs := newRecordingServer(t, geminiSSEFixture)
		defer rs.close()
		_, err := runSimpleStream(t, rs.srv.URL, llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := rs.headers.Get("Authorization"); got != "" {
			t.Errorf("Authorization sent: %q", got)
		}
	})

	t.Run("url path includes model and streamGenerateContent", func(t *testing.T) {
		rs := newRecordingServer(t, geminiSSEFixture)
		defer rs.close()
		_, err := runSimpleStream(t, rs.srv.URL, llmrouter.ChatRequest{
			Model:    "gemini-1.5-pro",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		if !strings.Contains(rs.urlPath, "/models/gemini-1.5-pro:streamGenerateContent") {
			t.Errorf("url = %q, missing /models/<model>:streamGenerateContent", rs.urlPath)
		}
	})

	t.Run("url has alt=sse query", func(t *testing.T) {
		rs := newRecordingServer(t, geminiSSEFixture)
		defer rs.close()
		_, err := runSimpleStream(t, rs.srv.URL, llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		if !strings.Contains(rs.urlPath, "alt=sse") {
			t.Errorf("url = %q, missing alt=sse", rs.urlPath)
		}
	})

	t.Run("content-type is application/json", func(t *testing.T) {
		rs := newRecordingServer(t, geminiSSEFixture)
		defer rs.close()
		_, err := runSimpleStream(t, rs.srv.URL, llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := rs.headers.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q", got)
		}
	})
}

// ---------------------------------------------------------------------
// SSE translation
// ---------------------------------------------------------------------

func TestCompletionStream_TranslatesGeminiSSE(t *testing.T) {
	srv := newFixtureServer(t, geminiSSEFixture)
	defer srv.Close()

	chunks, err := runSimpleStream(t, srv.URL, llmrouter.ChatRequest{
		Model:    "gemini-1.5-flash",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	t.Run("emits primer + two deltas + finish", func(t *testing.T) {
		if len(chunks) != 4 {
			t.Fatalf("got %d chunks, want 4: %+v", len(chunks), chunks)
		}
	})
	if len(chunks) < 4 {
		return
	}

	t.Run("primer is role=assistant with empty content", func(t *testing.T) {
		if chunks[0].Choices[0].Delta.Role != "assistant" {
			t.Errorf("primer role = %q", chunks[0].Choices[0].Delta.Role)
		}
		if chunks[0].Choices[0].Delta.Content != "" {
			t.Errorf("primer content = %q", chunks[0].Choices[0].Delta.Content)
		}
	})

	t.Run("first delta is Hello", func(t *testing.T) {
		if chunks[1].Choices[0].Delta.Content != "Hello" {
			t.Errorf("chunk[1] = %q", chunks[1].Choices[0].Delta.Content)
		}
	})

	t.Run("second delta is space-world", func(t *testing.T) {
		if chunks[2].Choices[0].Delta.Content != " world" {
			t.Errorf("chunk[2] = %q", chunks[2].Choices[0].Delta.Content)
		}
	})

	t.Run("final chunk has finish reason stop", func(t *testing.T) {
		if chunks[3].Choices[0].FinishReason != "stop" {
			t.Errorf("finish = %q", chunks[3].Choices[0].FinishReason)
		}
	})

	t.Run("final chunk carries usage", func(t *testing.T) {
		u := chunks[3].Usage
		if u == nil {
			t.Fatal("usage nil")
		}
		if u.PromptTokens != 11 || u.CompletionTokens != 5 || u.TotalTokens != 16 {
			t.Errorf("usage = %+v, want {11,5,16}", *u)
		}
	})

	t.Run("stable chatcmpl id across chunks", func(t *testing.T) {
		first := chunks[0].ID
		if !strings.HasPrefix(first, "chatcmpl-") {
			t.Errorf("id = %q", first)
		}
		for i, c := range chunks {
			if c.ID != first {
				t.Errorf("chunk[%d].ID = %q, want %q", i, c.ID, first)
			}
		}
	})

	t.Run("every chunk has Object=chat.completion.chunk", func(t *testing.T) {
		for i, c := range chunks {
			if c.Object != "chat.completion.chunk" {
				t.Errorf("chunk[%d].Object = %q", i, c.Object)
			}
		}
	})

	t.Run("every chunk has non-empty Raw", func(t *testing.T) {
		for i, c := range chunks {
			if len(c.Raw) == 0 {
				t.Errorf("chunk[%d].Raw empty", i)
			}
		}
	})

	t.Run("every chunk carries model", func(t *testing.T) {
		for i, c := range chunks {
			if c.Model != "gemini-1.5-flash" {
				t.Errorf("chunk[%d].Model = %q", i, c.Model)
			}
		}
	})
}

// ---------------------------------------------------------------------
// finishReason mapping
// ---------------------------------------------------------------------

func TestMapFinishReason(t *testing.T) {
	cases := map[string]string{
		"STOP":       "stop",
		"MAX_TOKENS": "length",
		"SAFETY":     "content_filter",
		"RECITATION": "content_filter",
		"OTHER":      "stop",
		"":           "stop",
	}
	for in, want := range cases {
		t.Run("finishReason="+in, func(t *testing.T) {
			got := mapFinishReason(in)
			if got != want {
				t.Errorf("mapFinishReason(%q) = %q, want %q", in, got, want)
			}
		})
	}
}

func TestFinishReason_IntegrationMaxTokens(t *testing.T) {
	body := `data: {"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":4,"totalTokenCount":7}}

`
	srv := newFixtureServer(t, body)
	defer srv.Close()
	chunks, err := runSimpleStream(t, srv.URL, llmrouter.ChatRequest{
		Model:    "gemini-1.5-flash",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if len(chunks) == 0 || chunks[len(chunks)-1].Choices[0].FinishReason != "length" {
		t.Errorf("finish = %v", chunks)
	}
}

func TestFinishReason_IntegrationSafety(t *testing.T) {
	body := `data: {"candidates":[{"content":{"role":"model","parts":[{"text":""}]},"finishReason":"SAFETY"}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":0,"totalTokenCount":2}}

`
	srv := newFixtureServer(t, body)
	defer srv.Close()
	chunks, err := runSimpleStream(t, srv.URL, llmrouter.ChatRequest{
		Model:    "gemini-1.5-flash",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if len(chunks) == 0 || chunks[len(chunks)-1].Choices[0].FinishReason != "content_filter" {
		t.Errorf("finish = %v", chunks)
	}
}

// ---------------------------------------------------------------------
// Body translation
// ---------------------------------------------------------------------

func TestBuildGeminiBody(t *testing.T) {
	t.Run("defaults max output tokens to 4096", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"maxOutputTokens":4096`) {
			t.Errorf("missing default maxOutputTokens=4096: %s", body)
		}
	})

	t.Run("honours typed MaxTokens", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model:     "gemini-1.5-flash",
			MaxTokens: 256,
			Messages:  []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"maxOutputTokens":256`) {
			t.Errorf("missing maxOutputTokens=256: %s", body)
		}
	})

	t.Run("system messages lift to systemInstruction", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model: "gemini-1.5-flash",
			Messages: []llmrouter.Message{
				llmrouter.TextMessage("system", "be nice"),
				llmrouter.TextMessage("user", "hi"),
			},
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"systemInstruction"`) {
			t.Errorf("missing systemInstruction: %s", body)
		}
		if !strings.Contains(string(body), `"text":"be nice"`) {
			t.Errorf("system text missing: %s", body)
		}
	})

	t.Run("multiple system messages join with double newline", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model: "gemini-1.5-flash",
			Messages: []llmrouter.Message{
				llmrouter.TextMessage("system", "rule 1"),
				llmrouter.TextMessage("system", "rule 2"),
				llmrouter.TextMessage("user", "hi"),
			},
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `rule 1\n\nrule 2`) {
			t.Errorf("system not joined: %s", body)
		}
	})

	t.Run("system role does not leak into contents", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model: "gemini-1.5-flash",
			Messages: []llmrouter.Message{
				llmrouter.TextMessage("system", "be nice"),
				llmrouter.TextMessage("user", "hi"),
			},
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		// Decode and inspect contents roles to be precise.
		var parsed struct {
			Contents []struct {
				Role string `json:"role"`
			} `json:"contents"`
		}
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		for _, c := range parsed.Contents {
			if c.Role == "system" {
				t.Errorf("system role leaked into contents: %s", body)
			}
		}
	})

	t.Run("assistant role becomes model", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model: "gemini-1.5-flash",
			Messages: []llmrouter.Message{
				llmrouter.TextMessage("user", "hi"),
				llmrouter.TextMessage("assistant", "hello"),
				llmrouter.TextMessage("user", "again"),
			},
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		var parsed struct {
			Contents []struct {
				Role string `json:"role"`
			} `json:"contents"`
		}
		if err := json.Unmarshal(body, &parsed); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if len(parsed.Contents) != 3 {
			t.Fatalf("contents = %d, want 3", len(parsed.Contents))
		}
		want := []string{"user", "model", "user"}
		for i, c := range parsed.Contents {
			if c.Role != want[i] {
				t.Errorf("contents[%d].Role = %q, want %q", i, c.Role, want[i])
			}
		}
	})

	t.Run("typed temperature honoured", func(t *testing.T) {
		temp := 0.42
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model:       "gemini-1.5-flash",
			Temperature: &temp,
			Messages:    []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"temperature":0.42`) {
			t.Errorf("temperature missing: %s", body)
		}
	})

	t.Run("typed topP honoured", func(t *testing.T) {
		tp := 0.9
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			TopP:     &tp,
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"topP":0.9`) {
			t.Errorf("topP missing: %s", body)
		}
	})

	t.Run("typed stop becomes stopSequences", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			Stop:     []string{"###", "END"},
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"stopSequences":["###","END"]`) {
			t.Errorf("stopSequences missing: %s", body)
		}
	})

	t.Run("Raw temperature lifted into generationConfig", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
			Raw:      json.RawMessage(`{"temperature":0.7}`),
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"temperature":0.7`) {
			t.Errorf("Raw temperature not lifted: %s", body)
		}
	})

	t.Run("Raw top_p becomes topP", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
			Raw:      json.RawMessage(`{"top_p":0.8}`),
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"topP":0.8`) {
			t.Errorf("Raw top_p not lifted: %s", body)
		}
	})

	t.Run("Raw top_k becomes topK", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
			Raw:      json.RawMessage(`{"top_k":40}`),
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"topK":40`) {
			t.Errorf("Raw top_k not lifted: %s", body)
		}
	})

	t.Run("Raw stop becomes stopSequences", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
			Raw:      json.RawMessage(`{"stop":["FOO"]}`),
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"stopSequences":["FOO"]`) {
			t.Errorf("Raw stop not lifted as stopSequences: %s", body)
		}
	})

	t.Run("Raw takes precedence over typed fields", func(t *testing.T) {
		temp := 0.1
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model:       "gemini-1.5-flash",
			Temperature: &temp,
			Messages:    []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
			Raw:         json.RawMessage(`{"temperature":0.9}`),
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"temperature":0.9`) {
			t.Errorf("Raw should win: %s", body)
		}
		if strings.Contains(string(body), `"temperature":0.1`) {
			t.Errorf("typed leaked: %s", body)
		}
	})

	t.Run("malformed Raw is tolerated", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
			Raw:      json.RawMessage(`{not json`),
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"contents"`) {
			t.Errorf("body missing contents: %s", body)
		}
	})

	t.Run("generationConfig always present", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"generationConfig"`) {
			t.Errorf("generationConfig missing: %s", body)
		}
	})

	t.Run("user message text preserved in parts", func(t *testing.T) {
		body, err := buildGeminiBody(llmrouter.ChatRequest{
			Model:    "gemini-1.5-flash",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hello there")},
		})
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		if !strings.Contains(string(body), `"text":"hello there"`) {
			t.Errorf("user text missing: %s", body)
		}
	})
}

// ---------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------

func TestErrorMapping(t *testing.T) {
	cases := []struct {
		name string
		code int
	}{
		{"400 bad request", http.StatusBadRequest},
		{"401 unauthorized", http.StatusUnauthorized},
		{"429 too many", http.StatusTooManyRequests},
		{"500 internal", http.StatusInternalServerError},
		{"503 unavailable", http.StatusServiceUnavailable},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.code)
				_, _ = io.WriteString(w, `{"error":"boom"}`)
			}))
			defer srv.Close()

			p, err := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
				Model:    "gemini-1.5-flash",
				Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
			})
			if err == nil {
				t.Fatal("expected error")
			}
			var ue *llmrouter.ErrUpstream
			if !errors.As(err, &ue) {
				t.Fatalf("err = %v, want *ErrUpstream", err)
			}
			if ue.StatusCode != tc.code {
				t.Errorf("StatusCode = %d, want %d", ue.StatusCode, tc.code)
			}
			if ue.Provider != "gemini" {
				t.Errorf("Provider = %q", ue.Provider)
			}
			if !strings.Contains(ue.Body, "boom") {
				t.Errorf("Body = %q", ue.Body)
			}
		})
	}
}

func TestErrorBodyIsCapped(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		// Write 16 KiB of body — provider should only read 8 KiB.
		_, _ = io.WriteString(w, strings.Repeat("x", 16*1024))
	}))
	defer srv.Close()

	p, err := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "gemini-1.5-flash",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	var ue *llmrouter.ErrUpstream
	if !errors.As(err, &ue) {
		t.Fatalf("err = %v", err)
	}
	if len(ue.Body) > 8*1024 {
		t.Errorf("body len = %d, want <= 8 KiB", len(ue.Body))
	}
}

// ---------------------------------------------------------------------
// Malformed SSE tolerance
// ---------------------------------------------------------------------

func TestMalformedSSEEventTolerated(t *testing.T) {
	body := `data: {not valid json

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}

`
	srv := newFixtureServer(t, body)
	defer srv.Close()

	chunks, err := runSimpleStream(t, srv.URL, llmrouter.ChatRequest{
		Model:    "gemini-1.5-flash",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("got no chunks; malformed event should be skipped, not fatal")
	}
	// We expect at least a primer + content + finish chunk from the
	// valid event.
	if chunks[len(chunks)-1].Choices[0].FinishReason != "stop" {
		t.Errorf("finish = %q", chunks[len(chunks)-1].Choices[0].FinishReason)
	}
}

func TestBlankLinesIgnored(t *testing.T) {
	body := `

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}


`
	srv := newFixtureServer(t, body)
	defer srv.Close()
	chunks, err := runSimpleStream(t, srv.URL, llmrouter.ChatRequest{
		Model:    "gemini-1.5-flash",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected chunks")
	}
}

func TestNonDataLinesIgnored(t *testing.T) {
	body := `: keepalive comment
event: ignored
data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}

`
	srv := newFixtureServer(t, body)
	defer srv.Close()
	chunks, err := runSimpleStream(t, srv.URL, llmrouter.ChatRequest{
		Model:    "gemini-1.5-flash",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected chunks despite event/comment lines")
	}
}

// ---------------------------------------------------------------------
// usageFromMetadata
// ---------------------------------------------------------------------

func TestUsageFromMetadata(t *testing.T) {
	t.Run("prefers totalTokenCount when provided", func(t *testing.T) {
		u := usageFromMetadata(&usageMetadata{
			PromptTokenCount: 5, CandidatesTokenCount: 7, TotalTokenCount: 99,
		})
		if u.TotalTokens != 99 {
			t.Errorf("total = %d, want 99", u.TotalTokens)
		}
	})

	t.Run("falls back to prompt + candidates when total is zero", func(t *testing.T) {
		u := usageFromMetadata(&usageMetadata{
			PromptTokenCount: 5, CandidatesTokenCount: 7,
		})
		if u.TotalTokens != 12 {
			t.Errorf("total = %d, want 12", u.TotalTokens)
		}
	})

	t.Run("prompt and completion mapped", func(t *testing.T) {
		u := usageFromMetadata(&usageMetadata{
			PromptTokenCount: 3, CandidatesTokenCount: 4, TotalTokenCount: 7,
		})
		if u.PromptTokens != 3 || u.CompletionTokens != 4 {
			t.Errorf("usage = %+v", *u)
		}
	})
}

// ---------------------------------------------------------------------
// geminiRole helper
// ---------------------------------------------------------------------

func TestGeminiRole(t *testing.T) {
	cases := map[string]string{
		"user":      "user",
		"assistant": "model",
		"system":    "user", // system is lifted before this is called, but the fallback should still be "user"
		"other":     "user",
	}
	for in, want := range cases {
		t.Run("role="+in, func(t *testing.T) {
			if got := geminiRole(in); got != want {
				t.Errorf("geminiRole(%q) = %q, want %q", in, got, want)
			}
		})
	}
}

// ---------------------------------------------------------------------
// Context cancellation
// ---------------------------------------------------------------------

func TestUsageOnlyTerminalFrameEmitsSyntheticFinish(t *testing.T) {
	body := `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]}}]}

data: {"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}

`
	srv := newFixtureServer(t, body)
	defer srv.Close()
	chunks, err := runSimpleStream(t, srv.URL, llmrouter.ChatRequest{
		Model:    "gemini-1.5-flash",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected chunks")
	}
	last := chunks[len(chunks)-1]
	if last.Choices[0].FinishReason != "stop" {
		t.Errorf("finish = %q, want stop", last.Choices[0].FinishReason)
	}
	if last.Usage == nil || last.Usage.PromptTokens != 3 || last.Usage.CompletionTokens != 2 {
		t.Errorf("usage = %+v", last.Usage)
	}
}

func TestHTTPClientErrorIsWrapped(t *testing.T) {
	// Point at a closed server to force a connect error.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	srv.Close()

	p, err := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "gemini-1.5-flash",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "gemini") {
		t.Errorf("err = %v, want wrapped with gemini prefix", err)
	}
}

func TestInvalidBaseURLBuildsBadRequest(t *testing.T) {
	// Base URL is technically parseable but model contains a space → http.NewRequest fails.
	p, err := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL("http://example.invalid"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "bad model\x7f",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestContextCancelStopsStream(t *testing.T) {
	// Server hangs forever after writing one chunk.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		_, _ = io.WriteString(w, `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]}}]}`+"\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		<-r.Context().Done()
	}))
	defer srv.Close()

	p, err := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "gemini-1.5-flash",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	// Read the first chunk(s), then cancel.
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()
	for range stream.Chunks() {
		// drain
	}
	// stream.Err may be nil or ctx.Err — both are valid clean shutdowns.
	_ = stream.Err()
}
