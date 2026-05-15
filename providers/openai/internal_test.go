package openai

// White-box tests for the internal helpers in the openai provider:
// buildRequestBody, decodeChunk, readUpstreamErrorBody, pumpSSE behaviors
// that are easier to exercise directly than through the public surface.

import (
	"bytes"
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

// ---------------------------------------------------------------------------
// buildRequestBody
// ---------------------------------------------------------------------------

func TestBuildRequestBody_ForcesStreamTrue(t *testing.T) {
	cases := []struct {
		name string
		req  llmrouter.ChatRequest
	}{
		{"stream-omitted", llmrouter.ChatRequest{Model: "gpt-4o-mini", Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")}}},
		{"stream-explicit-false", llmrouter.ChatRequest{Model: "gpt-4o-mini", Stream: false, Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")}}},
		{"stream-explicit-true", llmrouter.ChatRequest{Model: "gpt-4o-mini", Stream: true, Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := buildRequestBody(tc.req)
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			var m map[string]any
			if err := json.Unmarshal(b, &m); err != nil {
				t.Fatalf("invalid json: %v", err)
			}
			if v, ok := m["stream"].(bool); !ok || !v {
				t.Fatalf("stream not true: %v", m["stream"])
			}
		})
	}
}

func TestBuildRequestBody_DefaultIncludeUsage(t *testing.T) {
	b, err := buildRequestBody(llmrouter.ChatRequest{
		Model:    "gpt-4o-mini",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	so, ok := m["stream_options"].(map[string]any)
	if !ok {
		t.Fatalf("stream_options missing or wrong type: %T", m["stream_options"])
	}
	if v, ok := so["include_usage"].(bool); !ok || !v {
		t.Fatalf("include_usage not true: %v", so["include_usage"])
	}
}

func TestBuildRequestBody_PreservesCallerStreamOptions(t *testing.T) {
	raw := json.RawMessage(`{"model":"x","messages":[],"stream_options":{"include_usage":false,"custom":true}}`)
	b, err := buildRequestBody(llmrouter.ChatRequest{Model: "x", Raw: raw})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	so := m["stream_options"].(map[string]any)
	if so["include_usage"] != false {
		t.Errorf("caller include_usage overridden: %v", so["include_usage"])
	}
	if so["custom"] != true {
		t.Errorf("caller custom flag lost")
	}
}

func TestBuildRequestBody_ModelOverridesRawModel(t *testing.T) {
	raw := json.RawMessage(`{"model":"raw-model","messages":[]}`)
	b, err := buildRequestBody(llmrouter.ChatRequest{Model: "typed-model", Raw: raw})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	if m["model"] != "typed-model" {
		t.Fatalf("model = %v, want 'typed-model'", m["model"])
	}
}

func TestBuildRequestBody_EmptyModelKeepsRawModel(t *testing.T) {
	raw := json.RawMessage(`{"model":"raw-model","messages":[]}`)
	b, err := buildRequestBody(llmrouter.ChatRequest{Raw: raw})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	if m["model"] != "raw-model" {
		t.Fatalf("model = %v", m["model"])
	}
}

func TestBuildRequestBody_RawPreservesUnmodelledFields(t *testing.T) {
	raw := json.RawMessage(`{"model":"x","messages":[],"tools":[{"name":"x"}],"response_format":{"type":"json_object"}}`)
	b, err := buildRequestBody(llmrouter.ChatRequest{Model: "x", Raw: raw})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(string(b), `"tools"`) {
		t.Errorf("tools dropped: %s", b)
	}
	if !strings.Contains(string(b), `"response_format"`) {
		t.Errorf("response_format dropped: %s", b)
	}
}

func TestBuildRequestBody_InvalidRawReturnsError(t *testing.T) {
	cases := []json.RawMessage{
		json.RawMessage(`not json`),
		json.RawMessage(`[1,2,3]`), // not an object
	}
	for i, raw := range cases {
		t.Run(string(rune('a'+i)), func(t *testing.T) {
			_, err := buildRequestBody(llmrouter.ChatRequest{Model: "x", Raw: raw})
			if err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestBuildRequestBody_NonRawIncludesTypedFields(t *testing.T) {
	temp := 0.5
	top := 0.95
	b, err := buildRequestBody(llmrouter.ChatRequest{
		Model:       "gpt-4o",
		Messages:    []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		MaxTokens:   500,
		Temperature: &temp,
		TopP:        &top,
		Stop:        []string{"END"},
		User:        "u-1",
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	s := string(b)
	for _, want := range []string{
		`"model":"gpt-4o"`,
		`"max_tokens":500`,
		`"temperature":0.5`,
		`"top_p":0.95`,
		`"stop":["END"]`,
		`"user":"u-1"`,
		`"stream":true`,
	} {
		if !strings.Contains(s, want) {
			t.Errorf("missing %q in %s", want, s)
		}
	}
}

// ---------------------------------------------------------------------------
// decodeChunk
// ---------------------------------------------------------------------------

func TestDecodeChunk_ValidPayloads(t *testing.T) {
	cases := []struct {
		name    string
		payload string
		want    func(t *testing.T, c llmrouter.Chunk)
	}{
		{
			"role-only-delta",
			`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}`,
			func(t *testing.T, c llmrouter.Chunk) {
				if c.Choices[0].Delta.Role != "assistant" {
					t.Errorf("role")
				}
			},
		},
		{
			"content-delta",
			`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
			func(t *testing.T, c llmrouter.Chunk) {
				if c.Choices[0].Delta.Content != "hi" {
					t.Errorf("content")
				}
			},
		},
		{
			"finish-stop",
			`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			func(t *testing.T, c llmrouter.Chunk) {
				if c.Choices[0].FinishReason != "stop" {
					t.Errorf("finish")
				}
			},
		},
		{
			"with-usage",
			`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}`,
			func(t *testing.T, c llmrouter.Chunk) {
				if c.Usage == nil || c.Usage.TotalTokens != 13 {
					t.Errorf("usage = %+v", c.Usage)
				}
			},
		},
		{
			"no-choices",
			`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[]}`,
			func(t *testing.T, c llmrouter.Chunk) {
				if len(c.Choices) != 0 {
					t.Errorf("choices = %d", len(c.Choices))
				}
			},
		},
		{
			"two-choices",
			`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"a"}},{"index":1,"delta":{"content":"b"}}]}`,
			func(t *testing.T, c llmrouter.Chunk) {
				if len(c.Choices) != 2 {
					t.Fatalf("choices = %d", len(c.Choices))
				}
				if c.Choices[0].Delta.Content != "a" || c.Choices[1].Delta.Content != "b" {
					t.Errorf("contents = %q %q", c.Choices[0].Delta.Content, c.Choices[1].Delta.Content)
				}
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, ok := decodeChunk(tc.payload)
			if !ok {
				t.Fatal("decodeChunk returned ok=false")
			}
			if string(c.Raw) != tc.payload {
				t.Errorf("Raw mismatch:\n got=%s\nwant=%s", c.Raw, tc.payload)
			}
			tc.want(t, c)
		})
	}
}

func TestDecodeChunk_HeaderFieldsPreserved(t *testing.T) {
	payload := `{"id":"chatcmpl-xyz","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-mini","choices":[]}`
	c, ok := decodeChunk(payload)
	if !ok {
		t.Fatal("ok=false")
	}
	if c.ID != "chatcmpl-xyz" {
		t.Errorf("ID = %q", c.ID)
	}
	if c.Object != "chat.completion.chunk" {
		t.Errorf("Object = %q", c.Object)
	}
	if c.Created != 1700000000 {
		t.Errorf("Created = %d", c.Created)
	}
	if c.Model != "gpt-4o-mini" {
		t.Errorf("Model = %q", c.Model)
	}
}

func TestDecodeChunk_InvalidJSONReturnsFalse(t *testing.T) {
	cases := []string{
		"",
		"not json",
		`{"id":`,
		`{`,
	}
	for _, p := range cases {
		t.Run("payload-len-"+strings.Repeat("x", len(p)), func(t *testing.T) {
			_, ok := decodeChunk(p)
			if ok {
				t.Fatalf("expected ok=false for %q", p)
			}
		})
	}
}

func TestDecodeChunk_MissingOptionalFieldsTolerated(t *testing.T) {
	cases := []string{
		`{}`,
		`{"choices":[]}`,
		`{"id":"x"}`,
		`{"choices":[{"index":0}]}`,
	}
	for i, p := range cases {
		t.Run(string(rune('a'+i)), func(t *testing.T) {
			_, ok := decodeChunk(p)
			if !ok {
				t.Fatalf("expected ok=true for permissive payload %q", p)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// readUpstreamErrorBody
// ---------------------------------------------------------------------------

func TestReadUpstreamErrorBody(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"basic", `{"error":"bad"}`, `{"error":"bad"}`},
		{"trimmed", "  hello  ", "hello"},
		{"empty", "", ""},
		{"only-whitespace", "   \n", ""},
		{"truncated-at-1kb", strings.Repeat("a", 2048), strings.Repeat("a", 1024)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := readUpstreamErrorBody(strings.NewReader(tc.in))
			if got != tc.want {
				t.Fatalf("got len=%d %q, want len=%d %q", len(got), got, len(tc.want), tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// End-to-end SSE behaviors via the public CompletionStream surface.
// ---------------------------------------------------------------------------

func newFakeServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, body)
	}))
}

func newConfiguredProvider(t *testing.T, url string) *Provider {
	t.Helper()
	p, err := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(url))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func runStreamCollect(t *testing.T, p *Provider) ([]llmrouter.Chunk, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "gpt-4o-mini",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		return nil, err
	}
	var got []llmrouter.Chunk
	for c := range stream.Chunks() {
		got = append(got, c)
	}
	return got, stream.Err()
}

func TestSSE_MalformedChunksSkipped(t *testing.T) {
	body := strings.Join([]string{
		`data: not json`,
		``,
		`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	srv := newFakeServer(t, body)
	defer srv.Close()
	p := newConfiguredProvider(t, srv.URL)
	got, err := runStreamCollect(t, p)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d chunks, want 1 (malformed dropped)", len(got))
	}
	if got[0].Choices[0].Delta.Content != "hi" {
		t.Errorf("content = %q", got[0].Choices[0].Delta.Content)
	}
}

func TestSSE_CommentsAndOtherEventTypesDropped(t *testing.T) {
	body := strings.Join([]string{
		`:heartbeat`,
		``,
		`event: ping`,
		`data: {"ping":true}`,
		``,
		`id: 123`,
		``,
		`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	srv := newFakeServer(t, body)
	defer srv.Close()
	p := newConfiguredProvider(t, srv.URL)
	got, err := runStreamCollect(t, p)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	// `event: ping` lines aren't `data:`-prefixed, so they're dropped.
	// The middle `data: {"ping":true}` block IS a valid JSON object so it will pass through;
	// the test asserts that the *content-bearing* chunk arrives and finishes cleanly.
	foundContent := false
	for _, c := range got {
		for _, ch := range c.Choices {
			if ch.Delta.Content == "hi" {
				foundContent = true
			}
		}
	}
	if !foundContent {
		t.Fatalf("expected 'hi' content chunk, got %#v", got)
	}
}

func TestSSE_MultiLineDataIsJoinedWithNewline(t *testing.T) {
	// One JSON object split across two `data:` lines.
	body := "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\"," +
		"\n" +
		"data: \"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}" +
		"\n\n" +
		"data: [DONE]\n\n"
	srv := newFakeServer(t, body)
	defer srv.Close()
	p := newConfiguredProvider(t, srv.URL)
	got, err := runStreamCollect(t, p)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}
	if got[0].Choices[0].Delta.Content != "hi" {
		t.Errorf("content = %q", got[0].Choices[0].Delta.Content)
	}
}

func TestSSE_DataWithoutSpaceAccepted(t *testing.T) {
	body := "data:{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: [DONE]\n\n"
	srv := newFakeServer(t, body)
	defer srv.Close()
	p := newConfiguredProvider(t, srv.URL)
	got, err := runStreamCollect(t, p)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}
}

func TestSSE_DoneTerminatesEvenWithoutFinalChunk(t *testing.T) {
	body := "data: [DONE]\n\n"
	srv := newFakeServer(t, body)
	defer srv.Close()
	p := newConfiguredProvider(t, srv.URL)
	got, err := runStreamCollect(t, p)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("len = %d, want 0", len(got))
	}
}

func TestSSE_StreamEndsCleanlyWhenServerClosesWithoutDone(t *testing.T) {
	body := "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n"
	srv := newFakeServer(t, body)
	defer srv.Close()
	p := newConfiguredProvider(t, srv.URL)
	got, err := runStreamCollect(t, p)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("len = %d", len(got))
	}
}

func TestCompletionStream_ContextCancellationCutsStreamOff(t *testing.T) {
	// Server that streams chunks slowly, forever.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, _ := w.(http.Flusher)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for i := 0; i < 1000; i++ {
			select {
			case <-r.Context().Done():
				return
			default:
			}
			_, _ = io.WriteString(w, `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"."}}]}`+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(2 * time.Millisecond)
		}
	}))
	defer srv.Close()

	p := newConfiguredProvider(t, srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "gpt-4o-mini",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	got := 0
	for c := range stream.Chunks() {
		_ = c
		got++
		if got >= 3 {
			cancel()
		}
	}
	if err := stream.Err(); err == nil {
		t.Fatal("expected context-canceled error")
	}
}

func TestCompletionStream_DefaultBaseURL(t *testing.T) {
	p, err := New(llmrouter.WithAPIKey("k"))
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if p.cfg.BaseURL != defaultBaseURL {
		t.Fatalf("BaseURL = %q, want %q", p.cfg.BaseURL, defaultBaseURL)
	}
}

func TestCompletionStream_NetworkErrorReturned(t *testing.T) {
	// Point at a closed port.
	p, err := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL("http://127.0.0.1:1"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "x",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("expected network error")
	}
	var ue *llmrouter.ErrUpstream
	if errors.As(err, &ue) {
		t.Fatalf("network err should not be ErrUpstream: %T", err)
	}
}

func TestCompletionStream_400And500BothMapToErrUpstream(t *testing.T) {
	cases := []int{400, 401, 403, 404, 429, 500, 502, 503, 504}
	for _, code := range cases {
		t.Run(http.StatusText(code), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(code)
				_, _ = io.WriteString(w, `{"error":"x"}`)
			}))
			defer srv.Close()
			p := newConfiguredProvider(t, srv.URL)
			_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{Model: "x", Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")}})
			if err == nil {
				t.Fatal("expected error")
			}
			var ue *llmrouter.ErrUpstream
			if !errors.As(err, &ue) {
				t.Fatalf("not ErrUpstream: %T %v", err, err)
			}
			if ue.StatusCode != code {
				t.Errorf("StatusCode = %d, want %d", ue.StatusCode, code)
			}
			if ue.Provider != "openai" {
				t.Errorf("Provider = %q", ue.Provider)
			}
		})
	}
}

func TestCompletionStream_AuthHeaderUsesBearer(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	p := newConfiguredProvider(t, srv.URL)
	_, _ = runStreamCollect(t, p)
	if gotAuth != "Bearer k" {
		t.Fatalf("Authorization = %q, want 'Bearer k'", gotAuth)
	}
}

func TestCompletionStream_AcceptHeader(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("Accept")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	p := newConfiguredProvider(t, srv.URL)
	_, _ = runStreamCollect(t, p)
	if got != "text/event-stream" {
		t.Fatalf("Accept = %q", got)
	}
}

func TestCompletionStream_ContentTypeHeader(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("Content-Type")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	p := newConfiguredProvider(t, srv.URL)
	_, _ = runStreamCollect(t, p)
	if got != "application/json" {
		t.Fatalf("Content-Type = %q", got)
	}
}

func TestCompletionStream_RequestBodyShape(t *testing.T) {
	var gotBody bytes.Buffer
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(&gotBody, r.Body)
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	p := newConfiguredProvider(t, srv.URL)
	_, _ = runStreamCollect(t, p)
	var m map[string]any
	if err := json.Unmarshal(gotBody.Bytes(), &m); err != nil {
		t.Fatalf("invalid body: %v %s", err, gotBody.String())
	}
	if m["model"] != "gpt-4o-mini" {
		t.Errorf("model = %v", m["model"])
	}
	if m["stream"] != true {
		t.Errorf("stream = %v", m["stream"])
	}
	if msgs, _ := m["messages"].([]any); len(msgs) != 1 {
		t.Errorf("messages len = %d", len(msgs))
	}
}

func TestCompletionStream_PathIsChatCompletions(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.URL.Path
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	p := newConfiguredProvider(t, srv.URL)
	_, _ = runStreamCollect(t, p)
	if got != "/chat/completions" {
		t.Fatalf("Path = %q", got)
	}
}

func TestCompletionStream_RawPassthroughCarriesPath(t *testing.T) {
	var gotBody bytes.Buffer
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(&gotBody, r.Body)
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	p := newConfiguredProvider(t, srv.URL)
	ctx := context.Background()
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model: "gpt-4o",
		Raw:   json.RawMessage(`{"model":"old","messages":[{"role":"user","content":"raw"}],"tools":[{"name":"t"}]}`),
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	for range stream.Chunks() {
	}
	if !strings.Contains(gotBody.String(), `"tools"`) {
		t.Errorf("tools dropped")
	}
	if !strings.Contains(gotBody.String(), `"model":"gpt-4o"`) {
		t.Errorf("model rewrite missed")
	}
}

func TestCompletionStream_TrailingSlashOnBaseURLNormalized(t *testing.T) {
	srv := newFakeServer(t, "data: [DONE]\n\n")
	defer srv.Close()
	p, err := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL+"////"))
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	_, err = runStreamCollect(t, p)
	if err != nil {
		t.Fatalf("expected stream OK, got %v", err)
	}
}

func TestNew_ErrInvalidConfigOnMissingKey(t *testing.T) {
	_, err := New()
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Fatalf("err = %v, want wraps ErrInvalidConfig", err)
	}
}

func TestNew_PropagatesOptionError(t *testing.T) {
	_, err := New(llmrouter.WithTimeout(-1))
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestProvider_NameIsStable(t *testing.T) {
	p, _ := New(llmrouter.WithAPIKey("k"))
	for i := 0; i < 5; i++ {
		if p.Name() != "openai" {
			t.Fatalf("Name drift: %q", p.Name())
		}
	}
}
