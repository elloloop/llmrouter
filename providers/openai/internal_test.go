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

// ---------------------------------------------------------------------------
// errorPayload — recognise in-band SSE error envelopes
// ---------------------------------------------------------------------------

func TestErrorPayload_RecognisedShapes(t *testing.T) {
	cases := []struct {
		name        string
		payload     string
		wantIsErr   bool
		wantContain string
	}{
		{
			"canonical-object-with-message",
			`{"error":{"message":"context overflow","type":"context_length_exceeded","code":"x"}}`,
			true,
			"context overflow",
		},
		{
			"canonical-object-type-prefix",
			`{"error":{"message":"too long","type":"context_length_exceeded"}}`,
			true,
			"context_length_exceeded: too long",
		},
		{
			"canonical-object-type-only",
			`{"error":{"type":"overloaded_error"}}`,
			true,
			"overloaded_error",
		},
		{
			"canonical-object-code-only",
			`{"error":{"code":"rate_limited"}}`,
			true,
			"rate_limited",
		},
		{
			"plain-string-error",
			`{"error":"upstream rate limited"}`,
			true,
			"upstream rate limited",
		},
		{
			"non-error-envelope-skipped",
			`{"random":"data"}`,
			false,
			"",
		},
		{
			"choices-shaped-skipped",
			`{"choices":[]}`,
			false,
			"",
		},
		{
			"non-object-skipped",
			`[1,2,3]`,
			false,
			"",
		},
		{
			"empty-skipped",
			``,
			false,
			"",
		},
		{
			"garbage-skipped",
			`not json`,
			false,
			"",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			msg, isErr := errorPayload(tc.payload)
			if isErr != tc.wantIsErr {
				t.Fatalf("isErr = %v, want %v (msg=%q)", isErr, tc.wantIsErr, msg)
			}
			if tc.wantIsErr && !strings.Contains(msg, tc.wantContain) {
				t.Errorf("msg = %q, want contains %q", msg, tc.wantContain)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// SSE — mid-stream error envelope terminates with ErrUpstream
// ---------------------------------------------------------------------------

func TestSSE_MidStreamErrorTerminatesStream(t *testing.T) {
	validChunk := `data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`
	cases := []struct {
		name         string
		errorLine    string
		wantBody     string
		wantProvider string
	}{
		{
			"canonical-error-object",
			`data: {"error":{"message":"context overflow","type":"context_length_exceeded"}}`,
			"context overflow",
			"openai",
		},
		{
			"plain-string-error",
			`data: {"error":"upstream rate limited"}`,
			"upstream rate limited",
			"openai",
		},
		{
			"type-only-error",
			`data: {"error":{"type":"overloaded_error"}}`,
			"overloaded_error",
			"openai",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Build body: one valid chunk, then error envelope, then a
			// trailing chunk the consumer should NEVER see because the
			// stream must have terminated already.
			body := strings.Join([]string{
				validChunk,
				``,
				tc.errorLine,
				``,
				`data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"AFTER"}}]}`,
				``,
				`data: [DONE]`,
				``,
			}, "\n")
			srv := newFakeServer(t, body)
			defer srv.Close()
			p := newConfiguredProvider(t, srv.URL)
			got, err := runStreamCollect(t, p)
			if err == nil {
				t.Fatalf("expected error, got nil (chunks=%d)", len(got))
			}
			var ue *llmrouter.ErrUpstream
			if !errors.As(err, &ue) {
				t.Fatalf("err = %T %v, want *ErrUpstream", err, err)
			}
			if ue.Provider != tc.wantProvider {
				t.Errorf("Provider = %q, want %q", ue.Provider, tc.wantProvider)
			}
			if ue.StatusCode != 0 {
				t.Errorf("StatusCode = %d, want 0 (mid-stream marker)", ue.StatusCode)
			}
			if !strings.Contains(ue.Body, tc.wantBody) {
				t.Errorf("Body = %q, want contains %q", ue.Body, tc.wantBody)
			}
			// At least the first valid chunk must have been delivered.
			if len(got) < 1 {
				t.Fatalf("expected >=1 chunk delivered before error, got %d", len(got))
			}
			if got[0].Choices[0].Delta.Content != "hi" {
				t.Errorf("first chunk content = %q, want 'hi'", got[0].Choices[0].Delta.Content)
			}
			// The post-error chunk must NOT have been delivered.
			for _, c := range got {
				for _, ch := range c.Choices {
					if ch.Delta.Content == "AFTER" {
						t.Errorf("post-error chunk leaked through: %+v", c)
					}
				}
			}
		})
	}
}

func TestSSE_MidStreamErrorMessageMentionsMidStream(t *testing.T) {
	body := "data: {\"error\":{\"message\":\"boom\",\"type\":\"overloaded_error\"}}\n\n"
	srv := newFakeServer(t, body)
	defer srv.Close()
	p := newConfiguredProvider(t, srv.URL)
	_, err := runStreamCollect(t, p)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "mid-stream") {
		t.Errorf("error %q should mention mid-stream", err.Error())
	}
}

func TestSSE_NonErrorMalformedJSONStillSkipped(t *testing.T) {
	// Regression: {"random":"data"} is not an error envelope and must
	// continue to be silently dropped (existing behaviour).
	body := strings.Join([]string{
		`data: {"random":"data"}`,
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
		t.Fatalf("expected clean stream, got err = %v", err)
	}
	// The random-data payload is parseable as a chat completion (no
	// choices, no usage), so the decoder accepts it and emits a chunk.
	// What matters here is that no ErrUpstream surfaces.
	foundHi := false
	for _, c := range got {
		for _, ch := range c.Choices {
			if ch.Delta.Content == "hi" {
				foundHi = true
			}
		}
	}
	if !foundHi {
		t.Errorf("expected 'hi' chunk, got %+v", got)
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

// ---------------------------------------------------------------------------
// decodeChunk — tool_calls in delta
// ---------------------------------------------------------------------------

func TestDecodeChunk_ToolCalls_FirstFragmentWithNameAndID(t *testing.T) {
	payload := `{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}`
	c, ok := decodeChunk(payload)
	if !ok {
		t.Fatal("ok=false")
	}
	tcs := c.Choices[0].Delta.ToolCalls
	if len(tcs) != 1 {
		t.Fatalf("len = %d", len(tcs))
	}
	if tcs[0].Index != 0 {
		t.Errorf("Index = %d", tcs[0].Index)
	}
	if tcs[0].ID != "call_1" {
		t.Errorf("ID = %q", tcs[0].ID)
	}
	if tcs[0].Type != "function" {
		t.Errorf("Type = %q", tcs[0].Type)
	}
	if tcs[0].Function == nil || tcs[0].Function.Name != "get_weather" {
		t.Errorf("Function = %+v", tcs[0].Function)
	}
}

func TestDecodeChunk_ToolCalls_ArgumentFragment(t *testing.T) {
	payload := `{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}`
	c, ok := decodeChunk(payload)
	if !ok {
		t.Fatal("ok=false")
	}
	tcs := c.Choices[0].Delta.ToolCalls
	if len(tcs) != 1 {
		t.Fatalf("len = %d", len(tcs))
	}
	if tcs[0].Function == nil || tcs[0].Function.Arguments != `{"city":` {
		t.Errorf("Arguments = %+v", tcs[0].Function)
	}
}

func TestDecodeChunk_ToolCalls_NoToolCallsAbsent(t *testing.T) {
	payload := `{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}`
	c, ok := decodeChunk(payload)
	if !ok {
		t.Fatal("ok=false")
	}
	if c.Choices[0].Delta.ToolCalls != nil {
		t.Errorf("ToolCalls should be nil, got %+v", c.Choices[0].Delta.ToolCalls)
	}
}

func TestDecodeChunk_ToolCalls_MultipleToolsByIndex(t *testing.T) {
	payload := `{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f1"}},{"index":1,"id":"b","type":"function","function":{"name":"f2"}}]}}]}`
	c, ok := decodeChunk(payload)
	if !ok {
		t.Fatal("ok=false")
	}
	tcs := c.Choices[0].Delta.ToolCalls
	if len(tcs) != 2 {
		t.Fatalf("len = %d", len(tcs))
	}
	if tcs[0].Index != 0 || tcs[0].Function.Name != "f1" {
		t.Errorf("tcs[0] = %+v", tcs[0])
	}
	if tcs[1].Index != 1 || tcs[1].Function.Name != "f2" {
		t.Errorf("tcs[1] = %+v", tcs[1])
	}
}

func TestDecodeChunk_ToolCalls_ArgumentsConcatenateAcrossChunks(t *testing.T) {
	// Simulate a stream of fragments — each chunk decoded independently;
	// callers concatenate by Index. Verifies that decodeChunk preserves
	// each fragment as-is rather than mutating/joining it.
	fragments := []string{
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"add","arguments":""}}]}}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":"}}]}}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1"}}]}}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":",\"b\":2}"}}]}}]}`,
	}
	var concatenated string
	var name string
	var id string
	for _, p := range fragments {
		c, ok := decodeChunk(p)
		if !ok {
			t.Fatalf("decodeChunk failed for %s", p)
		}
		tcs := c.Choices[0].Delta.ToolCalls
		if len(tcs) != 1 || tcs[0].Index != 0 {
			t.Fatalf("unexpected tool_calls shape: %+v", tcs)
		}
		if tcs[0].ID != "" {
			id = tcs[0].ID
		}
		if tcs[0].Function != nil {
			if tcs[0].Function.Name != "" {
				name = tcs[0].Function.Name
			}
			concatenated += tcs[0].Function.Arguments
		}
	}
	if id != "c1" {
		t.Errorf("id = %q", id)
	}
	if name != "add" {
		t.Errorf("name = %q", name)
	}
	if concatenated != `{"a":1,"b":2}` {
		t.Errorf("concatenated = %q", concatenated)
	}
}

func TestDecodeChunk_ToolCalls_IndexCorrelationAcrossChunks(t *testing.T) {
	// Two tool calls interleaved across chunks; check Index identifies them.
	frag1 := `{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"f0"}}]}}]}`
	frag2 := `{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"b","type":"function","function":{"name":"f1"}}]}}]}`
	frag3 := `{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"args0"}}]}}]}`
	frag4 := `{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"args1"}}]}}]}`

	args := map[int]string{}
	names := map[int]string{}
	for _, p := range []string{frag1, frag2, frag3, frag4} {
		c, ok := decodeChunk(p)
		if !ok {
			t.Fatalf("decodeChunk failed")
		}
		for _, tc := range c.Choices[0].Delta.ToolCalls {
			if tc.Function != nil {
				if tc.Function.Name != "" {
					names[tc.Index] = tc.Function.Name
				}
				args[tc.Index] += tc.Function.Arguments
			}
		}
	}
	if names[0] != "f0" || names[1] != "f1" {
		t.Errorf("names = %v", names)
	}
	if args[0] != "args0" || args[1] != "args1" {
		t.Errorf("args = %v", args)
	}
}

func TestDecodeChunk_ToolCalls_NilFunctionTolerated(t *testing.T) {
	// Some upstreams emit `tool_calls` entries with no function block on
	// terminal/scaffolding chunks. Should decode without panicking.
	payload := `{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0}]}}]}`
	c, ok := decodeChunk(payload)
	if !ok {
		t.Fatal("ok=false")
	}
	tcs := c.Choices[0].Delta.ToolCalls
	if len(tcs) != 1 {
		t.Fatalf("len = %d", len(tcs))
	}
	if tcs[0].Function != nil {
		t.Errorf("Function should be nil: %+v", tcs[0].Function)
	}
}

func TestDecodeChunk_ToolCalls_FinishReasonToolCalls(t *testing.T) {
	payload := `{"id":"x","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`
	c, ok := decodeChunk(payload)
	if !ok {
		t.Fatal("ok=false")
	}
	if c.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q", c.Choices[0].FinishReason)
	}
}

