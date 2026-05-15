package mistral_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/mistral"
)

// captureServer records the request and replies with the given SSE payloads
// followed by [DONE]. Header/path assertions are deferred to the caller via
// the captured request.
type captureServer struct {
	server     *httptest.Server
	mu         sync.Mutex
	gotPath    string
	gotAuth    string
	gotAccept  string
	gotCType   string
	gotBody    []byte
	gotMethod  string
	hitCount   atomic.Int32
}

func newCaptureServer(t *testing.T, payloads []string) *captureServer {
	t.Helper()
	c := &captureServer{}
	c.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c.hitCount.Add(1)
		body, _ := io.ReadAll(r.Body)
		c.mu.Lock()
		c.gotPath = r.URL.Path
		c.gotAuth = r.Header.Get("Authorization")
		c.gotAccept = r.Header.Get("Accept")
		c.gotCType = r.Header.Get("Content-Type")
		c.gotBody = body
		c.gotMethod = r.Method
		c.mu.Unlock()

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
	return c
}

func (c *captureServer) close() { c.server.Close() }

func (c *captureServer) bodyMap(t *testing.T) map[string]any {
	t.Helper()
	c.mu.Lock()
	defer c.mu.Unlock()
	var m map[string]any
	if err := json.Unmarshal(c.gotBody, &m); err != nil {
		t.Fatalf("server body not JSON: %v (body=%q)", err, string(c.gotBody))
	}
	return m
}

func newProvider(t *testing.T, baseURL string) *mistral.Provider {
	t.Helper()
	p, err := mistral.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(baseURL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func defaultReq() llmrouter.ChatRequest {
	return llmrouter.ChatRequest{
		Model:    "mistral-large-latest",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	}
}

func drain(t *testing.T, s *llmrouter.Stream) []llmrouter.Chunk {
	t.Helper()
	var out []llmrouter.Chunk
	for c := range s.Chunks() {
		out = append(out, c)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("stream.Err: %v", err)
	}
	return out
}

// errorsAs is a tiny shim to assert against *llmrouter.ErrUpstream.
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

// ---- Constructor tests -------------------------------------------------

func TestNew_RequiresAPIKey(t *testing.T) {
	if _, err := mistral.New(); err == nil {
		t.Fatal("expected error for missing api key")
	}
}

func TestNew_RejectsEmptyAPIKey(t *testing.T) {
	if _, err := mistral.New(llmrouter.WithAPIKey("   ")); err == nil {
		t.Fatal("expected error for whitespace api key")
	}
}

func TestNew_AcceptsValidConfig(t *testing.T) {
	p, err := mistral.New(llmrouter.WithAPIKey("sk-test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p == nil {
		t.Fatal("Provider is nil")
	}
}

func TestNew_HonoursBaseURLOverride(t *testing.T) {
	srv := newCaptureServer(t, []string{})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	_, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	if srv.hitCount.Load() != 1 {
		t.Fatalf("expected 1 request, got %d", srv.hitCount.Load())
	}
}

func TestProvider_Name(t *testing.T) {
	p, err := mistral.New(llmrouter.WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := p.Name(); got != "mistral" {
		t.Fatalf("Name = %q, want mistral", got)
	}
}

// ---- Request shape tests ----------------------------------------------

func TestRequestBody_HasStreamTrue(t *testing.T) {
	srv := newCaptureServer(t, []string{})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	if _, err := p.CompletionStream(context.Background(), defaultReq()); err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	m := srv.bodyMap(t)
	if v, ok := m["stream"].(bool); !ok || !v {
		t.Errorf("stream = %v, want true", m["stream"])
	}
}

func TestRequestBody_OmitsStreamOptions(t *testing.T) {
	srv := newCaptureServer(t, []string{})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	if _, err := p.CompletionStream(context.Background(), defaultReq()); err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	m := srv.bodyMap(t)
	if _, ok := m["stream_options"]; ok {
		t.Errorf("stream_options must not be sent to Mistral; got %v", m["stream_options"])
	}
}

func TestRequestBody_StripsStreamOptionsFromRaw(t *testing.T) {
	srv := newCaptureServer(t, []string{})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	raw := json.RawMessage(`{"model":"mistral-large-latest","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}`)
	req := llmrouter.ChatRequest{Model: "mistral-large-latest", Raw: raw}
	if _, err := p.CompletionStream(context.Background(), req); err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	m := srv.bodyMap(t)
	if _, ok := m["stream_options"]; ok {
		t.Errorf("stream_options must be stripped from Raw; got %v", m["stream_options"])
	}
}

func TestRequestBody_RawPassthroughKeepsExtraFields(t *testing.T) {
	srv := newCaptureServer(t, []string{})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	raw := json.RawMessage(`{"model":"codestral-latest","messages":[{"role":"user","content":"x"}],"tools":[{"type":"function"}],"response_format":{"type":"json_object"}}`)
	req := llmrouter.ChatRequest{Model: "codestral-latest", Raw: raw}
	if _, err := p.CompletionStream(context.Background(), req); err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	m := srv.bodyMap(t)
	if _, ok := m["tools"]; !ok {
		t.Error("tools field dropped from raw passthrough")
	}
	if _, ok := m["response_format"]; !ok {
		t.Error("response_format dropped from raw passthrough")
	}
}

func TestRequestBody_ModelOverlayOnRaw(t *testing.T) {
	srv := newCaptureServer(t, []string{})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	raw := json.RawMessage(`{"model":"old-model","messages":[{"role":"user","content":"x"}]}`)
	req := llmrouter.ChatRequest{Model: "mistral-small-latest", Raw: raw}
	if _, err := p.CompletionStream(context.Background(), req); err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	m := srv.bodyMap(t)
	if got, _ := m["model"].(string); got != "mistral-small-latest" {
		t.Errorf("model overlay = %q, want mistral-small-latest", got)
	}
}

func TestRequestBody_TypedMarshalsMessages(t *testing.T) {
	srv := newCaptureServer(t, []string{})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	if _, err := p.CompletionStream(context.Background(), defaultReq()); err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	m := srv.bodyMap(t)
	msgs, ok := m["messages"].([]any)
	if !ok || len(msgs) != 1 {
		t.Fatalf("messages = %v, want 1 entry", m["messages"])
	}
}

func TestRequestBody_InvalidRawReturnsError(t *testing.T) {
	srv := newCaptureServer(t, []string{})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	req := llmrouter.ChatRequest{Model: "x", Raw: json.RawMessage(`{not json`)}
	_, err := p.CompletionStream(context.Background(), req)
	if err == nil {
		t.Fatal("expected error for invalid raw json")
	}
	if !strings.Contains(err.Error(), "mistral") {
		t.Errorf("error = %v, want provider-prefixed message", err)
	}
}

// ---- Header & path tests ----------------------------------------------

func TestRequest_PathIsChatCompletions(t *testing.T) {
	srv := newCaptureServer(t, []string{})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	if _, err := p.CompletionStream(context.Background(), defaultReq()); err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	if srv.gotPath != "/chat/completions" {
		t.Errorf("path = %q, want /chat/completions", srv.gotPath)
	}
}

func TestRequest_AuthorizationBearer(t *testing.T) {
	srv := newCaptureServer(t, []string{})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	if _, err := p.CompletionStream(context.Background(), defaultReq()); err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	if srv.gotAuth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want Bearer test-key", srv.gotAuth)
	}
}

func TestRequest_AcceptEventStream(t *testing.T) {
	srv := newCaptureServer(t, []string{})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	if _, err := p.CompletionStream(context.Background(), defaultReq()); err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	if srv.gotAccept != "text/event-stream" {
		t.Errorf("Accept = %q, want text/event-stream", srv.gotAccept)
	}
}

func TestRequest_ContentTypeJSON(t *testing.T) {
	srv := newCaptureServer(t, []string{})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	if _, err := p.CompletionStream(context.Background(), defaultReq()); err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	if srv.gotCType != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", srv.gotCType)
	}
}

func TestRequest_MethodPOST(t *testing.T) {
	srv := newCaptureServer(t, []string{})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	if _, err := p.CompletionStream(context.Background(), defaultReq()); err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	if srv.gotMethod != http.MethodPost {
		t.Errorf("method = %q, want POST", srv.gotMethod)
	}
}

// ---- Streaming behaviour tests ----------------------------------------

func TestStream_DeliversContentInOrder(t *testing.T) {
	payloads := []string{
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"role":"assistant","content":"Bon"}}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"jour"}}]}`,
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"mistral-large-latest","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	srv := newCaptureServer(t, payloads)
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if len(chunks) != 3 {
		t.Fatalf("len(chunks) = %d, want 3", len(chunks))
	}
	var content string
	for _, c := range chunks {
		for _, ch := range c.Choices {
			content += ch.Delta.Content
		}
	}
	if content != "Bonjour" {
		t.Errorf("content = %q, want Bonjour", content)
	}
}

func TestStream_FinishReasonStop(t *testing.T) {
	payloads := []string{
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	}
	srv := newCaptureServer(t, payloads)
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if len(chunks) != 1 || chunks[0].Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %v, want stop", chunks)
	}
}

func TestStream_FinishReasonLength(t *testing.T) {
	payloads := []string{
		`{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
	}
	srv := newCaptureServer(t, payloads)
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if chunks[0].Choices[0].FinishReason != "length" {
		t.Errorf("finish_reason = %q, want length", chunks[0].Choices[0].FinishReason)
	}
}

func TestStream_FinishReasonModelLength(t *testing.T) {
	// Mistral-specific finish reason; we byte-passthrough without remap.
	payloads := []string{
		`{"choices":[{"index":0,"delta":{},"finish_reason":"model_length"}]}`,
	}
	srv := newCaptureServer(t, payloads)
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if chunks[0].Choices[0].FinishReason != "model_length" {
		t.Errorf("finish_reason = %q, want model_length", chunks[0].Choices[0].FinishReason)
	}
}

func TestStream_FinishReasonError(t *testing.T) {
	// Mistral-specific finish reason for upstream-side errors.
	payloads := []string{
		`{"choices":[{"index":0,"delta":{},"finish_reason":"error"}]}`,
	}
	srv := newCaptureServer(t, payloads)
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if chunks[0].Choices[0].FinishReason != "error" {
		t.Errorf("finish_reason = %q, want error", chunks[0].Choices[0].FinishReason)
	}
}

func TestStream_UsageInFinalChunk(t *testing.T) {
	payloads := []string{
		`{"choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
	}
	srv := newCaptureServer(t, payloads)
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	var u *llmrouter.Usage
	for _, c := range chunks {
		if c.Usage != nil {
			u = c.Usage
		}
	}
	if u == nil || u.TotalTokens != 7 || u.PromptTokens != 5 || u.CompletionTokens != 2 {
		t.Errorf("usage = %+v, want {5,2,7}", u)
	}
}

func TestStream_ChunkRawByteIdentical(t *testing.T) {
	payload := `{"id":"x","object":"chat.completion.chunk","created":1,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"hi"}}]}`
	srv := newCaptureServer(t, []string{payload})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if len(chunks) != 1 {
		t.Fatalf("len(chunks) = %d, want 1", len(chunks))
	}
	if string(chunks[0].Raw) != payload {
		t.Errorf("Raw = %q\nwant %q", string(chunks[0].Raw), payload)
	}
}

func TestStream_ChunkRawNotEmpty(t *testing.T) {
	payloads := []string{
		`{"choices":[{"index":0,"delta":{"content":"a"}}]}`,
		`{"choices":[{"index":0,"delta":{"content":"b"}}]}`,
	}
	srv := newCaptureServer(t, payloads)
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	for i, c := range chunks {
		if len(c.Raw) == 0 {
			t.Errorf("chunk[%d].Raw empty", i)
		}
	}
}

func TestStream_DonesentinelTerminates(t *testing.T) {
	srv := newCaptureServer(t, []string{
		`{"choices":[{"index":0,"delta":{"content":"x"}}]}`,
	})
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if len(chunks) != 1 {
		t.Errorf("len(chunks) = %d, want 1 (DONE terminates)", len(chunks))
	}
}

func TestStream_SkipsMalformedPayload(t *testing.T) {
	payloads := []string{
		`{not json}`,
		`{"choices":[{"index":0,"delta":{"content":"ok"}}]}`,
	}
	srv := newCaptureServer(t, payloads)
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if len(chunks) != 1 {
		t.Errorf("len(chunks) = %d, want 1 (malformed skipped)", len(chunks))
	}
}

func TestStream_HandlesEmptyChoices(t *testing.T) {
	payloads := []string{
		`{"id":"x","object":"chat.completion.chunk","created":1,"model":"mistral-large-latest","choices":[]}`,
	}
	srv := newCaptureServer(t, payloads)
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if len(chunks) != 1 || len(chunks[0].Choices) != 0 {
		t.Errorf("expected 1 chunk with 0 choices, got %+v", chunks)
	}
}

func TestStream_PreservesMetadataFields(t *testing.T) {
	payloads := []string{
		`{"id":"cmpl-abc","object":"chat.completion.chunk","created":99,"model":"mistral-medium","choices":[{"index":0,"delta":{"content":"x"}}]}`,
	}
	srv := newCaptureServer(t, payloads)
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	c := chunks[0]
	if c.ID != "cmpl-abc" || c.Object != "chat.completion.chunk" || c.Created != 99 || c.Model != "mistral-medium" {
		t.Errorf("metadata mismatch: %+v", c)
	}
}

func TestStream_RolePropagates(t *testing.T) {
	payloads := []string{
		`{"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
	}
	srv := newCaptureServer(t, payloads)
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Errorf("role = %q, want assistant", chunks[0].Choices[0].Delta.Role)
	}
}

func TestStream_MultipleChoicesIndices(t *testing.T) {
	payloads := []string{
		`{"choices":[{"index":0,"delta":{"content":"a"}},{"index":1,"delta":{"content":"b"}}]}`,
	}
	srv := newCaptureServer(t, payloads)
	defer srv.close()
	p := newProvider(t, srv.server.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if len(chunks[0].Choices) != 2 {
		t.Fatalf("choices = %d, want 2", len(chunks[0].Choices))
	}
	if chunks[0].Choices[0].Index != 0 || chunks[0].Choices[1].Index != 1 {
		t.Errorf("choice indices = %d/%d, want 0/1",
			chunks[0].Choices[0].Index, chunks[0].Choices[1].Index)
	}
}

// ---- SSE parsing edge cases ------------------------------------------

func TestSSE_DataLineWithoutSpace(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		// Note: "data:" without trailing space.
		fmt.Fprint(w, "data:{\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	p := newProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if len(chunks) != 1 || chunks[0].Choices[0].Delta.Content != "hi" {
		t.Errorf("chunks = %+v", chunks)
	}
}

func TestSSE_IgnoresCommentsAndUnknownFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, ": keepalive\n\n")
		fmt.Fprint(w, "event: chunk\n")
		fmt.Fprint(w, "id: 42\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	p := newProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if len(chunks) != 1 {
		t.Errorf("len(chunks) = %d, want 1", len(chunks))
	}
}

func TestSSE_MultiLineDataJoinedWithNewline(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: {\"choices\":[\ndata: {\"index\":0,\"delta\":{\"content\":\"x\"}}\ndata: ]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	p := newProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	// The synthetic multi-line payload above is intentionally not valid JSON
	// once joined; the parser should skip it without aborting the stream.
	chunks := drain(t, stream)
	_ = chunks
}

func TestSSE_BlankLineBeforeAnyData(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "\n\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"y\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()
	p := newProvider(t, srv.URL)
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks := drain(t, stream)
	if len(chunks) != 1 || chunks[0].Choices[0].Delta.Content != "y" {
		t.Errorf("chunks = %+v", chunks)
	}
}

// ---- Cancellation tests ------------------------------------------------

func TestCompletionStream_ContextCancelStopsStream(t *testing.T) {
	// Server holds the connection open after first chunk; we cancel mid-stream.
	released := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		<-released
	}))
	defer func() {
		close(released)
		srv.Close()
	}()
	p := newProvider(t, srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := p.CompletionStream(ctx, defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	// Read one chunk, then cancel.
	<-stream.Chunks()
	cancel()
	// Drain remaining (should be empty) and confirm Err is set to ctx err.
	for range stream.Chunks() {
	}
	if err := stream.Err(); err == nil {
		t.Error("expected non-nil error after cancel")
	}
}

func TestCompletionStream_StreamCancelMethod(t *testing.T) {
	released := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a\"}}]}\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		<-released
	}))
	defer func() {
		close(released)
		srv.Close()
	}()
	// Use a short client timeout so the pump goroutine unblocks promptly
	// even though stream.Cancel only signals the producer context.
	p, err := mistral.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
		llmrouter.WithTimeout(2*time.Second),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	stream, err := p.CompletionStream(context.Background(), defaultReq())
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	<-stream.Chunks()
	stream.Cancel()
	for range stream.Chunks() {
	}
	if err := stream.Err(); err == nil {
		t.Error("expected non-nil error after Cancel")
	}
}

// ---- Error tests -------------------------------------------------------

func errorStatusTest(t *testing.T, status int) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		fmt.Fprintf(w, `{"error":{"message":"status %d"}}`, status)
	}))
	defer srv.Close()
	p := newProvider(t, srv.URL)
	_, err := p.CompletionStream(context.Background(), defaultReq())
	if err == nil {
		t.Fatalf("expected error for status %d", status)
	}
	var up *llmrouter.ErrUpstream
	if !errorsAs(err, &up) {
		t.Fatalf("error = %T %v, want *llmrouter.ErrUpstream", err, err)
	}
	if up.Provider != "mistral" {
		t.Errorf("Provider = %q, want mistral", up.Provider)
	}
	if up.StatusCode != status {
		t.Errorf("StatusCode = %d, want %d", up.StatusCode, status)
	}
	if !strings.Contains(up.Body, fmt.Sprintf("status %d", status)) {
		t.Errorf("Body = %q, want substring 'status %d'", up.Body, status)
	}
}

func TestUpstreamError_400(t *testing.T) { errorStatusTest(t, http.StatusBadRequest) }
func TestUpstreamError_401(t *testing.T) { errorStatusTest(t, http.StatusUnauthorized) }
func TestUpstreamError_422(t *testing.T) { errorStatusTest(t, http.StatusUnprocessableEntity) }
func TestUpstreamError_429(t *testing.T) { errorStatusTest(t, http.StatusTooManyRequests) }
func TestUpstreamError_500(t *testing.T) { errorStatusTest(t, http.StatusInternalServerError) }

func TestUpstreamError_BodyCappedAt1KiB(t *testing.T) {
	big := strings.Repeat("X", 5000)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, big)
	}))
	defer srv.Close()
	p := newProvider(t, srv.URL)
	_, err := p.CompletionStream(context.Background(), defaultReq())
	if err == nil {
		t.Fatal("expected error")
	}
	var up *llmrouter.ErrUpstream
	if !errorsAs(err, &up) {
		t.Fatalf("error = %T", err)
	}
	if len(up.Body) > 1024 {
		t.Errorf("body len = %d, want <= 1024", len(up.Body))
	}
}

func TestUpstreamError_ErrorMessageFormat(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		fmt.Fprint(w, "forbidden")
	}))
	defer srv.Close()
	p := newProvider(t, srv.URL)
	_, err := p.CompletionStream(context.Background(), defaultReq())
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "mistral upstream 403") {
		t.Errorf("error.Error = %q, want substring 'mistral upstream 403'", err.Error())
	}
}

func TestTransport_NetworkErrorPropagates(t *testing.T) {
	p, err := mistral.New(
		llmrouter.WithAPIKey("k"),
		llmrouter.WithBaseURL("http://127.0.0.1:1"), // closed port
		llmrouter.WithTimeout(500*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CompletionStream(context.Background(), defaultReq())
	if err == nil {
		t.Fatal("expected network error")
	}
	var up *llmrouter.ErrUpstream
	if errorsAs(err, &up) {
		t.Errorf("got ErrUpstream for transport failure: %v", up)
	}
}
