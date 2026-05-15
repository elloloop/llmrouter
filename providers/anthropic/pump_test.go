package anthropic

import (
	"context"
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
// mapStopReason
// ---------------------------------------------------------------------------

func TestMapStopReason_Table(t *testing.T) {
	cases := map[string]string{
		"end_turn":           "stop",
		"stop_sequence":      "stop",
		"max_tokens":         "length",
		"tool_use":           "tool_calls",
		"":                   "stop",
		"unknown_thing":      "stop",
		"refusal":            "stop",
		"completed":          "stop",
		"system_interrupt":   "stop",
		"END_TURN":           "stop", // case-sensitive: not matched → default
		"  end_turn  ":       "stop", // not trimmed → default
	}
	for in, want := range cases {
		t.Run("in="+strings.ReplaceAll(in, " ", "_"), func(t *testing.T) {
			if got := mapStopReason(in); got != want {
				t.Fatalf("mapStopReason(%q) = %q, want %q", in, got, want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// handleEvent — message_start
// ---------------------------------------------------------------------------

// captureHooks builds a synchronous ProducerHooks pair for unit-testing
// handleEvent in isolation. Returns a pointer to the slice of captured
// chunks so callers see the appends made by Send.
func captureHooks() (*[]llmrouter.Chunk, llmrouter.ProducerHooks) {
	var sent []llmrouter.Chunk
	cancelled := false
	hooks := llmrouter.ProducerHooks{
		Send: func(c llmrouter.Chunk) bool {
			if cancelled {
				return false
			}
			sent = append(sent, c)
			return true
		},
		Finish: func(err error) { _ = err },
	}
	return &sent, hooks
}

func TestHandleEvent_MessageStart_EmitsRolePrimer(t *testing.T) {
	st := &pumpState{model: "claude", chatID: "chatcmpl-1", created: 100}
	sent, hooks := captureHooks()
	done, err := handleEvent(context.Background(), "message_start",
		`{"type":"message_start","message":{"id":"msg_1","model":"claude-3-5-sonnet","usage":{"input_tokens":7,"output_tokens":0}}}`,
		st, hooks)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if done {
		t.Fatal("done=true unexpected")
	}
	if len(*sent) != 1 {
		t.Fatalf("got %d chunks, want 1", len(*sent))
	}
	c := (*sent)[0]
	if c.Choices[0].Delta.Role != "assistant" {
		t.Errorf("role = %q", c.Choices[0].Delta.Role)
	}
	if c.Choices[0].Delta.Content != "" {
		t.Errorf("content = %q, want empty", c.Choices[0].Delta.Content)
	}
	if st.inputTokens != 7 {
		t.Errorf("state.inputTokens = %d", st.inputTokens)
	}
	if st.model != "claude-3-5-sonnet" {
		t.Errorf("state.model = %q", st.model)
	}
}

func TestHandleEvent_MessageStart_WithoutModelKeepsExisting(t *testing.T) {
	st := &pumpState{model: "preset", chatID: "x", created: 1}
	_, hooks := captureHooks()
	_, err := handleEvent(context.Background(), "message_start",
		`{"type":"message_start","message":{"id":"msg","usage":{"input_tokens":3}}}`,
		st, hooks)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if st.model != "preset" {
		t.Errorf("state.model overridden: %q", st.model)
	}
}

func TestHandleEvent_MessageStart_MalformedTolerated(t *testing.T) {
	st := &pumpState{model: "preset"}
	sent, hooks := captureHooks()
	// Even with malformed payload, we still emit the role primer.
	_, err := handleEvent(context.Background(), "message_start", `garbage`, st, hooks)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(*sent) != 1 {
		t.Fatalf("got %d", len(*sent))
	}
}

// ---------------------------------------------------------------------------
// handleEvent — content_block_delta
// ---------------------------------------------------------------------------

func TestHandleEvent_ContentBlockDelta_TextDeltaEmitted(t *testing.T) {
	st := &pumpState{model: "claude", chatID: "x", created: 1}
	sent, hooks := captureHooks()
	_, err := handleEvent(context.Background(), "content_block_delta",
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
		st, hooks)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(*sent) != 1 {
		t.Fatalf("got %d", len(*sent))
	}
	if (*sent)[0].Choices[0].Delta.Content != "Hello" {
		t.Errorf("content = %q", (*sent)[0].Choices[0].Delta.Content)
	}
}

func TestHandleEvent_ContentBlockDelta_Skipped(t *testing.T) {
	cases := []struct {
		name    string
		payload string
	}{
		{"empty-text", `{"type":"content_block_delta","delta":{"type":"text_delta","text":""}}`},
		{"thinking-delta", `{"type":"content_block_delta","delta":{"type":"thinking_delta","text":"silent"}}`},
		{"input-json-delta", `{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{"}}`},
		{"unknown-type", `{"type":"content_block_delta","delta":{"type":"future_thing","text":"x"}}`},
		{"malformed", `not json`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := &pumpState{model: "x"}
			sent, hooks := captureHooks()
			_, err := handleEvent(context.Background(), "content_block_delta", tc.payload, st, hooks)
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			if len(*sent) != 0 {
				t.Fatalf("expected 0 chunks, got %d", len(*sent))
			}
		})
	}
}

// ---------------------------------------------------------------------------
// handleEvent — message_delta
// ---------------------------------------------------------------------------

func TestHandleEvent_MessageDelta_StopReasonMapping(t *testing.T) {
	cases := []struct {
		name   string
		reason string
		want   string
	}{
		{"end_turn", "end_turn", "stop"},
		{"stop_sequence", "stop_sequence", "stop"},
		{"max_tokens", "max_tokens", "length"},
		{"tool_use", "tool_use", "tool_calls"},
		{"unknown", "weird_thing", "stop"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := &pumpState{model: "claude", chatID: "x", created: 1, inputTokens: 12}
			sent, hooks := captureHooks()
			payload := `{"type":"message_delta","delta":{"stop_reason":"` + tc.reason + `"},"usage":{"output_tokens":7}}`
			_, err := handleEvent(context.Background(), "message_delta", payload, st, hooks)
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			if len(*sent) != 1 {
				t.Fatalf("got %d chunks", len(*sent))
			}
			c := (*sent)[0]
			if c.Choices[0].FinishReason != tc.want {
				t.Errorf("FinishReason = %q, want %q", c.Choices[0].FinishReason, tc.want)
			}
			if c.Usage == nil {
				t.Fatal("Usage should be populated on finish chunk")
			}
			if c.Usage.PromptTokens != 12 || c.Usage.CompletionTokens != 7 || c.Usage.TotalTokens != 19 {
				t.Errorf("Usage = %+v", *c.Usage)
			}
		})
	}
}

func TestHandleEvent_MessageDelta_NoStopReasonNoEmission(t *testing.T) {
	st := &pumpState{model: "claude"}
	sent, hooks := captureHooks()
	_, err := handleEvent(context.Background(), "message_delta",
		`{"type":"message_delta","delta":{},"usage":{"output_tokens":2}}`, st, hooks)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(*sent) != 0 {
		t.Fatalf("got %d, want 0", len(*sent))
	}
	if st.outputTokens != 2 {
		t.Errorf("outputTokens = %d", st.outputTokens)
	}
}

func TestHandleEvent_MessageDelta_MalformedTolerated(t *testing.T) {
	st := &pumpState{}
	_, hooks := captureHooks()
	_, err := handleEvent(context.Background(), "message_delta", `not json`, st, hooks)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
}

// ---------------------------------------------------------------------------
// handleEvent — message_stop
// ---------------------------------------------------------------------------

func TestHandleEvent_MessageStop_ReturnsDone(t *testing.T) {
	st := &pumpState{}
	_, hooks := captureHooks()
	done, err := handleEvent(context.Background(), "message_stop", `{"type":"message_stop"}`, st, hooks)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if !done {
		t.Fatal("expected done=true")
	}
}

func TestHandleEvent_MessageStop_EmptyPayloadOK(t *testing.T) {
	st := &pumpState{}
	_, hooks := captureHooks()
	done, err := handleEvent(context.Background(), "message_stop", ``, st, hooks)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if !done {
		t.Fatal("expected done=true")
	}
}

// ---------------------------------------------------------------------------
// handleEvent — ignored event types
// ---------------------------------------------------------------------------

func TestHandleEvent_IgnoredEventTypes(t *testing.T) {
	cases := []string{
		"ping",
		"content_block_start",
		"content_block_stop",
		"thinking_delta",
		"some_future_event",
		"",
	}
	for _, et := range cases {
		t.Run("event="+et, func(t *testing.T) {
			st := &pumpState{}
			sent, hooks := captureHooks()
			done, err := handleEvent(context.Background(), et, `{"any":"thing"}`, st, hooks)
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			if done {
				t.Fatal("unexpected done=true")
			}
			if len(*sent) != 0 {
				t.Fatalf("got %d chunks, want 0", len(*sent))
			}
		})
	}
}

// ---------------------------------------------------------------------------
// newChunk / currentUsage
// ---------------------------------------------------------------------------

func TestNewChunk_FieldsAndRaw(t *testing.T) {
	st := &pumpState{model: "claude", chatID: "chatcmpl-99", created: 42}
	c := newChunk(st, llmrouter.Delta{Content: "hi"}, "stop")
	if c.ID != "chatcmpl-99" {
		t.Errorf("ID = %q", c.ID)
	}
	if c.Object != "chat.completion.chunk" {
		t.Errorf("Object = %q", c.Object)
	}
	if c.Created != 42 {
		t.Errorf("Created = %d", c.Created)
	}
	if c.Model != "claude" {
		t.Errorf("Model = %q", c.Model)
	}
	if len(c.Choices) != 1 {
		t.Fatalf("Choices = %d", len(c.Choices))
	}
	if c.Choices[0].FinishReason != "stop" {
		t.Errorf("FinishReason = %q", c.Choices[0].FinishReason)
	}
	if len(c.Raw) == 0 {
		t.Errorf("Raw empty")
	}
}

func TestCurrentUsage(t *testing.T) {
	cases := []struct {
		name string
		st   pumpState
		want *llmrouter.Usage
	}{
		{"both-zero", pumpState{}, nil},
		{"input-only", pumpState{inputTokens: 5}, &llmrouter.Usage{PromptTokens: 5, TotalTokens: 5}},
		{"output-only", pumpState{outputTokens: 3}, &llmrouter.Usage{CompletionTokens: 3, TotalTokens: 3}},
		{"both", pumpState{inputTokens: 4, outputTokens: 6}, &llmrouter.Usage{PromptTokens: 4, CompletionTokens: 6, TotalTokens: 10}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := currentUsage(&tc.st)
			if (got == nil) != (tc.want == nil) {
				t.Fatalf("nil-ness mismatch: got=%v want=%v", got, tc.want)
			}
			if got != nil && *got != *tc.want {
				t.Fatalf("got %+v, want %+v", *got, *tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// End-to-end: CompletionStream public surface — headers, model rewriting,
// custom HTTP client, context cancellation, default base URL, error status.
// ---------------------------------------------------------------------------

func TestNew_RequiresAPIKey(t *testing.T) {
	_, err := New()
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Fatalf("err = %v, want wraps ErrInvalidConfig", err)
	}
}

func TestNew_DefaultBaseURL(t *testing.T) {
	p, err := New(llmrouter.WithAPIKey("k"))
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if p.cfg.BaseURL != defaultBaseURL {
		t.Fatalf("BaseURL = %q, want %q", p.cfg.BaseURL, defaultBaseURL)
	}
}

func TestProvider_NameStable(t *testing.T) {
	p, _ := New(llmrouter.WithAPIKey("k"))
	for i := 0; i < 5; i++ {
		if p.Name() != providerName {
			t.Fatalf("Name drift: %q", p.Name())
		}
	}
}

func TestCompletionStream_VariousErrorStatusCodes(t *testing.T) {
	cases := []int{400, 401, 403, 404, 429, 500, 502, 503, 504, 529}
	for _, code := range cases {
		t.Run(http.StatusText(code), func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(code)
				_, _ = io.WriteString(w, `{"error":{"type":"x"}}`)
			}))
			defer srv.Close()
			p, _ := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
			_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
				Model:    "claude",
				Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
			})
			if err == nil {
				t.Fatal("expected error")
			}
			var ue *llmrouter.ErrUpstream
			if !errors.As(err, &ue) {
				t.Fatalf("err = %T %v", err, err)
			}
			if ue.StatusCode != code {
				t.Errorf("StatusCode = %d", ue.StatusCode)
			}
			if ue.Provider != providerName {
				t.Errorf("Provider = %q", ue.Provider)
			}
		})
	}
}

func TestCompletionStream_NetworkErrorReturned(t *testing.T) {
	p, _ := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL("http://127.0.0.1:1"))
	_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("expected network error")
	}
	var ue *llmrouter.ErrUpstream
	if errors.As(err, &ue) {
		t.Fatalf("should not be ErrUpstream: %T", err)
	}
}

func TestCompletionStream_HeadersAreSet(t *testing.T) {
	var gotAPIKey, gotVersion, gotAccept, gotCT string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAPIKey = r.Header.Get("x-api-key")
		gotVersion = r.Header.Get("anthropic-version")
		gotAccept = r.Header.Get("Accept")
		gotCT = r.Header.Get("Content-Type")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message_stop\ndata: {}\n\n")
	}))
	defer srv.Close()
	p, _ := New(llmrouter.WithAPIKey("secret-key"), llmrouter.WithBaseURL(srv.URL))
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	for range stream.Chunks() {
	}
	if gotAPIKey != "secret-key" {
		t.Errorf("x-api-key = %q", gotAPIKey)
	}
	if gotVersion != anthropicVersion {
		t.Errorf("anthropic-version = %q", gotVersion)
	}
	if gotAccept != "text/event-stream" {
		t.Errorf("Accept = %q", gotAccept)
	}
	if gotCT != "application/json" {
		t.Errorf("Content-Type = %q", gotCT)
	}
}

func TestCompletionStream_PathSuffix(t *testing.T) {
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		_, _ = io.WriteString(w, "event: message_stop\ndata: {}\n\n")
	}))
	defer srv.Close()
	p, _ := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	for range stream.Chunks() {
	}
	if !strings.HasSuffix(gotPath, "/messages") {
		t.Fatalf("path = %q, want suffix /messages", gotPath)
	}
}

func TestCompletionStream_ContextCancellationStopsStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, _ := w.(http.Flusher)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for i := 0; i < 100; i++ {
			select {
			case <-r.Context().Done():
				return
			default:
			}
			_, _ = io.WriteString(w, "event: content_block_delta\n"+
				`data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"."}}`+"\n\n")
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(2 * time.Millisecond)
		}
	}))
	defer srv.Close()
	p, _ := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	got := 0
	for range stream.Chunks() {
		got++
		if got >= 3 {
			cancel()
		}
	}
	if err := stream.Err(); err == nil {
		t.Fatal("expected error after cancel")
	}
}

// ---------------------------------------------------------------------------
// Full-stream translation: every SSE event type in one fixture, including
// thinking_delta and ping which must be ignored without breaking the stream.
// ---------------------------------------------------------------------------

const fullFixture = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"claude-3-5-sonnet","usage":{"input_tokens":10,"output_tokens":0}}}

event: ping
data: {}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","text":"silent thoughts"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}

event: message_stop
data: {"type":"message_stop"}

`

func TestPump_FullFixtureTranslatesCorrectly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, fullFixture)
	}))
	defer srv.Close()
	p, _ := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	var chunks []llmrouter.Chunk
	for c := range stream.Chunks() {
		chunks = append(chunks, c)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("Err = %v", err)
	}
	// primer + 2 text + finish = 4
	if len(chunks) != 4 {
		t.Fatalf("got %d chunks, want 4: %+v", len(chunks), chunks)
	}
	if chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Errorf("primer role")
	}
	if chunks[1].Choices[0].Delta.Content != "Hello" {
		t.Errorf("chunk[1] = %q", chunks[1].Choices[0].Delta.Content)
	}
	if chunks[2].Choices[0].Delta.Content != " world" {
		t.Errorf("chunk[2] = %q", chunks[2].Choices[0].Delta.Content)
	}
	if chunks[3].Choices[0].FinishReason != "stop" {
		t.Errorf("final.FinishReason = %q", chunks[3].Choices[0].FinishReason)
	}
	if chunks[3].Usage == nil || chunks[3].Usage.TotalTokens != 15 {
		t.Fatalf("Usage = %+v", chunks[3].Usage)
	}
}

func TestPump_AllChunksShareStableID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, fullFixture)
	}))
	defer srv.Close()
	p, _ := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	var ids []string
	for c := range stream.Chunks() {
		ids = append(ids, c.ID)
	}
	if len(ids) == 0 {
		t.Fatal("no chunks")
	}
	for i, id := range ids {
		if !strings.HasPrefix(id, "chatcmpl-") {
			t.Errorf("chunk[%d].ID = %q, want chatcmpl- prefix", i, id)
		}
		if id != ids[0] {
			t.Errorf("chunk[%d].ID drift: %q vs %q", i, id, ids[0])
		}
	}
}

func TestPump_RawByteFieldPopulated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, fullFixture)
	}))
	defer srv.Close()
	p, _ := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(srv.URL))
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	for c := range stream.Chunks() {
		if len(c.Raw) == 0 {
			t.Errorf("Raw empty on chunk %+v", c)
		}
	}
}
