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

// anthropicSSEFixture is a complete /v1/messages SSE response covering
// message_start, two text deltas, message_delta with usage + stop_reason,
// and message_stop.
const anthropicSSEFixture = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"claude-3-5-sonnet-20241022","usage":{"input_tokens":12,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}

`

func newFixtureServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") == "" {
			t.Errorf("missing x-api-key header")
		}
		if got := r.Header.Get("anthropic-version"); got != anthropicVersion {
			t.Errorf("anthropic-version = %q, want %q", got, anthropicVersion)
		}
		if got := r.Header.Get("Accept"); got != "text/event-stream" {
			t.Errorf("Accept = %q, want text/event-stream", got)
		}
		if !strings.HasSuffix(r.URL.Path, "/messages") {
			t.Errorf("path = %q, want .../messages", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, body)
	}))
}

func TestCompletionStream_TranslatesAnthropicSSEToOpenAIChunks(t *testing.T) {
	srv := newFixtureServer(t, anthropicSSEFixture)
	defer srv.Close()

	p, err := New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "claude-3-5-sonnet-20241022",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}

	var chunks []llmrouter.Chunk
	for c := range stream.Chunks() {
		chunks = append(chunks, c)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err: %v", err)
	}

	if len(chunks) != 4 {
		t.Fatalf("got %d chunks, want 4 (primer + 2 deltas + finish): %+v", len(chunks), chunks)
	}

	// 1. role primer
	if chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Errorf("primer role = %q, want assistant", chunks[0].Choices[0].Delta.Role)
	}
	if chunks[0].Choices[0].Delta.Content != "" {
		t.Errorf("primer content = %q, want empty", chunks[0].Choices[0].Delta.Content)
	}

	// 2-3. text deltas
	if chunks[1].Choices[0].Delta.Content != "Hello" {
		t.Errorf("chunk[1].content = %q, want Hello", chunks[1].Choices[0].Delta.Content)
	}
	if chunks[2].Choices[0].Delta.Content != " world" {
		t.Errorf("chunk[2].content = %q, want ' world'", chunks[2].Choices[0].Delta.Content)
	}

	// 4. finish
	final := chunks[3]
	if final.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", final.Choices[0].FinishReason)
	}
	if final.Usage == nil {
		t.Fatalf("final.Usage = nil, want populated")
	}
	if final.Usage.PromptTokens != 12 || final.Usage.CompletionTokens != 7 || final.Usage.TotalTokens != 19 {
		t.Errorf("usage = %+v, want {12,7,19}", *final.Usage)
	}

	// Stable IDs across chunks.
	for i, c := range chunks {
		if c.ID == "" || !strings.HasPrefix(c.ID, "chatcmpl-") {
			t.Errorf("chunk[%d].ID = %q, want chatcmpl-...", i, c.ID)
		}
		if c.ID != chunks[0].ID {
			t.Errorf("chunk[%d].ID = %q, want %q (stable)", i, c.ID, chunks[0].ID)
		}
		if c.Object != "chat.completion.chunk" {
			t.Errorf("chunk[%d].Object = %q", i, c.Object)
		}
		if len(c.Raw) == 0 {
			t.Errorf("chunk[%d].Raw is empty", i)
		}
	}
}

func TestCompletionStream_UpstreamErrorReturnsErrUpstream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":"invalid api key"}`)
	}))
	defer srv.Close()

	p, err := New(llmrouter.WithAPIKey("bad"), llmrouter.WithBaseURL(srv.URL))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "claude-3-5-sonnet-20241022",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	var ue *llmrouter.ErrUpstream
	if !errors.As(err, &ue) {
		t.Fatalf("err = %v, want *llmrouter.ErrUpstream", err)
	}
	if ue.StatusCode != http.StatusUnauthorized {
		t.Errorf("StatusCode = %d", ue.StatusCode)
	}
	if ue.Provider != "anthropic" {
		t.Errorf("Provider = %q", ue.Provider)
	}
}

func TestBuildAnthropicBody_LiftsSystemAndDefaultsMaxTokens(t *testing.T) {
	body, err := buildAnthropicBody(llmrouter.ChatRequest{
		Model: "claude-3-5-sonnet",
		Messages: []llmrouter.Message{
			llmrouter.TextMessage("system", "you are nice"),
			llmrouter.TextMessage("system", "be concise"),
			llmrouter.TextMessage("user", "hi"),
		},
	})
	if err != nil {
		t.Fatalf("buildAnthropicBody: %v", err)
	}
	s := string(body)
	if !strings.Contains(s, `"max_tokens":4096`) {
		t.Errorf("missing default max_tokens=4096: %s", s)
	}
	if !strings.Contains(s, `"system":"you are nice\n\nbe concise"`) {
		t.Errorf("system not lifted+joined: %s", s)
	}
	if !strings.Contains(s, `"stream":true`) {
		t.Errorf("stream flag missing: %s", s)
	}
	if strings.Contains(s, `"role":"system"`) {
		t.Errorf("system role leaked into messages: %s", s)
	}
}

func TestMapStopReason(t *testing.T) {
	cases := map[string]string{
		"end_turn":      "stop",
		"stop_sequence": "stop",
		"max_tokens":    "length",
		"tool_use":      "tool_calls",
		"weird":         "stop",
	}
	for in, want := range cases {
		if got := mapStopReason(in); got != want {
			t.Errorf("mapStopReason(%q) = %q, want %q", in, got, want)
		}
	}
}
