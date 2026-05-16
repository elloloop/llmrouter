// Package llmrouter is a polyglot Go client for LLM providers.
//
// One OpenAI-shaped API surface fans out to OpenAI, Anthropic, Azure
// OpenAI, AWS Bedrock, Google Vertex (planned), and any OpenAI-compatible
// endpoint (Groq, Together, OpenRouter, self-hosted) via a configurable
// base URL.
//
// Streaming-first design with proper context cancellation propagation.
//
// Quick start:
//
//	import (
//		"context"
//		"github.com/elloloop/llmrouter"
//		"github.com/elloloop/llmrouter/providers/openai"
//	)
//
//	p, _ := openai.New(llmrouter.WithAPIKey("sk-..."))
//	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
//		Model: "gpt-4o-mini",
//		Messages: []llmrouter.Message{{Role: "user", Content: "hi"}},
//	})
//	if err != nil { /* handle */ }
//	for chunk := range stream.Chunks() {
//		fmt.Print(chunk.Delta.Content)
//	}
//	if err := stream.Err(); err != nil { /* upstream error */ }
package llmrouter

import (
	"context"
	"encoding/json"
)

// Provider is the upstream-LLM contract. Each implementation translates
// to/from the OpenAI /v1/chat/completions wire format internally.
type Provider interface {
	// Name returns the provider's stable id (e.g. "openai", "anthropic").
	Name() string

	// CompletionStream issues a streaming chat completion request. The
	// returned Stream yields normalized chunks until upstream finishes,
	// the context cancels, or an error occurs. Callers must drain the
	// stream (or cancel ctx) to release resources.
	CompletionStream(ctx context.Context, req ChatRequest) (*Stream, error)
}

// ChatRequest is the OpenAI-shaped request. Raw, if non-nil, is used by
// passthrough providers (OpenAI-compatible) to forward the original JSON
// without re-serializing fields they don't model (tools, vision, etc.).
type ChatRequest struct {
	Model       string          `json:"model"`
	Messages    []Message       `json:"messages"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	Stop        []string        `json:"stop,omitempty"`
	User        string          `json:"user,omitempty"`
	Stream      bool            `json:"stream,omitempty"`

	// ResponseSchema asks the model to produce output that strictly matches
	// a JSON Schema. OpenAI maps to response_format={"type":"json_schema",
	// "json_schema":{...}}. Anthropic translates to forced tool-use with a
	// synthetic tool matching the schema. Providers that don't support
	// schema-coerced output ignore this field.
	ResponseSchema *ResponseSchema `json:"response_schema,omitempty"`

	Tools      []Tool          `json:"tools,omitempty"`
	ToolChoice *ToolChoice     `json:"tool_choice,omitempty"`
	Raw        json.RawMessage `json:"-"`
}

// ResponseSchema constrains the model's output to a JSON Schema. The
// Name is sent to providers that require a schema identifier (OpenAI);
// providers that don't need it ignore it.
type ResponseSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Strict      bool            `json:"strict,omitempty"`
	Schema      json.RawMessage `json:"schema"`
}

// Tool describes a callable function/tool. OpenAI-shaped (the Anthropic
// provider translates).
type Tool struct {
	Type     string       `json:"type"` // always "function" for v0.1
	Function ToolFunction `json:"function"`
}

// ToolFunction is the function descriptor inside a Tool.
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"` // JSON Schema
}

// ToolChoice controls whether/which tool the model must call.
// One of: "auto" | "none" | "required" | {type:"function", function:{name:"..."}}
type ToolChoice struct {
	Mode     string `json:"-"` // "auto" | "none" | "required" | "specific"
	Function string `json:"-"` // set when Mode == "specific"
}

// MarshalJSON renders the ToolChoice as either a string or the OpenAI
// object form for specific function selection.
func (tc ToolChoice) MarshalJSON() ([]byte, error) {
	switch tc.Mode {
	case "specific":
		return json.Marshal(map[string]any{
			"type":     "function",
			"function": map[string]string{"name": tc.Function},
		})
	case "auto", "none", "required":
		return json.Marshal(tc.Mode)
	default:
		return json.Marshal("auto")
	}
}

// Message is one item in a chat completion request. Content is
// json.RawMessage so multimodal content arrays pass through unchanged.
//
// ToolCallID is set on Messages with Role == "tool" to correlate the
// result back to the model's earlier tool_call (OpenAI semantics).
// Anthropic uses tool_use_id in its content-block shape; the anthropic
// provider translates ToolCallID -> tool_use_id.
//
// Name is the OpenAI legacy "function name on tool result" field; it is
// preserved as a passthrough for callers that still set it.
type Message struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
}

// TextMessage is a convenience constructor for plain-text messages.
func TextMessage(role, text string) Message {
	b, _ := json.Marshal(text)
	return Message{Role: role, Content: b}
}

// ToolResultMessage builds a "tool" role message carrying the result of
// a previous tool call. content is the tool's textual output; toolCallID
// must match the model's earlier tool_call id.
func ToolResultMessage(toolCallID, content string) Message {
	b, _ := json.Marshal(content)
	return Message{
		Role:       "tool",
		Content:    b,
		ToolCallID: toolCallID,
	}
}

// PlainText extracts text from a Message. For multimodal content arrays
// it concatenates the "text" parts and ignores image bytes.
func (m Message) PlainText() string {
	if len(m.Content) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(m.Content, &parts); err == nil {
		var out string
		for _, p := range parts {
			if p.Type == "text" {
				out += p.Text
			}
		}
		return out
	}
	return ""
}

// Chunk is one streaming delta, normalized to OpenAI shape. Most fields
// match the OpenAI Chat Completion chunk format; Usage is populated only
// on the final chunk when the upstream reports it.
type Chunk struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"`
	Created int64     `json:"created"`
	Model   string    `json:"model"`
	Choices []Choice  `json:"choices"`
	Usage   *Usage    `json:"usage,omitempty"`
	// Raw is the original wire-format JSON for this chunk. Passthrough
	// providers populate this; consumers that want to forward bytes
	// unmodified should prefer Raw over re-marshaling the typed fields.
	Raw json.RawMessage `json:"-"`
}

// Choice is one streaming choice within a Chunk.
type Choice struct {
	Index        int    `json:"index"`
	Delta        Delta  `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// Delta is the incremental content within a streaming Choice.
//
// Thinking is Anthropic-specific; it carries the model's internal reasoning
// stream when the upstream emits "thinking_delta" events. OpenAI's o*
// reasoning models do not stream reasoning tokens, so this field stays
// empty for them.
type Delta struct {
	Role      string          `json:"role,omitempty"`
	Content   string          `json:"content,omitempty"`
	Thinking  string          `json:"thinking,omitempty"`
	ToolCalls []ToolCallDelta `json:"tool_calls,omitempty"`
}

// ToolCallDelta is one incremental tool-call fragment. Mirrors OpenAI's
// streaming shape: each tool_call has an Index that correlates fragments
// across multiple chunks; Function.Arguments streams as a JSON fragment
// that callers must concatenate by Index.
type ToolCallDelta struct {
	Index    int                    `json:"index"`
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"` // "function"
	Function *ToolCallFunctionDelta `json:"function,omitempty"`
}

// ToolCallFunctionDelta is the function-call body of a streaming
// ToolCallDelta. Name typically appears on the first fragment for an
// Index; Arguments streams incrementally as JSON text.
type ToolCallFunctionDelta struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"` // incrementally streamed JSON fragment
}

// Usage is the token-count summary reported by the upstream.
//
// CachedPromptTokens / CacheCreationTokens are Anthropic-specific and
// reflect prompt caching: how many prompt tokens were served from cache
// (read) and how many were written into the cache (creation). They are
// zero for providers that don't support prompt caching.
type Usage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	CachedPromptTokens  int `json:"cached_prompt_tokens,omitempty"`
	CacheCreationTokens int `json:"cache_creation_tokens,omitempty"`
}
