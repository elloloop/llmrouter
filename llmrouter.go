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
	Raw         json.RawMessage `json:"-"`
}

// Message is one item in a chat completion request. Content is
// json.RawMessage so multimodal content arrays pass through unchanged.
type Message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// TextMessage is a convenience constructor for plain-text messages.
func TextMessage(role, text string) Message {
	b, _ := json.Marshal(text)
	return Message{Role: role, Content: b}
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
type Delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// Usage is the token-count summary reported by the upstream.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
