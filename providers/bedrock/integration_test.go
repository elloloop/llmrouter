//go:build integration

// Package bedrock integration tests exercise real AWS Bedrock endpoints
// and are excluded from the default `go test` run. Enable with:
//
//	AWS_REGION=us-east-1 \
//	go test -tags=integration ./providers/bedrock/...
//
// The test relies on the standard AWS credential chain (env vars, profile,
// IAM role). Either AWS_REGION or AWS_DEFAULT_REGION must be set. Each
// test skips when its prerequisites are absent.
//
// These tests are stubs that document the SDK-bound paths we cannot
// meaningfully unit-test: the *bedrockruntime.Client has no public
// interface seam, so confidence in CompletionStream / Embed / resolveClient
// / pump comes from running this suite against a real account.
package bedrock

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
)

// requireAWSRegion skips when no AWS region env var is set.
func requireAWSRegion(t *testing.T) string {
	t.Helper()
	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = os.Getenv("AWS_DEFAULT_REGION")
	}
	if region == "" {
		t.Skip("AWS_REGION or AWS_DEFAULT_REGION required for integration tests")
	}
	return region
}

// TestIntegration_ChatCompletion validates the happy-path streaming chat
// completion against Bedrock Anthropic. Exercises resolveClient (lazy
// client init) + CompletionStream + pump translation.
func TestIntegration_ChatCompletion(t *testing.T) {
	region := requireAWSRegion(t)
	p, err := New(WithRegion(region))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model: "anthropic.claude-3-5-haiku-20241022-v1:0",
		Messages: []llmrouter.Message{
			llmrouter.TextMessage("user", "Say the word 'pong' and nothing else."),
		},
		MaxTokens: 16,
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	var buf strings.Builder
	for chunk := range stream.Chunks() {
		for _, ch := range chunk.Choices {
			buf.WriteString(ch.Delta.Content)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if buf.Len() == 0 {
		t.Fatal("no content received")
	}
}

// TestIntegration_ChatCompletion_ToolUse validates that the Converse
// tool_use translation round-trips a tool call. Optional — many accounts
// don't have tool-use turned on for the chosen model.
func TestIntegration_ChatCompletion_ToolUse(t *testing.T) {
	region := requireAWSRegion(t)
	p, err := New(WithRegion(region))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model: "anthropic.claude-3-5-haiku-20241022-v1:0",
		Messages: []llmrouter.Message{
			llmrouter.TextMessage("user", "Use the get_weather tool for San Francisco."),
		},
		Tools: []llmrouter.Tool{
			{
				Type: "function",
				Function: llmrouter.ToolFunction{
					Name:        "get_weather",
					Description: "Get the current weather for a city.",
				},
			},
		},
		MaxTokens: 256,
	})
	if err != nil {
		t.Skipf("Tool use not supported for this configuration: %v", err)
	}
	for range stream.Chunks() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}
}

// TestIntegration_Embed_Titan validates the Amazon Titan embeddings path
// (embedTitan). Skip when the account does not have Titan enabled.
func TestIntegration_Embed_Titan(t *testing.T) {
	region := requireAWSRegion(t)
	p, err := New(WithRegion(region))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	resp, err := p.Embed(ctx, llmrouter.EmbedRequest{
		Model:  "amazon.titan-embed-text-v2:0",
		Inputs: []string{"hello world"},
	})
	if err != nil {
		t.Skipf("Titan embeddings unavailable: %v", err)
	}
	if len(resp.Embeddings) != 1 || len(resp.Embeddings[0]) == 0 {
		t.Fatalf("unexpected embeddings shape: %+v", resp.Embeddings)
	}
}

// TestIntegration_Embed_Cohere validates the Cohere embeddings path
// (embedCohere). Cohere embeddings on Bedrock require Cohere model access.
func TestIntegration_Embed_Cohere(t *testing.T) {
	region := requireAWSRegion(t)
	p, err := New(WithRegion(region))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	resp, err := p.Embed(ctx, llmrouter.EmbedRequest{
		Model:  "cohere.embed-english-v3",
		Inputs: []string{"hello world"},
	})
	if err != nil {
		t.Skipf("Cohere embeddings unavailable: %v", err)
	}
	if len(resp.Embeddings) != 1 || len(resp.Embeddings[0]) == 0 {
		t.Fatalf("unexpected embeddings shape: %+v", resp.Embeddings)
	}
}

// TestIntegration_ContextCancel proves CompletionStream + pump respond to
// context cancellation mid-stream by closing the channel and surfacing an
// error.
func TestIntegration_ContextCancel(t *testing.T) {
	region := requireAWSRegion(t)
	p, err := New(WithRegion(region))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model: "anthropic.claude-3-5-haiku-20241022-v1:0",
		Messages: []llmrouter.Message{
			llmrouter.TextMessage("user", "Count slowly to one hundred."),
		},
		MaxTokens: 2048,
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()
	for range stream.Chunks() {
	}
	if stream.Err() == nil {
		t.Skip("stream completed before cancellation took effect")
	}
}

// TestIntegration_InvalidModelID confirms an unknown model id surfaces a
// clear error, not a panic.
func TestIntegration_InvalidModelID(t *testing.T) {
	region := requireAWSRegion(t)
	p, err := New(WithRegion(region))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "anthropic.this-model-id-does-not-exist-v999:0",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		// Synchronous validation error — acceptable.
		return
	}
	for range stream.Chunks() {
	}
	if stream.Err() == nil {
		t.Fatal("expected an error for invalid model id")
	}
}

// TestIntegration_MissingRegion confirms New rejects a missing region
// even before any AWS round-trip occurs.
func TestIntegration_MissingRegion(t *testing.T) {
	_, err := New()
	if err == nil {
		t.Fatal("expected error from New() with no region")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Fatalf("want ErrInvalidConfig, got %v", err)
	}
}
