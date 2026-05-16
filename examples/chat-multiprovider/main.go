// chat-multiprovider runs the same prompt against OpenAI, Anthropic, and
// Gemini side-by-side and prints each provider's full streamed answer.
//
// Requires:
//
//	OPENAI_API_KEY     — for the OpenAI call
//	ANTHROPIC_API_KEY  — for the Anthropic call
//	GEMINI_API_KEY     — for the Gemini call (Google AI Studio)
//
// Usage:
//
//	go run ./examples/chat-multiprovider
//
// Output: three labeled sections (openai, anthropic, gemini) each
// containing that provider's full response. Calls run sequentially so
// the output stays readable in a terminal; swap to goroutines if you
// prefer parallel fan-out.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/anthropic"
	"github.com/elloloop/llmrouter/providers/gemini"
	"github.com/elloloop/llmrouter/providers/openai"
)

const prompt = "In one sentence, what is a goroutine?"

func main() {
	mustEnv := func(name string) string {
		v := os.Getenv(name)
		if v == "" {
			log.Fatalf("%s required", name)
		}
		return v
	}
	openaiKey := mustEnv("OPENAI_API_KEY")
	anthropicKey := mustEnv("ANTHROPIC_API_KEY")
	geminiKey := mustEnv("GEMINI_API_KEY")

	ctx := context.Background()

	openaiProvider, err := openai.New(llmrouter.WithAPIKey(openaiKey))
	if err != nil {
		log.Fatalf("openai: %v", err)
	}
	anthropicProvider, err := anthropic.New(llmrouter.WithAPIKey(anthropicKey))
	if err != nil {
		log.Fatalf("anthropic: %v", err)
	}
	geminiProvider, err := gemini.New(llmrouter.WithAPIKey(geminiKey))
	if err != nil {
		log.Fatalf("gemini: %v", err)
	}

	runs := []struct {
		label    string
		provider llmrouter.Provider
		model    string
	}{
		{"openai", openaiProvider, "gpt-4o-mini"},
		{"anthropic", anthropicProvider, "claude-3-5-haiku-latest"},
		{"gemini", geminiProvider, "gemini-2.0-flash-exp"},
	}

	for _, run := range runs {
		fmt.Printf("=== %s (%s) ===\n", run.label, run.model)
		answer, err := completeOnce(ctx, run.provider, run.model)
		if err != nil {
			fmt.Printf("error: %v\n\n", err)
			continue
		}
		fmt.Println(strings.TrimSpace(answer))
		fmt.Println()
	}
}

// completeOnce drains a streaming completion into a single string. It
// blocks until the stream finishes or returns an error.
func completeOnce(ctx context.Context, provider llmrouter.Provider, model string) (string, error) {
	stream, err := provider.CompletionStream(ctx, llmrouter.ChatRequest{
		Model: model,
		Messages: []llmrouter.Message{
			llmrouter.TextMessage("user", prompt),
		},
		MaxTokens: 256,
	})
	if err != nil {
		return "", err
	}
	var buf strings.Builder
	for chunk := range stream.Chunks() {
		for _, choice := range chunk.Choices {
			buf.WriteString(choice.Delta.Content)
		}
	}
	if err := stream.Err(); err != nil {
		return "", err
	}
	return buf.String(), nil
}
