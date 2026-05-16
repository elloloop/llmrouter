// chat-streaming streams a chat completion from OpenAI using a prompt
// read from stdin and prints the assistant delta tokens as they arrive.
//
// Requires:
//
//	OPENAI_API_KEY  — your OpenAI API key
//
// Usage:
//
//	echo "tell me a joke about goroutines" | go run ./examples/chat-streaming
//
// Output: the streamed assistant text, byte-by-byte, followed by a
// newline and a one-line usage summary (prompt/completion tokens).
package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/openai"
)

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY required")
	}

	prompt, err := io.ReadAll(bufio.NewReader(os.Stdin))
	if err != nil {
		log.Fatalf("read stdin: %v", err)
	}
	if len(prompt) == 0 {
		log.Fatal("no prompt on stdin (pipe one in: echo 'hi' | go run .)")
	}

	provider, err := openai.New(llmrouter.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("build provider: %v", err)
	}

	ctx := context.Background()
	stream, err := provider.CompletionStream(ctx, llmrouter.ChatRequest{
		Model: "gpt-4o-mini",
		Messages: []llmrouter.Message{
			llmrouter.TextMessage("user", string(prompt)),
		},
	})
	if err != nil {
		log.Fatalf("open stream: %v", err)
	}

	var usage *llmrouter.Usage
	for chunk := range stream.Chunks() {
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				fmt.Print(choice.Delta.Content)
			}
		}
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
	}
	if err := stream.Err(); err != nil {
		log.Fatalf("stream: %v", err)
	}
	fmt.Println()
	if usage != nil {
		fmt.Fprintf(os.Stderr, "usage: prompt=%d completion=%d total=%d\n",
			usage.PromptTokens, usage.CompletionTokens, usage.TotalTokens)
	}
}
