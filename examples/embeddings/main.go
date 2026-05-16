// embeddings embeds three short documents with OpenAI's
// text-embedding-3-small model and prints the first 8 dimensions of
// each returned vector.
//
// Requires:
//
//	OPENAI_API_KEY  — your OpenAI API key
//
// Usage:
//
//	go run ./examples/embeddings
//
// Output: three lines, one per input document, each showing the
// truncated vector prefix and the total dimensionality.
package main

import (
	"context"
	"fmt"
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

	provider, err := openai.New(llmrouter.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("build provider: %v", err)
	}

	docs := []string{
		"The mitochondrion is the powerhouse of the cell.",
		"Go's goroutines are lightweight threads managed by the runtime.",
		"A llama is a domesticated South American camelid.",
	}

	ctx := context.Background()
	resp, err := provider.Embed(ctx, llmrouter.EmbedRequest{
		Model:  "text-embedding-3-small",
		Inputs: docs,
	})
	if err != nil {
		log.Fatalf("embed: %v", err)
	}

	for i, doc := range docs {
		vec := resp.Embeddings[i]
		fmt.Printf("doc=%q\n  dims=%d first8=%v\n", doc, len(vec), vec[:min(8, len(vec))])
	}
	if resp.Usage != nil {
		fmt.Printf("usage: prompt_tokens=%d\n", resp.Usage.PromptTokens)
	}
}
