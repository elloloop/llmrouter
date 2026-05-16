// rerank-cohere re-orders five candidate documents against a query
// using Cohere's rerank-v3.5 model and prints the relevance-sorted
// results.
//
// Requires:
//
//	COHERE_API_KEY  — your Cohere API key
//
// Usage:
//
//	go run ./examples/rerank-cohere
//
// Output: a list of results sorted by relevance, one per line, with the
// original index, the relevance score, and the document text echoed
// back.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/cohere"
)

func main() {
	apiKey := os.Getenv("COHERE_API_KEY")
	if apiKey == "" {
		log.Fatal("COHERE_API_KEY required")
	}

	provider, err := cohere.New(llmrouter.WithAPIKey(apiKey))
	if err != nil {
		log.Fatalf("build provider: %v", err)
	}

	query := "How do goroutines and channels work together?"
	docs := []string{
		"Goroutines are lightweight threads managed by the Go runtime; channels are typed conduits used to communicate between them.",
		"Llamas are domesticated South American camelids, often confused with alpacas.",
		"In Go, sync.Mutex protects shared state from concurrent access, complementing channel-based synchronization.",
		"The mitochondrion produces ATP and is referred to as the cell's powerhouse.",
		"Channels in Go can be buffered or unbuffered; unbuffered channels synchronize sender and receiver.",
	}

	ctx := context.Background()
	resp, err := provider.Rerank(ctx, llmrouter.RerankRequest{
		Model:           "rerank-v3.5",
		Query:           query,
		Documents:       docs,
		ReturnDocuments: true,
	})
	if err != nil {
		log.Fatalf("rerank: %v", err)
	}

	fmt.Printf("query: %s\n\n", query)
	for rank, r := range resp.Results {
		fmt.Printf("#%d  score=%.4f  orig_idx=%d\n  %s\n",
			rank+1, r.RelevanceScore, r.Index, r.Document)
	}
}
