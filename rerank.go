// Package-doc additions live in llmrouter.go.
package llmrouter

import (
	"context"
	"encoding/json"
)

// Reranker re-orders a list of documents by relevance to a query. Used
// for RAG retrieval refinement. Implementations are concurrency-safe.
type Reranker interface {
	Rerank(ctx context.Context, req RerankRequest) (*RerankResponse, error)
}

// RerankRequest describes a rerank call.
type RerankRequest struct {
	// Model identifier (e.g. "rerank-v3.5", "rerank-english-v3.0").
	Model string `json:"model"`

	// Query is the search query the documents will be scored against.
	Query string `json:"query"`

	// Documents is the candidate list to re-rank.
	Documents []string `json:"documents"`

	// TopN, if > 0, limits the response to the top-N highest scoring
	// documents. Zero means return all.
	TopN int `json:"top_n,omitempty"`

	// ReturnDocuments, when true, asks the provider to echo the document
	// text back on each result (otherwise only the index is returned).
	ReturnDocuments bool `json:"return_documents,omitempty"`

	// User is the OpenAI-style end-user identifier.
	User string `json:"user,omitempty"`

	// Raw is forwarded for byte-passthrough callers; Model + Query +
	// Documents are overlaid.
	Raw json.RawMessage `json:"-"`
}

// RerankResponse carries the re-scored documents.
type RerankResponse struct {
	// Model is the resolved model id echoed by the provider.
	Model string `json:"model"`

	// Results is the re-ranked list. Each result.Index references the
	// position of the document in the original RerankRequest.Documents
	// slice. Sorted by RelevanceScore descending.
	Results []RerankResult `json:"results"`

	// Usage carries token counts when the provider returns them.
	Usage *Usage `json:"usage,omitempty"`

	// Raw is the original wire-format JSON for the response.
	Raw json.RawMessage `json:"-"`
}

// RerankResult is one entry in RerankResponse.Results.
type RerankResult struct {
	// Index into the original RerankRequest.Documents slice.
	Index int `json:"index"`

	// RelevanceScore in [0,1] — higher = more relevant.
	RelevanceScore float32 `json:"relevance_score"`

	// Document optionally carries the text back when caller asked for it.
	// Empty when ReturnDocuments was false.
	Document string `json:"document,omitempty"`
}
