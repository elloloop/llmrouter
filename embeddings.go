// Package-doc additions live in llmrouter.go.
package llmrouter

import (
	"context"
	"encoding/json"
)

// Embedder produces vector embeddings for one or more text inputs.
// Implementations are concurrency-safe.
type Embedder interface {
	Embed(ctx context.Context, req EmbedRequest) (*EmbedResponse, error)
}

// EmbedRequest is the polymorphic embedding-request shape. Vendor-specific
// extras travel through Raw (byte passthrough) or Extras (typed map).
type EmbedRequest struct {
	// Model identifier ("text-embedding-3-small", "embed-english-v3.0", ...).
	Model string `json:"model"`

	// Inputs is the list of strings to embed. Most providers accept arrays.
	// For single-string providers (some legacy endpoints) the implementation
	// batches client-side.
	Inputs []string `json:"input,omitempty"`

	// Dimensions optionally requests a lower-dimensional output (supported
	// by OpenAI text-embedding-3-* and a few others). Zero means default.
	Dimensions int `json:"dimensions,omitempty"`

	// TaskType is the Gemini/Vertex/Voyage task hint
	// ("RETRIEVAL_QUERY" | "RETRIEVAL_DOCUMENT" | "SEMANTIC_SIMILARITY" |
	//  "CLASSIFICATION" | "CLUSTERING" | "QUESTION_ANSWERING" |
	//  "FACT_VERIFICATION"). Empty means provider default.
	TaskType string `json:"task_type,omitempty"`

	// EncodingFormat is "float" (default) or "base64". Providers that only
	// support one format ignore this.
	EncodingFormat string `json:"encoding_format,omitempty"`

	// User is the OpenAI-style end-user identifier (telemetry/abuse).
	User string `json:"user,omitempty"`

	// Raw, if non-nil, is forwarded as the outgoing JSON body with Model
	// overlaid. Use this for vendor-specific fields not modelled above
	// (Cohere's input_type, Voyage's input_type, etc.).
	Raw json.RawMessage `json:"-"`
}

// EmbedResponse carries the vectors returned by the provider, plus token
// usage when available.
type EmbedResponse struct {
	// Model is the resolved model id echoed by the provider.
	Model string `json:"model"`

	// Embeddings is index-aligned with EmbedRequest.Inputs: the embedding
	// at position i corresponds to input i. Length equals len(req.Inputs).
	Embeddings [][]float32 `json:"embeddings"`

	// Usage carries token counts when the provider returns them. May be nil.
	// Embedding providers populate PromptTokens (sometimes also TotalTokens).
	Usage *Usage `json:"usage,omitempty"`

	// Raw is the original wire-format JSON. Provided so passthrough callers
	// can forward bytes without re-marshaling.
	Raw json.RawMessage `json:"-"`
}
