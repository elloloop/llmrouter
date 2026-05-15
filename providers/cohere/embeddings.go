package cohere

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/elloloop/llmrouter"
)

// Embed implements llmrouter.Embedder against the Cohere /v2/embed endpoint.
//
// Translation notes:
//   - req.Inputs is forwarded as Cohere's "texts" field.
//   - req.TaskType (OpenAI/Vertex vocabulary) maps to Cohere's "input_type":
//     RETRIEVAL_QUERY -> search_query, RETRIEVAL_DOCUMENT -> search_document,
//     SEMANTIC_SIMILARITY -> classification, CLASSIFICATION -> classification,
//     CLUSTERING -> clustering. Anything else (including empty) defaults to
//     "search_document", Cohere's most-common use case.
//   - "embedding_types" is fixed to ["float"]; only the float vectors are
//     extracted from the response.
//   - req.Dimensions (when > 0) is sent as "output_dimension".
//   - req.Raw, when set, is reused as the outgoing body with Model and
//     texts overlaid, preserving any vendor-specific keys (e.g. truncate).
//
// Non-2xx responses are surfaced as *llmrouter.ErrUpstream with up to 8 KiB
// of the response body retained for diagnostics.
func (p *Provider) Embed(ctx context.Context, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	body, err := buildEmbedRequestBody(req)
	if err != nil {
		return nil, err
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.cfg.BaseURL+"/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "application/json")
	hreq.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("cohere: http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, errBodyCap))
		return nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: resp.StatusCode,
			Body:       string(b),
		}
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("cohere: read embed body: %w", err)
	}
	return decodeEmbedResponse(raw)
}

// buildEmbedRequestBody assembles the outgoing JSON body for /v2/embed.
// When req.Raw is set, the raw map is reused with model + texts overlaid
// so vendor extras (e.g. truncate, embedding_types overrides) survive.
func buildEmbedRequestBody(req llmrouter.EmbedRequest) ([]byte, error) {
	m := map[string]any{}
	if len(req.Raw) > 0 {
		if err := json.Unmarshal(req.Raw, &m); err != nil {
			return nil, fmt.Errorf("cohere: invalid raw embed request: %w", err)
		}
	}

	if req.Model != "" {
		m["model"] = req.Model
	}
	if len(req.Inputs) > 0 {
		m["texts"] = req.Inputs
	}
	m["input_type"] = mapTaskTypeToInputType(req.TaskType)
	if _, ok := m["embedding_types"]; !ok {
		m["embedding_types"] = []string{"float"}
	}
	if req.Dimensions > 0 {
		m["output_dimension"] = req.Dimensions
	}
	return json.Marshal(m)
}

// mapTaskTypeToInputType translates the OpenAI/Vertex TaskType vocabulary
// to Cohere's "input_type" enum. Unknown / empty values fall through to
// "search_document", which is Cohere's most-common default.
func mapTaskTypeToInputType(taskType string) string {
	switch taskType {
	case "RETRIEVAL_QUERY":
		return "search_query"
	case "RETRIEVAL_DOCUMENT":
		return "search_document"
	case "SEMANTIC_SIMILARITY":
		return "classification"
	case "CLASSIFICATION":
		return "classification"
	case "CLUSTERING":
		return "clustering"
	default:
		return "search_document"
	}
}

// decodeEmbedResponse parses Cohere's /v2/embed JSON into the normalized
// EmbedResponse. Only the float vectors are extracted; other embedding
// types in the response (int8, uint8, binary, ubinary) are ignored.
func decodeEmbedResponse(raw []byte) (*llmrouter.EmbedResponse, error) {
	var wire struct {
		ID         string `json:"id"`
		Model      string `json:"model"`
		Embeddings struct {
			Float [][]float32 `json:"float"`
		} `json:"embeddings"`
		BilledUnits *struct {
			InputTokens int `json:"input_tokens"`
		} `json:"billed_units"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("cohere: decode embed: %w", err)
	}

	out := &llmrouter.EmbedResponse{
		Model:      wire.Model,
		Embeddings: wire.Embeddings.Float,
		Raw:        json.RawMessage(raw),
	}
	if wire.BilledUnits != nil {
		out.Usage = &llmrouter.Usage{
			PromptTokens: wire.BilledUnits.InputTokens,
			TotalTokens:  wire.BilledUnits.InputTokens,
		}
	}
	if out.Embeddings == nil {
		out.Embeddings = [][]float32{}
	}
	return out, nil
}
