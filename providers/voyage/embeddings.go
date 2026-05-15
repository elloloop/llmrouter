package voyage

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"

	"github.com/elloloop/llmrouter"
)

// defaultModel is used when EmbedRequest.Model is empty. voyage-3 is the
// general-purpose flagship model recommended for most retrieval workloads.
const defaultModel = "voyage-3"

// Embed implements llmrouter.Embedder against Voyage AI's POST /v1/embeddings.
//
// Translation notes:
//   - req.Inputs is forwarded as Voyage's "input" field.
//   - req.Model defaults to "voyage-3" when empty.
//   - req.TaskType (OpenAI/Vertex vocabulary) maps to Voyage's "input_type":
//     RETRIEVAL_QUERY, QUESTION_ANSWERING, FACT_VERIFICATION -> "query"
//     RETRIEVAL_DOCUMENT, CLUSTERING, CLASSIFICATION,
//     SEMANTIC_SIMILARITY -> "document"
//     Anything else (including empty) -> input_type omitted, letting
//     Voyage apply its own default.
//   - req.Dimensions (when > 0) is sent as "output_dimension".
//   - req.EncodingFormat (when set) is forwarded as "encoding_format".
//   - req.Raw, when set, is reused as the outgoing body and merged OVER the
//     fields above, so vendor-specific keys (e.g. truncation) survive and
//     callers can override anything.
//
// Non-2xx responses are surfaced as *llmrouter.ErrUpstream with up to 8 KiB
// of the response body retained for diagnostics.
func (p *Provider) Embed(ctx context.Context, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	body, err := buildEmbedRequestBody(req)
	if err != nil {
		return nil, err
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.cfg.BaseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "application/json")
	hreq.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("voyage: http: %w", err)
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
		return nil, fmt.Errorf("voyage: read embed body: %w", err)
	}
	return decodeEmbedResponse(raw)
}

// buildEmbedRequestBody assembles the outgoing JSON body for /v1/embeddings.
// Fields the caller did NOT set are omitted entirely so Voyage applies its
// own defaults. When req.Raw is provided it is merged OVER the computed
// fields, allowing callers full control.
func buildEmbedRequestBody(req llmrouter.EmbedRequest) ([]byte, error) {
	m := map[string]any{}

	if len(req.Inputs) > 0 {
		m["input"] = req.Inputs
	}

	model := req.Model
	if model == "" {
		model = defaultModel
	}
	m["model"] = model

	if inputType := mapTaskTypeToInputType(req.TaskType); inputType != "" {
		m["input_type"] = inputType
	}
	if req.Dimensions > 0 {
		m["output_dimension"] = req.Dimensions
	}
	if req.EncodingFormat != "" {
		m["encoding_format"] = req.EncodingFormat
	}

	if len(req.Raw) > 0 {
		var overrides map[string]json.RawMessage
		if err := json.Unmarshal(req.Raw, &overrides); err != nil {
			return nil, fmt.Errorf("voyage: invalid raw embed request: %w", err)
		}
		for k, v := range overrides {
			m[k] = v
		}
	}

	return json.Marshal(m)
}

// mapTaskTypeToInputType translates the OpenAI/Vertex TaskType vocabulary
// to Voyage's "input_type" enum. Returns "" for empty/unknown values so the
// caller knows to omit the field from the outgoing body.
func mapTaskTypeToInputType(taskType string) string {
	switch taskType {
	case "RETRIEVAL_QUERY", "QUESTION_ANSWERING", "FACT_VERIFICATION":
		return "query"
	case "RETRIEVAL_DOCUMENT", "CLUSTERING", "CLASSIFICATION", "SEMANTIC_SIMILARITY":
		return "document"
	default:
		return ""
	}
}

// voyageEmbeddingDatum mirrors a single element of Voyage's response "data" array.
type voyageEmbeddingDatum struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// decodeEmbedResponse parses Voyage's /v1/embeddings JSON into the normalized
// EmbedResponse. The "data" array is sorted by index so the returned slice
// is positionally aligned with the request inputs even if the API returns
// items out of order.
func decodeEmbedResponse(raw []byte) (*llmrouter.EmbedResponse, error) {
	var wire struct {
		Object string                 `json:"object"`
		Data   []voyageEmbeddingDatum `json:"data"`
		Model  string                 `json:"model"`
		Usage  *struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("voyage: decode embed: %w", err)
	}

	sort.Slice(wire.Data, func(i, j int) bool {
		return wire.Data[i].Index < wire.Data[j].Index
	})

	embeddings := make([][]float32, len(wire.Data))
	for i, d := range wire.Data {
		embeddings[i] = d.Embedding
	}

	out := &llmrouter.EmbedResponse{
		Model:      wire.Model,
		Embeddings: embeddings,
		Raw:        json.RawMessage(raw),
	}
	if wire.Usage != nil {
		out.Usage = &llmrouter.Usage{
			PromptTokens: wire.Usage.TotalTokens,
			TotalTokens:  wire.Usage.TotalTokens,
		}
	}
	return out, nil
}
