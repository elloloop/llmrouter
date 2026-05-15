package mistral

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

// Embed implements llmrouter.Embedder against Mistral's /embeddings
// endpoint, which is OpenAI-shaped at the wire level (same request body,
// same response body). Notes:
//   - req.TaskType is ignored (Mistral has no task-type concept). It is
//     deliberately stripped from any caller-supplied Raw body too.
//   - req.Dimensions is dropped: Mistral's embedding endpoint does not
//     support the "dimensions" parameter as of late 2025. Stripped from
//     Raw bodies as well to avoid surprise 400s.
//   - req.Raw is honoured with Model and input overlaid.
//
// Non-2xx responses are surfaced as *llmrouter.ErrUpstream with a snippet
// of the response body retained for diagnostics.
func (p *Provider) Embed(ctx context.Context, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	body, err := buildEmbedRequestBody(req)
	if err != nil {
		return nil, err
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.cfg.BaseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "application/json")
	hreq.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		snippet := readUpstreamErrorBody(resp.Body)
		return nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: resp.StatusCode,
			Body:       snippet,
		}
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("mistral: read embeddings body: %w", err)
	}
	return decodeEmbedResponse(raw)
}

// buildEmbedRequestBody assembles the outgoing JSON for /embeddings.
// task_type and dimensions are stripped — Mistral does not support either
// and a stray field can produce a 400. encoding_format defaults to "float".
func buildEmbedRequestBody(req llmrouter.EmbedRequest) ([]byte, error) {
	m := map[string]any{}
	if len(req.Raw) > 0 {
		if err := json.Unmarshal(req.Raw, &m); err != nil {
			return nil, fmt.Errorf("mistral: invalid raw embed request: %w", err)
		}
	}

	if req.Model != "" {
		m["model"] = req.Model
	}
	if len(req.Inputs) > 0 {
		m["input"] = req.Inputs
	}
	if _, ok := m["encoding_format"]; !ok {
		format := req.EncodingFormat
		if format == "" {
			format = "float"
		}
		m["encoding_format"] = format
	}
	// Mistral does not understand these — drop unconditionally so callers
	// can't accidentally trigger upstream 400s by setting them.
	delete(m, "task_type")
	delete(m, "dimensions")
	return json.Marshal(m)
}

// decodeEmbedResponse parses the Mistral OpenAI-shape embeddings response,
// sorting vectors by index for safety.
func decodeEmbedResponse(raw []byte) (*llmrouter.EmbedResponse, error) {
	var wire struct {
		Object string `json:"object"`
		Model  string `json:"model"`
		Data   []struct {
			Object    string    `json:"object"`
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
		Usage *struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("mistral: decode embeddings: %w", err)
	}

	sort.SliceStable(wire.Data, func(i, j int) bool {
		return wire.Data[i].Index < wire.Data[j].Index
	})
	embeddings := make([][]float32, 0, len(wire.Data))
	for _, d := range wire.Data {
		embeddings = append(embeddings, d.Embedding)
	}

	out := &llmrouter.EmbedResponse{
		Model:      wire.Model,
		Embeddings: embeddings,
		Raw:        json.RawMessage(raw),
	}
	if wire.Usage != nil {
		out.Usage = &llmrouter.Usage{
			PromptTokens: wire.Usage.PromptTokens,
			TotalTokens:  wire.Usage.TotalTokens,
		}
	}
	return out, nil
}
