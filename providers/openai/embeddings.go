package openai

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

// Embed implements llmrouter.Embedder against the OpenAI /embeddings
// endpoint. The Raw field of the request is honoured (vendor extras are
// preserved); Model and Input are always overlaid from the typed fields.
//
// Non-2xx responses are surfaced as *llmrouter.ErrUpstream with up to
// 1 KiB of the response body retained for diagnostics.
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
			Provider:   "openai",
			StatusCode: resp.StatusCode,
			Body:       snippet,
		}
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("openai: read embeddings body: %w", err)
	}
	return decodeEmbeddingsResponse(raw)
}

// buildEmbedRequestBody assembles the outgoing JSON. If req.Raw is set, it
// is reused with Model + Input overlaid from the typed fields, preserving
// any vendor-specific keys. Otherwise, the typed request is marshaled.
func buildEmbedRequestBody(req llmrouter.EmbedRequest) ([]byte, error) {
	var m map[string]json.RawMessage
	if len(req.Raw) > 0 {
		if err := json.Unmarshal(req.Raw, &m); err != nil {
			return nil, fmt.Errorf("openai: invalid raw embed request: %w", err)
		}
	} else {
		raw, err := json.Marshal(req)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, err
		}
	}

	if req.Model != "" {
		mb, err := json.Marshal(req.Model)
		if err != nil {
			return nil, err
		}
		m["model"] = mb
	}
	if len(req.Inputs) > 0 {
		ib, err := json.Marshal(req.Inputs)
		if err != nil {
			return nil, err
		}
		m["input"] = ib
	}
	// Strip nil/empty fields that json.Marshal of EmbedRequest may emit
	// but that we don't want to forward unless explicitly set.
	delete(m, "task_type")
	if req.TaskType != "" {
		tb, _ := json.Marshal(req.TaskType)
		m["task_type"] = tb
	}
	return json.Marshal(m)
}

// decodeEmbeddingsResponse parses the OpenAI embeddings JSON into the
// normalized EmbedResponse, sorting embeddings by index for safety.
func decodeEmbeddingsResponse(raw []byte) (*llmrouter.EmbedResponse, error) {
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
		return nil, fmt.Errorf("openai: decode embeddings: %w", err)
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
