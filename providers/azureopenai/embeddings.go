package azureopenai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"

	"github.com/elloloop/llmrouter"
)

// Embed implements llmrouter.Embedder against an Azure OpenAI
// deployment's /embeddings endpoint. The URL is deployment-scoped and
// the api-version query parameter is included. Auth is api-key OR AAD
// bearer, matching the existing chat path.
//
// Non-2xx responses are surfaced as *llmrouter.ErrUpstream.
func (p *Provider) Embed(ctx context.Context, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	body, err := buildEmbedRequestBody(req)
	if err != nil {
		return nil, err
	}

	endpoint := fmt.Sprintf("%s/openai/deployments/%s/embeddings?api-version=%s",
		strings.TrimRight(p.cfg.BaseURL, "/"), p.deployment, p.apiVersion)

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "application/json")
	if err := p.applyAuth(ctx, hreq); err != nil {
		return nil, err
	}

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		snippet := readUpstreamErrorBody(resp.Body)
		return nil, &llmrouter.ErrUpstream{
			Provider:   "azureopenai",
			StatusCode: resp.StatusCode,
			Body:       snippet,
		}
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("azureopenai: read embeddings body: %w", err)
	}
	return decodeEmbeddingsResponse(raw)
}

// applyAuth sets the Authorization or api-key header on the request,
// matching the resolution logic used by the chat path.
func (p *Provider) applyAuth(ctx context.Context, hreq *http.Request) error {
	if p.aadSource != nil {
		token, terr := p.aadSource(ctx)
		if terr != nil {
			return fmt.Errorf("azureopenai: AAD token source: %w", terr)
		}
		if strings.TrimSpace(token) == "" {
			return fmt.Errorf("azureopenai: AAD token source returned empty token")
		}
		hreq.Header.Set("Authorization", "Bearer "+token)
		return nil
	}
	hreq.Header.Set("api-key", p.cfg.APIKey)
	return nil
}

// buildEmbedRequestBody assembles the outgoing JSON body. The shape is
// identical to OpenAI's embeddings endpoint: Azure accepts the same
// payload (Model is overlaid even though the deployment in the URL
// selects the actual model).
func buildEmbedRequestBody(req llmrouter.EmbedRequest) ([]byte, error) {
	var m map[string]json.RawMessage
	if len(req.Raw) > 0 {
		if err := json.Unmarshal(req.Raw, &m); err != nil {
			return nil, fmt.Errorf("azureopenai: invalid raw embed request: %w", err)
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
	delete(m, "task_type")
	if req.TaskType != "" {
		tb, _ := json.Marshal(req.TaskType)
		m["task_type"] = tb
	}
	return json.Marshal(m)
}

// decodeEmbeddingsResponse parses the upstream JSON, sorting embeddings
// by index for safety, and preserves the raw bytes in Raw.
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
		return nil, fmt.Errorf("azureopenai: decode embeddings: %w", err)
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
