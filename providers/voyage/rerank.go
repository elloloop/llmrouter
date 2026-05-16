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

// defaultRerankModel is used when RerankRequest.Model is empty. rerank-2 is
// Voyage's flagship general-purpose rerank model.
const defaultRerankModel = "rerank-2"

// Rerank implements llmrouter.Reranker against Voyage AI's POST /v1/rerank.
//
// Translation notes:
//   - req.Model defaults to "rerank-2" when empty.
//   - req.TopN is sent as Voyage's "top_k" (Voyage's naming, NOT top_n).
//   - req.ReturnDocuments is sent as "return_documents".
//   - req.Raw, when set, is merged OVER the computed fields so vendor
//     extras (e.g. "truncation") survive and callers can override.
//
// Voyage stores each result's document text as a plain string at the top
// level of the result object (different from Cohere's nested shape).
//
// Non-2xx responses are surfaced as *llmrouter.ErrUpstream with up to 8 KiB
// of the response body retained for diagnostics.
func (p *Provider) Rerank(ctx context.Context, req llmrouter.RerankRequest) (*llmrouter.RerankResponse, error) {
	body, err := buildRerankRequestBody(req)
	if err != nil {
		return nil, err
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.cfg.BaseURL+"/v1/rerank", bytes.NewReader(body))
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
		return nil, fmt.Errorf("voyage: read rerank body: %w", err)
	}
	return decodeRerankResponse(raw)
}

// buildRerankRequestBody assembles the outgoing JSON body for /v1/rerank.
// When req.Raw is set, the raw fields are merged OVER the computed fields
// so vendor extras survive and callers can override anything.
func buildRerankRequestBody(req llmrouter.RerankRequest) ([]byte, error) {
	model := req.Model
	if model == "" {
		model = defaultRerankModel
	}
	m := map[string]any{
		"model":            model,
		"query":            req.Query,
		"documents":        req.Documents,
		"return_documents": req.ReturnDocuments,
	}
	if req.TopN > 0 {
		m["top_k"] = req.TopN
	}

	if len(req.Raw) > 0 {
		var overrides map[string]json.RawMessage
		if err := json.Unmarshal(req.Raw, &overrides); err != nil {
			return nil, fmt.Errorf("voyage: invalid raw rerank request: %w", err)
		}
		for k, v := range overrides {
			m[k] = v
		}
	}

	return json.Marshal(m)
}

// voyageRerankResult mirrors a single element of Voyage's /v1/rerank "data"
// array. Document is a plain string (not nested) at this level.
type voyageRerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float32 `json:"relevance_score"`
	Document       string  `json:"document"`
}

// decodeRerankResponse parses Voyage's /v1/rerank JSON into the normalized
// RerankResponse. Results are sorted by RelevanceScore descending. Usage
// is populated from the top-level usage.total_tokens field as PromptTokens
// — Voyage doesn't distinguish prompt vs completion on rerank.
func decodeRerankResponse(raw []byte) (*llmrouter.RerankResponse, error) {
	var wire struct {
		Data  []voyageRerankResult `json:"data"`
		Model string               `json:"model"`
		Usage *struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("voyage: decode rerank: %w", err)
	}

	results := make([]llmrouter.RerankResult, len(wire.Data))
	for i, d := range wire.Data {
		results[i] = llmrouter.RerankResult{
			Index:          d.Index,
			RelevanceScore: d.RelevanceScore,
			Document:       d.Document,
		}
	}
	sort.SliceStable(results, func(i, j int) bool {
		return results[i].RelevanceScore > results[j].RelevanceScore
	})

	out := &llmrouter.RerankResponse{
		Model:   wire.Model,
		Results: results,
		Raw:     json.RawMessage(raw),
	}
	if wire.Usage != nil {
		out.Usage = &llmrouter.Usage{
			PromptTokens: wire.Usage.TotalTokens,
			TotalTokens:  wire.Usage.TotalTokens,
		}
	}
	return out, nil
}
