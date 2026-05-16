package cohere

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

// defaultRerankModel is used when RerankRequest.Model is empty. rerank-v3.5
// is Cohere's flagship rerank model as of 2025.
const defaultRerankModel = "rerank-v3.5"

// Rerank implements llmrouter.Reranker against Cohere's POST /v2/rerank.
//
// Translation notes:
//   - req.Model defaults to "rerank-v3.5" when empty.
//   - req.TopN, when > 0, is sent as Cohere's "top_n".
//   - req.ReturnDocuments is sent as "return_documents".
//   - req.Raw, when set, is merged OVER the computed fields so vendor
//     extras (e.g. "max_chunks_per_doc") survive and callers can override.
//
// Cohere v2 nests document text inside an object: {"document": {"text": "..."}}.
// The decoder flattens that into RerankResult.Document. Cohere does not
// return token counts on rerank, so Usage stays nil.
//
// Non-2xx responses are surfaced as *llmrouter.ErrUpstream with up to 8 KiB
// of the response body retained for diagnostics.
func (p *Provider) Rerank(ctx context.Context, req llmrouter.RerankRequest) (*llmrouter.RerankResponse, error) {
	body, err := buildRerankRequestBody(req)
	if err != nil {
		return nil, err
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.cfg.BaseURL+"/rerank", bytes.NewReader(body))
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
		return nil, fmt.Errorf("cohere: read rerank body: %w", err)
	}
	return decodeRerankResponse(raw, resolveRerankModel(req.Model))
}

// buildRerankRequestBody assembles the outgoing JSON body for /v2/rerank.
// When req.Raw is set, the raw fields are merged OVER the computed fields
// so vendor extras survive and callers can override anything.
func buildRerankRequestBody(req llmrouter.RerankRequest) ([]byte, error) {
	m := map[string]any{
		"model":            resolveRerankModel(req.Model),
		"query":            req.Query,
		"documents":        req.Documents,
		"return_documents": req.ReturnDocuments,
	}
	if req.TopN > 0 {
		m["top_n"] = req.TopN
	}

	if len(req.Raw) > 0 {
		var overrides map[string]json.RawMessage
		if err := json.Unmarshal(req.Raw, &overrides); err != nil {
			return nil, fmt.Errorf("cohere: invalid raw rerank request: %w", err)
		}
		for k, v := range overrides {
			m[k] = v
		}
	}

	return json.Marshal(m)
}

// resolveRerankModel returns the supplied model or the default when empty.
func resolveRerankModel(model string) string {
	if model == "" {
		return defaultRerankModel
	}
	return model
}

// cohereRerankResult mirrors a single element of Cohere v2's /rerank
// "results" array. The document field is an object, not a plain string.
type cohereRerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float32 `json:"relevance_score"`
	Document       *struct {
		Text string `json:"text"`
	} `json:"document"`
}

// decodeRerankResponse parses Cohere's /v2/rerank JSON into the normalized
// RerankResponse. Results are sorted by RelevanceScore descending. The
// response does NOT echo the model, so the resolved request model is used.
func decodeRerankResponse(raw []byte, model string) (*llmrouter.RerankResponse, error) {
	var wire struct {
		ID      string               `json:"id"`
		Results []cohereRerankResult `json:"results"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("cohere: decode rerank: %w", err)
	}

	results := make([]llmrouter.RerankResult, len(wire.Results))
	for i, r := range wire.Results {
		out := llmrouter.RerankResult{
			Index:          r.Index,
			RelevanceScore: r.RelevanceScore,
		}
		if r.Document != nil {
			out.Document = r.Document.Text
		}
		results[i] = out
	}
	sort.SliceStable(results, func(i, j int) bool {
		return results[i].RelevanceScore > results[j].RelevanceScore
	})

	return &llmrouter.RerankResponse{
		Model:   model,
		Results: results,
		Raw:     json.RawMessage(raw),
	}, nil
}
