package together

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

const (
	// defaultRerankModel is used when RerankRequest.Model is empty.
	// Salesforce/Llama-Rank-V1 is Together's most popular general-purpose
	// rerank model.
	defaultRerankModel = "Salesforce/Llama-Rank-V1"

	// rerankErrBodyCap limits how much of an error response body is kept
	// on *llmrouter.ErrUpstream for diagnostics.
	rerankErrBodyCap = 8 * 1024
)

// Rerank implements llmrouter.Reranker against Together AI's POST /rerank
// (relative to the configured base URL, which defaults to
// https://api.together.xyz/v1).
//
// Together hosts Cohere/Salesforce/etc rerank models via this endpoint and
// follows Cohere's wire shape: results carry the document text nested
// inside an object — {"document": {"text": "..."}}.
//
// Translation notes:
//   - req.Model defaults to "Salesforce/Llama-Rank-V1" when empty.
//   - req.TopN is sent as Together's "top_n".
//   - req.ReturnDocuments is sent as "return_documents".
//   - req.Raw, when set, is merged OVER the computed fields.
//
// Non-2xx responses are surfaced as *llmrouter.ErrUpstream{Provider:"together"}.
func (p *Provider) Rerank(ctx context.Context, req llmrouter.RerankRequest) (*llmrouter.RerankResponse, error) {
	body, err := buildRerankRequestBody(req)
	if err != nil {
		return nil, err
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.baseURL+"/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "application/json")
	hreq.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.httpClient.Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("together: http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, rerankErrBodyCap))
		return nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: resp.StatusCode,
			Body:       string(b),
		}
	}

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("together: read rerank body: %w", err)
	}
	return decodeRerankResponse(raw, resolveRerankModel(req.Model))
}

// buildRerankRequestBody assembles the outgoing JSON body for /rerank.
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
			return nil, fmt.Errorf("together: invalid raw rerank request: %w", err)
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

// togetherRerankResult mirrors a single element of Together's /rerank
// "results" array. Together adopts Cohere's nested document shape:
// {"document": {"text": "..."}}.
type togetherRerankResult struct {
	Index          int     `json:"index"`
	RelevanceScore float32 `json:"relevance_score"`
	Document       *struct {
		Text string `json:"text"`
	} `json:"document"`
}

// decodeRerankResponse parses Together's /rerank JSON into the normalized
// RerankResponse. Results are sorted by RelevanceScore descending. The
// response does not echo the model, so the resolved request model is used.
func decodeRerankResponse(raw []byte, model string) (*llmrouter.RerankResponse, error) {
	var wire struct {
		Object  string                 `json:"object"`
		Results []togetherRerankResult `json:"results"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, fmt.Errorf("together: decode rerank: %w", err)
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
