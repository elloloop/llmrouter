package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/elloloop/llmrouter"
)

const (
	// embedSingleSuffix is the URL suffix for the single-content embed
	// endpoint on the Gemini AI Studio (generativelanguage) API.
	embedSingleSuffix = ":embedContent"
	// embedBatchSuffix is the URL suffix for the batch-content embed endpoint.
	embedBatchSuffix = ":batchEmbedContents"

	// errBodyCap caps the bytes preserved from an upstream error response.
	errBodyCap = 8 * 1024
)

// Embed implements llmrouter.Embedder against Google AI Studio's embedding
// endpoints. Single-input requests hit :embedContent; multi-input requests
// hit :batchEmbedContents. TaskType (if non-empty) and Dimensions (if > 0)
// are passed verbatim.
//
// Gemini does NOT return token usage on embedding endpoints — Usage is nil.
// Non-2xx responses become *llmrouter.ErrUpstream with up to 8 KiB of body.
func (p *Provider) Embed(ctx context.Context, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	if req.Model == "" {
		return nil, fmt.Errorf("gemini: embed requires model")
	}
	if len(req.Inputs) == 0 {
		return nil, fmt.Errorf("gemini: embed requires at least one input")
	}
	if len(req.Inputs) == 1 {
		return p.embedSingle(ctx, req)
	}
	return p.embedBatch(ctx, req)
}

// embedSingle issues a POST to :embedContent for a single input string.
func (p *Provider) embedSingle(ctx context.Context, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	body, err := buildEmbedSingleBody(req)
	if err != nil {
		return nil, fmt.Errorf("gemini: build embed body: %w", err)
	}
	url := fmt.Sprintf("%s/models/%s%s", p.cfg.BaseURL, req.Model, embedSingleSuffix)
	raw, err := p.doEmbed(ctx, url, body)
	if err != nil {
		return nil, err
	}
	return decodeSingleResponse(req.Model, raw)
}

// embedBatch issues a POST to :batchEmbedContents for multi-input requests.
func (p *Provider) embedBatch(ctx context.Context, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	body, err := buildEmbedBatchBody(req)
	if err != nil {
		return nil, fmt.Errorf("gemini: build batch embed body: %w", err)
	}
	url := fmt.Sprintf("%s/models/%s%s", p.cfg.BaseURL, req.Model, embedBatchSuffix)
	raw, err := p.doEmbed(ctx, url, body)
	if err != nil {
		return nil, err
	}
	return decodeBatchResponse(req.Model, raw)
}

// doEmbed performs the POST + non-2xx handling. Returns the raw response
// bytes on success.
func (p *Provider) doEmbed(ctx context.Context, url string, body []byte) ([]byte, error) {
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "application/json")
	hreq.Header.Set(apiKeyHeader, p.cfg.APIKey)

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("gemini: http: %w", err)
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
	return io.ReadAll(resp.Body)
}

// buildEmbedSingleBody marshals a single-input embed body. The "model" path
// reference uses "models/<id>" per the Gemini contract. Raw is overlaid on
// top of the typed body so callers can inject vendor-specific extras.
func buildEmbedSingleBody(req llmrouter.EmbedRequest) ([]byte, error) {
	body := map[string]any{
		"model":   "models/" + req.Model,
		"content": textContent(req.Inputs[0]),
	}
	if req.TaskType != "" {
		body["taskType"] = req.TaskType
	}
	if req.Dimensions > 0 {
		body["outputDimensionality"] = req.Dimensions
	}
	return marshalWithRaw(body, req.Raw)
}

// buildEmbedBatchBody marshals a batch embed body, one inner request per
// input. TaskType is applied uniformly across inputs.
func buildEmbedBatchBody(req llmrouter.EmbedRequest) ([]byte, error) {
	inner := make([]map[string]any, 0, len(req.Inputs))
	for _, in := range req.Inputs {
		r := map[string]any{
			"model":   "models/" + req.Model,
			"content": textContent(in),
		}
		if req.TaskType != "" {
			r["taskType"] = req.TaskType
		}
		if req.Dimensions > 0 {
			r["outputDimensionality"] = req.Dimensions
		}
		inner = append(inner, r)
	}
	body := map[string]any{"requests": inner}
	return marshalWithRaw(body, req.Raw)
}

// textContent returns the {"parts":[{"text":...}]} content shape used by
// the Gemini embed endpoints.
func textContent(text string) map[string]any {
	return map[string]any{
		"parts": []map[string]any{{"text": text}},
	}
}

// marshalWithRaw overlays req.Raw vendor keys on top of the typed body so
// callers can inject vendor extras the typed API doesn't model.
func marshalWithRaw(body map[string]any, raw json.RawMessage) ([]byte, error) {
	if len(raw) > 0 {
		var extra map[string]json.RawMessage
		if err := json.Unmarshal(raw, &extra); err != nil {
			return nil, fmt.Errorf("invalid raw body: %w", err)
		}
		for k, v := range extra {
			if _, exists := body[k]; exists {
				continue
			}
			body[k] = v
		}
	}
	return json.Marshal(body)
}

// embedSingleWire mirrors {"embedding":{"values":[...]}}.
type embedSingleWire struct {
	Embedding struct {
		Values []float32 `json:"values"`
	} `json:"embedding"`
}

// embedBatchWire mirrors {"embeddings":[{"values":[...]}, ...]}.
type embedBatchWire struct {
	Embeddings []struct {
		Values []float32 `json:"values"`
	} `json:"embeddings"`
}

// decodeSingleResponse parses a :embedContent response and wraps it as an
// EmbedResponse with a single 1-element Embeddings slice.
func decodeSingleResponse(model string, raw []byte) (*llmrouter.EmbedResponse, error) {
	var w embedSingleWire
	if err := json.Unmarshal(raw, &w); err != nil {
		return nil, fmt.Errorf("gemini: decode embed response: %w", err)
	}
	return &llmrouter.EmbedResponse{
		Model:      model,
		Embeddings: [][]float32{w.Embedding.Values},
		Raw:        json.RawMessage(raw),
	}, nil
}

// decodeBatchResponse parses a :batchEmbedContents response. The output is
// index-aligned with the inputs.
func decodeBatchResponse(model string, raw []byte) (*llmrouter.EmbedResponse, error) {
	var w embedBatchWire
	if err := json.Unmarshal(raw, &w); err != nil {
		return nil, fmt.Errorf("gemini: decode batch embed response: %w", err)
	}
	out := make([][]float32, len(w.Embeddings))
	for i, e := range w.Embeddings {
		out[i] = e.Values
	}
	return &llmrouter.EmbedResponse{
		Model:      model,
		Embeddings: out,
		Raw:        json.RawMessage(raw),
	}, nil
}
