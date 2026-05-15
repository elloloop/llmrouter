package vertex

import (
	"context"
	"errors"
	"fmt"

	"google.golang.org/genai"

	"github.com/elloloop/llmrouter"
)

// Embed implements llmrouter.Embedder against Vertex AI's EmbedContent
// method exposed by google.golang.org/genai. Each input is sent as a
// separate single-content embed request because the wire format requires
// one TaskType per Content and the SDK's slice API takes a single config.
//
// Vertex does not surface token usage on embeddings, so EmbedResponse.Usage
// is always nil. SDK errors are mapped to *llmrouter.ErrUpstream.
func (p *Provider) Embed(ctx context.Context, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	if req.Model == "" {
		return nil, fmt.Errorf("vertex: embed requires model")
	}
	if len(req.Inputs) == 0 {
		return nil, fmt.Errorf("vertex: embed requires at least one input")
	}

	cfg := buildEmbedConfig(req)

	out := make([][]float32, len(req.Inputs))
	for i, input := range req.Inputs {
		contents := []*genai.Content{{
			Parts: []*genai.Part{{Text: input}},
		}}
		resp, err := p.client.Models.EmbedContent(ctx, req.Model, contents, cfg)
		if err != nil {
			return nil, wrapSDKError(err)
		}
		values, err := extractFirstEmbedding(resp)
		if err != nil {
			return nil, fmt.Errorf("vertex: input %d: %w", i, err)
		}
		out[i] = values
	}

	return &llmrouter.EmbedResponse{
		Model:      req.Model,
		Embeddings: out,
	}, nil
}

// buildEmbedConfig assembles an EmbedContentConfig from the typed request.
// TaskType and OutputDimensionality are forwarded verbatim when supplied.
func buildEmbedConfig(req llmrouter.EmbedRequest) *genai.EmbedContentConfig {
	cfg := &genai.EmbedContentConfig{}
	if req.TaskType != "" {
		cfg.TaskType = req.TaskType
	}
	if req.Dimensions > 0 {
		dims := int32(req.Dimensions)
		cfg.OutputDimensionality = &dims
	}
	return cfg
}

// extractFirstEmbedding pulls the first ContentEmbedding values from a
// response, returning an error if the slice is empty or nil. (Vertex's
// EmbedContent returns one embedding per call when given one Content.)
func extractFirstEmbedding(resp *genai.EmbedContentResponse) ([]float32, error) {
	if resp == nil || len(resp.Embeddings) == 0 {
		return nil, errors.New("vertex: empty embeddings response")
	}
	emb := resp.Embeddings[0]
	if emb == nil {
		return nil, errors.New("vertex: nil embedding entry")
	}
	return emb.Values, nil
}
