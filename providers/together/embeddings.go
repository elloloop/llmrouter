package together

import (
	"context"

	"github.com/elloloop/llmrouter"
)

// Embed delegates to the inner openai provider — Together AI's
// /v1/embeddings endpoint is OpenAI-shaped, so no translation is required.
// Any *llmrouter.ErrUpstream surfaced by the inner provider has its
// Provider field rewritten from "openai" to "together", matching the
// pattern used for chat completions.
func (p *Provider) Embed(ctx context.Context, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	resp, err := p.inner.Embed(ctx, req)
	if err != nil {
		return nil, rewriteUpstreamProvider(err)
	}
	return resp, nil
}
