package deepseek

import (
	"context"
	"errors"

	"github.com/elloloop/llmrouter"
)

// Embed delegates to the underlying OpenAI-compatible /v1/embeddings
// endpoint at DeepSeek (e.g. the deepseek-embedding model family).
//
// Any *llmrouter.ErrUpstream surfaced by the inner provider has its
// Provider field rewritten from "openai" to "deepseek", matching the
// pattern used for chat completions so downstream code, logs, and
// metrics can attribute the failure correctly.
func (p *Provider) Embed(ctx context.Context, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	resp, err := p.inner.Embed(ctx, req)
	if err != nil {
		var ue *llmrouter.ErrUpstream
		if errors.As(err, &ue) && ue.Provider == "openai" {
			return nil, &llmrouter.ErrUpstream{
				Provider:   providerName,
				StatusCode: ue.StatusCode,
				Body:       ue.Body,
			}
		}
		return nil, err
	}
	return resp, nil
}
