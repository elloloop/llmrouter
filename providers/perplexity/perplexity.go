// Package perplexity is a thin wrapper around the openai provider that
// targets Perplexity's OpenAI-compatible Chat Completions API.
//
// Perplexity exposes an OpenAI-compatible endpoint at
// https://api.perplexity.ai, so this package delegates all transport,
// SSE parsing, and chunk normalization to providers/openai. The only
// behavioural differences are:
//
//   - Name() returns "perplexity".
//   - The default base URL points at Perplexity.
//   - *llmrouter.ErrUpstream values surfaced by the inner provider have
//     their Provider field rewritten from "openai" to "perplexity" so
//     downstream code, logs, and metrics can attribute the failure
//     correctly.
//
// Quick start:
//
//	p, err := perplexity.New(llmrouter.WithAPIKey(os.Getenv("PERPLEXITY_API_KEY")))
package perplexity

import (
	"context"
	"errors"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/openai"
)

// DefaultBaseURL is Perplexity's OpenAI-compatible Chat Completions
// endpoint.
const DefaultBaseURL = "https://api.perplexity.ai"

// providerName is the stable id reported by Name() and stamped on any
// *llmrouter.ErrUpstream surfaced from the inner provider.
const providerName = "perplexity"

// Provider implements llmrouter.Provider against Perplexity by
// delegating to a configured openai.Provider.
type Provider struct {
	inner *openai.Provider
}

// New constructs a Perplexity provider. The default base URL targets
// Perplexity; pass llmrouter.WithBaseURL to override (e.g. for a
// self-hosted proxy).
func New(opts ...llmrouter.Option) (*Provider, error) {
	inner, err := openai.New(prependDefaultBaseURL(opts)...)
	if err != nil {
		return nil, err
	}
	return &Provider{inner: inner}, nil
}

// Name returns the provider id.
func (p *Provider) Name() string { return providerName }

// CompletionStream delegates to the inner openai provider and rewrites
// any *llmrouter.ErrUpstream so its Provider field reads "perplexity".
func (p *Provider) CompletionStream(ctx context.Context, req llmrouter.ChatRequest) (*llmrouter.Stream, error) {
	stream, err := p.inner.CompletionStream(ctx, req)
	if err != nil {
		return nil, rewriteUpstreamProvider(err)
	}
	return stream, nil
}

// prependDefaultBaseURL returns a new option slice with our default base
// URL applied first; any user-supplied llmrouter.WithBaseURL runs later
// and overrides it.
func prependDefaultBaseURL(opts []llmrouter.Option) []llmrouter.Option {
	out := make([]llmrouter.Option, 0, len(opts)+1)
	out = append(out, llmrouter.WithBaseURL(DefaultBaseURL))
	out = append(out, opts...)
	return out
}

// rewriteUpstreamProvider returns err with any *llmrouter.ErrUpstream
// reassigned to providerName when its Provider field is "openai". Other
// errors pass through untouched.
func rewriteUpstreamProvider(err error) error {
	var ue *llmrouter.ErrUpstream
	if errors.As(err, &ue) && ue.Provider == "openai" {
		return &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: ue.StatusCode,
			Body:       ue.Body,
		}
	}
	return err
}
