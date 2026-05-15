// Package together is a thin wrapper around the openai provider that
// targets Together AI's OpenAI-compatible Chat Completions API.
//
// Together AI exposes a fully OpenAI-compatible endpoint at
// https://api.together.xyz/v1, so this package delegates all transport,
// SSE parsing, and chunk normalization to providers/openai. The only
// behavioural differences are:
//
//   - Name() returns "together".
//   - The default base URL points at Together AI.
//   - *llmrouter.ErrUpstream values surfaced by the inner provider have
//     their Provider field rewritten from "openai" to "together" so
//     downstream code, logs, and metrics can attribute the failure
//     correctly.
//
// Quick start:
//
//	p, err := together.New(llmrouter.WithAPIKey(os.Getenv("TOGETHER_API_KEY")))
package together

import (
	"context"
	"errors"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/openai"
)

// DefaultBaseURL is Together AI's OpenAI-compatible Chat Completions
// endpoint.
const DefaultBaseURL = "https://api.together.xyz/v1"

// providerName is the stable id reported by Name() and stamped on any
// *llmrouter.ErrUpstream surfaced from the inner provider.
const providerName = "together"

// Provider implements llmrouter.Provider against Together AI by
// delegating to a configured openai.Provider.
type Provider struct {
	inner *openai.Provider
}

// New constructs a Together AI provider. The default base URL targets
// Together AI; pass llmrouter.WithBaseURL to override (e.g. for a
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
// any *llmrouter.ErrUpstream so its Provider field reads "together".
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
