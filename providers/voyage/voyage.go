// Package voyage implements the llmrouter.Embedder interface against
// Voyage AI's /v1/embeddings API. Voyage AI is the Anthropic-recommended
// embeddings path for Claude users and was recently acquired by MongoDB.
//
// The provider exposes only the Embed method — Voyage is an embeddings-only
// service and does not offer a chat/completions endpoint.
package voyage

import (
	"fmt"

	"github.com/elloloop/llmrouter"
)

const (
	// DefaultBaseURL is Voyage AI's production API host.
	DefaultBaseURL = "https://api.voyageai.com"
	providerName   = "voyage"
	errBodyCap     = 8 * 1024 // 8 KiB
)

// Provider talks to Voyage AI's /v1/embeddings endpoint. Construct with New.
type Provider struct {
	cfg *llmrouter.Config
}

// New builds a Provider from llmrouter options. WithAPIKey is required.
// BaseURL defaults to DefaultBaseURL when not supplied.
func New(opts ...llmrouter.Option) (*Provider, error) {
	cfg, err := llmrouter.NewConfig(opts...)
	if err != nil {
		return nil, err
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("%w: voyage requires an api key", llmrouter.ErrInvalidConfig)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	return &Provider{cfg: cfg}, nil
}

// Name returns the provider id.
func (p *Provider) Name() string { return providerName }
