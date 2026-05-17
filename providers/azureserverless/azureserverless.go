// Package azureserverless implements llmrouter.Provider for non-OpenAI
// models hosted on Azure AI Foundry via the Serverless API
// (Models-as-a-Service, MaaS) deployment mode.
//
// Covers any OpenAI-compatible chat-completions model in Foundry's
// marketplace: Llama (Meta), Mistral, Cohere, Phi (Microsoft), Jais,
// Nemotron (NVIDIA), DeepSeek, etc.
//
// Use providers/azureopenai for OpenAI's own GPT models on Foundry
// (different URL shape: needs api-version query + /openai/deployments
// path).
// Use providers/azureanthropic for Claude on Foundry (different body
// shape: native Anthropic /messages).
// Use this package for everything else on Foundry.
//
// Two URL patterns are supported, both via llmrouter.WithBaseURL:
//
//  1. Deployment-scoped (single model per deployment):
//     https://<deployment-name>.<region>.models.ai.azure.com
//     The deployment + region are baked into the hostname; the chat
//     request's Model field is echoed back unchanged.
//
//  2. Hub-scoped (single hub, multiple models):
//     https://<hub-endpoint>
//     The Model field in the body picks which deployment.
//
// Authentication is either an Azure api-key header (Azure-style, not
// OpenAI's Authorization: Bearer) or an AAD bearer token via
// WithAADToken. Exactly one is required.
//
// Usage:
//
//	p, err := azureserverless.New(
//	    llmrouter.WithAPIKey("..."),
//	    llmrouter.WithBaseURL("https://my-llama.eastus.models.ai.azure.com"),
//	)
package azureserverless

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/elloloop/llmrouter"
)

// providerName is the stable id returned by Name().
const providerName = "azureserverless"

// Extra config keys used by this provider.
const (
	extraKeyAADSource = "azureserverless.aad-source"
)

// AADTokenSource produces a fresh AAD bearer token for each request.
// The source is invoked once per CompletionStream call so callers can
// refresh credentials transparently.
type AADTokenSource = func(ctx context.Context) (string, error)

// Provider implements llmrouter.Provider against Azure AI Foundry's
// Serverless API endpoints.
type Provider struct {
	cfg       *llmrouter.Config
	aadSource AADTokenSource
}

// WithAADToken configures AAD bearer-token auth via a token source that
// is invoked once per request. Mutually exclusive with
// llmrouter.WithAPIKey.
func WithAADToken(tokenSource AADTokenSource) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		if tokenSource == nil {
			return errors.New("azureserverless: AAD token source cannot be nil")
		}
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[extraKeyAADSource] = tokenSource
		return nil
	}
}

// New constructs an Azure Serverless provider. A base URL plus exactly
// one of llmrouter.WithAPIKey or WithAADToken is required. There is no
// default base URL — the caller must supply their deployment or hub
// hostname.
func New(opts ...llmrouter.Option) (*Provider, error) {
	cfg, err := llmrouter.NewConfig(opts...)
	if err != nil {
		return nil, err
	}

	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("%w: azureserverless: base url required (https://<deployment>.<region>.models.ai.azure.com or hub endpoint)", llmrouter.ErrInvalidConfig)
	}

	aadSource, _ := cfg.Extra[extraKeyAADSource].(AADTokenSource)
	hasAPIKey := cfg.APIKey != ""
	hasAAD := aadSource != nil
	if hasAPIKey && hasAAD {
		return nil, fmt.Errorf("%w: azureserverless: cannot specify both api key and AAD token source", llmrouter.ErrInvalidConfig)
	}
	if !hasAPIKey && !hasAAD {
		return nil, fmt.Errorf("%w: azureserverless: api key or AAD token source required", llmrouter.ErrInvalidConfig)
	}

	return &Provider{
		cfg:       cfg,
		aadSource: aadSource,
	}, nil
}

// Name returns the stable provider id.
func (p *Provider) Name() string { return providerName }

// CompletionStream opens a streaming chat completion against the
// configured Foundry Serverless endpoint and returns a llmrouter.Stream
// that yields normalized chunks. The wire format on the way out and
// back in is OpenAI-compatible; bytes are preserved end-to-end via
// Chunk.Raw for passthrough callers.
func (p *Provider) CompletionStream(ctx context.Context, req llmrouter.ChatRequest) (*llmrouter.Stream, error) {
	body, err := buildRequestBody(req)
	if err != nil {
		return nil, err
	}

	endpoint := strings.TrimRight(p.cfg.BaseURL, "/") + "/v1/chat/completions"

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")

	if err := p.applyAuth(ctx, hreq); err != nil {
		return nil, err
	}

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		snippet := readUpstreamErrorBody(resp.Body)
		resp.Body.Close()
		return nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: resp.StatusCode,
			Body:       snippet,
		}
	}

	stream, sctx, hooks := llmrouter.NewStream(ctx)
	go pumpSSE(sctx, resp, hooks)
	return stream, nil
}

// applyAuth sets the request's auth header. AAD takes precedence when
// configured; otherwise the api-key header is set. AAD sources are
// called per request so callers can refresh tokens transparently;
// errors and empty tokens fail loudly rather than silently producing
// an unauthenticated request.
func (p *Provider) applyAuth(ctx context.Context, hreq *http.Request) error {
	if p.aadSource != nil {
		token, terr := p.aadSource(ctx)
		if terr != nil {
			return fmt.Errorf("azureserverless: AAD token source: %w", terr)
		}
		if strings.TrimSpace(token) == "" {
			return fmt.Errorf("azureserverless: AAD token source returned empty token")
		}
		hreq.Header.Set("Authorization", "Bearer "+token)
		return nil
	}
	hreq.Header.Set("api-key", p.cfg.APIKey)
	return nil
}
