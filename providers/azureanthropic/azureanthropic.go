// Package azureanthropic implements llmrouter.Provider for Claude
// models hosted on Azure AI Foundry. The transport layer is Azure-style
// (deployment-scoped URL + api-version query + api-key OR AAD bearer),
// but the request body and SSE event stream are identical to direct
// Anthropic's /v1/messages API.
//
// Use providers/anthropic for direct api.anthropic.com access.
// Use providers/azureopenai for GPT models on Foundry.
// Use this package for Claude models on Foundry.
//
// Two URL variants — caller chooses:
//
//  1. Deployment-scoped (recommended):
//     https://<resource>.services.ai.azure.com/openai/deployments/<deployment>/messages?api-version=<v>
//     Pass WithDeployment("...") to opt in.
//
//  2. Resource-scoped (no deployment in URL):
//     https://<resource>.services.ai.azure.com/openai/v1/messages?api-version=<v>
//     Omit WithDeployment; the model name in the body identifies the variant.
//
// Auth is xor: provide exactly one of llmrouter.WithAPIKey or
// WithAADToken. API key flows as the Azure-specific "api-key" header
// (NOT "x-api-key"); AAD token flows as "Authorization: Bearer ...".
// The "anthropic-version" header is never sent — Azure manages the API
// version via the query parameter.
package azureanthropic

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/elloloop/llmrouter"
)

const (
	providerName      = "azureanthropic"
	defaultMaxTokens  = 4096
	scannerBufferSize = 1024 * 1024 // 1 MiB

	extraKeyDeployment = "azureanthropic.deployment"
	extraKeyAPIVersion = "azureanthropic.api-version"
	extraKeyAADSource  = "azureanthropic.aad-source"
)

// AADTokenSource is the caller-supplied function that mints an AAD
// bearer token. Called per request — implement caching and refresh in
// the source if the upstream cost matters.
type AADTokenSource = func(ctx context.Context) (string, error)

// Provider talks to Claude on Azure AI Foundry. Construct with New.
type Provider struct {
	cfg        *llmrouter.Config
	deployment string
	apiVersion string
	aadSource  AADTokenSource
}

// WithDeployment selects the Foundry deployment whose name appears in
// the URL. Optional: when omitted, requests target the resource-scoped
// /openai/v1/messages endpoint and the body's "model" field selects the
// Claude variant.
func WithDeployment(name string) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		name = strings.TrimSpace(name)
		if name == "" {
			return errors.New("azureanthropic: deployment name cannot be empty")
		}
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[extraKeyDeployment] = name
		return nil
	}
}

// WithAPIVersion sets the api-version query parameter. Required.
func WithAPIVersion(version string) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		version = strings.TrimSpace(version)
		if version == "" {
			return errors.New("azureanthropic: api-version cannot be empty")
		}
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[extraKeyAPIVersion] = version
		return nil
	}
}

// WithAADToken configures AAD bearer-token auth via a token source that
// is called once per request. Mutually exclusive with WithAPIKey.
func WithAADToken(source AADTokenSource) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		if source == nil {
			return errors.New("azureanthropic: AAD token source cannot be nil")
		}
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[extraKeyAADSource] = source
		return nil
	}
}

// New builds an azureanthropic Provider.
//
// Required options:
//   - llmrouter.WithBaseURL("https://<resource>.services.ai.azure.com")
//   - WithAPIVersion("2024-10-21")
//   - exactly one of llmrouter.WithAPIKey(...) OR WithAADToken(...)
//
// Optional:
//   - WithDeployment("my-deployment") — uses deployment-scoped URL.
//   - llmrouter.WithHTTPClient(...) / llmrouter.WithTimeout(...).
func New(opts ...llmrouter.Option) (*Provider, error) {
	cfg, err := llmrouter.NewConfig(opts...)
	if err != nil {
		return nil, err
	}

	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("%w: azureanthropic: base url required (https://<resource>.services.ai.azure.com)", llmrouter.ErrInvalidConfig)
	}

	apiVersion, err := requireExtraString(cfg, extraKeyAPIVersion, "api-version")
	if err != nil {
		return nil, err
	}

	deployment, _ := cfg.Extra[extraKeyDeployment].(string)
	aadSource, _ := cfg.Extra[extraKeyAADSource].(AADTokenSource)

	hasAPIKey := cfg.APIKey != ""
	hasAAD := aadSource != nil
	if hasAPIKey && hasAAD {
		return nil, fmt.Errorf("%w: azureanthropic: cannot specify both api key and AAD token source", llmrouter.ErrInvalidConfig)
	}
	if !hasAPIKey && !hasAAD {
		return nil, fmt.Errorf("%w: azureanthropic: api key or AAD token source required", llmrouter.ErrInvalidConfig)
	}

	return &Provider{
		cfg:        cfg,
		deployment: deployment,
		apiVersion: apiVersion,
		aadSource:  aadSource,
	}, nil
}

// requireExtraString fetches a required string value from cfg.Extra.
func requireExtraString(cfg *llmrouter.Config, key, label string) (string, error) {
	if cfg.Extra == nil {
		return "", fmt.Errorf("%w: azureanthropic: %s required", llmrouter.ErrInvalidConfig, label)
	}
	v, ok := cfg.Extra[key]
	if !ok {
		return "", fmt.Errorf("%w: azureanthropic: %s required", llmrouter.ErrInvalidConfig, label)
	}
	s, ok := v.(string)
	if !ok || strings.TrimSpace(s) == "" {
		return "", fmt.Errorf("%w: azureanthropic: %s must be a non-empty string", llmrouter.ErrInvalidConfig, label)
	}
	return s, nil
}

// Name returns the provider id.
func (p *Provider) Name() string { return providerName }

// CompletionStream issues a POST to the deployment-scoped /messages URL
// (or /openai/v1/messages when no deployment is configured) with the
// Anthropic body shape and translates the SSE response.
func (p *Provider) CompletionStream(ctx context.Context, req llmrouter.ChatRequest) (*llmrouter.Stream, error) {
	body, err := buildBody(req)
	if err != nil {
		return nil, fmt.Errorf("azureanthropic: build request: %w", err)
	}

	url := p.endpointURL()

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
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
		return nil, fmt.Errorf("azureanthropic: http: %w", err)
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 8*1024))
		return nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: resp.StatusCode,
			Body:       string(b),
		}
	}

	stream, sctx, hooks := llmrouter.NewStream(ctx)
	go pump(sctx, resp, req.Model, hooks)
	return stream, nil
}

// endpointURL composes the request URL. When a deployment is configured
// the deployment-scoped path is used; otherwise the resource-scoped
// /openai/v1/messages path. The api-version query parameter is always
// appended.
func (p *Provider) endpointURL() string {
	base := strings.TrimRight(p.cfg.BaseURL, "/")
	if p.deployment != "" {
		return fmt.Sprintf("%s/openai/deployments/%s/messages?api-version=%s",
			base, p.deployment, p.apiVersion)
	}
	return fmt.Sprintf("%s/openai/v1/messages?api-version=%s", base, p.apiVersion)
}

// applyAuth attaches either the api-key header or an AAD bearer token
// to the outgoing request. Exactly one path is active per Provider
// (enforced in New).
func (p *Provider) applyAuth(ctx context.Context, hreq *http.Request) error {
	if p.aadSource != nil {
		token, err := p.aadSource(ctx)
		if err != nil {
			return fmt.Errorf("azureanthropic: AAD token source: %w", err)
		}
		if strings.TrimSpace(token) == "" {
			return fmt.Errorf("azureanthropic: AAD token source returned empty token")
		}
		hreq.Header.Set("Authorization", "Bearer "+token)
		return nil
	}
	hreq.Header.Set("api-key", p.cfg.APIKey)
	return nil
}
