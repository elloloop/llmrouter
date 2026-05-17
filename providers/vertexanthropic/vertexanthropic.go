// Package vertexanthropic implements llmrouter.Provider for Claude
// models hosted on Google Vertex AI Model Garden. The transport layer
// is Vertex-style (ADC bearer token + project/region URL), but the
// request body and SSE event stream are native Anthropic.
//
// Use providers/anthropic for direct api.anthropic.com.
// Use providers/azureanthropic for Claude on Azure AI Foundry.
// Use providers/bedrock for Claude on AWS Bedrock (Converse API).
// Use this package for Claude on Vertex Model Garden.
//
// Endpoint shape:
//
//	POST https://<region>-aiplatform.googleapis.com/v1/projects/<project>/locations/<region>/publishers/anthropic/models/<model>:streamRawPredict
//
// Notes:
//   - Auth is GCP ADC: an OAuth2 access token sent as
//     "Authorization: Bearer <token>". Mint it via google.golang.org/api
//     or any equivalent — this package is intentionally a pure HTTP
//     provider and does NOT import the GCP SDK.
//   - Vertex requires the field "anthropic_version":"vertex-2023-10-16"
//     in the request body (not in a header). buildBody injects it.
//   - Model id uses an @-version suffix on Vertex
//     (e.g. "claude-3-5-sonnet-v2@20241022"), distinct from direct
//     Anthropic's dash-version suffix. Pass the Vertex form verbatim in
//     ChatRequest.Model — it is interpolated into the URL.
package vertexanthropic

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
	providerName       = "vertexanthropic"
	defaultMaxTokens   = 4096
	scannerBufferSize  = 1024 * 1024 // 1 MiB
	vertexAnthropicVer = "vertex-2023-10-16"

	extraKeyProject     = "vertexanthropic.project"
	extraKeyRegion      = "vertexanthropic.region"
	extraKeyTokenSource = "vertexanthropic.token-source"
	extraKeyAccessToken = "vertexanthropic.access-token"
)

// TokenSource is the caller-supplied function that mints a GCP access
// token. Called per request — implement caching and refresh inside the
// source if performance matters.
type TokenSource = func(ctx context.Context) (string, error)

// Provider talks to Claude on Vertex Model Garden. Construct with New.
type Provider struct {
	cfg         *llmrouter.Config
	project     string
	region      string
	tokenSource TokenSource
	accessToken string
}

// WithProject sets the GCP project that owns the Vertex deployment.
// Required.
func WithProject(project string) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		project = strings.TrimSpace(project)
		if project == "" {
			return errors.New("vertexanthropic: project cannot be empty")
		}
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[extraKeyProject] = project
		return nil
	}
}

// WithRegion sets the Vertex region (e.g. "us-central1"). Required.
func WithRegion(region string) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		region = strings.TrimSpace(region)
		if region == "" {
			return errors.New("vertexanthropic: region cannot be empty")
		}
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[extraKeyRegion] = region
		return nil
	}
}

// WithTokenSource configures ADC-style bearer-token auth via a token
// source called once per request. Preferred over WithAccessToken because
// the source can implement caching and refresh. Mutually exclusive with
// WithAccessToken.
func WithTokenSource(src TokenSource) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		if src == nil {
			return errors.New("vertexanthropic: token source cannot be nil")
		}
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[extraKeyTokenSource] = src
		return nil
	}
}

// WithAccessToken supplies a static GCP access token. Convenient for
// short-lived tokens (e.g. minted in CI). Mutually exclusive with
// WithTokenSource.
func WithAccessToken(token string) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		token = strings.TrimSpace(token)
		if token == "" {
			return errors.New("vertexanthropic: access token cannot be empty")
		}
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[extraKeyAccessToken] = token
		return nil
	}
}

// New builds a vertexanthropic Provider.
//
// Required options:
//   - WithProject("my-gcp-project")
//   - WithRegion("us-central1") or another Vertex-supported region
//   - exactly one of WithTokenSource(...) OR WithAccessToken(...)
//
// Optional:
//   - llmrouter.WithBaseURL("https://<custom>") — overrides the
//     regional default https://<region>-aiplatform.googleapis.com.
//   - llmrouter.WithHTTPClient(...) / llmrouter.WithTimeout(...).
//
// Rejects llmrouter.WithAPIKey explicitly — Vertex uses Google ADC, not
// API keys, and accepting one would silently misconfigure auth.
func New(opts ...llmrouter.Option) (*Provider, error) {
	cfg, err := llmrouter.NewConfig(opts...)
	if err != nil {
		return nil, err
	}

	if cfg.APIKey != "" {
		return nil, fmt.Errorf("%w: vertexanthropic: api key not supported (use WithTokenSource or WithAccessToken — Vertex requires GCP ADC)", llmrouter.ErrInvalidConfig)
	}

	project, err := requireExtraString(cfg, extraKeyProject, "project")
	if err != nil {
		return nil, err
	}
	region, err := requireExtraString(cfg, extraKeyRegion, "region")
	if err != nil {
		return nil, err
	}

	tokenSource, _ := cfg.Extra[extraKeyTokenSource].(TokenSource)
	accessToken, _ := cfg.Extra[extraKeyAccessToken].(string)

	hasSource := tokenSource != nil
	hasStatic := accessToken != ""
	if hasSource && hasStatic {
		return nil, fmt.Errorf("%w: vertexanthropic: cannot specify both token source and static access token", llmrouter.ErrInvalidConfig)
	}
	if !hasSource && !hasStatic {
		return nil, fmt.Errorf("%w: vertexanthropic: token source or access token required", llmrouter.ErrInvalidConfig)
	}

	return &Provider{
		cfg:         cfg,
		project:     project,
		region:      region,
		tokenSource: tokenSource,
		accessToken: accessToken,
	}, nil
}

// requireExtraString fetches a required string from cfg.Extra.
func requireExtraString(cfg *llmrouter.Config, key, label string) (string, error) {
	if cfg.Extra == nil {
		return "", fmt.Errorf("%w: vertexanthropic: %s required", llmrouter.ErrInvalidConfig, label)
	}
	v, ok := cfg.Extra[key]
	if !ok {
		return "", fmt.Errorf("%w: vertexanthropic: %s required", llmrouter.ErrInvalidConfig, label)
	}
	s, ok := v.(string)
	if !ok || strings.TrimSpace(s) == "" {
		return "", fmt.Errorf("%w: vertexanthropic: %s must be a non-empty string", llmrouter.ErrInvalidConfig, label)
	}
	return s, nil
}

// Name returns the provider id.
func (p *Provider) Name() string { return providerName }

// CompletionStream issues a POST to the Vertex :streamRawPredict
// endpoint for the configured project/region and the request's model,
// then translates the Anthropic SSE response into normalised
// llmrouter Chunks.
func (p *Provider) CompletionStream(ctx context.Context, req llmrouter.ChatRequest) (*llmrouter.Stream, error) {
	body, err := buildBody(req)
	if err != nil {
		return nil, fmt.Errorf("vertexanthropic: build request: %w", err)
	}

	url := p.endpointURL(req.Model)

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
		return nil, fmt.Errorf("vertexanthropic: http: %w", err)
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

// endpointURL composes the Vertex raw-predict URL for the given model.
// Honors a caller-supplied base URL override; otherwise derives the
// regional Vertex hostname.
func (p *Provider) endpointURL(model string) string {
	base := strings.TrimRight(p.cfg.BaseURL, "/")
	if base == "" {
		base = fmt.Sprintf("https://%s-aiplatform.googleapis.com", p.region)
	}
	return fmt.Sprintf(
		"%s/v1/projects/%s/locations/%s/publishers/anthropic/models/%s:streamRawPredict",
		base, p.project, p.region, model,
	)
}

// applyAuth attaches the Authorization: Bearer header using either the
// caller-supplied token source or the static access token (exactly one
// is active per Provider — enforced in New).
func (p *Provider) applyAuth(ctx context.Context, hreq *http.Request) error {
	token := p.accessToken
	if p.tokenSource != nil {
		t, err := p.tokenSource(ctx)
		if err != nil {
			return fmt.Errorf("vertexanthropic: token source: %w", err)
		}
		token = t
	}
	if strings.TrimSpace(token) == "" {
		return fmt.Errorf("vertexanthropic: token source returned empty token")
	}
	hreq.Header.Set("Authorization", "Bearer "+token)
	return nil
}
