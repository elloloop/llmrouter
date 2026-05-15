// Package openrouter is a thin wrapper around the openai provider that
// targets OpenRouter's OpenAI-compatible Chat Completions API.
//
// OpenRouter is fully OpenAI-compatible at the wire level
// (https://openrouter.ai/api/v1/chat/completions), so this package
// delegates all transport, SSE parsing, and chunk normalization to
// providers/openai. The only behavioural differences are:
//
//   - Name() returns "openrouter".
//   - The default base URL points at OpenRouter.
//   - WithReferer and WithTitle inject the OpenRouter attribution
//     headers (HTTP-Referer and X-Title) on every outbound request via
//     a wrapping RoundTripper. These are required by some
//     referrer-restricted models and recommended for analytics.
//   - *llmrouter.ErrUpstream values surfaced by the inner provider have
//     their Provider field rewritten from "openai" to "openrouter" so
//     downstream code, logs, and metrics can attribute the failure
//     correctly.
//
// Quick start:
//
//	p, err := openrouter.New(
//	    llmrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
//	    openrouter.WithReferer("https://example.com"),
//	    openrouter.WithTitle("My App"),
//	)
package openrouter

import (
	"context"
	"errors"
	"net/http"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/openai"
)

// DefaultBaseURL is OpenRouter's OpenAI-compatible Chat Completions endpoint.
const DefaultBaseURL = "https://openrouter.ai/api/v1"

// providerName is the stable id reported by Name() and stamped on any
// *llmrouter.ErrUpstream surfaced from the inner provider.
const providerName = "openrouter"

// Config.Extra keys used to thread OpenRouter-specific options through
// the generic option pipeline.
const (
	extraReferer = "openrouter.referer"
	extraTitle   = "openrouter.title"
)

// WithReferer sets the HTTP-Referer header sent on every outbound
// request. OpenRouter requires this for some referrer-restricted models
// and uses it for analytics. The value should be the URL of the app
// making the request.
func WithReferer(url string) llmrouter.Option {
	return llmrouter.WithExtra(extraReferer, url)
}

// WithTitle sets the X-Title header sent on every outbound request.
// OpenRouter uses this as a human-readable name in its analytics
// dashboards.
func WithTitle(title string) llmrouter.Option {
	return llmrouter.WithExtra(extraTitle, title)
}

// Provider implements llmrouter.Provider against OpenRouter by
// delegating to a configured openai.Provider.
type Provider struct {
	inner *openai.Provider
}

// New constructs an OpenRouter provider. The default base URL targets
// OpenRouter; pass llmrouter.WithBaseURL to override (e.g. for a
// self-hosted proxy).
//
// Use WithReferer and WithTitle to attach OpenRouter attribution
// headers; without them some referrer-restricted models will refuse to
// route the request.
func New(opts ...llmrouter.Option) (*Provider, error) {
	probe, err := llmrouter.NewConfig(opts...)
	if err != nil {
		return nil, err
	}
	referer, _ := probe.Extra[extraReferer].(string)
	title, _ := probe.Extra[extraTitle].(string)

	baseClient := probe.HTTP()
	wrapped := wrapHTTPClient(baseClient, referer, title)

	// Our defaults (BaseURL, wrapped HTTPClient) run first; caller
	// options run after and may override either.
	all := make([]llmrouter.Option, 0, len(opts)+2)
	all = append(all, llmrouter.WithBaseURL(DefaultBaseURL))
	all = append(all, llmrouter.WithHTTPClient(wrapped))
	all = append(all, opts...)

	inner, err := openai.New(all...)
	if err != nil {
		return nil, err
	}
	return &Provider{inner: inner}, nil
}

// Name returns the provider id.
func (p *Provider) Name() string { return providerName }

// CompletionStream delegates to the inner openai provider and rewrites
// any *llmrouter.ErrUpstream so its Provider field reads "openrouter".
func (p *Provider) CompletionStream(ctx context.Context, req llmrouter.ChatRequest) (*llmrouter.Stream, error) {
	stream, err := p.inner.CompletionStream(ctx, req)
	if err != nil {
		return nil, rewriteUpstreamProvider(err)
	}
	return stream, nil
}

// wrapHTTPClient returns a new *http.Client whose Transport injects the
// OpenRouter attribution headers before delegating to the original
// client's transport (or http.DefaultTransport when none was set).
// Timeout is preserved from the source client.
func wrapHTTPClient(src *http.Client, referer, title string) *http.Client {
	return &http.Client{
		Transport: &headerTransport{
			next:    src.Transport,
			referer: referer,
			title:   title,
		},
		Timeout: src.Timeout,
	}
}

// headerTransport is an http.RoundTripper that stamps the OpenRouter
// attribution headers (HTTP-Referer, X-Title) on every outbound request
// before delegating to next.
type headerTransport struct {
	next    http.RoundTripper
	referer string
	title   string
}

// RoundTrip implements http.RoundTripper. Empty referer or title values
// are skipped so callers can opt into either header independently.
func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if t.referer != "" {
		req.Header.Set("HTTP-Referer", t.referer)
	}
	if t.title != "" {
		req.Header.Set("X-Title", t.title)
	}
	next := t.next
	if next == nil {
		next = http.DefaultTransport
	}
	return next.RoundTrip(req)
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
