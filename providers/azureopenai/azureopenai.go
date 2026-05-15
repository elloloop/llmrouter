// Package azureopenai implements llmrouter.Provider against the Azure
// OpenAI Service. Unlike vanilla OpenAI, Azure scopes the URL by
// deployment (not model), requires an api-version query parameter, and
// authenticates via either an api-key header or an AAD bearer token.
//
// Usage:
//
//	p, err := azureopenai.New(
//	    llmrouter.WithAPIKey("..."),
//	    llmrouter.WithBaseURL("https://my-resource.openai.azure.com"),
//	    azureopenai.WithDeployment("my-deploy"),
//	    azureopenai.WithAPIVersion("2024-10-21"),
//	)
//
// The chat request's Model field is ignored for routing (the deployment
// in the URL selects the model) but is echoed back in returned chunks.
package azureopenai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/elloloop/llmrouter"
)

// Extra config keys used by this provider.
const (
	extraKeyDeployment = "azureopenai.deployment"
	extraKeyAPIVersion = "azureopenai.api-version"
	extraKeyAADSource  = "azureopenai.aad-source"
)

// AADTokenSource produces a fresh AAD bearer token for each request.
type AADTokenSource func(ctx context.Context) (string, error)

// Provider implements llmrouter.Provider against Azure OpenAI.
type Provider struct {
	cfg        *llmrouter.Config
	deployment string
	apiVersion string
	aadSource  AADTokenSource
}

// WithDeployment selects the Azure OpenAI deployment name that appears in
// the URL path. Required.
func WithDeployment(name string) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		name = strings.TrimSpace(name)
		if name == "" {
			return errors.New("azureopenai: deployment name cannot be empty")
		}
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[extraKeyDeployment] = name
		return nil
	}
}

// WithAPIVersion sets the api-version query parameter. Required (Azure
// rejects requests without it).
func WithAPIVersion(version string) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		version = strings.TrimSpace(version)
		if version == "" {
			return errors.New("azureopenai: api-version cannot be empty")
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
func WithAADToken(tokenSource AADTokenSource) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		if tokenSource == nil {
			return errors.New("azureopenai: AAD token source cannot be nil")
		}
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[extraKeyAADSource] = tokenSource
		return nil
	}
}

// New constructs an Azure OpenAI provider. APIKey or WithAADToken is
// required (exactly one); deployment, base URL, and api-version are also
// required.
func New(opts ...llmrouter.Option) (*Provider, error) {
	cfg, err := llmrouter.NewConfig(opts...)
	if err != nil {
		return nil, err
	}

	deployment, err := requireExtraString(cfg, extraKeyDeployment, "deployment")
	if err != nil {
		return nil, err
	}
	apiVersion, err := requireExtraString(cfg, extraKeyAPIVersion, "api-version")
	if err != nil {
		return nil, err
	}
	if cfg.BaseURL == "" {
		return nil, fmt.Errorf("%w: azureopenai: base url required (https://<resource>.openai.azure.com)", llmrouter.ErrInvalidConfig)
	}

	aadSource, _ := cfg.Extra[extraKeyAADSource].(AADTokenSource)
	hasAPIKey := cfg.APIKey != ""
	hasAAD := aadSource != nil
	if hasAPIKey && hasAAD {
		return nil, fmt.Errorf("%w: azureopenai: cannot specify both api key and AAD token source", llmrouter.ErrInvalidConfig)
	}
	if !hasAPIKey && !hasAAD {
		return nil, fmt.Errorf("%w: azureopenai: api key or AAD token source required", llmrouter.ErrInvalidConfig)
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
		return "", fmt.Errorf("%w: azureopenai: %s required", llmrouter.ErrInvalidConfig, label)
	}
	v, ok := cfg.Extra[key]
	if !ok {
		return "", fmt.Errorf("%w: azureopenai: %s required", llmrouter.ErrInvalidConfig, label)
	}
	s, ok := v.(string)
	if !ok || strings.TrimSpace(s) == "" {
		return "", fmt.Errorf("%w: azureopenai: %s must be a non-empty string", llmrouter.ErrInvalidConfig, label)
	}
	return s, nil
}

// Name returns the provider id.
func (p *Provider) Name() string { return "azureopenai" }

// CompletionStream opens a streaming chat completion against the Azure
// OpenAI deployment and returns a llmrouter.Stream that yields normalized
// chunks.
func (p *Provider) CompletionStream(ctx context.Context, req llmrouter.ChatRequest) (*llmrouter.Stream, error) {
	body, err := buildRequestBody(req)
	if err != nil {
		return nil, err
	}

	endpoint := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s",
		strings.TrimRight(p.cfg.BaseURL, "/"), p.deployment, p.apiVersion)

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")

	if p.aadSource != nil {
		token, terr := p.aadSource(ctx)
		if terr != nil {
			return nil, fmt.Errorf("azureopenai: AAD token source: %w", terr)
		}
		if strings.TrimSpace(token) == "" {
			return nil, fmt.Errorf("azureopenai: AAD token source returned empty token")
		}
		hreq.Header.Set("Authorization", "Bearer "+token)
	} else {
		hreq.Header.Set("api-key", p.cfg.APIKey)
	}

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		snippet := readUpstreamErrorBody(resp.Body)
		resp.Body.Close()
		return nil, &llmrouter.ErrUpstream{
			Provider:   "azureopenai",
			StatusCode: resp.StatusCode,
			Body:       snippet,
		}
	}

	stream, sctx, hooks := llmrouter.NewStream(ctx)
	go pumpSSE(sctx, resp, hooks)
	return stream, nil
}

// buildRequestBody assembles the outgoing JSON. If req.Raw is supplied,
// it is reused (passthrough); otherwise the typed ChatRequest is
// marshaled. In both cases stream:true and stream_options.include_usage
// are forced on. The Model field is overlaid only when non-empty — Azure
// ignores it for routing but echoes it back in chunks.
func buildRequestBody(req llmrouter.ChatRequest) ([]byte, error) {
	var m map[string]json.RawMessage
	if len(req.Raw) > 0 {
		if err := json.Unmarshal(req.Raw, &m); err != nil {
			return nil, fmt.Errorf("azureopenai: invalid raw request: %w", err)
		}
	} else {
		typed := req
		typed.Stream = true
		raw, err := json.Marshal(typed)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(raw, &m); err != nil {
			return nil, err
		}
	}

	if req.Model != "" {
		mb, err := json.Marshal(req.Model)
		if err != nil {
			return nil, err
		}
		m["model"] = mb
	}
	m["stream"] = json.RawMessage(`true`)
	if _, ok := m["stream_options"]; !ok {
		m["stream_options"] = json.RawMessage(`{"include_usage":true}`)
	}
	return json.Marshal(m)
}

// pumpSSE reads the SSE stream from the upstream response, decodes each
// event into a llmrouter.Chunk, and forwards it through the producer
// hooks. Always calls hooks.Finish exactly once.
func pumpSSE(ctx context.Context, resp *http.Response, hooks llmrouter.ProducerHooks) {
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var dataLines []string
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			hooks.Finish(ctx.Err())
			return
		default:
		}

		line := scanner.Text()
		if line == "" {
			if len(dataLines) == 0 {
				continue
			}
			payload := strings.Join(dataLines, "\n")
			dataLines = dataLines[:0]

			if payload == "[DONE]" {
				hooks.Finish(nil)
				return
			}
			chunk, ok := decodeChunk(payload)
			if !ok {
				// Unparseable payload — skip rather than abort the stream.
				continue
			}
			if !hooks.Send(chunk) {
				hooks.Finish(ctx.Err())
				return
			}
			continue
		}
		if strings.HasPrefix(line, "data: ") {
			dataLines = append(dataLines, strings.TrimPrefix(line, "data: "))
		} else if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimPrefix(line, "data:"))
		}
		// Drop comments, heartbeats, and other event types.
	}

	if err := scanner.Err(); err != nil && !errors.Is(err, io.EOF) {
		hooks.Finish(fmt.Errorf("azureopenai: read stream: %w", err))
		return
	}
	hooks.Finish(nil)
}

// decodeChunk parses one SSE payload into a llmrouter.Chunk while
// preserving the original bytes in Chunk.Raw.
func decodeChunk(payload string) (llmrouter.Chunk, bool) {
	var wire struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			Index int `json:"index"`
			Delta struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"delta"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(payload), &wire); err != nil {
		return llmrouter.Chunk{}, false
	}

	choices := make([]llmrouter.Choice, 0, len(wire.Choices))
	for _, c := range wire.Choices {
		choices = append(choices, llmrouter.Choice{
			Index: c.Index,
			Delta: llmrouter.Delta{
				Role:    c.Delta.Role,
				Content: c.Delta.Content,
			},
			FinishReason: c.FinishReason,
		})
	}

	chunk := llmrouter.Chunk{
		ID:      wire.ID,
		Object:  wire.Object,
		Created: wire.Created,
		Model:   wire.Model,
		Choices: choices,
		Raw:     json.RawMessage(payload),
	}
	if wire.Usage != nil {
		chunk.Usage = &llmrouter.Usage{
			PromptTokens:     wire.Usage.PromptTokens,
			CompletionTokens: wire.Usage.CompletionTokens,
			TotalTokens:      wire.Usage.TotalTokens,
		}
	}
	return chunk, true
}

// readUpstreamErrorBody reads up to 1KB of the error response body for
// inclusion in ErrUpstream.
func readUpstreamErrorBody(r io.Reader) string {
	const limit = 1024
	buf := make([]byte, limit)
	n, _ := io.ReadFull(io.LimitReader(r, limit), buf)
	return strings.TrimSpace(string(buf[:n]))
}
