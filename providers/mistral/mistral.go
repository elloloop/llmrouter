// Package mistral implements llmrouter.Provider for Mistral AI's chat
// completions API (Mistral Large, Codestral, Mixtral, etc).
//
// Mistral's HTTP wire format is OpenAI-compatible at the chat-completions
// endpoint: same request body shape, same SSE chunk shape, terminated by
// the `[DONE]` sentinel. This provider is therefore a thin variant of the
// openai provider with one critical difference: Mistral rejects requests
// that include `stream_options.include_usage`. We never inject it, and we
// strip it if the caller supplies it via ChatRequest.Raw. Mistral models
// emit a usage block in the final chunk regardless.
//
// Streaming-only in v0.1. Codestral's fill-in-the-middle (FIM) endpoint
// is out of scope.
package mistral

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

const (
	defaultBaseURL = "https://api.mistral.ai/v1"
	providerName   = "mistral"
)

// Provider implements llmrouter.Provider against the Mistral chat
// completions wire format.
type Provider struct {
	cfg *llmrouter.Config
}

// New constructs a Mistral provider. APIKey is required. BaseURL defaults
// to https://api.mistral.ai/v1 — override with llmrouter.WithBaseURL to
// target a Mistral-compatible self-hosted endpoint.
func New(opts ...llmrouter.Option) (*Provider, error) {
	cfg, err := llmrouter.NewConfig(opts...)
	if err != nil {
		return nil, err
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("%w: api key required", llmrouter.ErrInvalidConfig)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	return &Provider{cfg: cfg}, nil
}

// Name returns the provider id.
func (p *Provider) Name() string { return providerName }

// CompletionStream opens a streaming chat completion against the
// configured Mistral base URL and returns a llmrouter.Stream that yields
// normalized chunks.
func (p *Provider) CompletionStream(ctx context.Context, req llmrouter.ChatRequest) (*llmrouter.Stream, error) {
	body, err := buildRequestBody(req)
	if err != nil {
		return nil, err
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.cfg.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	hreq.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)

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

// buildRequestBody assembles the outgoing JSON. If req.Raw is supplied
// (passthrough mode), it is reused with the model field rewritten;
// otherwise the typed ChatRequest is marshaled. In both cases we force
// streaming on. Unlike the openai provider, we DO NOT inject
// stream_options.include_usage — Mistral rejects that field — and we
// strip it if the caller supplied it via Raw.
func buildRequestBody(req llmrouter.ChatRequest) ([]byte, error) {
	var m map[string]json.RawMessage
	if len(req.Raw) > 0 {
		if err := json.Unmarshal(req.Raw, &m); err != nil {
			return nil, fmt.Errorf("mistral: invalid raw request: %w", err)
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
	// Mistral rejects stream_options.include_usage. Strip the entire
	// stream_options block — Mistral emits usage in the final chunk
	// regardless, so there is nothing to gain from passing it through.
	delete(m, "stream_options")
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
		hooks.Finish(fmt.Errorf("mistral: read stream: %w", err))
		return
	}
	hooks.Finish(nil)
}

// decodeChunk parses one SSE payload into a llmrouter.Chunk while
// preserving the original bytes in Chunk.Raw. Mistral's chunk wire format
// is byte-identical to OpenAI's; finish_reason values such as
// "model_length" and "error" pass through unchanged (we do not remap).
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

// readUpstreamErrorBody reads up to 1KiB of the error response body for
// inclusion in ErrUpstream.
func readUpstreamErrorBody(r io.Reader) string {
	const limit = 1024
	buf := make([]byte, limit)
	n, _ := io.ReadFull(io.LimitReader(r, limit), buf)
	return strings.TrimSpace(string(buf[:n]))
}
