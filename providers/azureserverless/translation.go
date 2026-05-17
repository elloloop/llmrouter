package azureserverless

// This file mirrors the OpenAI-compatible chat-completions body builder
// and SSE pump used by providers/openai. Azure AI Foundry's Serverless
// API endpoints speak the same wire format, so duplicating the
// translation here keeps this provider self-contained — no cross-
// provider imports, no shared mutable state — at the cost of a small
// amount of copy. Any tweak to chunk decoding, SSE framing, or the
// mid-stream error envelope semantics should be made in lockstep with
// providers/openai/openai.go (functions buildRequestBody, pumpSSE,
// decodeChunk, errorPayload, readUpstreamErrorBody).

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/elloloop/llmrouter"
)

// buildRequestBody assembles the outgoing JSON. If req.Raw is supplied
// (passthrough mode), it is reused; otherwise the typed ChatRequest is
// marshaled. In both cases stream:true and stream_options.include_usage
// are forced on so the upstream emits a final usage block. When
// req.Model is set it overlays any model in Raw — useful for hub-scoped
// endpoints where the body's model field selects the deployment.
func buildRequestBody(req llmrouter.ChatRequest) ([]byte, error) {
	var m map[string]json.RawMessage
	if len(req.Raw) > 0 {
		if err := json.Unmarshal(req.Raw, &m); err != nil {
			return nil, fmt.Errorf("azureserverless: invalid raw request: %w", err)
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
// hooks. Always calls hooks.Finish exactly once. Mid-stream error
// envelopes (data: {"error":...}) terminate with a typed ErrUpstream
// carrying StatusCode 0.
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
			// Detect mid-stream error envelopes before normal chunk
			// decoding. Some Foundry deployments and compatibility
			// proxies emit `data: {"error": ...}` on rate-limit,
			// quota, or overflow conditions and these must surface as
			// a typed error rather than being silently dropped.
			if msg, isErr := errorPayload(payload); isErr {
				hooks.Finish(&llmrouter.ErrUpstream{
					Provider:   providerName,
					StatusCode: 0,
					Body:       msg,
				})
				return
			}
			chunk, ok := decodeChunk(payload)
			if !ok {
				// Unparseable payload — skip rather than abort.
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
		hooks.Finish(fmt.Errorf("azureserverless: read stream: %w", err))
		return
	}
	hooks.Finish(nil)
}

// decodeChunk parses one SSE payload into a llmrouter.Chunk while
// preserving the original bytes in Chunk.Raw for byte-identical
// passthrough. Recognises tool_call fragments too — same shape as
// OpenAI.
func decodeChunk(payload string) (llmrouter.Chunk, bool) {
	var wire struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			Index int `json:"index"`
			Delta struct {
				Role      string `json:"role"`
				Content   string `json:"content"`
				ToolCalls []struct {
					Index    int    `json:"index"`
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function *struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
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
		var toolCalls []llmrouter.ToolCallDelta
		if len(c.Delta.ToolCalls) > 0 {
			toolCalls = make([]llmrouter.ToolCallDelta, 0, len(c.Delta.ToolCalls))
			for _, tc := range c.Delta.ToolCalls {
				out := llmrouter.ToolCallDelta{
					Index: tc.Index,
					ID:    tc.ID,
					Type:  tc.Type,
				}
				if tc.Function != nil {
					out.Function = &llmrouter.ToolCallFunctionDelta{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					}
				}
				toolCalls = append(toolCalls, out)
			}
		}
		choices = append(choices, llmrouter.Choice{
			Index: c.Index,
			Delta: llmrouter.Delta{
				Role:      c.Delta.Role,
				Content:   c.Delta.Content,
				ToolCalls: toolCalls,
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

// errorPayload inspects an SSE data payload and reports whether it
// represents an upstream error envelope. Two wire shapes are
// recognised: the canonical OpenAI {"error":{"message":..., "type":...,
// "code":...}} object form, and the simpler {"error":"plain string"}
// shape used by some compatibility proxies. Anything else returns
// isErr=false so the caller falls back to the skip-this-payload
// behaviour.
func errorPayload(payload string) (msg string, isErr bool) {
	trimmed := strings.TrimSpace(payload)
	if trimmed == "" || trimmed[0] != '{' {
		return "", false
	}
	var probe struct {
		Error json.RawMessage `json:"error"`
	}
	if err := json.Unmarshal([]byte(trimmed), &probe); err != nil {
		return "", false
	}
	if len(probe.Error) == 0 {
		return "", false
	}
	// Try canonical object shape first.
	var obj struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	}
	if err := json.Unmarshal(probe.Error, &obj); err == nil && (obj.Message != "" || obj.Type != "" || obj.Code != "") {
		out := obj.Message
		if obj.Type != "" {
			if out == "" {
				out = obj.Type
			} else {
				out = obj.Type + ": " + out
			}
		}
		if out == "" {
			out = obj.Code
		}
		return out, true
	}
	// Fall back to string shape.
	var s string
	if err := json.Unmarshal(probe.Error, &s); err == nil && s != "" {
		return s, true
	}
	// Field present but unrecognised — still treat as an error so the
	// caller surfaces it rather than silently dropping it.
	return strings.TrimSpace(string(probe.Error)), true
}

// readUpstreamErrorBody reads up to 1 KiB of the error response body
// for inclusion in ErrUpstream. The cap protects against pathological
// upstream responses pinning memory.
func readUpstreamErrorBody(r io.Reader) string {
	const limit = 1024
	buf := make([]byte, limit)
	n, _ := io.ReadFull(io.LimitReader(r, limit), buf)
	return strings.TrimSpace(string(buf[:n]))
}
