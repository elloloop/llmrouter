// Package cohere implements the llmrouter.Provider interface against
// Cohere's /v2/chat API (Command R+, Command R, etc). The provider accepts
// OpenAI-shaped requests (llmrouter.ChatRequest), translates them to
// Cohere's v2 wire format (which is itself OpenAI-shaped, so translation
// is shallow), then re-encodes the Cohere SSE event stream as
// OpenAI-shaped streaming chunks (llmrouter.Chunk).
package cohere

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/elloloop/llmrouter"
)

const (
	defaultBaseURL    = "https://api.cohere.com/v2"
	defaultMaxTokens  = 4096
	scannerBufferSize = 1024 * 1024 // 1 MiB
	providerName      = "cohere"
	errBodyCap        = 8 * 1024 // 8 KiB
)

// Provider talks to Cohere /v2/chat. Construct with New.
type Provider struct {
	cfg *llmrouter.Config
}

// New builds a Provider from llmrouter options. WithAPIKey is required.
func New(opts ...llmrouter.Option) (*Provider, error) {
	cfg, err := llmrouter.NewConfig(opts...)
	if err != nil {
		return nil, err
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("%w: cohere requires an api key", llmrouter.ErrInvalidConfig)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	return &Provider{cfg: cfg}, nil
}

// Name returns the provider id.
func (p *Provider) Name() string { return providerName }

// CompletionStream issues a streaming /v2/chat request and returns a
// Stream that yields OpenAI-shaped chunks.
func (p *Provider) CompletionStream(ctx context.Context, req llmrouter.ChatRequest) (*llmrouter.Stream, error) {
	body, err := buildCohereBody(req)
	if err != nil {
		return nil, fmt.Errorf("cohere: build request: %w", err)
	}

	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		p.cfg.BaseURL+"/chat", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	hreq.Header.Set("Authorization", "Bearer "+p.cfg.APIKey)

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("cohere: http: %w", err)
	}
	if resp.StatusCode >= 400 {
		defer resp.Body.Close()
		b, _ := io.ReadAll(io.LimitReader(resp.Body, errBodyCap))
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

// buildCohereBody converts an OpenAI-shaped ChatRequest to a Cohere v2
// /chat JSON body. Cohere v2 takes messages with roles directly, so
// system messages are NOT lifted to a top-level field — they remain in
// the messages array unchanged.
func buildCohereBody(req llmrouter.ChatRequest) ([]byte, error) {
	type coMsg struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	msgs := make([]coMsg, 0, len(req.Messages))
	for _, m := range req.Messages {
		msgs = append(msgs, coMsg{Role: m.Role, Content: m.Content})
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}

	out := map[string]any{
		"model":      req.Model,
		"stream":     true,
		"messages":   msgs,
		"max_tokens": maxTokens,
	}

	// Pull optional tuning knobs from the original raw body if present.
	// Cohere v2 uses "p" for nucleus, "k" for top-k, "stop_sequences" for stop.
	if len(req.Raw) > 0 {
		applyRawKnobs(req.Raw, out)
	} else {
		applyTypedKnobs(req, out)
	}

	return json.Marshal(out)
}

// applyRawKnobs reads tuning knobs from the original raw body. Accepts
// both OpenAI-style keys (top_p, top_k, stop) and Cohere-native keys
// (p, k, stop_sequences).
func applyRawKnobs(raw json.RawMessage, out map[string]any) {
	var src map[string]json.RawMessage
	if err := json.Unmarshal(raw, &src); err != nil {
		return
	}
	if v, ok := src["temperature"]; ok {
		out["temperature"] = v
	}
	if v, ok := src["p"]; ok {
		out["p"] = v
	} else if v, ok := src["top_p"]; ok {
		out["p"] = v
	}
	if v, ok := src["k"]; ok {
		out["k"] = v
	} else if v, ok := src["top_k"]; ok {
		out["k"] = v
	}
	if v, ok := src["stop_sequences"]; ok {
		out["stop_sequences"] = v
	} else if v, ok := src["stop"]; ok {
		out["stop_sequences"] = v
	}
	if v, ok := src["max_tokens"]; ok {
		out["max_tokens"] = v
	}
}

// applyTypedKnobs falls back to the typed fields on ChatRequest when no
// raw body was supplied.
func applyTypedKnobs(req llmrouter.ChatRequest, out map[string]any) {
	if req.Temperature != nil {
		out["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		out["p"] = *req.TopP
	}
	if len(req.Stop) > 0 {
		out["stop_sequences"] = req.Stop
	}
}

// pump reads Cohere SSE events from resp.Body and emits OpenAI-shaped
// chunks via hooks until the stream terminates. Always calls hooks.Finish.
func pump(ctx context.Context, resp *http.Response, model string, hooks llmrouter.ProducerHooks) {
	defer resp.Body.Close()

	chatID := "chatcmpl-" + uuid.NewString()
	created := time.Now().Unix()

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), scannerBufferSize)

	var (
		state     = &pumpState{model: model, chatID: chatID, created: created}
		evType    string
		dataLines []string
	)

	flush := func() (done bool, err error) {
		if len(dataLines) == 0 {
			return false, nil
		}
		payload := strings.Join(dataLines, "\n")
		dataLines = dataLines[:0]
		return handleEvent(ctx, evType, payload, state, hooks)
	}

	for scanner.Scan() {
		if ctx.Err() != nil {
			hooks.Finish(ctx.Err())
			return
		}
		line := scanner.Text()
		if line == "" {
			done, err := flush()
			if err != nil {
				hooks.Finish(err)
				return
			}
			if done {
				hooks.Finish(nil)
				return
			}
			evType = ""
			continue
		}
		switch {
		case strings.HasPrefix(line, "event: "):
			evType = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			dataLines = append(dataLines, strings.TrimPrefix(line, "data: "))
		case strings.HasPrefix(line, "data:"):
			dataLines = append(dataLines, strings.TrimPrefix(line, "data:"))
		}
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		hooks.Finish(fmt.Errorf("cohere: read stream: %w", err))
		return
	}
	hooks.Finish(nil)
}

// pumpState carries identifiers and usage accumulated across events.
type pumpState struct {
	chatID       string
	created      int64
	model        string
	inputTokens  int
	outputTokens int
}

// handleEvent processes a single Cohere SSE event and emits the
// corresponding OpenAI-shaped chunk(s). Returns done=true on message-end.
func handleEvent(ctx context.Context, evType, payload string, st *pumpState, hooks llmrouter.ProducerHooks) (done bool, err error) {
	switch evType {
	case "message-start":
		chunk := newChunk(st, llmrouter.Delta{Role: "assistant", Content: ""}, "")
		if !hooks.Send(chunk) {
			return false, ctx.Err()
		}
		return false, nil

	case "content-delta":
		var ev struct {
			Delta struct {
				Message struct {
					Content struct {
						Text string `json:"text"`
					} `json:"content"`
				} `json:"message"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			return false, nil // tolerate malformed individual events
		}
		if ev.Delta.Message.Content.Text == "" {
			return false, nil
		}
		chunk := newChunk(st, llmrouter.Delta{Content: ev.Delta.Message.Content.Text}, "")
		if !hooks.Send(chunk) {
			return false, ctx.Err()
		}
		return false, nil

	case "message-end":
		var ev struct {
			Delta struct {
				FinishReason string `json:"finish_reason"`
				Usage        struct {
					BilledUnits struct {
						InputTokens  int `json:"input_tokens"`
						OutputTokens int `json:"output_tokens"`
					} `json:"billed_units"`
				} `json:"usage"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			return true, nil
		}
		if ev.Delta.Usage.BilledUnits.InputTokens > 0 {
			st.inputTokens = ev.Delta.Usage.BilledUnits.InputTokens
		}
		if ev.Delta.Usage.BilledUnits.OutputTokens > 0 {
			st.outputTokens = ev.Delta.Usage.BilledUnits.OutputTokens
		}
		finish := mapFinishReason(ev.Delta.FinishReason)
		chunk := newChunk(st, llmrouter.Delta{}, finish)
		chunk.Usage = currentUsage(st)
		if raw, err := json.Marshal(chunk); err == nil {
			chunk.Raw = raw
		}
		if !hooks.Send(chunk) {
			return true, ctx.Err()
		}
		return true, nil

	default:
		// tool-call-start, tool-call-delta, tool-call-end, content-start,
		// content-end and any other event types are silently skipped in v0.1.
		return false, nil
	}
}

// newChunk builds a normalized OpenAI-shaped Chunk and pre-populates Raw
// with its JSON form.
func newChunk(st *pumpState, delta llmrouter.Delta, finish string) llmrouter.Chunk {
	c := llmrouter.Chunk{
		ID:      st.chatID,
		Object:  "chat.completion.chunk",
		Created: st.created,
		Model:   st.model,
		Choices: []llmrouter.Choice{{
			Index:        0,
			Delta:        delta,
			FinishReason: finish,
		}},
	}
	if raw, err := json.Marshal(c); err == nil {
		c.Raw = raw
	}
	return c
}

// currentUsage returns the accumulated token usage, or nil if nothing is
// known yet.
func currentUsage(st *pumpState) *llmrouter.Usage {
	if st.inputTokens == 0 && st.outputTokens == 0 {
		return nil
	}
	return &llmrouter.Usage{
		PromptTokens:     st.inputTokens,
		CompletionTokens: st.outputTokens,
		TotalTokens:      st.inputTokens + st.outputTokens,
	}
}

// mapFinishReason converts a Cohere finish_reason to the OpenAI
// finish_reason vocabulary.
func mapFinishReason(r string) string {
	switch r {
	case "COMPLETE", "STOP_SEQUENCE":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "TOOL_CALL":
		return "tool_calls"
	default:
		return "stop"
	}
}
