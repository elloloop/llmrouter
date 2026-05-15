// Package gemini implements the llmrouter.Provider interface against
// Google AI Studio's generativelanguage.googleapis.com endpoint (the
// consumer-facing Gemini API). This is distinct from Vertex AI: AI Studio
// uses a simple API key, while Vertex requires GCP project + ADC.
//
// The provider accepts OpenAI-shaped requests (llmrouter.ChatRequest),
// translates them to the Gemini GenerateContentRequest wire format, and
// re-encodes the SSE response as OpenAI-shaped streaming chunks
// (llmrouter.Chunk).
package gemini

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
	defaultBaseURL    = "https://generativelanguage.googleapis.com/v1beta"
	defaultMaxTokens  = 4096
	scannerBufferSize = 1024 * 1024 // 1 MiB
	providerName      = "gemini"
	apiKeyHeader      = "x-goog-api-key"
)

// Provider talks to Google AI Studio's generativelanguage.googleapis.com.
// Construct with New.
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
		return nil, fmt.Errorf("%w: gemini requires an api key", llmrouter.ErrInvalidConfig)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = defaultBaseURL
	}
	return &Provider{cfg: cfg}, nil
}

// Name returns the provider id.
func (p *Provider) Name() string { return providerName }

// CompletionStream issues a streaming :streamGenerateContent request and
// returns a Stream that yields OpenAI-shaped chunks.
func (p *Provider) CompletionStream(ctx context.Context, req llmrouter.ChatRequest) (*llmrouter.Stream, error) {
	body, err := buildGeminiBody(req)
	if err != nil {
		return nil, fmt.Errorf("gemini: build request: %w", err)
	}

	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse", p.cfg.BaseURL, req.Model)
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	hreq.Header.Set(apiKeyHeader, p.cfg.APIKey)

	resp, err := p.cfg.HTTP().Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("gemini: http: %w", err)
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

// buildGeminiBody converts an OpenAI-shaped ChatRequest to the Gemini
// GenerateContentRequest JSON body.
func buildGeminiBody(req llmrouter.ChatRequest) ([]byte, error) {
	type part struct {
		Text string `json:"text"`
	}
	type content struct {
		Role  string `json:"role"`
		Parts []part `json:"parts"`
	}
	type sysInstruction struct {
		Parts []part `json:"parts"`
	}

	var systemText string
	contents := make([]content, 0, len(req.Messages))
	for _, m := range req.Messages {
		if m.Role == "system" {
			if systemText != "" {
				systemText += "\n\n"
			}
			systemText += m.PlainText()
			continue
		}
		role := geminiRole(m.Role)
		contents = append(contents, content{
			Role:  role,
			Parts: []part{{Text: m.PlainText()}},
		})
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}

	genConfig := map[string]any{
		"maxOutputTokens": maxTokens,
	}

	// Honour Raw tuning knobs (OpenAI-shaped keys) first; fall back to
	// typed fields on ChatRequest. Mirrors the Anthropic provider.
	if len(req.Raw) > 0 {
		var src map[string]json.RawMessage
		if err := json.Unmarshal(req.Raw, &src); err == nil {
			liftRawTuning(src, genConfig)
		}
	} else {
		if req.Temperature != nil {
			genConfig["temperature"] = *req.Temperature
		}
		if req.TopP != nil {
			genConfig["topP"] = *req.TopP
		}
		if len(req.Stop) > 0 {
			genConfig["stopSequences"] = req.Stop
		}
	}

	out := map[string]any{
		"contents":         contents,
		"generationConfig": genConfig,
	}
	if systemText != "" {
		out["systemInstruction"] = sysInstruction{Parts: []part{{Text: systemText}}}
	}

	return json.Marshal(out)
}

// liftRawTuning copies tuning knobs from an OpenAI-shaped Raw body into
// the Gemini generationConfig map, renaming keys as needed.
func liftRawTuning(src map[string]json.RawMessage, genConfig map[string]any) {
	mapping := map[string]string{
		"temperature": "temperature",
		"top_p":       "topP",
		"top_k":       "topK",
		"stop":        "stopSequences",
	}
	for openAIKey, geminiKey := range mapping {
		v, ok := src[openAIKey]
		if !ok {
			continue
		}
		genConfig[geminiKey] = v
	}
}

// geminiRole converts an OpenAI role to a Gemini role.
func geminiRole(role string) string {
	switch role {
	case "assistant":
		return "model"
	default:
		return "user"
	}
}

// pump reads Gemini SSE events from resp.Body and emits OpenAI-shaped
// chunks via hooks until the stream terminates. Always calls hooks.Finish.
func pump(ctx context.Context, resp *http.Response, model string, hooks llmrouter.ProducerHooks) {
	defer resp.Body.Close()

	chatID := "chatcmpl-" + uuid.NewString()
	created := time.Now().Unix()

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), scannerBufferSize)

	st := &pumpState{model: model, chatID: chatID, created: created}

	for scanner.Scan() {
		if ctx.Err() != nil {
			hooks.Finish(ctx.Err())
			return
		}
		line := scanner.Text()
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}
		if !handlePayload(ctx, payload, st, hooks) {
			return
		}
	}
	if err := scanner.Err(); err != nil && err != io.EOF {
		hooks.Finish(fmt.Errorf("gemini: read stream: %w", err))
		return
	}
	hooks.Finish(nil)
}

// pumpState carries identifiers and per-stream flags across SSE events.
type pumpState struct {
	chatID      string
	created     int64
	model       string
	rolePrimed  bool
	finishSent  bool
}

// generateContentResponse mirrors the subset of Gemini's response shape
// that this provider needs.
type generateContentResponse struct {
	Candidates    []candidate    `json:"candidates"`
	UsageMetadata *usageMetadata `json:"usageMetadata,omitempty"`
}

type candidate struct {
	Content      candidateContent `json:"content"`
	FinishReason string           `json:"finishReason,omitempty"`
}

type candidateContent struct {
	Role  string          `json:"role"`
	Parts []candidatePart `json:"parts"`
}

type candidatePart struct {
	Text string `json:"text"`
}

type usageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// handlePayload decodes one SSE data payload into a GenerateContentResponse
// and emits OpenAI-shaped chunks. Returns false if the consumer cancelled
// or the stream finished and hooks.Finish has been called.
func handlePayload(ctx context.Context, payload string, st *pumpState, hooks llmrouter.ProducerHooks) bool {
	var resp generateContentResponse
	if err := json.Unmarshal([]byte(payload), &resp); err != nil {
		// Tolerate malformed individual events.
		return true
	}

	for _, cand := range resp.Candidates {
		if !st.rolePrimed && cand.Content.Role != "" {
			st.rolePrimed = true
			chunk := newChunk(st, llmrouter.Delta{Role: "assistant"}, "")
			if !hooks.Send(chunk) {
				hooks.Finish(ctx.Err())
				return false
			}
		}
		for _, p := range cand.Content.Parts {
			if p.Text == "" {
				continue
			}
			chunk := newChunk(st, llmrouter.Delta{Content: p.Text}, "")
			if !hooks.Send(chunk) {
				hooks.Finish(ctx.Err())
				return false
			}
		}
		if cand.FinishReason != "" && !st.finishSent {
			st.finishSent = true
			finish := mapFinishReason(cand.FinishReason)
			chunk := newChunk(st, llmrouter.Delta{}, finish)
			if resp.UsageMetadata != nil {
				chunk.Usage = usageFromMetadata(resp.UsageMetadata)
			}
			if raw, err := json.Marshal(chunk); err == nil {
				chunk.Raw = raw
			}
			if !hooks.Send(chunk) {
				hooks.Finish(ctx.Err())
				return false
			}
		}
	}

	// If usage arrived without a candidate finishReason (terminal-only
	// usage frame), surface it as a synthetic finish chunk.
	if len(resp.Candidates) == 0 && resp.UsageMetadata != nil && !st.finishSent {
		st.finishSent = true
		chunk := newChunk(st, llmrouter.Delta{}, "stop")
		chunk.Usage = usageFromMetadata(resp.UsageMetadata)
		if raw, err := json.Marshal(chunk); err == nil {
			chunk.Raw = raw
		}
		if !hooks.Send(chunk) {
			hooks.Finish(ctx.Err())
			return false
		}
	}

	return true
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

// usageFromMetadata converts Gemini usageMetadata to the OpenAI-shaped
// Usage block.
func usageFromMetadata(u *usageMetadata) *llmrouter.Usage {
	total := u.TotalTokenCount
	if total == 0 {
		total = u.PromptTokenCount + u.CandidatesTokenCount
	}
	return &llmrouter.Usage{
		PromptTokens:     u.PromptTokenCount,
		CompletionTokens: u.CandidatesTokenCount,
		TotalTokens:      total,
	}
}

// mapFinishReason converts a Gemini finishReason to the OpenAI
// finish_reason vocabulary.
func mapFinishReason(r string) string {
	switch r {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION":
		return "content_filter"
	default:
		return "stop"
	}
}
