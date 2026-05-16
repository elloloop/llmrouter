// translation.go — Anthropic /v1/messages body assembly + SSE event
// translation. Duplicated from providers/anthropic/anthropic.go (NOT a
// public dependency). The translation logic is byte-identical because
// Foundry-hosted Claude speaks the same wire protocol as direct
// Anthropic — only the transport (auth + URL + api-version) differs.
//
// When this file diverges from providers/anthropic/anthropic.go, sync
// the changes back. A future refactor may extract the shared logic
// into providers/internal/anthropicwire and import it from both.

package azureanthropic

import (
	"bufio"
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

// buildBody converts an OpenAI-shaped ChatRequest to the Anthropic
// /v1/messages JSON body. Mirrors providers/anthropic.buildAnthropicBody.
func buildBody(req llmrouter.ChatRequest) ([]byte, error) {
	type anMsg struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	var system string
	msgs := make([]anMsg, 0, len(req.Messages))
	for _, m := range req.Messages {
		if m.Role == "tool" {
			block := map[string]any{
				"type":        "tool_result",
				"tool_use_id": m.ToolCallID,
				"content":     m.PlainText(),
			}
			raw, _ := json.Marshal([]any{block})
			msgs = append(msgs, anMsg{Role: "user", Content: raw})
			continue
		}
		if m.Role == "system" {
			if system != "" {
				system += "\n\n"
			}
			system += m.PlainText()
			continue
		}
		content := translateMultipartContent(m.Content)
		msgs = append(msgs, anMsg{Role: m.Role, Content: content})
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}

	out := map[string]any{
		"model":      req.Model,
		"max_tokens": maxTokens,
		"stream":     true,
		"messages":   msgs,
	}
	if system != "" {
		out["system"] = system
	}

	if len(req.Tools) > 0 {
		atools := make([]map[string]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			at := map[string]any{"name": t.Function.Name}
			if t.Function.Description != "" {
				at["description"] = t.Function.Description
			}
			if len(t.Function.Parameters) > 0 {
				at["input_schema"] = json.RawMessage(t.Function.Parameters)
			}
			atools = append(atools, at)
		}
		out["tools"] = atools
	}

	if req.ToolChoice != nil {
		switch req.ToolChoice.Mode {
		case "auto":
			out["tool_choice"] = map[string]any{"type": "auto"}
		case "none":
			out["tool_choice"] = map[string]any{"type": "none"}
		case "required":
			out["tool_choice"] = map[string]any{"type": "any"}
		case "specific":
			out["tool_choice"] = map[string]any{
				"type": "tool",
				"name": req.ToolChoice.Function,
			}
		default:
			out["tool_choice"] = map[string]any{"type": "auto"}
		}
	}

	applyResponseSchema(out, req)

	if len(req.Raw) > 0 {
		var src map[string]json.RawMessage
		if err := json.Unmarshal(req.Raw, &src); err == nil {
			for _, k := range []string{"temperature", "top_p", "top_k", "stop"} {
				v, ok := src[k]
				if !ok {
					continue
				}
				if k == "stop" {
					out["stop_sequences"] = v
				} else {
					out[k] = v
				}
			}
			if mv, ok := src["model"]; ok {
				// Raw passthrough overlays Model when present.
				out["model"] = mv
			}
		}
	} else {
		if req.Temperature != nil {
			out["temperature"] = *req.Temperature
		}
		if req.TopP != nil {
			out["top_p"] = *req.TopP
		}
		if len(req.Stop) > 0 {
			out["stop_sequences"] = req.Stop
		}
	}

	return json.Marshal(out)
}

// applyResponseSchema injects a synthetic tool + forced tool_choice into
// the outgoing Anthropic body so the model is constrained to emit JSON
// matching the schema.
func applyResponseSchema(out map[string]any, req llmrouter.ChatRequest) {
	if req.ResponseSchema == nil {
		return
	}
	if len(req.Tools) > 0 && req.ToolChoice != nil {
		return
	}
	if req.ResponseSchema.Name == "" {
		return
	}
	tool := map[string]any{"name": req.ResponseSchema.Name}
	if req.ResponseSchema.Description != "" {
		tool["description"] = req.ResponseSchema.Description
	}
	if len(req.ResponseSchema.Schema) > 0 {
		tool["input_schema"] = json.RawMessage(req.ResponseSchema.Schema)
	}
	switch existing := out["tools"].(type) {
	case []map[string]any:
		out["tools"] = append(existing, tool)
	default:
		out["tools"] = []map[string]any{tool}
	}
	out["tool_choice"] = map[string]any{
		"type": "tool",
		"name": req.ResponseSchema.Name,
	}
}

// translateMultipartContent converts OpenAI-shaped multipart content
// blocks to Anthropic's content-block shape. Plain-string content and
// unrecognized shapes pass through unchanged.
func translateMultipartContent(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return raw
	}
	var blocks []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return raw
	}
	out := make([]map[string]any, 0, len(blocks))
	for _, b := range blocks {
		var typ string
		if rawType, ok := b["type"]; ok {
			_ = json.Unmarshal(rawType, &typ)
		}
		switch typ {
		case "text":
			var text string
			_ = json.Unmarshal(b["text"], &text)
			out = append(out, map[string]any{"type": "text", "text": text})
		case "image_url":
			var iu struct {
				URL string `json:"url"`
			}
			_ = json.Unmarshal(b["image_url"], &iu)
			if strings.HasPrefix(iu.URL, "data:") {
				rest := strings.TrimPrefix(iu.URL, "data:")
				mediaType := ""
				data := ""
				if idx := strings.Index(rest, ";base64,"); idx >= 0 {
					mediaType = rest[:idx]
					data = rest[idx+len(";base64,"):]
				}
				out = append(out, map[string]any{
					"type": "image",
					"source": map[string]any{
						"type":       "base64",
						"media_type": mediaType,
						"data":       data,
					},
				})
			} else {
				out = append(out, map[string]any{
					"type": "image",
					"source": map[string]any{
						"type": "url",
						"url":  iu.URL,
					},
				})
			}
		default:
			passthrough := map[string]any{}
			for k, v := range b {
				passthrough[k] = json.RawMessage(v)
			}
			out = append(out, passthrough)
		}
	}
	enc, err := json.Marshal(out)
	if err != nil {
		return raw
	}
	return enc
}

// pump reads Anthropic SSE events from resp.Body and emits OpenAI-shaped
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
		hooks.Finish(fmt.Errorf("azureanthropic: read stream: %w", err))
		return
	}
	hooks.Finish(nil)
}

// pumpState carries identifiers and usage accumulated across events.
type pumpState struct {
	chatID              string
	created             int64
	model               string
	inputTokens         int
	outputTokens        int
	cachedPromptTokens  int
	cacheCreationTokens int
	contentBlockIndex   int
}

// handleEvent processes a single Anthropic SSE event and emits the
// corresponding OpenAI-shaped chunk(s). Returns done=true on message_stop.
func handleEvent(ctx context.Context, evType, payload string, st *pumpState, hooks llmrouter.ProducerHooks) (done bool, err error) {
	switch evType {
	case "message_start":
		var ev struct {
			Message struct {
				Model string `json:"model"`
				Usage struct {
					InputTokens              int `json:"input_tokens"`
					OutputTokens             int `json:"output_tokens"`
					CacheReadInputTokens     int `json:"cache_read_input_tokens"`
					CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
				} `json:"usage"`
			} `json:"message"`
		}
		_ = json.Unmarshal([]byte(payload), &ev)
		if ev.Message.Usage.InputTokens > 0 {
			st.inputTokens = ev.Message.Usage.InputTokens
		}
		if ev.Message.Usage.CacheReadInputTokens > 0 {
			st.cachedPromptTokens = ev.Message.Usage.CacheReadInputTokens
		}
		if ev.Message.Usage.CacheCreationInputTokens > 0 {
			st.cacheCreationTokens = ev.Message.Usage.CacheCreationInputTokens
		}
		if ev.Message.Model != "" {
			st.model = ev.Message.Model
		}
		chunk := newChunk(st, llmrouter.Delta{Role: "assistant", Content: ""}, "")
		if !sendChunk(ctx, hooks, chunk) {
			return false, ctx.Err()
		}
		return false, nil

	case "content_block_start":
		var ev struct {
			Index        int `json:"index"`
			ContentBlock struct {
				Type string `json:"type"`
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"content_block"`
		}
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			return false, nil
		}
		st.contentBlockIndex = ev.Index
		if ev.ContentBlock.Type != "tool_use" {
			return false, nil
		}
		chunk := newChunk(st, llmrouter.Delta{
			ToolCalls: []llmrouter.ToolCallDelta{{
				Index: ev.Index,
				ID:    ev.ContentBlock.ID,
				Type:  "function",
				Function: &llmrouter.ToolCallFunctionDelta{
					Name: ev.ContentBlock.Name,
				},
			}},
		}, "")
		if !sendChunk(ctx, hooks, chunk) {
			return false, ctx.Err()
		}
		return false, nil

	case "content_block_delta":
		var ev struct {
			Index int `json:"index"`
			Delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			return false, nil
		}
		blockIdx := ev.Index
		switch ev.Delta.Type {
		case "text_delta":
			if ev.Delta.Text == "" {
				return false, nil
			}
			chunk := newChunk(st, llmrouter.Delta{Content: ev.Delta.Text}, "")
			if !sendChunk(ctx, hooks, chunk) {
				return false, ctx.Err()
			}
			return false, nil
		case "thinking_delta":
			if ev.Delta.Text == "" {
				return false, nil
			}
			chunk := newChunk(st, llmrouter.Delta{Thinking: ev.Delta.Text}, "")
			if !sendChunk(ctx, hooks, chunk) {
				return false, ctx.Err()
			}
			return false, nil
		case "input_json_delta":
			if ev.Delta.PartialJSON == "" {
				return false, nil
			}
			chunk := newChunk(st, llmrouter.Delta{
				ToolCalls: []llmrouter.ToolCallDelta{{
					Index: blockIdx,
					Function: &llmrouter.ToolCallFunctionDelta{
						Arguments: ev.Delta.PartialJSON,
					},
				}},
			}, "")
			if !sendChunk(ctx, hooks, chunk) {
				return false, ctx.Err()
			}
			return false, nil
		default:
			return false, nil
		}

	case "message_delta":
		var ev struct {
			Delta struct {
				StopReason string `json:"stop_reason"`
			} `json:"delta"`
			Usage struct {
				OutputTokens             int `json:"output_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			return false, nil
		}
		if ev.Usage.OutputTokens > 0 {
			st.outputTokens = ev.Usage.OutputTokens
		}
		if ev.Usage.CacheReadInputTokens > 0 {
			st.cachedPromptTokens = ev.Usage.CacheReadInputTokens
		}
		if ev.Usage.CacheCreationInputTokens > 0 {
			st.cacheCreationTokens = ev.Usage.CacheCreationInputTokens
		}
		if ev.Delta.StopReason != "" {
			finish := mapStopReason(ev.Delta.StopReason)
			chunk := newChunk(st, llmrouter.Delta{}, finish)
			chunk.Usage = currentUsage(st)
			if raw, err := json.Marshal(chunk); err == nil {
				chunk.Raw = raw
			}
			if !sendChunk(ctx, hooks, chunk) {
				return false, ctx.Err()
			}
		}
		return false, nil

	case "message_stop":
		return true, nil

	case "error":
		var ev struct {
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		_ = json.Unmarshal([]byte(payload), &ev)
		body := ev.Error.Message
		if ev.Error.Type != "" {
			if body == "" {
				body = ev.Error.Type
			} else {
				body = ev.Error.Type + ": " + body
			}
		}
		if body == "" {
			body = strings.TrimSpace(payload)
		}
		return false, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: 0,
			Body:       body,
		}

	default:
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
	if st.inputTokens == 0 && st.outputTokens == 0 &&
		st.cachedPromptTokens == 0 && st.cacheCreationTokens == 0 {
		return nil
	}
	return &llmrouter.Usage{
		PromptTokens:        st.inputTokens,
		CompletionTokens:    st.outputTokens,
		TotalTokens:         st.inputTokens + st.outputTokens,
		CachedPromptTokens:  st.cachedPromptTokens,
		CacheCreationTokens: st.cacheCreationTokens,
	}
}

// sendChunk forwards the chunk to the consumer, returning false if the
// consumer has cancelled.
func sendChunk(_ context.Context, hooks llmrouter.ProducerHooks, c llmrouter.Chunk) bool {
	return hooks.Send(c)
}

// mapStopReason converts an Anthropic stop_reason to the OpenAI
// finish_reason vocabulary.
func mapStopReason(r string) string {
	switch r {
	case "end_turn", "stop_sequence":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	default:
		return "stop"
	}
}
