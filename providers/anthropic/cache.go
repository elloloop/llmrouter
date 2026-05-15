package anthropic

import (
	"encoding/json"

	"github.com/elloloop/llmrouter"
)

// CacheControl wraps a Message's Content with Anthropic's
// cache_control:{"type":"ephemeral"} annotation. The returned Message
// produces content as a JSON array with cache_control on the last block.
//
// If the input message is already multipart (JSON array of blocks),
// cache_control is overlaid on the last block; otherwise the plain-text
// content is wrapped in a single text block.
func CacheControl(m llmrouter.Message) llmrouter.Message {
	// Try to parse existing content as a multipart array.
	var existing []map[string]json.RawMessage
	if err := json.Unmarshal(m.Content, &existing); err == nil && len(existing) > 0 {
		blocks := make([]map[string]any, 0, len(existing))
		for i, b := range existing {
			out := map[string]any{}
			for k, v := range b {
				out[k] = json.RawMessage(v)
			}
			if i == len(existing)-1 {
				out["cache_control"] = map[string]string{"type": "ephemeral"}
			}
			blocks = append(blocks, out)
		}
		raw, _ := json.Marshal(blocks)
		return llmrouter.Message{Role: m.Role, Content: raw}
	}

	// Fall back to plain-text wrapping.
	text := m.PlainText()
	blocks := []map[string]any{{
		"type":          "text",
		"text":          text,
		"cache_control": map[string]string{"type": "ephemeral"},
	}}
	raw, _ := json.Marshal(blocks)
	return llmrouter.Message{Role: m.Role, Content: raw}
}
