package llmrouter

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
)

// ContentPart is one segment of multimodal Message content. Use the
// constructor helpers (Text, ImageURL, ImageBytes) and pass them to
// MultipartMessage. Providers serialize parts to their native shapes.
type ContentPart struct {
	Type      string // "text" | "image_url" | "image_bytes"
	Text      string
	URL       string
	MediaType string // e.g. "image/png", required for image_bytes
	Data      []byte
}

// Text builds a text ContentPart.
func Text(s string) ContentPart {
	return ContentPart{Type: "text", Text: s}
}

// ImageURL builds an image ContentPart that references a remote URL.
func ImageURL(url string) ContentPart {
	return ContentPart{Type: "image_url", URL: url}
}

// ImageBytes builds an image ContentPart from raw bytes; mediaType (e.g.
// "image/png") is required so the data URL is well-formed.
func ImageBytes(data []byte, mediaType string) ContentPart {
	return ContentPart{Type: "image_bytes", Data: data, MediaType: mediaType}
}

// MultipartMessage builds a Message with OpenAI-shaped multipart content.
// Anthropic translates this to its own content-block shape at request time.
func MultipartMessage(role string, parts ...ContentPart) Message {
	blocks := make([]map[string]any, 0, len(parts))
	for _, p := range parts {
		switch p.Type {
		case "text":
			blocks = append(blocks, map[string]any{"type": "text", "text": p.Text})
		case "image_url":
			blocks = append(blocks, map[string]any{
				"type":      "image_url",
				"image_url": map[string]string{"url": p.URL},
			})
		case "image_bytes":
			// OpenAI accepts data URLs.
			dataURL := fmt.Sprintf("data:%s;base64,%s", p.MediaType, base64.StdEncoding.EncodeToString(p.Data))
			blocks = append(blocks, map[string]any{
				"type":      "image_url",
				"image_url": map[string]string{"url": dataURL},
			})
		}
	}
	raw, _ := json.Marshal(blocks)
	return Message{Role: role, Content: raw}
}
