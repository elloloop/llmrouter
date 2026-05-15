package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
)

// ---------------------------------------------------------------------------
// CacheControl — plain-text wrapping
// ---------------------------------------------------------------------------

func TestCacheControl_WrapsPlainTextMessage(t *testing.T) {
	cases := []struct {
		name string
		text string
	}{
		{"basic", "hello"},
		{"empty", ""},
		{"unicode", "こんにちは"},
		{"multiline", "a\nb"},
		{"emoji", "🚀"},
		{"long", strings.Repeat("x", 1024)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := llmrouter.TextMessage("user", tc.text)
			out := CacheControl(in)
			if out.Role != "user" {
				t.Errorf("Role = %q", out.Role)
			}
			var blocks []map[string]any
			if err := json.Unmarshal(out.Content, &blocks); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if len(blocks) != 1 {
				t.Fatalf("blocks = %d", len(blocks))
			}
			if blocks[0]["type"] != "text" {
				t.Errorf("type = %v", blocks[0]["type"])
			}
			if blocks[0]["text"] != tc.text {
				t.Errorf("text = %v, want %q", blocks[0]["text"], tc.text)
			}
			cc, ok := blocks[0]["cache_control"].(map[string]any)
			if !ok {
				t.Fatalf("cache_control missing or wrong type: %v", blocks[0]["cache_control"])
			}
			if cc["type"] != "ephemeral" {
				t.Errorf("cache_control.type = %v", cc["type"])
			}
		})
	}
}

func TestCacheControl_RoleVariations(t *testing.T) {
	cases := []string{"user", "assistant", "system", "tool"}
	for _, role := range cases {
		t.Run("role="+role, func(t *testing.T) {
			in := llmrouter.TextMessage(role, "x")
			out := CacheControl(in)
			if out.Role != role {
				t.Errorf("Role = %q, want %q", out.Role, role)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// CacheControl — multipart overlay
// ---------------------------------------------------------------------------

func TestCacheControl_OverlaysOnLastBlockOfMultipart(t *testing.T) {
	in := llmrouter.MultipartMessage("user",
		llmrouter.Text("first"),
		llmrouter.Text("second"),
		llmrouter.Text("third"),
	)
	out := CacheControl(in)
	var blocks []map[string]any
	if err := json.Unmarshal(out.Content, &blocks); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(blocks) != 3 {
		t.Fatalf("blocks = %d, want 3", len(blocks))
	}
	if _, ok := blocks[0]["cache_control"]; ok {
		t.Errorf("block 0 has cache_control: %+v", blocks[0])
	}
	if _, ok := blocks[1]["cache_control"]; ok {
		t.Errorf("block 1 has cache_control: %+v", blocks[1])
	}
	cc, ok := blocks[2]["cache_control"].(map[string]any)
	if !ok {
		t.Fatalf("block 2 missing cache_control: %+v", blocks[2])
	}
	if cc["type"] != "ephemeral" {
		t.Errorf("cache_control.type = %v", cc["type"])
	}
	// Original text fields should be preserved.
	if blocks[0]["text"] != "first" || blocks[1]["text"] != "second" || blocks[2]["text"] != "third" {
		t.Errorf("text preservation: %+v", blocks)
	}
}

func TestCacheControl_OverlayOnSingleBlockMultipart(t *testing.T) {
	in := llmrouter.MultipartMessage("user", llmrouter.Text("only"))
	out := CacheControl(in)
	var blocks []map[string]any
	if err := json.Unmarshal(out.Content, &blocks); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("blocks = %d", len(blocks))
	}
	if _, ok := blocks[0]["cache_control"]; !ok {
		t.Errorf("cache_control missing on single block")
	}
}

func TestCacheControl_OverlayPreservesImageBlock(t *testing.T) {
	in := llmrouter.MultipartMessage("user",
		llmrouter.Text("description:"),
		llmrouter.ImageURL("https://example.com/x.png"),
	)
	out := CacheControl(in)
	var blocks []map[string]any
	if err := json.Unmarshal(out.Content, &blocks); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("blocks = %d", len(blocks))
	}
	// last block is image_url; cache_control should still be applied.
	if blocks[1]["type"] != "image_url" {
		t.Errorf("type = %v", blocks[1]["type"])
	}
	if _, ok := blocks[1]["cache_control"]; !ok {
		t.Errorf("cache_control missing on image block")
	}
	// the image_url payload should still be there
	if _, ok := blocks[1]["image_url"]; !ok {
		t.Errorf("image_url field stripped: %+v", blocks[1])
	}
}

func TestCacheControl_DoesNotMutateInput(t *testing.T) {
	original := llmrouter.TextMessage("user", "hello")
	originalContent := string(original.Content)
	_ = CacheControl(original)
	if string(original.Content) != originalContent {
		t.Errorf("input mutated: %q vs %q", string(original.Content), originalContent)
	}
}

func TestCacheControl_EphemeralIsCanonical(t *testing.T) {
	// Documented contract: Anthropic only supports the "ephemeral" cache
	// type today, so CacheControl always emits that exact string.
	out := CacheControl(llmrouter.TextMessage("user", "x"))
	if !strings.Contains(string(out.Content), `"type":"ephemeral"`) {
		t.Errorf("ephemeral missing: %s", out.Content)
	}
}
