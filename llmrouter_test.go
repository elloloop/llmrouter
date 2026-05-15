package llmrouter_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
)

// ---------------------------------------------------------------------------
// TextMessage
// ---------------------------------------------------------------------------

func TestTextMessage_Role(t *testing.T) {
	cases := []struct {
		name string
		role string
	}{
		{"user", "user"},
		{"assistant", "assistant"},
		{"system", "system"},
		{"tool", "tool"},
		{"empty", ""},
		{"unicode-role", "ユーザー"},
		{"whitespace-role", "  user  "},
		{"capitalised", "User"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := llmrouter.TextMessage(tc.role, "hi")
			if m.Role != tc.role {
				t.Fatalf("Role = %q, want %q", m.Role, tc.role)
			}
		})
	}
}

func TestTextMessage_Content_RoundTripsAsJSONString(t *testing.T) {
	cases := []struct {
		name string
		text string
	}{
		{"basic", "hello world"},
		{"empty", ""},
		{"unicode", "héllo 🌍 こんにちは"},
		{"quotes", `she said "hi"`},
		{"backslashes", `a\b\c`},
		{"newlines", "line1\nline2\nline3"},
		{"tabs", "a\tb\tc"},
		{"control-chars", "a\x00b"},
		{"emoji-only", "🚀🎉🤖"},
		{"long", strings.Repeat("x", 5000)},
		{"json-like", `{"injected":true}`},
		{"angle-brackets", "<script>alert(1)</script>"},
		{"rtl", "مرحبا بالعالم"},
		{"mixed-scripts", "Hello мир 世界"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := llmrouter.TextMessage("user", tc.text)
			var got string
			if err := json.Unmarshal(m.Content, &got); err != nil {
				t.Fatalf("Content not a JSON string: %v (raw=%s)", err, string(m.Content))
			}
			if got != tc.text {
				t.Fatalf("round-trip mismatch:\n got=%q\nwant=%q", got, tc.text)
			}
		})
	}
}

func TestTextMessage_MarshalsAsObject(t *testing.T) {
	m := llmrouter.TextMessage("user", "hi")
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	want := `{"role":"user","content":"hi"}`
	if string(b) != want {
		t.Fatalf("got %s, want %s", b, want)
	}
}

// ---------------------------------------------------------------------------
// Message.PlainText
// ---------------------------------------------------------------------------

func TestPlainText_StringContent(t *testing.T) {
	cases := []struct {
		name string
		text string
	}{
		{"basic", "hello"},
		{"empty", ""},
		{"unicode", "こんにちは"},
		{"multiline", "a\nb"},
		{"quotes", `"hi"`},
		{"emoji", "🌍"},
		{"long", strings.Repeat("z", 2048)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := llmrouter.TextMessage("user", tc.text)
			if got := m.PlainText(); got != tc.text {
				t.Fatalf("got %q, want %q", got, tc.text)
			}
		})
	}
}

func TestPlainText_MultipartContent(t *testing.T) {
	cases := []struct {
		name    string
		content string
		want    string
	}{
		{
			"single-text-part",
			`[{"type":"text","text":"hi"}]`,
			"hi",
		},
		{
			"two-text-parts",
			`[{"type":"text","text":"hello "},{"type":"text","text":"world"}]`,
			"hello world",
		},
		{
			"text-with-image-parts-skips-images",
			`[{"type":"text","text":"see: "},{"type":"image_url","image_url":{"url":"x"}},{"type":"text","text":"end"}]`,
			"see: end",
		},
		{
			"image-only-parts",
			`[{"type":"image_url","image_url":{"url":"x"}}]`,
			"",
		},
		{
			"unknown-type-only",
			`[{"type":"audio"}]`,
			"",
		},
		{
			"empty-array",
			`[]`,
			"",
		},
		{
			"text-with-empty-text",
			`[{"type":"text","text":""}]`,
			"",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := llmrouter.Message{Role: "user", Content: json.RawMessage(tc.content)}
			if got := m.PlainText(); got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestPlainText_EmptyOrMalformed(t *testing.T) {
	cases := []struct {
		name    string
		content json.RawMessage
		want    string
	}{
		{"nil-content", nil, ""},
		{"empty-bytes", json.RawMessage{}, ""},
		{"invalid-json", json.RawMessage(`not json`), ""},
		{"number-content", json.RawMessage(`42`), ""},
		{"bool-content", json.RawMessage(`true`), ""},
		{"object-content", json.RawMessage(`{"foo":"bar"}`), ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := llmrouter.Message{Role: "user", Content: tc.content}
			if got := m.PlainText(); got != tc.want {
				t.Fatalf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// ChatRequest JSON marshaling
// ---------------------------------------------------------------------------

func TestChatRequest_OmitemptyFields(t *testing.T) {
	req := llmrouter.ChatRequest{
		Model:    "gpt-4o-mini",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(b)
	for _, k := range []string{"max_tokens", "temperature", "top_p", "stop", `"user":`, `"stream":`} {
		if strings.Contains(s, k) {
			t.Errorf("expected %q omitted, got %s", k, s)
		}
	}
	if !strings.Contains(s, `"model":"gpt-4o-mini"`) {
		t.Errorf("missing model: %s", s)
	}
}

func TestChatRequest_AllFieldsRoundTrip(t *testing.T) {
	temp := 0.7
	top := 0.9
	req := llmrouter.ChatRequest{
		Model:       "claude-3-5-sonnet",
		Messages:    []llmrouter.Message{llmrouter.TextMessage("user", "hello")},
		MaxTokens:   256,
		Temperature: &temp,
		TopP:        &top,
		Stop:        []string{"\n\n", "END"},
		User:        "user-abc",
		Stream:      true,
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got llmrouter.ChatRequest
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Model != req.Model {
		t.Errorf("Model mismatch: %q vs %q", got.Model, req.Model)
	}
	if got.MaxTokens != req.MaxTokens {
		t.Errorf("MaxTokens mismatch: %d vs %d", got.MaxTokens, req.MaxTokens)
	}
	if got.Temperature == nil || *got.Temperature != temp {
		t.Errorf("Temperature mismatch")
	}
	if got.TopP == nil || *got.TopP != top {
		t.Errorf("TopP mismatch")
	}
	if len(got.Stop) != 2 || got.Stop[0] != "\n\n" || got.Stop[1] != "END" {
		t.Errorf("Stop mismatch: %#v", got.Stop)
	}
	if got.User != "user-abc" {
		t.Errorf("User mismatch: %q", got.User)
	}
	if !got.Stream {
		t.Errorf("Stream mismatch")
	}
}

func TestChatRequest_RawFieldIsNotMarshaled(t *testing.T) {
	req := llmrouter.ChatRequest{
		Model:    "gpt-4o",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		Raw:      json.RawMessage(`{"unmodelled":true}`),
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "unmodelled") {
		t.Fatalf("Raw leaked into output: %s", b)
	}
	if strings.Contains(string(b), `"Raw"`) || strings.Contains(string(b), `"raw"`) {
		t.Fatalf("Raw field name leaked: %s", b)
	}
}

// ---------------------------------------------------------------------------
// Chunk / Choice / Delta / Usage JSON
// ---------------------------------------------------------------------------

func TestChunk_RoundTrip(t *testing.T) {
	cases := []struct {
		name string
		c    llmrouter.Chunk
	}{
		{
			"empty",
			llmrouter.Chunk{},
		},
		{
			"single-delta",
			llmrouter.Chunk{
				ID:      "chatcmpl-1",
				Object:  "chat.completion.chunk",
				Created: 123456,
				Model:   "gpt-4o-mini",
				Choices: []llmrouter.Choice{{
					Index: 0,
					Delta: llmrouter.Delta{Role: "assistant", Content: "hi"},
				}},
			},
		},
		{
			"finish-stop",
			llmrouter.Chunk{
				ID: "x", Object: "y", Created: 1, Model: "m",
				Choices: []llmrouter.Choice{{
					Index: 0, FinishReason: "stop", Delta: llmrouter.Delta{},
				}},
			},
		},
		{
			"finish-length",
			llmrouter.Chunk{
				ID: "x", Object: "y", Created: 1, Model: "m",
				Choices: []llmrouter.Choice{{Index: 0, FinishReason: "length", Delta: llmrouter.Delta{}}},
			},
		},
		{
			"with-usage",
			llmrouter.Chunk{
				ID: "x", Object: "y", Created: 1, Model: "m",
				Choices: []llmrouter.Choice{{Index: 0, Delta: llmrouter.Delta{}}},
				Usage:   &llmrouter.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
			},
		},
		{
			"multiple-choices",
			llmrouter.Chunk{
				ID: "x", Object: "y", Created: 1, Model: "m",
				Choices: []llmrouter.Choice{
					{Index: 0, Delta: llmrouter.Delta{Content: "a"}},
					{Index: 1, Delta: llmrouter.Delta{Content: "b"}},
				},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.c)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got llmrouter.Chunk
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got.ID != tc.c.ID || got.Object != tc.c.Object || got.Created != tc.c.Created || got.Model != tc.c.Model {
				t.Errorf("header fields mismatch: %#v vs %#v", got, tc.c)
			}
			if len(got.Choices) != len(tc.c.Choices) {
				t.Fatalf("choice count: got %d, want %d", len(got.Choices), len(tc.c.Choices))
			}
			for i := range got.Choices {
				if got.Choices[i].Index != tc.c.Choices[i].Index {
					t.Errorf("Choice[%d].Index mismatch", i)
				}
				if got.Choices[i].Delta.Content != tc.c.Choices[i].Delta.Content {
					t.Errorf("Choice[%d].Delta.Content mismatch", i)
				}
				if got.Choices[i].FinishReason != tc.c.Choices[i].FinishReason {
					t.Errorf("Choice[%d].FinishReason mismatch", i)
				}
			}
			if (got.Usage == nil) != (tc.c.Usage == nil) {
				t.Errorf("Usage nil-ness mismatch")
			}
			if tc.c.Usage != nil && *got.Usage != *tc.c.Usage {
				t.Errorf("Usage mismatch: %+v vs %+v", got.Usage, tc.c.Usage)
			}
		})
	}
}

func TestChunk_RawNotMarshaled(t *testing.T) {
	c := llmrouter.Chunk{ID: "x", Raw: json.RawMessage(`{"secret":true}`)}
	b, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "secret") {
		t.Fatalf("Raw leaked: %s", b)
	}
}

func TestChoice_FinishReasonOmitempty(t *testing.T) {
	c := llmrouter.Choice{Index: 0, Delta: llmrouter.Delta{Content: "x"}}
	b, _ := json.Marshal(c)
	if strings.Contains(string(b), "finish_reason") {
		t.Fatalf("finish_reason not omitted when empty: %s", b)
	}
}

func TestDelta_OmitemptyBehavior(t *testing.T) {
	cases := []struct {
		name string
		d    llmrouter.Delta
		want string
	}{
		{"empty", llmrouter.Delta{}, `{}`},
		{"only-role", llmrouter.Delta{Role: "assistant"}, `{"role":"assistant"}`},
		{"only-content", llmrouter.Delta{Content: "hi"}, `{"content":"hi"}`},
		{"both", llmrouter.Delta{Role: "assistant", Content: "hi"}, `{"role":"assistant","content":"hi"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, _ := json.Marshal(tc.d)
			if string(b) != tc.want {
				t.Fatalf("got %s, want %s", b, tc.want)
			}
		})
	}
}

func TestUsage_RoundTrip(t *testing.T) {
	cases := []llmrouter.Usage{
		{0, 0, 0},
		{1, 1, 2},
		{1000, 500, 1500},
		{2147483647, 0, 2147483647},
	}
	for i, u := range cases {
		t.Run(string(rune('a'+i)), func(t *testing.T) {
			b, err := json.Marshal(u)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got llmrouter.Usage
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got != u {
				t.Fatalf("got %+v, want %+v", got, u)
			}
		})
	}
}
