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
		{PromptTokens: 0, CompletionTokens: 0, TotalTokens: 0},
		{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
		{PromptTokens: 1000, CompletionTokens: 500, TotalTokens: 1500},
		{PromptTokens: 2147483647, CompletionTokens: 0, TotalTokens: 2147483647},
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

// ---------------------------------------------------------------------------
// ToolChoice.MarshalJSON
// ---------------------------------------------------------------------------

func TestToolChoice_MarshalJSON(t *testing.T) {
	cases := []struct {
		name string
		tc   llmrouter.ToolChoice
		want string
	}{
		{"auto", llmrouter.ToolChoice{Mode: "auto"}, `"auto"`},
		{"none", llmrouter.ToolChoice{Mode: "none"}, `"none"`},
		{"required", llmrouter.ToolChoice{Mode: "required"}, `"required"`},
		{"specific", llmrouter.ToolChoice{Mode: "specific", Function: "get_weather"}, `{"function":{"name":"get_weather"},"type":"function"}`},
		{"specific-empty-name", llmrouter.ToolChoice{Mode: "specific"}, `{"function":{"name":""},"type":"function"}`},
		{"empty-mode-defaults-auto", llmrouter.ToolChoice{}, `"auto"`},
		{"unknown-mode-defaults-auto", llmrouter.ToolChoice{Mode: "xyz"}, `"auto"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.tc)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(b) != tc.want {
				t.Fatalf("got %s, want %s", b, tc.want)
			}
		})
	}
}

func TestToolChoice_MarshalsThroughPointer(t *testing.T) {
	cases := []struct {
		name string
		tc   *llmrouter.ToolChoice
	}{
		{"auto", &llmrouter.ToolChoice{Mode: "auto"}},
		{"required", &llmrouter.ToolChoice{Mode: "required"}},
		{"specific", &llmrouter.ToolChoice{Mode: "specific", Function: "f"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.tc)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if len(b) == 0 || string(b) == "null" {
				t.Fatalf("unexpected null: %s", b)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Tool / ToolFunction JSON round-trip
// ---------------------------------------------------------------------------

func TestTool_JSONRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		tool llmrouter.Tool
	}{
		{
			"basic-function",
			llmrouter.Tool{
				Type: "function",
				Function: llmrouter.ToolFunction{
					Name:        "get_weather",
					Description: "Get current weather",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
				},
			},
		},
		{
			"no-description",
			llmrouter.Tool{
				Type:     "function",
				Function: llmrouter.ToolFunction{Name: "ping"},
			},
		},
		{
			"no-parameters",
			llmrouter.Tool{
				Type:     "function",
				Function: llmrouter.ToolFunction{Name: "now", Description: "current time"},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.tool)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got llmrouter.Tool
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got.Type != tc.tool.Type || got.Function.Name != tc.tool.Function.Name {
				t.Errorf("mismatch: %+v vs %+v", got, tc.tool)
			}
		})
	}
}

func TestToolFunction_OmitemptyDescriptionAndParameters(t *testing.T) {
	tf := llmrouter.ToolFunction{Name: "x"}
	b, _ := json.Marshal(tf)
	if strings.Contains(string(b), "description") {
		t.Errorf("description not omitted: %s", b)
	}
	if strings.Contains(string(b), "parameters") {
		t.Errorf("parameters not omitted: %s", b)
	}
}

func TestChatRequest_ToolsAndToolChoiceMarshaled(t *testing.T) {
	req := llmrouter.ChatRequest{
		Model:    "gpt-4o-mini",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		Tools: []llmrouter.Tool{
			{Type: "function", Function: llmrouter.ToolFunction{Name: "f"}},
		},
		ToolChoice: &llmrouter.ToolChoice{Mode: "required"},
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(b)
	if !strings.Contains(s, `"tools":`) {
		t.Errorf("tools missing: %s", s)
	}
	if !strings.Contains(s, `"tool_choice":"required"`) {
		t.Errorf("tool_choice missing: %s", s)
	}
}

func TestChatRequest_ToolsOmittedWhenEmpty(t *testing.T) {
	req := llmrouter.ChatRequest{
		Model:    "gpt-4o",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	}
	b, _ := json.Marshal(req)
	s := string(b)
	if strings.Contains(s, "tools") {
		t.Errorf("tools should be omitted: %s", s)
	}
	if strings.Contains(s, "tool_choice") {
		t.Errorf("tool_choice should be omitted: %s", s)
	}
}

// ---------------------------------------------------------------------------
// ToolCallDelta JSON round-trip
// ---------------------------------------------------------------------------

func TestToolCallDelta_JSONRoundTrip(t *testing.T) {
	cases := []struct {
		name string
		tcd  llmrouter.ToolCallDelta
	}{
		{
			"full",
			llmrouter.ToolCallDelta{
				Index: 0,
				ID:    "call_abc",
				Type:  "function",
				Function: &llmrouter.ToolCallFunctionDelta{
					Name:      "get_weather",
					Arguments: `{"city":`,
				},
			},
		},
		{
			"index-only",
			llmrouter.ToolCallDelta{Index: 2},
		},
		{
			"arguments-fragment",
			llmrouter.ToolCallDelta{
				Index:    1,
				Function: &llmrouter.ToolCallFunctionDelta{Arguments: `"NYC"}`},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.tcd)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got llmrouter.ToolCallDelta
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got.Index != tc.tcd.Index || got.ID != tc.tcd.ID || got.Type != tc.tcd.Type {
				t.Errorf("header mismatch: %+v vs %+v", got, tc.tcd)
			}
			if (got.Function == nil) != (tc.tcd.Function == nil) {
				t.Errorf("Function nilness mismatch")
			}
			if got.Function != nil && tc.tcd.Function != nil {
				if got.Function.Name != tc.tcd.Function.Name || got.Function.Arguments != tc.tcd.Function.Arguments {
					t.Errorf("Function mismatch: %+v vs %+v", got.Function, tc.tcd.Function)
				}
			}
		})
	}
}

func TestDelta_WithToolCallsMarshalShape(t *testing.T) {
	d := llmrouter.Delta{
		ToolCalls: []llmrouter.ToolCallDelta{
			{Index: 0, ID: "x", Type: "function", Function: &llmrouter.ToolCallFunctionDelta{Name: "f"}},
		},
	}
	b, _ := json.Marshal(d)
	s := string(b)
	if !strings.Contains(s, `"tool_calls"`) {
		t.Errorf("tool_calls missing: %s", s)
	}
	if !strings.Contains(s, `"index":0`) {
		t.Errorf("index missing: %s", s)
	}
}

func TestDelta_ToolCallsOmittedWhenEmpty(t *testing.T) {
	d := llmrouter.Delta{Content: "hi"}
	b, _ := json.Marshal(d)
	if strings.Contains(string(b), "tool_calls") {
		t.Errorf("tool_calls should be omitted: %s", b)
	}
}

func TestDelta_ThinkingMarshaling(t *testing.T) {
	cases := []struct {
		name string
		d    llmrouter.Delta
		want string
	}{
		{"empty-thinking-omitted", llmrouter.Delta{Content: "x"}, `{"content":"x"}`},
		{"thinking-set", llmrouter.Delta{Thinking: "reasoning"}, `{"thinking":"reasoning"}`},
		{"both", llmrouter.Delta{Content: "x", Thinking: "y"}, `{"content":"x","thinking":"y"}`},
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

// ---------------------------------------------------------------------------
// Usage cache token fields
// ---------------------------------------------------------------------------

func TestUsage_CacheTokenFields(t *testing.T) {
	u := llmrouter.Usage{
		PromptTokens:        100,
		CompletionTokens:    50,
		TotalTokens:         150,
		CachedPromptTokens:  80,
		CacheCreationTokens: 20,
	}
	b, err := json.Marshal(u)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(b)
	if !strings.Contains(s, `"cached_prompt_tokens":80`) {
		t.Errorf("cached_prompt_tokens missing: %s", s)
	}
	if !strings.Contains(s, `"cache_creation_tokens":20`) {
		t.Errorf("cache_creation_tokens missing: %s", s)
	}
	var got llmrouter.Usage
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got != u {
		t.Errorf("round-trip mismatch: %+v vs %+v", got, u)
	}
}

func TestUsage_CacheTokensOmittedWhenZero(t *testing.T) {
	u := llmrouter.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}
	b, _ := json.Marshal(u)
	s := string(b)
	if strings.Contains(s, "cached_prompt_tokens") {
		t.Errorf("cached_prompt_tokens leaked: %s", s)
	}
	if strings.Contains(s, "cache_creation_tokens") {
		t.Errorf("cache_creation_tokens leaked: %s", s)
	}
}

// ---------------------------------------------------------------------------
// ContentPart constructors and MultipartMessage
// ---------------------------------------------------------------------------

func TestText_Constructor(t *testing.T) {
	cases := []string{"", "hi", "héllo", "multi\nline", "🎉"}
	for _, s := range cases {
		t.Run(s, func(t *testing.T) {
			p := llmrouter.Text(s)
			if p.Type != "text" {
				t.Errorf("Type = %q", p.Type)
			}
			if p.Text != s {
				t.Errorf("Text = %q", p.Text)
			}
		})
	}
}

func TestImageURL_Constructor(t *testing.T) {
	cases := []string{
		"https://example.com/a.png",
		"https://example.com/b.jpg",
		"http://localhost/c",
		"",
	}
	for _, u := range cases {
		t.Run(u, func(t *testing.T) {
			p := llmrouter.ImageURL(u)
			if p.Type != "image_url" {
				t.Errorf("Type = %q", p.Type)
			}
			if p.URL != u {
				t.Errorf("URL = %q", p.URL)
			}
		})
	}
}

func TestImageBytes_Constructor(t *testing.T) {
	cases := []struct {
		name      string
		data      []byte
		mediaType string
	}{
		{"png-1byte", []byte{1}, "image/png"},
		{"png-empty", []byte{}, "image/png"},
		{"jpg-many", []byte{1, 2, 3, 4, 5}, "image/jpeg"},
		{"webp", []byte{9, 9, 9}, "image/webp"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := llmrouter.ImageBytes(tc.data, tc.mediaType)
			if p.Type != "image_bytes" {
				t.Errorf("Type = %q", p.Type)
			}
			if p.MediaType != tc.mediaType {
				t.Errorf("MediaType = %q", p.MediaType)
			}
			if len(p.Data) != len(tc.data) {
				t.Errorf("Data len = %d", len(p.Data))
			}
		})
	}
}

func TestMultipartMessage_TextOnly(t *testing.T) {
	m := llmrouter.MultipartMessage("user", llmrouter.Text("hello"), llmrouter.Text(" world"))
	if m.Role != "user" {
		t.Errorf("Role = %q", m.Role)
	}
	var blocks []map[string]any
	if err := json.Unmarshal(m.Content, &blocks); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("blocks = %d", len(blocks))
	}
	if blocks[0]["type"] != "text" || blocks[0]["text"] != "hello" {
		t.Errorf("block[0] = %+v", blocks[0])
	}
	if blocks[1]["type"] != "text" || blocks[1]["text"] != " world" {
		t.Errorf("block[1] = %+v", blocks[1])
	}
}

func TestMultipartMessage_WithImageURL(t *testing.T) {
	m := llmrouter.MultipartMessage("user",
		llmrouter.Text("see:"),
		llmrouter.ImageURL("https://example.com/cat.png"),
	)
	var blocks []map[string]any
	if err := json.Unmarshal(m.Content, &blocks); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("blocks = %d", len(blocks))
	}
	if blocks[1]["type"] != "image_url" {
		t.Errorf("block[1].type = %v", blocks[1]["type"])
	}
	imgURL, _ := blocks[1]["image_url"].(map[string]any)
	if imgURL["url"] != "https://example.com/cat.png" {
		t.Errorf("url = %v", imgURL["url"])
	}
}

func TestMultipartMessage_WithImageBytes(t *testing.T) {
	data := []byte{0x89, 0x50, 0x4e, 0x47}
	m := llmrouter.MultipartMessage("user", llmrouter.ImageBytes(data, "image/png"))
	var blocks []map[string]any
	if err := json.Unmarshal(m.Content, &blocks); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("blocks = %d", len(blocks))
	}
	imgURL, _ := blocks[0]["image_url"].(map[string]any)
	url, _ := imgURL["url"].(string)
	if !strings.HasPrefix(url, "data:image/png;base64,") {
		t.Errorf("url prefix wrong: %s", url)
	}
	// The base64 of {0x89,0x50,0x4e,0x47} is "iVBORw=="
	if !strings.HasSuffix(url, "iVBORw==") {
		t.Errorf("url suffix wrong: %s", url)
	}
}

func TestMultipartMessage_EmptyParts(t *testing.T) {
	m := llmrouter.MultipartMessage("user")
	var blocks []map[string]any
	if err := json.Unmarshal(m.Content, &blocks); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(blocks) != 0 {
		t.Fatalf("blocks should be empty: %d", len(blocks))
	}
}

func TestMultipartMessage_RoleVariations(t *testing.T) {
	cases := []string{"user", "assistant", "system", "tool", ""}
	for _, role := range cases {
		t.Run("role="+role, func(t *testing.T) {
			m := llmrouter.MultipartMessage(role, llmrouter.Text("x"))
			if m.Role != role {
				t.Errorf("Role = %q", m.Role)
			}
		})
	}
}

func TestMultipartMessage_UnknownPartTypeSkipped(t *testing.T) {
	// Construct a ContentPart with an unrecognized Type.
	part := llmrouter.ContentPart{Type: "audio", Text: "ignored"}
	m := llmrouter.MultipartMessage("user", part, llmrouter.Text("kept"))
	var blocks []map[string]any
	if err := json.Unmarshal(m.Content, &blocks); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(blocks) != 1 {
		t.Fatalf("blocks = %d, want 1 (unknown skipped)", len(blocks))
	}
	if blocks[0]["text"] != "kept" {
		t.Errorf("kept block missing: %+v", blocks[0])
	}
}

// ---------------------------------------------------------------------------
// ToolResultMessage
// ---------------------------------------------------------------------------

func TestToolResultMessage_Constructs(t *testing.T) {
	cases := []struct {
		name       string
		toolCallID string
		content    string
	}{
		{"basic", "call_abc", "sunny"},
		{"json-content", "call_1", `{"weather":"sunny"}`},
		{"empty-content", "call_2", ""},
		{"unicode", "call_unicode", "結果: ok"},
		{"long-id", "call_" + strings.Repeat("x", 200), "ok"},
		{"with-quotes", "call_q", `she said "hi"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := llmrouter.ToolResultMessage(tc.toolCallID, tc.content)
			if m.Role != "tool" {
				t.Errorf("Role = %q, want tool", m.Role)
			}
			if m.ToolCallID != tc.toolCallID {
				t.Errorf("ToolCallID = %q, want %q", m.ToolCallID, tc.toolCallID)
			}
			var got string
			if err := json.Unmarshal(m.Content, &got); err != nil {
				t.Fatalf("content not a JSON string: %v (raw=%s)", err, string(m.Content))
			}
			if got != tc.content {
				t.Errorf("content mismatch: got=%q want=%q", got, tc.content)
			}
			if m.Name != "" {
				t.Errorf("Name should be empty, got %q", m.Name)
			}
		})
	}
}

func TestMessage_OmitemptyForToolFields(t *testing.T) {
	cases := []struct {
		name string
		msg  llmrouter.Message
	}{
		{"text-message", llmrouter.TextMessage("user", "hi")},
		{"assistant-empty", llmrouter.TextMessage("assistant", "")},
		{"system-message", llmrouter.TextMessage("system", "be terse")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := json.Marshal(tc.msg)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			s := string(b)
			if strings.Contains(s, "tool_call_id") {
				t.Errorf("tool_call_id should be omitted: %s", s)
			}
			if strings.Contains(s, `"name"`) {
				t.Errorf("name should be omitted: %s", s)
			}
		})
	}
}

func TestMessage_WithToolCallID_Marshals(t *testing.T) {
	cases := []struct {
		name       string
		toolCallID string
		content    string
		wantSub    string
	}{
		{"basic", "call_abc", "sunny", `"tool_call_id":"call_abc"`},
		{"numeric-id", "12345", "ok", `"tool_call_id":"12345"`},
		{"hyphenated", "call-id-1", "ok", `"tool_call_id":"call-id-1"`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := llmrouter.ToolResultMessage(tc.toolCallID, tc.content)
			b, err := json.Marshal(m)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if !strings.Contains(string(b), tc.wantSub) {
				t.Errorf("missing %q in %s", tc.wantSub, b)
			}
			if !strings.Contains(string(b), `"role":"tool"`) {
				t.Errorf("missing role=tool in %s", b)
			}
		})
	}
}
