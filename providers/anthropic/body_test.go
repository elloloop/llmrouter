package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
)

// ---------------------------------------------------------------------------
// buildAnthropicBody — system lifting
// ---------------------------------------------------------------------------

func TestBuildAnthropicBody_SystemLifting(t *testing.T) {
	cases := []struct {
		name     string
		messages []llmrouter.Message
		wantSys  string // "" means key absent
		wantMsgs int
	}{
		{
			"no-system",
			[]llmrouter.Message{
				llmrouter.TextMessage("user", "hi"),
			},
			"",
			1,
		},
		{
			"one-system",
			[]llmrouter.Message{
				llmrouter.TextMessage("system", "be concise"),
				llmrouter.TextMessage("user", "hi"),
			},
			"be concise",
			1,
		},
		{
			"two-systems-joined",
			[]llmrouter.Message{
				llmrouter.TextMessage("system", "a"),
				llmrouter.TextMessage("system", "b"),
				llmrouter.TextMessage("user", "hi"),
			},
			"a\n\nb",
			1,
		},
		{
			"three-systems-joined",
			[]llmrouter.Message{
				llmrouter.TextMessage("system", "x"),
				llmrouter.TextMessage("system", "y"),
				llmrouter.TextMessage("system", "z"),
				llmrouter.TextMessage("user", "hi"),
			},
			"x\n\ny\n\nz",
			1,
		},
		{
			"system-interleaved",
			[]llmrouter.Message{
				llmrouter.TextMessage("user", "u1"),
				llmrouter.TextMessage("assistant", "a1"),
				llmrouter.TextMessage("system", "sys"),
				llmrouter.TextMessage("user", "u2"),
			},
			"sys",
			3,
		},
		{
			"alternating-conversation",
			[]llmrouter.Message{
				llmrouter.TextMessage("system", "rules"),
				llmrouter.TextMessage("user", "u1"),
				llmrouter.TextMessage("assistant", "a1"),
				llmrouter.TextMessage("user", "u2"),
				llmrouter.TextMessage("assistant", "a2"),
				llmrouter.TextMessage("user", "u3"),
			},
			"rules",
			5,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := buildAnthropicBody(llmrouter.ChatRequest{
				Model:    "claude-3-5-sonnet",
				Messages: tc.messages,
			})
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			var m map[string]any
			if err := json.Unmarshal(b, &m); err != nil {
				t.Fatalf("invalid json: %v", err)
			}
			if tc.wantSys == "" {
				if _, ok := m["system"]; ok {
					t.Errorf("system should be absent: %s", b)
				}
			} else {
				if m["system"] != tc.wantSys {
					t.Errorf("system = %v, want %q", m["system"], tc.wantSys)
				}
			}
			msgs, _ := m["messages"].([]any)
			if len(msgs) != tc.wantMsgs {
				t.Errorf("messages len = %d, want %d", len(msgs), tc.wantMsgs)
			}
			// system role must never leak into messages array
			for _, raw := range msgs {
				if obj, ok := raw.(map[string]any); ok && obj["role"] == "system" {
					t.Errorf("system role leaked into messages")
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// buildAnthropicBody — max_tokens defaults
// ---------------------------------------------------------------------------

func TestBuildAnthropicBody_MaxTokens(t *testing.T) {
	cases := []struct {
		name string
		in   int
		want float64
	}{
		{"unset-defaults-4096", 0, 4096},
		{"negative-defaults-4096", -1, 4096},
		{"set-256", 256, 256},
		{"set-1", 1, 1},
		{"set-large", 200000, 200000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := buildAnthropicBody(llmrouter.ChatRequest{
				Model:     "claude",
				MaxTokens: tc.in,
				Messages:  []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
			})
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			var m map[string]any
			if err := json.Unmarshal(b, &m); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if m["max_tokens"] != tc.want {
				t.Fatalf("max_tokens = %v (%T), want %v", m["max_tokens"], m["max_tokens"], tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// buildAnthropicBody — stream flag and model
// ---------------------------------------------------------------------------

func TestBuildAnthropicBody_AlwaysStreamTrue(t *testing.T) {
	cases := []bool{false, true}
	for _, in := range cases {
		t.Run("stream-input="+boolStr(in), func(t *testing.T) {
			b, _ := buildAnthropicBody(llmrouter.ChatRequest{Model: "c", Stream: in, Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")}})
			if !strings.Contains(string(b), `"stream":true`) {
				t.Fatalf("missing stream:true: %s", b)
			}
		})
	}
}

func TestBuildAnthropicBody_ModelPresent(t *testing.T) {
	cases := []string{
		"claude-3-5-sonnet-20241022",
		"claude-3-opus-20240229",
		"claude-3-haiku-20240307",
		"claude-3-5-sonnet-latest",
	}
	for _, model := range cases {
		t.Run(model, func(t *testing.T) {
			b, err := buildAnthropicBody(llmrouter.ChatRequest{Model: model, Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")}})
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			if !strings.Contains(string(b), `"model":"`+model+`"`) {
				t.Fatalf("model missing: %s", b)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// buildAnthropicBody — typed tuning fields
// ---------------------------------------------------------------------------

func TestBuildAnthropicBody_TypedFields(t *testing.T) {
	temp := 0.3
	top := 0.85
	b, err := buildAnthropicBody(llmrouter.ChatRequest{
		Model:       "claude",
		Messages:    []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		Temperature: &temp,
		TopP:        &top,
		Stop:        []string{"\n\n", "END"},
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	s := string(b)
	if !strings.Contains(s, `"temperature":0.3`) {
		t.Errorf("temperature missing: %s", s)
	}
	if !strings.Contains(s, `"top_p":0.85`) {
		t.Errorf("top_p missing: %s", s)
	}
	if !strings.Contains(s, `"stop_sequences":["\n\n","END"]`) {
		t.Errorf("stop_sequences renamed/included: %s", s)
	}
	// The OpenAI-named "stop" key must NOT leak through.
	if strings.Contains(s, `"stop":`) && !strings.Contains(s, `"stop_sequences":`) {
		t.Errorf("stop key leaked: %s", s)
	}
}

func TestBuildAnthropicBody_TypedFields_OmittedWhenAbsent(t *testing.T) {
	b, err := buildAnthropicBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	s := string(b)
	for _, key := range []string{`"temperature"`, `"top_p"`, `"top_k"`, `"stop_sequences"`} {
		if strings.Contains(s, key) {
			t.Errorf("%s should be absent: %s", key, s)
		}
	}
}

// ---------------------------------------------------------------------------
// buildAnthropicBody — Raw passthrough lifts known knobs
// ---------------------------------------------------------------------------

func TestBuildAnthropicBody_RawTuningKnobsLifted(t *testing.T) {
	raw := json.RawMessage(`{
		"model":"x",
		"messages":[],
		"temperature":0.42,
		"top_p":0.9,
		"top_k":40,
		"stop":["X","Y"]
	}`)
	b, err := buildAnthropicBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		Raw:      raw,
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	s := string(b)
	for _, want := range []string{
		`"temperature":0.42`,
		`"top_p":0.9`,
		`"top_k":40`,
		`"stop_sequences":["X","Y"]`,
	} {
		if !strings.Contains(s, want) {
			t.Errorf("missing %q in %s", want, s)
		}
	}
	if strings.Contains(s, `"stop":`) && !strings.Contains(s, `"stop_sequences":`) {
		t.Errorf("raw 'stop' should be renamed to 'stop_sequences'")
	}
}

func TestBuildAnthropicBody_RawIgnoredWhenUnparseable(t *testing.T) {
	// Malformed raw should not cause buildAnthropicBody to fail — it
	// silently falls back to typed-field semantics.
	temp := 0.99
	b, err := buildAnthropicBody(llmrouter.ChatRequest{
		Model:       "claude",
		Messages:    []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		Temperature: &temp,
		Raw:         json.RawMessage(`not json`),
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	// With raw present the typed fields path is skipped; with malformed raw we
	// expect no temperature key to appear. The contract is "either-or".
	s := string(b)
	if strings.Contains(s, `"temperature":0.99`) {
		t.Errorf("typed temperature leaked into output despite raw being set (even if malformed): %s", s)
	}
}

// ---------------------------------------------------------------------------
// buildAnthropicBody — system from non-string content
// ---------------------------------------------------------------------------

func TestBuildAnthropicBody_SystemFromMultipartContent(t *testing.T) {
	system := llmrouter.Message{
		Role:    "system",
		Content: json.RawMessage(`[{"type":"text","text":"part1 "},{"type":"text","text":"part2"}]`),
	}
	b, err := buildAnthropicBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{system, llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(string(b), `"system":"part1 part2"`) {
		t.Errorf("expected concatenated system, got: %s", b)
	}
}

// ---------------------------------------------------------------------------
// buildAnthropicBody — tools
// ---------------------------------------------------------------------------

func TestBuildAnthropicBody_Tools_EmitsAnthropicShape(t *testing.T) {
	req := llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		Tools: []llmrouter.Tool{
			{
				Type: "function",
				Function: llmrouter.ToolFunction{
					Name:        "get_weather",
					Description: "Look up weather",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
				},
			},
		},
	}
	b, err := buildAnthropicBody(req)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	tools, ok := m["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v", m["tools"])
	}
	t0 := tools[0].(map[string]any)
	if t0["name"] != "get_weather" {
		t.Errorf("name = %v", t0["name"])
	}
	if t0["description"] != "Look up weather" {
		t.Errorf("description = %v", t0["description"])
	}
	if _, ok := t0["input_schema"]; !ok {
		t.Errorf("input_schema missing: %v", t0)
	}
	// Anthropic does NOT wrap with "function" / "parameters" -- check flat shape.
	if _, ok := t0["function"]; ok {
		t.Errorf("function wrapper leaked into Anthropic body: %v", t0)
	}
	if _, ok := t0["parameters"]; ok {
		t.Errorf("'parameters' should be 'input_schema': %v", t0)
	}
}

func TestBuildAnthropicBody_Tools_OmittedWhenEmpty(t *testing.T) {
	b, _ := buildAnthropicBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if strings.Contains(string(b), `"tools"`) {
		t.Errorf("tools key should be absent: %s", b)
	}
}

func TestBuildAnthropicBody_Tools_NoDescriptionOmitted(t *testing.T) {
	req := llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		Tools: []llmrouter.Tool{
			{Type: "function", Function: llmrouter.ToolFunction{Name: "ping"}},
		},
	}
	b, _ := buildAnthropicBody(req)
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	tools := m["tools"].([]any)
	t0 := tools[0].(map[string]any)
	if _, ok := t0["description"]; ok {
		t.Errorf("description should be absent: %+v", t0)
	}
}

func TestBuildAnthropicBody_Tools_MultipleTools(t *testing.T) {
	req := llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		Tools: []llmrouter.Tool{
			{Type: "function", Function: llmrouter.ToolFunction{Name: "f1"}},
			{Type: "function", Function: llmrouter.ToolFunction{Name: "f2"}},
			{Type: "function", Function: llmrouter.ToolFunction{Name: "f3"}},
		},
	}
	b, _ := buildAnthropicBody(req)
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	tools := m["tools"].([]any)
	if len(tools) != 3 {
		t.Fatalf("tools len = %d", len(tools))
	}
}

// ---------------------------------------------------------------------------
// buildAnthropicBody — tool_choice
// ---------------------------------------------------------------------------

func TestBuildAnthropicBody_ToolChoice_AllModes(t *testing.T) {
	cases := []struct {
		name       string
		choice     llmrouter.ToolChoice
		wantType   string
		wantName   string
		wantNoName bool
	}{
		{"auto", llmrouter.ToolChoice{Mode: "auto"}, "auto", "", true},
		{"none", llmrouter.ToolChoice{Mode: "none"}, "none", "", true},
		{"required", llmrouter.ToolChoice{Mode: "required"}, "any", "", true},
		{"specific", llmrouter.ToolChoice{Mode: "specific", Function: "get_weather"}, "tool", "get_weather", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := llmrouter.ChatRequest{
				Model:      "claude",
				Messages:   []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
				ToolChoice: &tc.choice,
			}
			b, err := buildAnthropicBody(req)
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			var m map[string]any
			if err := json.Unmarshal(b, &m); err != nil {
				t.Fatalf("invalid json: %v", err)
			}
			ch, ok := m["tool_choice"].(map[string]any)
			if !ok {
				t.Fatalf("tool_choice = %v", m["tool_choice"])
			}
			if ch["type"] != tc.wantType {
				t.Errorf("type = %v, want %s", ch["type"], tc.wantType)
			}
			if tc.wantNoName {
				if _, ok := ch["name"]; ok {
					t.Errorf("name should be absent: %+v", ch)
				}
			} else {
				if ch["name"] != tc.wantName {
					t.Errorf("name = %v, want %s", ch["name"], tc.wantName)
				}
			}
		})
	}
}

func TestBuildAnthropicBody_ToolChoice_NilOmitted(t *testing.T) {
	b, _ := buildAnthropicBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if strings.Contains(string(b), "tool_choice") {
		t.Errorf("tool_choice should be absent: %s", b)
	}
}

func TestBuildAnthropicBody_ToolChoice_UnknownModeDefaultsAuto(t *testing.T) {
	choice := llmrouter.ToolChoice{Mode: "unknown_thing"}
	b, _ := buildAnthropicBody(llmrouter.ChatRequest{
		Model:      "claude",
		Messages:   []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		ToolChoice: &choice,
	})
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	ch := m["tool_choice"].(map[string]any)
	if ch["type"] != "auto" {
		t.Errorf("type = %v, want auto", ch["type"])
	}
}

// ---------------------------------------------------------------------------
// buildAnthropicBody — multimodal translation
// ---------------------------------------------------------------------------

func TestBuildAnthropicBody_Multimodal_TextOnlyUnchanged(t *testing.T) {
	msg := llmrouter.MultipartMessage("user", llmrouter.Text("hello"))
	b, err := buildAnthropicBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{msg},
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(string(b), `"text":"hello"`) {
		t.Errorf("text block missing: %s", b)
	}
}

func TestBuildAnthropicBody_Multimodal_ImageURLTranslated(t *testing.T) {
	msg := llmrouter.MultipartMessage("user",
		llmrouter.Text("see:"),
		llmrouter.ImageURL("https://example.com/a.png"),
	)
	b, err := buildAnthropicBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{msg},
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(b, &body); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	msgs := body["messages"].([]any)
	m0 := msgs[0].(map[string]any)
	blocks := m0["content"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("blocks = %d", len(blocks))
	}
	imgBlock := blocks[1].(map[string]any)
	if imgBlock["type"] != "image" {
		t.Errorf("type = %v, want image", imgBlock["type"])
	}
	src := imgBlock["source"].(map[string]any)
	if src["type"] != "url" {
		t.Errorf("source.type = %v", src["type"])
	}
	if src["url"] != "https://example.com/a.png" {
		t.Errorf("source.url = %v", src["url"])
	}
	// Make sure the OpenAI-style image_url shape didn't leak through.
	if _, ok := imgBlock["image_url"]; ok {
		t.Errorf("image_url leaked: %+v", imgBlock)
	}
}

func TestBuildAnthropicBody_Multimodal_Base64DataURLTranslated(t *testing.T) {
	data := []byte{0x89, 0x50, 0x4e, 0x47}
	msg := llmrouter.MultipartMessage("user",
		llmrouter.Text("look:"),
		llmrouter.ImageBytes(data, "image/png"),
	)
	b, err := buildAnthropicBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{msg},
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(b, &body); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	msgs := body["messages"].([]any)
	m0 := msgs[0].(map[string]any)
	blocks := m0["content"].([]any)
	imgBlock := blocks[1].(map[string]any)
	if imgBlock["type"] != "image" {
		t.Errorf("type = %v", imgBlock["type"])
	}
	src := imgBlock["source"].(map[string]any)
	if src["type"] != "base64" {
		t.Errorf("source.type = %v, want base64", src["type"])
	}
	if src["media_type"] != "image/png" {
		t.Errorf("media_type = %v", src["media_type"])
	}
	if src["data"] != "iVBORw==" {
		t.Errorf("data = %v", src["data"])
	}
}

func TestBuildAnthropicBody_Multimodal_PlainStringContentUnchanged(t *testing.T) {
	// Existing tests already cover this via TextMessage, but assert
	// explicitly that the translation helper doesn't break plain strings.
	b, err := buildAnthropicBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "plain")},
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(string(b), `"content":"plain"`) {
		t.Errorf("plain string content not preserved: %s", b)
	}
}

func TestBuildAnthropicBody_Multimodal_UnknownBlockTypePassthrough(t *testing.T) {
	raw := json.RawMessage(`[{"type":"audio","audio":{"url":"x"}}]`)
	msg := llmrouter.Message{Role: "user", Content: raw}
	b, err := buildAnthropicBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{msg},
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(string(b), `"audio"`) {
		t.Errorf("unknown block dropped: %s", b)
	}
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}
