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
// helpers
// ---------------------------------------------------------------------------

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}
