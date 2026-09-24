package openaiwire

import (
	"encoding/json"
	"testing"

	"github.com/elloloop/llmrouter"
)

func TestSetResponseFormat_NilSchemaLeavesTheBodyAlone(t *testing.T) {
	body := map[string]json.RawMessage{"model": json.RawMessage(`"x"`)}
	if err := SetResponseFormat(body, nil); err != nil {
		t.Fatal(err)
	}
	if _, ok := body[keyResponseFormat]; ok || len(body) != 1 {
		t.Fatalf("body = %v, want untouched", body)
	}
}

func TestSetResponseFormat_RendersTheJSONSchemaEnvelope(t *testing.T) {
	body := map[string]json.RawMessage{}
	schema := &llmrouter.ResponseSchema{Name: "answer", Strict: true, Schema: json.RawMessage(`{"type":"object"}`)}
	if err := SetResponseFormat(body, schema); err != nil {
		t.Fatal(err)
	}
	var got struct {
		Type       string `json:"type"`
		JSONSchema struct {
			Name        string          `json:"name"`
			Strict      bool            `json:"strict"`
			Description *string         `json:"description"`
			Schema      json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	}
	if err := json.Unmarshal(body[keyResponseFormat], &got); err != nil {
		t.Fatal(err)
	}
	if got.Type != "json_schema" || got.JSONSchema.Name != "answer" || !got.JSONSchema.Strict ||
		string(got.JSONSchema.Schema) != `{"type":"object"}` || got.JSONSchema.Description != nil {
		t.Fatalf("response_format = %s", body[keyResponseFormat])
	}
}

func TestSetResponseFormat_ACallersOwnFormatWins(t *testing.T) {
	own := json.RawMessage(`{"type":"json_object"}`)
	body := map[string]json.RawMessage{keyResponseFormat: own}
	if err := SetResponseFormat(body, &llmrouter.ResponseSchema{Name: "answer"}); err != nil {
		t.Fatal(err)
	}
	if string(body[keyResponseFormat]) != string(own) {
		t.Fatalf("response_format = %s, want the caller's %s", body[keyResponseFormat], own)
	}
}

func TestUseMaxCompletionTokens(t *testing.T) {
	cases := []struct {
		name string
		in   map[string]json.RawMessage
		want map[string]string
	}{
		{"renames the deprecated field", map[string]json.RawMessage{keyMaxTokens: json.RawMessage(`500`)},
			map[string]string{keyMaxCompletionTokens: "500"}},
		{"keeps an explicit max_completion_tokens",
			map[string]json.RawMessage{keyMaxTokens: json.RawMessage(`500`), keyMaxCompletionTokens: json.RawMessage(`900`)},
			map[string]string{keyMaxCompletionTokens: "900"}},
		{"no limit, no field", map[string]json.RawMessage{}, map[string]string{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			UseMaxCompletionTokens(tc.in)
			if len(tc.in) != len(tc.want) {
				t.Fatalf("body = %v, want %v", tc.in, tc.want)
			}
			for k, v := range tc.want {
				if string(tc.in[k]) != v {
					t.Fatalf("%s = %s, want %s", k, tc.in[k], v)
				}
			}
		})
	}
}
