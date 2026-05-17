package gemini

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestMarshalSpeechWithRaw_NilRaw verifies the no-Raw path produces the
// canonical typed body unchanged.
func TestMarshalSpeechWithRaw_NilRaw(t *testing.T) {
	body := map[string]any{
		"contents": []map[string]any{
			{"parts": []map[string]any{{"text": "hi"}}},
		},
		"generationConfig": map[string]any{
			"responseModalities": []string{"AUDIO"},
		},
	}
	out, err := marshalSpeechWithRaw(body, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, ok := decoded["contents"]; !ok {
		t.Errorf("contents key missing in output: %s", out)
	}
	if _, ok := decoded["generationConfig"]; !ok {
		t.Errorf("generationConfig key missing in output: %s", out)
	}
}

// TestMarshalSpeechWithRaw_EmptyRaw covers the explicit-empty branch where
// raw is non-nil but zero-length — should behave identically to nil.
func TestMarshalSpeechWithRaw_EmptyRaw(t *testing.T) {
	body := map[string]any{"contents": []any{}}
	out, err := marshalSpeechWithRaw(body, json.RawMessage(""))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(string(out), `"contents"`) {
		t.Errorf("output missing contents: %s", out)
	}
}

// TestMarshalSpeechWithRaw_OverlayExtras adds caller-supplied vendor
// extras (a field that doesn't already exist on body) and confirms it
// flows through to the final JSON.
func TestMarshalSpeechWithRaw_OverlayExtras(t *testing.T) {
	body := map[string]any{
		"contents": []any{},
	}
	raw := json.RawMessage(`{"safetySettings":[{"category":"HARM_NONE","threshold":"OFF"}]}`)
	out, err := marshalSpeechWithRaw(body, raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, ok := decoded["safetySettings"]; !ok {
		t.Errorf("safetySettings should be overlaid: %s", out)
	}
}

// TestMarshalSpeechWithRaw_TypedKeysWin verifies that when Raw carries a
// key that already exists on the typed body, the typed value wins. The
// comment on the function reads "without clobbering known keys" — this
// nails that semantic down.
func TestMarshalSpeechWithRaw_TypedKeysWin(t *testing.T) {
	body := map[string]any{
		"contents": []map[string]any{
			{"parts": []map[string]any{{"text": "typed"}}},
		},
	}
	raw := json.RawMessage(`{"contents":"raw-wins"}`)
	out, err := marshalSpeechWithRaw(body, raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	// contents must be the typed []map form, not the raw string.
	if _, ok := decoded["contents"].(string); ok {
		t.Errorf("typed contents should win over Raw; got string: %s", out)
	}
	if _, ok := decoded["contents"].([]any); !ok {
		t.Errorf("typed contents should be an array; got %T: %s", decoded["contents"], out)
	}
}

// TestMarshalSpeechWithRaw_GenerationConfigCallerCannotOverride locks down
// that even a speechConfig override in Raw cannot defeat the typed
// generationConfig — Raw overlay is shallow (top-level keys only) by
// design.
func TestMarshalSpeechWithRaw_GenerationConfigCallerCannotOverride(t *testing.T) {
	body := map[string]any{
		"generationConfig": map[string]any{
			"responseModalities": []string{"AUDIO"},
		},
	}
	raw := json.RawMessage(`{"generationConfig":{"responseModalities":["TEXT"]}}`)
	out, err := marshalSpeechWithRaw(body, raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(out, &decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	gc, ok := decoded["generationConfig"].(map[string]any)
	if !ok {
		t.Fatalf("generationConfig missing: %s", out)
	}
	modalities, ok := gc["responseModalities"].([]any)
	if !ok || len(modalities) == 0 {
		t.Fatalf("responseModalities missing: %s", out)
	}
	if modalities[0] != "AUDIO" {
		t.Errorf("typed generationConfig should win; got modalities[0]=%v", modalities[0])
	}
}

// TestMarshalSpeechWithRaw_InvalidJSONErrors covers the error path when
// Raw is syntactically broken — the wrapper must surface "invalid raw
// body" via fmt.Errorf.
func TestMarshalSpeechWithRaw_InvalidJSONErrors(t *testing.T) {
	body := map[string]any{}
	raw := json.RawMessage(`{this is not json`)
	_, err := marshalSpeechWithRaw(body, raw)
	if err == nil {
		t.Fatal("expected error for malformed raw body")
	}
	if !strings.Contains(err.Error(), "invalid raw body") {
		t.Errorf("error should mention 'invalid raw body'; got %v", err)
	}
}
