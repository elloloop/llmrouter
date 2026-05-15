package bedrock

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestBedrockFamily(t *testing.T) {
	cases := []struct {
		name    string
		modelID string
		want    string
	}{
		{"titan-v1", "amazon.titan-embed-text-v1", familyTitan},
		{"titan-v2", "amazon.titan-embed-text-v2:0", familyTitan},
		{"titan-image", "amazon.titan-embed-image-v1", familyTitan},
		{"cohere-english", "cohere.embed-english-v3", familyCohere},
		{"cohere-multilingual", "cohere.embed-multilingual-v3", familyCohere},
		{"anthropic-claude", "anthropic.claude-3-sonnet", ""},
		{"empty", "", ""},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := bedrockFamily(tc.modelID)
			if got != tc.want {
				t.Fatalf("bedrockFamily(%q) = %q, want %q", tc.modelID, got, tc.want)
			}
		})
	}
}

func TestMapCohereInputType(t *testing.T) {
	cases := []struct {
		name     string
		taskType string
		want     string
	}{
		{"retrieval-query", "RETRIEVAL_QUERY", "search_query"},
		{"retrieval-document", "RETRIEVAL_DOCUMENT", "search_document"},
		{"semantic-similarity", "SEMANTIC_SIMILARITY", "classification"},
		{"classification", "CLASSIFICATION", "classification"},
		{"clustering", "CLUSTERING", "clustering"},
		{"empty-defaults-to-document", "", "search_document"},
		{"unknown-defaults-to-document", "UNKNOWN_TASK", "search_document"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := mapCohereInputType(tc.taskType)
			if got != tc.want {
				t.Fatalf("mapCohereInputType(%q) = %q, want %q", tc.taskType, got, tc.want)
			}
		})
	}
}

func TestIsTitanV2(t *testing.T) {
	cases := []struct {
		name    string
		modelID string
		want    bool
	}{
		{"v2-zero-suffix", "amazon.titan-embed-text-v2:0", true},
		{"v2-bare", "amazon.titan-embed-text-v2", true},
		{"v1", "amazon.titan-embed-text-v1", false},
		{"image", "amazon.titan-embed-image-v1", false},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := isTitanV2(tc.modelID)
			if got != tc.want {
				t.Fatalf("isTitanV2(%q) = %v, want %v", tc.modelID, got, tc.want)
			}
		})
	}
}

func TestBuildTitanRequest_V1OmitsDimensions(t *testing.T) {
	body, err := buildTitanRequest("hello world", 512, "amazon.titan-embed-text-v1")
	if err != nil {
		t.Fatalf("buildTitanRequest: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["inputText"] != "hello world" {
		t.Fatalf("inputText = %v", got["inputText"])
	}
	if _, ok := got["dimensions"]; ok {
		t.Fatalf("v1 should not emit dimensions, got %v", got)
	}
	if _, ok := got["normalize"]; ok {
		t.Fatalf("v1 should not emit normalize, got %v", got)
	}
}

func TestBuildTitanRequest_V2WithDimensions(t *testing.T) {
	body, err := buildTitanRequest("hi", 1024, "amazon.titan-embed-text-v2:0")
	if err != nil {
		t.Fatalf("buildTitanRequest: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["inputText"] != "hi" {
		t.Fatalf("inputText = %v", got["inputText"])
	}
	if d, ok := got["dimensions"].(float64); !ok || int(d) != 1024 {
		t.Fatalf("dimensions = %v, want 1024", got["dimensions"])
	}
	if n, ok := got["normalize"].(bool); !ok || !n {
		t.Fatalf("normalize = %v, want true", got["normalize"])
	}
}

func TestBuildTitanRequest_V2WithoutDimensions(t *testing.T) {
	body, err := buildTitanRequest("hello", 0, "amazon.titan-embed-text-v2:0")
	if err != nil {
		t.Fatalf("buildTitanRequest: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := got["dimensions"]; ok {
		t.Fatalf("zero dimensions must be omitted, got %v", got)
	}
	if n, ok := got["normalize"].(bool); !ok || !n {
		t.Fatalf("normalize = %v, want true", got["normalize"])
	}
}

func TestBuildCohereRequest_Shape(t *testing.T) {
	body, err := buildCohereRequest([]string{"a", "b", "c"}, "RETRIEVAL_QUERY")
	if err != nil {
		t.Fatalf("buildCohereRequest: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	texts, ok := got["texts"].([]any)
	if !ok || len(texts) != 3 {
		t.Fatalf("texts = %v", got["texts"])
	}
	if got["input_type"] != "search_query" {
		t.Fatalf("input_type = %v", got["input_type"])
	}
	etypes, ok := got["embedding_types"].([]any)
	if !ok || len(etypes) != 1 || etypes[0] != "float" {
		t.Fatalf("embedding_types = %v", got["embedding_types"])
	}
	if got["truncate"] != "END" {
		t.Fatalf("truncate = %v", got["truncate"])
	}
}

func TestBuildCohereRequest_DefaultTaskType(t *testing.T) {
	body, err := buildCohereRequest([]string{"x"}, "")
	if err != nil {
		t.Fatalf("buildCohereRequest: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got["input_type"] != "search_document" {
		t.Fatalf("default input_type = %v, want search_document", got["input_type"])
	}
}

func TestParseTitanResponse_Valid(t *testing.T) {
	raw := []byte(`{"embedding":[0.1,0.2,0.3],"inputTextTokenCount":7}`)
	vec, tokens, err := parseTitanResponse(raw)
	if err != nil {
		t.Fatalf("parseTitanResponse: %v", err)
	}
	if len(vec) != 3 || vec[0] != 0.1 || vec[1] != 0.2 || vec[2] != 0.3 {
		t.Fatalf("vec = %v", vec)
	}
	if tokens != 7 {
		t.Fatalf("tokens = %d, want 7", tokens)
	}
}

func TestParseTitanResponse_MalformedJSON(t *testing.T) {
	_, _, err := parseTitanResponse([]byte(`{not json`))
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "decode titan response") {
		t.Fatalf("error = %v", err)
	}
}

func TestParseTitanResponse_MissingEmbedding(t *testing.T) {
	_, _, err := parseTitanResponse([]byte(`{"inputTextTokenCount":3}`))
	if err == nil {
		t.Fatalf("expected error for missing embedding")
	}
}

func TestParseCohereResponse_TypedFloat(t *testing.T) {
	raw := []byte(`{"embeddings":{"float":[[0.1,0.2],[0.3,0.4]]},"id":"x","response_type":"embeddings_floats"}`)
	vecs, err := parseCohereResponse(raw)
	if err != nil {
		t.Fatalf("parseCohereResponse: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("len = %d, want 2", len(vecs))
	}
	if vecs[0][0] != 0.1 || vecs[1][1] != 0.4 {
		t.Fatalf("vecs = %v", vecs)
	}
}

func TestParseCohereResponse_LegacyArray(t *testing.T) {
	raw := []byte(`{"embeddings":[[0.5,0.6],[0.7,0.8]]}`)
	vecs, err := parseCohereResponse(raw)
	if err != nil {
		t.Fatalf("parseCohereResponse: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("len = %d, want 2", len(vecs))
	}
	if vecs[0][1] != 0.6 || vecs[1][0] != 0.7 {
		t.Fatalf("vecs = %v", vecs)
	}
}

func TestParseCohereResponse_SingleVector(t *testing.T) {
	raw := []byte(`{"embeddings":{"float":[[0.9,1.0,1.1]]}}`)
	vecs, err := parseCohereResponse(raw)
	if err != nil {
		t.Fatalf("parseCohereResponse: %v", err)
	}
	if len(vecs) != 1 || len(vecs[0]) != 3 {
		t.Fatalf("vecs = %v", vecs)
	}
}

func TestParseCohereResponse_MalformedJSON(t *testing.T) {
	_, err := parseCohereResponse([]byte(`{`))
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestParseCohereResponse_MissingEmbeddings(t *testing.T) {
	_, err := parseCohereResponse([]byte(`{"id":"x"}`))
	if err == nil {
		t.Fatalf("expected error for missing embeddings")
	}
}

func TestParseCohereResponse_UnrecognisedEmbeddingsShape(t *testing.T) {
	// embeddings is a string — neither object-with-float nor [][]float32.
	_, err := parseCohereResponse([]byte(`{"embeddings":"oops"}`))
	if err == nil {
		t.Fatalf("expected error for unrecognised embeddings shape")
	}
}

func TestParseCohereResponse_EmptyArray(t *testing.T) {
	_, err := parseCohereResponse([]byte(`{"embeddings":[]}`))
	if err == nil {
		t.Fatalf("expected error for empty embeddings array")
	}
}
