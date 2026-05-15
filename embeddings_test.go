package llmrouter_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
)

// fakeEmbedder is a compile-time-asserted implementation of Embedder used
// in tests below.
type fakeEmbedder struct {
	resp *llmrouter.EmbedResponse
	err  error
}

func (f *fakeEmbedder) Embed(ctx context.Context, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	return f.resp, f.err
}

// Compile-time assertion that *fakeEmbedder satisfies llmrouter.Embedder.
var _ llmrouter.Embedder = (*fakeEmbedder)(nil)

func TestEmbedder_InterfaceSatisfied(t *testing.T) {
	var e llmrouter.Embedder = &fakeEmbedder{}
	if e == nil {
		t.Fatal("nil Embedder")
	}
}

func TestEmbedder_CallReturnsResponse(t *testing.T) {
	want := &llmrouter.EmbedResponse{Model: "m", Embeddings: [][]float32{{1, 2, 3}}}
	e := &fakeEmbedder{resp: want}
	got, err := e.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"hi"}})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got != want {
		t.Fatalf("response mismatch")
	}
}

func TestEmbedder_CallReturnsError(t *testing.T) {
	want := errors.New("boom")
	e := &fakeEmbedder{err: want}
	_, err := e.Embed(context.Background(), llmrouter.EmbedRequest{})
	if !errors.Is(err, want) {
		t.Fatalf("err = %v want %v", err, want)
	}
}

func TestEmbedRequest_MarshalOmitsZeroOptionalFields(t *testing.T) {
	req := llmrouter.EmbedRequest{Model: "m"}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(b)
	for _, key := range []string{"input", "dimensions", "task_type", "encoding_format", "user"} {
		if strings.Contains(s, key) {
			t.Fatalf("expected %q to be omitted; got %s", key, s)
		}
	}
}

func TestEmbedRequest_MarshalIncludesPopulatedFields(t *testing.T) {
	req := llmrouter.EmbedRequest{
		Model:          "text-embedding-3-small",
		Inputs:         []string{"a", "b"},
		Dimensions:     256,
		TaskType:       "RETRIEVAL_QUERY",
		EncodingFormat: "float",
		User:           "u1",
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(b)
	for _, want := range []string{`"model":"text-embedding-3-small"`, `"input":["a","b"]`, `"dimensions":256`, `"task_type":"RETRIEVAL_QUERY"`, `"encoding_format":"float"`, `"user":"u1"`} {
		if !strings.Contains(s, want) {
			t.Fatalf("expected substring %q in %s", want, s)
		}
	}
}

func TestEmbedRequest_RawDoesNotMarshal(t *testing.T) {
	req := llmrouter.EmbedRequest{
		Model: "m",
		Raw:   json.RawMessage(`{"secret":"value"}`),
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "secret") {
		t.Fatalf("Raw must not leak into JSON: %s", b)
	}
	if strings.Contains(string(b), `"Raw"`) || strings.Contains(string(b), `"raw"`) {
		t.Fatalf("Raw key must not appear: %s", b)
	}
}

func TestEmbedRequest_RoundTrip(t *testing.T) {
	orig := llmrouter.EmbedRequest{
		Model:          "m",
		Inputs:         []string{"a", "b", "c"},
		Dimensions:     128,
		TaskType:       "CLASSIFICATION",
		EncodingFormat: "base64",
		User:           "user-1",
	}
	b, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var round llmrouter.EmbedRequest
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !reflect.DeepEqual(orig, round) {
		t.Fatalf("round-trip mismatch: %#v vs %#v", orig, round)
	}
}

func TestEmbedRequest_EmptyInputsOmitted(t *testing.T) {
	req := llmrouter.EmbedRequest{Model: "m", Inputs: nil}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "input") {
		t.Fatalf("nil Inputs should be omitted: %s", b)
	}
}

func TestEmbedRequest_DimensionsZeroOmitted(t *testing.T) {
	req := llmrouter.EmbedRequest{Model: "m", Dimensions: 0}
	b, _ := json.Marshal(req)
	if strings.Contains(string(b), "dimensions") {
		t.Fatalf("0 Dimensions should be omitted: %s", b)
	}
}

func TestEmbedRequest_SingleInputArray(t *testing.T) {
	req := llmrouter.EmbedRequest{Model: "m", Inputs: []string{"only"}}
	b, _ := json.Marshal(req)
	if !strings.Contains(string(b), `"input":["only"]`) {
		t.Fatalf("single input must serialise as one-element array: %s", b)
	}
}

func TestEmbedResponse_MarshalIncludesEmbeddings(t *testing.T) {
	resp := llmrouter.EmbedResponse{
		Model:      "m",
		Embeddings: [][]float32{{0.1, 0.2}, {0.3, 0.4}},
	}
	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), `"embeddings":[[0.1,0.2],[0.3,0.4]]`) {
		t.Fatalf("embeddings missing or wrong: %s", b)
	}
}

func TestEmbedResponse_UsageOmittedWhenNil(t *testing.T) {
	resp := llmrouter.EmbedResponse{Model: "m", Embeddings: [][]float32{{1}}}
	b, _ := json.Marshal(resp)
	if strings.Contains(string(b), "usage") {
		t.Fatalf("nil Usage should be omitted: %s", b)
	}
}

func TestEmbedResponse_UsageIncluded(t *testing.T) {
	resp := llmrouter.EmbedResponse{
		Model:      "m",
		Embeddings: [][]float32{{1}},
		Usage:      &llmrouter.Usage{PromptTokens: 10, TotalTokens: 10},
	}
	b, _ := json.Marshal(resp)
	if !strings.Contains(string(b), `"usage"`) {
		t.Fatalf("Usage should be included: %s", b)
	}
	if !strings.Contains(string(b), `"prompt_tokens":10`) {
		t.Fatalf("prompt_tokens missing: %s", b)
	}
}

func TestEmbedResponse_RawDoesNotMarshal(t *testing.T) {
	resp := llmrouter.EmbedResponse{
		Model:      "m",
		Embeddings: [][]float32{{1}},
		Raw:        json.RawMessage(`{"hidden":1}`),
	}
	b, _ := json.Marshal(resp)
	if strings.Contains(string(b), "hidden") {
		t.Fatalf("Raw leaked: %s", b)
	}
}

func TestEmbedResponse_RoundTripWithUsage(t *testing.T) {
	orig := llmrouter.EmbedResponse{
		Model:      "m",
		Embeddings: [][]float32{{0.5, 0.5}},
		Usage:      &llmrouter.Usage{PromptTokens: 3, TotalTokens: 3},
	}
	b, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var round llmrouter.EmbedResponse
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if round.Model != orig.Model {
		t.Fatalf("Model mismatch")
	}
	if !reflect.DeepEqual(round.Embeddings, orig.Embeddings) {
		t.Fatalf("Embeddings mismatch")
	}
	if round.Usage == nil || round.Usage.PromptTokens != 3 {
		t.Fatalf("Usage mismatch: %#v", round.Usage)
	}
}

func TestEmbedResponse_RoundTripWithoutUsage(t *testing.T) {
	orig := llmrouter.EmbedResponse{
		Model:      "m",
		Embeddings: [][]float32{{0.1}, {0.2}},
	}
	b, _ := json.Marshal(orig)
	var round llmrouter.EmbedResponse
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if round.Usage != nil {
		t.Fatalf("Usage should remain nil after round-trip without it: %#v", round.Usage)
	}
	if !reflect.DeepEqual(round.Embeddings, orig.Embeddings) {
		t.Fatalf("Embeddings mismatch")
	}
}

func TestEmbedRequest_RawFieldPresentButNotSerialised(t *testing.T) {
	req := llmrouter.EmbedRequest{
		Model: "m",
		Raw:   json.RawMessage(`{"input_type":"search_document"}`),
	}
	// Confirm Raw is readable on the struct itself.
	if len(req.Raw) == 0 {
		t.Fatal("Raw should be readable on the struct")
	}
	// But it must not be in the JSON.
	b, _ := json.Marshal(req)
	if strings.Contains(string(b), "input_type") {
		t.Fatalf("Raw must not be serialised: %s", b)
	}
}

func TestEmbedResponse_EmbeddingsLengthAlignment(t *testing.T) {
	// Doc says Embeddings is index-aligned with EmbedRequest.Inputs.
	// We can only test that the slice carries the values we put in.
	resp := llmrouter.EmbedResponse{
		Embeddings: [][]float32{{1}, {2}, {3}},
	}
	if len(resp.Embeddings) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(resp.Embeddings))
	}
}

func TestEmbedRequest_TaskTypeValues(t *testing.T) {
	values := []string{
		"RETRIEVAL_QUERY",
		"RETRIEVAL_DOCUMENT",
		"SEMANTIC_SIMILARITY",
		"CLASSIFICATION",
		"CLUSTERING",
		"QUESTION_ANSWERING",
		"FACT_VERIFICATION",
	}
	for _, v := range values {
		t.Run(v, func(t *testing.T) {
			req := llmrouter.EmbedRequest{Model: "m", TaskType: v}
			b, _ := json.Marshal(req)
			if !strings.Contains(string(b), v) {
				t.Fatalf("TaskType %q missing in JSON: %s", v, b)
			}
		})
	}
}
