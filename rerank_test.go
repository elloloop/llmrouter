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

// fakeReranker is a compile-time-asserted Reranker implementation used
// in tests below.
type fakeReranker struct {
	resp *llmrouter.RerankResponse
	err  error
}

func (f *fakeReranker) Rerank(ctx context.Context, req llmrouter.RerankRequest) (*llmrouter.RerankResponse, error) {
	return f.resp, f.err
}

// Compile-time assertion that *fakeReranker satisfies llmrouter.Reranker.
var _ llmrouter.Reranker = (*fakeReranker)(nil)

// ---------------------------------------------------------------------------
// Reranker interface
// ---------------------------------------------------------------------------

func TestReranker_InterfaceSatisfied(t *testing.T) {
	var r llmrouter.Reranker = &fakeReranker{resp: &llmrouter.RerankResponse{}}
	if _, err := r.Rerank(context.Background(), llmrouter.RerankRequest{Model: "m"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestReranker_CallReturnsResponse(t *testing.T) {
	want := &llmrouter.RerankResponse{Model: "m", Results: []llmrouter.RerankResult{
		{Index: 0, RelevanceScore: 0.9},
	}}
	r := &fakeReranker{resp: want}
	got, err := r.Rerank(context.Background(), llmrouter.RerankRequest{
		Model: "m", Query: "q", Documents: []string{"a"},
	})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if got != want {
		t.Fatal("response mismatch")
	}
}

func TestReranker_CallReturnsError(t *testing.T) {
	want := errors.New("rerank-boom")
	r := &fakeReranker{err: want}
	_, err := r.Rerank(context.Background(), llmrouter.RerankRequest{})
	if !errors.Is(err, want) {
		t.Fatalf("err = %v want %v", err, want)
	}
}

// ---------------------------------------------------------------------------
// RerankRequest JSON marshaling
// ---------------------------------------------------------------------------

func TestRerankRequest_OmitsZeroOptionalFields(t *testing.T) {
	req := llmrouter.RerankRequest{Model: "m", Query: "q"}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(b)
	for _, key := range []string{"top_n", "return_documents", "user"} {
		if strings.Contains(s, key) {
			t.Fatalf("expected %q to be omitted; got %s", key, s)
		}
	}
}

func TestRerankRequest_TopNZeroOmitted(t *testing.T) {
	req := llmrouter.RerankRequest{Model: "m", Query: "q", TopN: 0}
	b, _ := json.Marshal(req)
	if strings.Contains(string(b), "top_n") {
		t.Fatalf("TopN=0 should be omitted: %s", b)
	}
}

func TestRerankRequest_TopNPositiveIncluded(t *testing.T) {
	cases := []int{1, 5, 10, 100, 1000}
	for _, n := range cases {
		t.Run(string(rune('0'+(n%10)))+"-n", func(t *testing.T) {
			req := llmrouter.RerankRequest{Model: "m", Query: "q", TopN: n}
			b, _ := json.Marshal(req)
			if !strings.Contains(string(b), `"top_n":`) {
				t.Fatalf("top_n missing: %s", b)
			}
		})
	}
}

func TestRerankRequest_ReturnDocumentsFalseOmitted(t *testing.T) {
	req := llmrouter.RerankRequest{Model: "m", Query: "q", ReturnDocuments: false}
	b, _ := json.Marshal(req)
	if strings.Contains(string(b), "return_documents") {
		t.Fatalf("ReturnDocuments=false should be omitted: %s", b)
	}
}

func TestRerankRequest_ReturnDocumentsTrueIncluded(t *testing.T) {
	req := llmrouter.RerankRequest{Model: "m", Query: "q", ReturnDocuments: true}
	b, _ := json.Marshal(req)
	if !strings.Contains(string(b), `"return_documents":true`) {
		t.Fatalf("return_documents=true missing: %s", b)
	}
}

func TestRerankRequest_UserEmptyOmitted(t *testing.T) {
	req := llmrouter.RerankRequest{Model: "m", Query: "q", User: ""}
	b, _ := json.Marshal(req)
	if strings.Contains(string(b), `"user"`) {
		t.Fatalf("empty User should be omitted: %s", b)
	}
}

func TestRerankRequest_UserSetIncluded(t *testing.T) {
	cases := []string{"u1", "user-abc", "12345", "user@example.com"}
	for _, u := range cases {
		t.Run(u, func(t *testing.T) {
			req := llmrouter.RerankRequest{Model: "m", Query: "q", User: u}
			b, _ := json.Marshal(req)
			if !strings.Contains(string(b), `"user":"`+u+`"`) {
				t.Fatalf("user %q missing: %s", u, b)
			}
		})
	}
}

func TestRerankRequest_DocumentsAlwaysSerialised(t *testing.T) {
	cases := []struct {
		name string
		docs []string
		want string
	}{
		{"nil", nil, `"documents":null`},
		{"empty", []string{}, `"documents":[]`},
		{"single", []string{"a"}, `"documents":["a"]`},
		{"multi", []string{"a", "b", "c"}, `"documents":["a","b","c"]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := llmrouter.RerankRequest{Model: "m", Query: "q", Documents: tc.docs}
			b, _ := json.Marshal(req)
			if !strings.Contains(string(b), tc.want) {
				t.Fatalf("want %q in %s", tc.want, b)
			}
		})
	}
}

func TestRerankRequest_ModelAlwaysSerialised(t *testing.T) {
	cases := []string{"rerank-v3.5", "rerank-english-v3.0", "rerank-multilingual-v3.0", "x"}
	for _, m := range cases {
		t.Run(m, func(t *testing.T) {
			req := llmrouter.RerankRequest{Model: m, Query: "q"}
			b, _ := json.Marshal(req)
			if !strings.Contains(string(b), `"model":"`+m+`"`) {
				t.Fatalf("model %q missing: %s", m, b)
			}
		})
	}
}

func TestRerankRequest_QueryAlwaysSerialised(t *testing.T) {
	cases := []string{"hi", "", "what is the meaning of life?", "héllo 🌍"}
	for _, q := range cases {
		t.Run(q, func(t *testing.T) {
			req := llmrouter.RerankRequest{Model: "m", Query: q}
			b, _ := json.Marshal(req)
			var got llmrouter.RerankRequest
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got.Query != q {
				t.Fatalf("Query round-trip mismatch: %q vs %q", got.Query, q)
			}
		})
	}
}

func TestRerankRequest_RawDoesNotMarshal(t *testing.T) {
	req := llmrouter.RerankRequest{
		Model: "m",
		Query: "q",
		Raw:   json.RawMessage(`{"secret":"value"}`),
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b), "secret") {
		t.Fatalf("Raw leaked: %s", b)
	}
	if strings.Contains(string(b), `"Raw"`) || strings.Contains(string(b), `"raw"`) {
		t.Fatalf("Raw key leaked: %s", b)
	}
}

func TestRerankRequest_RawPresentOnStruct(t *testing.T) {
	req := llmrouter.RerankRequest{
		Model: "m",
		Raw:   json.RawMessage(`{"x":1}`),
	}
	if len(req.Raw) == 0 {
		t.Fatal("Raw should be readable on the struct")
	}
}

func TestRerankRequest_RoundTrip(t *testing.T) {
	orig := llmrouter.RerankRequest{
		Model:           "rerank-v3.5",
		Query:           "best laptops 2025",
		Documents:       []string{"a", "b", "c"},
		TopN:            3,
		ReturnDocuments: true,
		User:            "u1",
	}
	b, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var round llmrouter.RerankRequest
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !reflect.DeepEqual(orig, round) {
		t.Fatalf("round-trip mismatch:\norig=%#v\n got=%#v", orig, round)
	}
}

func TestRerankRequest_AllFieldsPresent(t *testing.T) {
	req := llmrouter.RerankRequest{
		Model:           "m",
		Query:           "q",
		Documents:       []string{"a"},
		TopN:            1,
		ReturnDocuments: true,
		User:            "u",
	}
	b, _ := json.Marshal(req)
	s := string(b)
	for _, want := range []string{
		`"model":"m"`,
		`"query":"q"`,
		`"documents":["a"]`,
		`"top_n":1`,
		`"return_documents":true`,
		`"user":"u"`,
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("want %q in %s", want, s)
		}
	}
}

// ---------------------------------------------------------------------------
// RerankResponse JSON marshaling
// ---------------------------------------------------------------------------

func TestRerankResponse_UsageOmittedWhenNil(t *testing.T) {
	resp := llmrouter.RerankResponse{
		Model:   "m",
		Results: []llmrouter.RerankResult{{Index: 0, RelevanceScore: 0.5}},
	}
	b, _ := json.Marshal(resp)
	if strings.Contains(string(b), "usage") {
		t.Fatalf("nil Usage should be omitted: %s", b)
	}
}

func TestRerankResponse_UsageIncluded(t *testing.T) {
	resp := llmrouter.RerankResponse{
		Model:   "m",
		Results: []llmrouter.RerankResult{{Index: 0, RelevanceScore: 0.5}},
		Usage:   &llmrouter.Usage{PromptTokens: 7, TotalTokens: 7},
	}
	b, _ := json.Marshal(resp)
	s := string(b)
	if !strings.Contains(s, `"usage"`) {
		t.Fatalf("Usage should be included: %s", s)
	}
	if !strings.Contains(s, `"prompt_tokens":7`) {
		t.Fatalf("prompt_tokens missing: %s", s)
	}
}

func TestRerankResponse_RawDoesNotMarshal(t *testing.T) {
	resp := llmrouter.RerankResponse{
		Model:   "m",
		Results: []llmrouter.RerankResult{{Index: 0}},
		Raw:     json.RawMessage(`{"hidden":1}`),
	}
	b, _ := json.Marshal(resp)
	if strings.Contains(string(b), "hidden") {
		t.Fatalf("Raw leaked: %s", b)
	}
}

func TestRerankResponse_RoundTripWithUsage(t *testing.T) {
	orig := llmrouter.RerankResponse{
		Model: "m",
		Results: []llmrouter.RerankResult{
			{Index: 0, RelevanceScore: 0.95},
			{Index: 2, RelevanceScore: 0.5},
		},
		Usage: &llmrouter.Usage{PromptTokens: 12, TotalTokens: 12},
	}
	b, err := json.Marshal(orig)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var round llmrouter.RerankResponse
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if round.Model != orig.Model {
		t.Fatalf("Model mismatch")
	}
	if !reflect.DeepEqual(round.Results, orig.Results) {
		t.Fatalf("Results mismatch")
	}
	if round.Usage == nil || round.Usage.PromptTokens != 12 {
		t.Fatalf("Usage mismatch: %#v", round.Usage)
	}
}

func TestRerankResponse_RoundTripWithoutUsage(t *testing.T) {
	orig := llmrouter.RerankResponse{
		Model: "m",
		Results: []llmrouter.RerankResult{
			{Index: 0, RelevanceScore: 0.9, Document: "a"},
		},
	}
	b, _ := json.Marshal(orig)
	var round llmrouter.RerankResponse
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if round.Usage != nil {
		t.Fatalf("Usage should remain nil after round-trip without it")
	}
	if !reflect.DeepEqual(round.Results, orig.Results) {
		t.Fatalf("Results mismatch: %#v vs %#v", round.Results, orig.Results)
	}
}

// ---------------------------------------------------------------------------
// RerankResult JSON marshaling
// ---------------------------------------------------------------------------

func TestRerankResult_DocumentOmittedWhenEmpty(t *testing.T) {
	r := llmrouter.RerankResult{Index: 0, RelevanceScore: 0.5}
	b, _ := json.Marshal(r)
	if strings.Contains(string(b), "document") {
		t.Fatalf("empty Document should be omitted: %s", b)
	}
}

func TestRerankResult_DocumentIncluded(t *testing.T) {
	cases := []string{"a", "the quick brown fox", "unicode 🌍", `"quoted"`}
	for _, doc := range cases {
		t.Run(doc, func(t *testing.T) {
			r := llmrouter.RerankResult{Index: 1, RelevanceScore: 0.7, Document: doc}
			b, _ := json.Marshal(r)
			var round llmrouter.RerankResult
			if err := json.Unmarshal(b, &round); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if round.Document != doc {
				t.Fatalf("Document mismatch: %q vs %q", round.Document, doc)
			}
		})
	}
}

func TestRerankResult_RoundTrip(t *testing.T) {
	cases := []llmrouter.RerankResult{
		{Index: 0, RelevanceScore: 0.0},
		{Index: 1, RelevanceScore: 1.0},
		{Index: 5, RelevanceScore: 0.5, Document: "mid"},
		{Index: 99, RelevanceScore: 0.99, Document: "near-top"},
	}
	for i, want := range cases {
		t.Run(string(rune('a'+i)), func(t *testing.T) {
			b, err := json.Marshal(want)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got llmrouter.RerankResult
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if got != want {
				t.Fatalf("got %+v want %+v", got, want)
			}
		})
	}
}

func TestRerankResult_IndexAndScoreSerialised(t *testing.T) {
	r := llmrouter.RerankResult{Index: 7, RelevanceScore: 0.42}
	b, _ := json.Marshal(r)
	s := string(b)
	if !strings.Contains(s, `"index":7`) {
		t.Fatalf("index missing: %s", s)
	}
	if !strings.Contains(s, `"relevance_score":0.42`) {
		t.Fatalf("relevance_score missing: %s", s)
	}
}

func TestRerankResult_ZeroIndexAlwaysPresent(t *testing.T) {
	// Index has no omitempty — the zero index 0 is a valid document
	// pointer and must always appear in the output.
	r := llmrouter.RerankResult{Index: 0, RelevanceScore: 0.5}
	b, _ := json.Marshal(r)
	if !strings.Contains(string(b), `"index":0`) {
		t.Fatalf("index=0 should always be present: %s", b)
	}
}

func TestRerankResult_ZeroScoreAlwaysPresent(t *testing.T) {
	// RelevanceScore has no omitempty either — a 0 score is meaningful
	// (perfectly irrelevant) and the caller needs to see it.
	r := llmrouter.RerankResult{Index: 3, RelevanceScore: 0}
	b, _ := json.Marshal(r)
	if !strings.Contains(string(b), `"relevance_score":0`) {
		t.Fatalf("relevance_score=0 should always be present: %s", b)
	}
}

func TestRerankResponse_ResultsLengthPreserved(t *testing.T) {
	cases := []int{0, 1, 3, 10, 100}
	for _, n := range cases {
		t.Run(string(rune('0'+(n%10))), func(t *testing.T) {
			results := make([]llmrouter.RerankResult, n)
			for i := range results {
				results[i] = llmrouter.RerankResult{Index: i, RelevanceScore: 1.0 / float32(n+1)}
			}
			resp := llmrouter.RerankResponse{Model: "m", Results: results}
			b, _ := json.Marshal(resp)
			var got llmrouter.RerankResponse
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if len(got.Results) != n {
				t.Fatalf("len = %d, want %d", len(got.Results), n)
			}
		})
	}
}

func TestRerankResponse_SortInvariantDocumentation(t *testing.T) {
	// The doc-comment says Results is "Sorted by RelevanceScore descending".
	// The type does not enforce ordering — providers sort. This test
	// documents that contract by constructing a sorted slice and
	// confirming the order is preserved through round-trip.
	resp := llmrouter.RerankResponse{
		Model: "m",
		Results: []llmrouter.RerankResult{
			{Index: 2, RelevanceScore: 0.95},
			{Index: 0, RelevanceScore: 0.7},
			{Index: 1, RelevanceScore: 0.3},
		},
	}
	b, _ := json.Marshal(resp)
	var got llmrouter.RerankResponse
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(got.Results) != 3 {
		t.Fatalf("len = %d", len(got.Results))
	}
	for i := 1; i < len(got.Results); i++ {
		if got.Results[i-1].RelevanceScore < got.Results[i].RelevanceScore {
			t.Fatalf("descending order broken at %d", i)
		}
	}
}

func TestRerankResponse_RawPresentOnStruct(t *testing.T) {
	resp := llmrouter.RerankResponse{
		Model: "m",
		Raw:   json.RawMessage(`{"id":"r1"}`),
	}
	if len(resp.Raw) == 0 {
		t.Fatal("Raw should be readable on struct")
	}
}

func TestRerankRequest_RawFieldPresentButNotSerialised(t *testing.T) {
	req := llmrouter.RerankRequest{
		Model: "m",
		Raw:   json.RawMessage(`{"input_type":"search_query"}`),
	}
	if len(req.Raw) == 0 {
		t.Fatal("Raw should be readable on struct")
	}
	b, _ := json.Marshal(req)
	if strings.Contains(string(b), "input_type") {
		t.Fatalf("Raw must not be serialised: %s", b)
	}
}
