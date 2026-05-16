package cohere

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
)

// cohereRerankFixture mirrors a real /v2/rerank response with five inputs
// scored out of order; results carry the nested document text shape that
// Cohere v2 uses.
const cohereRerankFixture = `{
  "id": "rerank-1",
  "results": [
    {"index": 2, "relevance_score": 0.97, "document": {"text": "carrot cake recipe"}},
    {"index": 0, "relevance_score": 0.81, "document": {"text": "apple pie"}},
    {"index": 4, "relevance_score": 0.42, "document": {"text": "fish tacos"}}
  ],
  "meta": {"billed_units": {"search_units": 1}}
}`

// cohereRerankFixtureNoDocs returns results with no nested document object
// (the shape Cohere emits when return_documents=false).
const cohereRerankFixtureNoDocs = `{
  "id": "rerank-2",
  "results": [
    {"index": 2, "relevance_score": 0.97},
    {"index": 0, "relevance_score": 0.81},
    {"index": 4, "relevance_score": 0.42}
  ]
}`

// cohereRerankFixtureUnsorted returns results in ascending score order to
// exercise the sort step inside decodeRerankResponse.
const cohereRerankFixtureUnsorted = `{
  "id": "rerank-3",
  "results": [
    {"index": 4, "relevance_score": 0.1},
    {"index": 0, "relevance_score": 0.5},
    {"index": 2, "relevance_score": 0.9}
  ]
}`

func newRerankServer(t *testing.T, status int, body string, capturedBody *[]byte, capturedPath *string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); !strings.HasPrefix(got, "Bearer ") {
			t.Errorf("Authorization = %q, want Bearer ...", got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", got)
		}
		if capturedPath != nil {
			*capturedPath = r.URL.Path
		}
		if capturedBody != nil {
			b, _ := io.ReadAll(r.Body)
			*capturedBody = b
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
}

func newRerankProvider(t *testing.T, url string) *Provider {
	t.Helper()
	p, err := New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(url),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func docs5() []string {
	return []string{"apple pie", "banana bread", "carrot cake recipe", "donut hole", "fish tacos"}
}

func TestRerank_HappyPath(t *testing.T) {
	t.Run("ReturnsThreeResultsSortedDescending", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model:           "rerank-v3.5",
			Query:           "dessert",
			Documents:       docs5(),
			TopN:            3,
			ReturnDocuments: true,
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if len(resp.Results) != 3 {
			t.Fatalf("len(results) = %d, want 3", len(resp.Results))
		}
		// Sorted desc.
		for i := 1; i < len(resp.Results); i++ {
			if resp.Results[i-1].RelevanceScore < resp.Results[i].RelevanceScore {
				t.Errorf("results not sorted desc at %d: %v < %v", i,
					resp.Results[i-1].RelevanceScore, resp.Results[i].RelevanceScore)
			}
		}
		if resp.Results[0].Index != 2 || resp.Results[0].RelevanceScore != 0.97 {
			t.Errorf("top result = %+v, want index=2 score=0.97", resp.Results[0])
		}
	})

	t.Run("ModelEchoedFromRequest", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-v3.5", Query: "q", Documents: docs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if resp.Model != "rerank-v3.5" {
			t.Errorf("Model = %q, want rerank-v3.5", resp.Model)
		}
	})

	t.Run("RawPopulatedFromWire", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-v3.5", Query: "q", Documents: docs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if len(resp.Raw) == 0 {
			t.Errorf("Raw is empty, want JSON")
		}
	})

	t.Run("SortsAscendingResults", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixtureUnsorted, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-v3.5", Query: "q", Documents: docs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if resp.Results[0].Index != 2 {
			t.Errorf("top index = %d, want 2", resp.Results[0].Index)
		}
	})

	t.Run("UsageNilOnRerank", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-v3.5", Query: "q", Documents: docs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if resp.Usage != nil {
			t.Errorf("Usage = %+v, want nil (cohere doesn't return tokens on rerank)", resp.Usage)
		}
	})
}

func TestRerank_DocumentBehaviour(t *testing.T) {
	t.Run("ReturnDocumentsTruePopulatesField", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-v3.5", Query: "q", Documents: docs5(), ReturnDocuments: true,
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if resp.Results[0].Document == "" {
			t.Errorf("Document is empty, want populated")
		}
		if resp.Results[0].Document != "carrot cake recipe" {
			t.Errorf("Document = %q, want carrot cake recipe", resp.Results[0].Document)
		}
	})

	t.Run("ReturnDocumentsFalseLeavesEmpty", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixtureNoDocs, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-v3.5", Query: "q", Documents: docs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		for i, r := range resp.Results {
			if r.Document != "" {
				t.Errorf("Results[%d].Document = %q, want empty", i, r.Document)
			}
		}
	})
}

func TestRerank_DefaultModel(t *testing.T) {
	t.Run("EmptyModelGetsDefault", func(t *testing.T) {
		var sent []byte
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(sent, &m)
		if m["model"] != "rerank-v3.5" {
			t.Errorf("sent model = %v, want rerank-v3.5", m["model"])
		}
		if resp.Model != "rerank-v3.5" {
			t.Errorf("resp.Model = %q, want rerank-v3.5", resp.Model)
		}
	})

	t.Run("ExplicitModelOverridesDefault", func(t *testing.T) {
		var sent []byte
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-english-v3.0", Query: "q", Documents: docs5(),
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(sent, &m)
		if m["model"] != "rerank-english-v3.0" {
			t.Errorf("sent model = %v, want rerank-english-v3.0", m["model"])
		}
	})
}

func TestRerank_RequestShape(t *testing.T) {
	t.Run("HitsRerankPath", func(t *testing.T) {
		var path string
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, nil, &path)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(),
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if !strings.HasSuffix(path, "/rerank") {
			t.Errorf("path = %q, want suffix /rerank", path)
		}
	})

	t.Run("SendsQuery", func(t *testing.T) {
		var sent []byte
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "favorite dessert", Documents: docs5(),
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(sent, &m)
		if m["query"] != "favorite dessert" {
			t.Errorf("query = %v, want 'favorite dessert'", m["query"])
		}
	})

	t.Run("SendsDocumentsArray", func(t *testing.T) {
		var sent []byte
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: []string{"alpha", "beta", "gamma"},
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(sent, &m)
		docs, ok := m["documents"].([]any)
		if !ok || len(docs) != 3 {
			t.Fatalf("documents = %v, want length 3", m["documents"])
		}
		if docs[0] != "alpha" || docs[2] != "gamma" {
			t.Errorf("documents = %v, want [alpha beta gamma]", docs)
		}
	})

	t.Run("SendsTopNWhenSet", func(t *testing.T) {
		var sent []byte
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(), TopN: 3,
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(sent, &m)
		if v, ok := m["top_n"].(float64); !ok || int(v) != 3 {
			t.Errorf("top_n = %v, want 3", m["top_n"])
		}
	})

	t.Run("OmitsTopNWhenZero", func(t *testing.T) {
		var sent []byte
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(),
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(sent, &m)
		if _, ok := m["top_n"]; ok {
			t.Errorf("top_n should be omitted when zero")
		}
	})

	t.Run("SendsReturnDocumentsFlag", func(t *testing.T) {
		var sent []byte
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(), ReturnDocuments: true,
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(sent, &m)
		if v, ok := m["return_documents"].(bool); !ok || !v {
			t.Errorf("return_documents = %v, want true", m["return_documents"])
		}
	})
}

func TestRerank_ErrUpstream(t *testing.T) {
	t.Run("4xx", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusBadRequest, `{"message":"bad"}`, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		_, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(),
		})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.Provider != "cohere" {
			t.Errorf("Provider = %q, want cohere", ue.Provider)
		}
		if ue.StatusCode != http.StatusBadRequest {
			t.Errorf("StatusCode = %d, want 400", ue.StatusCode)
		}
		if !strings.Contains(ue.Body, "bad") {
			t.Errorf("Body = %q, want to contain 'bad'", ue.Body)
		}
	})

	t.Run("5xx", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusInternalServerError, "boom", nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		_, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(),
		})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.StatusCode != http.StatusInternalServerError {
			t.Errorf("StatusCode = %d, want 500", ue.StatusCode)
		}
	})

	t.Run("BodyCappedAt8KiB", func(t *testing.T) {
		big := strings.Repeat("A", 16*1024)
		srv := newRerankServer(t, http.StatusBadGateway, big, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		_, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(),
		})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if len(ue.Body) > 8*1024 {
			t.Errorf("body length = %d, want <= %d", len(ue.Body), 8*1024)
		}
	})
}

func TestRerank_RawPassthrough(t *testing.T) {
	t.Run("RawPreservesVendorKeys", func(t *testing.T) {
		var sent []byte
		srv := newRerankServer(t, http.StatusOK, cohereRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		raw := json.RawMessage(`{"max_chunks_per_doc":10,"model":"old","query":"old-q"}`)
		_, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-v3.5", Query: "new-q", Documents: docs5(), Raw: raw,
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		if err := json.Unmarshal(sent, &m); err != nil {
			t.Fatalf("unmarshal sent: %v", err)
		}
		if v, ok := m["max_chunks_per_doc"].(float64); !ok || int(v) != 10 {
			t.Errorf("max_chunks_per_doc = %v, want 10 (raw passthrough)", m["max_chunks_per_doc"])
		}
		// Raw overlays on top — old wins because it was applied last.
		if m["model"] != "old" {
			t.Errorf("model = %v, want old (raw overrides typed)", m["model"])
		}
		if m["query"] != "old-q" {
			t.Errorf("query = %v, want old-q (raw overrides typed)", m["query"])
		}
	})

	t.Run("RawInvalidReturnsError", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		_, err = p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(),
			Raw: json.RawMessage(`not-json`),
		})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestRerank_AuthHeaderIsBearer(t *testing.T) {
	var seen string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, cohereRerankFixture)
	}))
	defer srv.Close()
	p := newRerankProvider(t, srv.URL)
	if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
		Query: "q", Documents: docs5(),
	}); err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if seen != "Bearer test-key" {
		t.Errorf("Authorization = %q, want %q", seen, "Bearer test-key")
	}
}

func TestRerank_EmptyResultsArray(t *testing.T) {
	body := `{"id":"e","results":[]}`
	srv := newRerankServer(t, http.StatusOK, body, nil, nil)
	defer srv.Close()
	p := newRerankProvider(t, srv.URL)
	resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
		Query: "q", Documents: docs5(),
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(resp.Results) != 0 {
		t.Errorf("len(Results) = %d, want 0", len(resp.Results))
	}
}

func TestRerank_ProviderImplementsReranker(t *testing.T) {
	var _ llmrouter.Reranker = (*Provider)(nil)
}
