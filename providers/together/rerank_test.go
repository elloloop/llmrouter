package together_test

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
	"github.com/elloloop/llmrouter/providers/together"
)

// togetherRerankFixture mirrors a Together /rerank response. Together
// follows Cohere's wire shape with the document text nested.
const togetherRerankFixture = `{
  "object": "rerank",
  "results": [
    {"index": 2, "relevance_score": 0.97, "document": {"text": "carrot cake recipe"}},
    {"index": 0, "relevance_score": 0.81, "document": {"text": "apple pie"}},
    {"index": 4, "relevance_score": 0.42, "document": {"text": "fish tacos"}}
  ]
}`

const togetherRerankFixtureNoDocs = `{
  "object": "rerank",
  "results": [
    {"index": 2, "relevance_score": 0.97},
    {"index": 0, "relevance_score": 0.81},
    {"index": 4, "relevance_score": 0.42}
  ]
}`

const togetherRerankFixtureUnsorted = `{
  "object": "rerank",
  "results": [
    {"index": 4, "relevance_score": 0.1},
    {"index": 0, "relevance_score": 0.5},
    {"index": 2, "relevance_score": 0.9}
  ]
}`

type rerankRecorded struct {
	path string
	auth string
	body []byte
}

func newTogetherRerankServer(t *testing.T, status int, body string, rec *rerankRecorded) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rec != nil {
			rec.path = r.URL.Path
			rec.auth = r.Header.Get("Authorization")
			b, _ := io.ReadAll(r.Body)
			rec.body = b
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
}

func newTogetherRerankProvider(t *testing.T, url string) *together.Provider {
	t.Helper()
	p, err := together.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(url),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func togetherDocs5() []string {
	return []string{"apple pie", "banana bread", "carrot cake recipe", "donut hole", "fish tacos"}
}

func TestRerank_HappyPath(t *testing.T) {
	t.Run("ReturnsThreeResultsSortedDescending", func(t *testing.T) {
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixture, nil)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "Salesforce/Llama-Rank-V1", Query: "dessert", Documents: togetherDocs5(),
			TopN: 3, ReturnDocuments: true,
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if len(resp.Results) != 3 {
			t.Fatalf("len(results) = %d, want 3", len(resp.Results))
		}
		for i := 1; i < len(resp.Results); i++ {
			if resp.Results[i-1].RelevanceScore < resp.Results[i].RelevanceScore {
				t.Errorf("results not sorted desc at %d", i)
			}
		}
		if resp.Results[0].Index != 2 || resp.Results[0].RelevanceScore != 0.97 {
			t.Errorf("top result = %+v, want index=2 score=0.97", resp.Results[0])
		}
	})

	t.Run("ModelEchoedFromRequest", func(t *testing.T) {
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixture, nil)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "Salesforce/Llama-Rank-V1", Query: "q", Documents: togetherDocs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if resp.Model != "Salesforce/Llama-Rank-V1" {
			t.Errorf("Model = %q, want Salesforce/Llama-Rank-V1", resp.Model)
		}
	})

	t.Run("RawPopulatedFromWire", func(t *testing.T) {
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixture, nil)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if len(resp.Raw) == 0 {
			t.Errorf("Raw is empty")
		}
	})

	t.Run("SortsAscendingResults", func(t *testing.T) {
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixtureUnsorted, nil)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if resp.Results[0].Index != 2 {
			t.Errorf("top index = %d, want 2", resp.Results[0].Index)
		}
	})
}

func TestRerank_DocumentBehaviour(t *testing.T) {
	t.Run("ReturnDocumentsTruePopulatesField", func(t *testing.T) {
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixture, nil)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(), ReturnDocuments: true,
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if resp.Results[0].Document != "carrot cake recipe" {
			t.Errorf("Document = %q, want carrot cake recipe", resp.Results[0].Document)
		}
	})

	t.Run("ReturnDocumentsFalseLeavesEmpty", func(t *testing.T) {
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixtureNoDocs, nil)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(),
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
		rec := &rerankRecorded{}
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixture, rec)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(rec.body, &m)
		if m["model"] != "Salesforce/Llama-Rank-V1" {
			t.Errorf("sent model = %v, want Salesforce/Llama-Rank-V1", m["model"])
		}
		if resp.Model != "Salesforce/Llama-Rank-V1" {
			t.Errorf("resp.Model = %q, want Salesforce/Llama-Rank-V1", resp.Model)
		}
	})

	t.Run("ExplicitModelOverridesDefault", func(t *testing.T) {
		rec := &rerankRecorded{}
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixture, rec)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "mixedbread-ai/Mxbai-Rerank-Large-V1", Query: "q", Documents: togetherDocs5(),
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(rec.body, &m)
		if m["model"] != "mixedbread-ai/Mxbai-Rerank-Large-V1" {
			t.Errorf("sent model = %v, want mixedbread-ai/Mxbai-Rerank-Large-V1", m["model"])
		}
	})
}

func TestRerank_RequestShape(t *testing.T) {
	t.Run("HitsRerankPath", func(t *testing.T) {
		rec := &rerankRecorded{}
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixture, rec)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(),
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if !strings.HasSuffix(rec.path, "/rerank") {
			t.Errorf("path = %q, want suffix /rerank", rec.path)
		}
	})

	t.Run("SendsQueryAndDocuments", func(t *testing.T) {
		rec := &rerankRecorded{}
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixture, rec)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "favorite dessert", Documents: []string{"alpha", "beta"},
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(rec.body, &m)
		if m["query"] != "favorite dessert" {
			t.Errorf("query = %v, want 'favorite dessert'", m["query"])
		}
		docs, ok := m["documents"].([]any)
		if !ok || len(docs) != 2 || docs[0] != "alpha" || docs[1] != "beta" {
			t.Errorf("documents = %v, want [alpha beta]", m["documents"])
		}
	})

	t.Run("SendsTopNWhenSet", func(t *testing.T) {
		rec := &rerankRecorded{}
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixture, rec)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(), TopN: 3,
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(rec.body, &m)
		if v, ok := m["top_n"].(float64); !ok || int(v) != 3 {
			t.Errorf("top_n = %v, want 3", m["top_n"])
		}
	})

	t.Run("OmitsTopNWhenZero", func(t *testing.T) {
		rec := &rerankRecorded{}
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixture, rec)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(),
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(rec.body, &m)
		if _, ok := m["top_n"]; ok {
			t.Errorf("top_n should be omitted when zero")
		}
	})

	t.Run("SendsReturnDocumentsFlag", func(t *testing.T) {
		rec := &rerankRecorded{}
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixture, rec)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(), ReturnDocuments: true,
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(rec.body, &m)
		if v, ok := m["return_documents"].(bool); !ok || !v {
			t.Errorf("return_documents = %v, want true", m["return_documents"])
		}
	})

	t.Run("ForwardsBearerAuth", func(t *testing.T) {
		rec := &rerankRecorded{}
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixture, rec)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(),
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if rec.auth != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", rec.auth)
		}
	})
}

func TestRerank_ErrUpstream(t *testing.T) {
	t.Run("4xx_ProviderIsTogether", func(t *testing.T) {
		srv := newTogetherRerankServer(t, http.StatusBadRequest, `{"error":"bad"}`, nil)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		_, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(),
		})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.Provider != "together" {
			t.Errorf("Provider = %q, want together", ue.Provider)
		}
		if ue.StatusCode != http.StatusBadRequest {
			t.Errorf("StatusCode = %d, want 400", ue.StatusCode)
		}
		if !strings.Contains(ue.Body, "bad") {
			t.Errorf("Body = %q, want to contain 'bad'", ue.Body)
		}
	})

	t.Run("5xx", func(t *testing.T) {
		srv := newTogetherRerankServer(t, http.StatusInternalServerError, "boom", nil)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		_, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(),
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
		srv := newTogetherRerankServer(t, http.StatusBadGateway, big, nil)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		_, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(),
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
		rec := &rerankRecorded{}
		srv := newTogetherRerankServer(t, http.StatusOK, togetherRerankFixture, rec)
		defer srv.Close()
		p := newTogetherRerankProvider(t, srv.URL)
		raw := json.RawMessage(`{"rank_fields":["title"],"model":"old"}`)
		_, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "Salesforce/Llama-Rank-V1", Query: "q", Documents: togetherDocs5(), Raw: raw,
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(rec.body, &m)
		rf, ok := m["rank_fields"].([]any)
		if !ok || len(rf) != 1 || rf[0] != "title" {
			t.Errorf("rank_fields = %v, want [title] (raw passthrough)", m["rank_fields"])
		}
		if m["model"] != "old" {
			t.Errorf("model = %v, want old (raw overrides typed)", m["model"])
		}
	})

	t.Run("RawInvalidReturnsError", func(t *testing.T) {
		p, err := together.New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		_, err = p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: togetherDocs5(),
			Raw: json.RawMessage(`not-json`),
		})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestRerank_EmptyResultsArray(t *testing.T) {
	body := `{"object":"rerank","results":[]}`
	srv := newTogetherRerankServer(t, http.StatusOK, body, nil)
	defer srv.Close()
	p := newTogetherRerankProvider(t, srv.URL)
	resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
		Query: "q", Documents: togetherDocs5(),
	})
	if err != nil {
		t.Fatalf("Rerank: %v", err)
	}
	if len(resp.Results) != 0 {
		t.Errorf("len(Results) = %d, want 0", len(resp.Results))
	}
}

func TestRerank_ProviderImplementsReranker(t *testing.T) {
	var _ llmrouter.Reranker = (*together.Provider)(nil)
}
