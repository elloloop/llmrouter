package voyage

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

// voyageRerankFixture mirrors a real /v1/rerank response. Voyage stores
// each result's document text as a plain string at the top level of the
// result object (NOT nested like Cohere).
const voyageRerankFixture = `{
  "object": "list",
  "data": [
    {"index": 2, "relevance_score": 0.97, "document": "carrot cake recipe"},
    {"index": 0, "relevance_score": 0.81, "document": "apple pie"},
    {"index": 4, "relevance_score": 0.42, "document": "fish tacos"}
  ],
  "model": "rerank-2",
  "usage": {"total_tokens": 123}
}`

const voyageRerankFixtureNoDocs = `{
  "object": "list",
  "data": [
    {"index": 2, "relevance_score": 0.97},
    {"index": 0, "relevance_score": 0.81},
    {"index": 4, "relevance_score": 0.42}
  ],
  "model": "rerank-2",
  "usage": {"total_tokens": 50}
}`

const voyageRerankFixtureUnsorted = `{
  "object": "list",
  "data": [
    {"index": 4, "relevance_score": 0.1, "document": "fish tacos"},
    {"index": 0, "relevance_score": 0.5, "document": "apple pie"},
    {"index": 2, "relevance_score": 0.9, "document": "carrot cake"}
  ],
  "model": "rerank-2"
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
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-2", Query: "dessert", Documents: docs5(), TopN: 3, ReturnDocuments: true,
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

	t.Run("ModelEchoedFromResponse", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-2", Query: "q", Documents: docs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if resp.Model != "rerank-2" {
			t.Errorf("Model = %q, want rerank-2", resp.Model)
		}
	})

	t.Run("RawPopulatedFromWire", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-2", Query: "q", Documents: docs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if len(resp.Raw) == 0 {
			t.Errorf("Raw is empty, want JSON")
		}
	})

	t.Run("SortsAscendingResults", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixtureUnsorted, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-2", Query: "q", Documents: docs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if resp.Results[0].Index != 2 {
			t.Errorf("top index = %d, want 2", resp.Results[0].Index)
		}
	})

	t.Run("UsagePopulatedFromTotalTokens", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-2", Query: "q", Documents: docs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if resp.Usage == nil {
			t.Fatal("Usage is nil")
		}
		if resp.Usage.PromptTokens != 123 {
			t.Errorf("PromptTokens = %d, want 123", resp.Usage.PromptTokens)
		}
		if resp.Usage.TotalTokens != 123 {
			t.Errorf("TotalTokens = %d, want 123", resp.Usage.TotalTokens)
		}
	})

	t.Run("UsageNilWhenMissing", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixtureUnsorted, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-2", Query: "q", Documents: docs5(),
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if resp.Usage != nil {
			t.Errorf("Usage = %+v, want nil", resp.Usage)
		}
	})
}

func TestRerank_DocumentBehaviour(t *testing.T) {
	t.Run("ReturnDocumentsTruePopulatesField", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-2", Query: "q", Documents: docs5(), ReturnDocuments: true,
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if resp.Results[0].Document != "carrot cake recipe" {
			t.Errorf("Document = %q, want carrot cake recipe", resp.Results[0].Document)
		}
	})

	t.Run("ReturnDocumentsFalseLeavesEmpty", func(t *testing.T) {
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixtureNoDocs, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		resp, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-2", Query: "q", Documents: docs5(),
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
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(),
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(sent, &m)
		if m["model"] != "rerank-2" {
			t.Errorf("sent model = %v, want rerank-2", m["model"])
		}
	})

	t.Run("ExplicitModelOverridesDefault", func(t *testing.T) {
		var sent []byte
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-lite-1", Query: "q", Documents: docs5(),
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(sent, &m)
		if m["model"] != "rerank-lite-1" {
			t.Errorf("sent model = %v, want rerank-lite-1", m["model"])
		}
	})
}

func TestRerank_RequestShape(t *testing.T) {
	t.Run("HitsRerankPath", func(t *testing.T) {
		var path string
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, nil, &path)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(),
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		if !strings.HasSuffix(path, "/v1/rerank") {
			t.Errorf("path = %q, want suffix /v1/rerank", path)
		}
	})

	t.Run("SendsQuery", func(t *testing.T) {
		var sent []byte
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, &sent, nil)
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
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: []string{"alpha", "beta"},
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(sent, &m)
		docs, ok := m["documents"].([]any)
		if !ok || len(docs) != 2 {
			t.Fatalf("documents = %v, want length 2", m["documents"])
		}
		if docs[0] != "alpha" || docs[1] != "beta" {
			t.Errorf("documents = %v, want [alpha beta]", docs)
		}
	})

	t.Run("SendsTopKWhenSet", func(t *testing.T) {
		var sent []byte
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(), TopN: 3,
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(sent, &m)
		if v, ok := m["top_k"].(float64); !ok || int(v) != 3 {
			t.Errorf("top_k = %v, want 3", m["top_k"])
		}
		if _, ok := m["top_n"]; ok {
			t.Errorf("top_n should NOT be sent (voyage uses top_k)")
		}
	})

	t.Run("OmitsTopKWhenZero", func(t *testing.T) {
		var sent []byte
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		if _, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(),
		}); err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(sent, &m)
		if _, ok := m["top_k"]; ok {
			t.Errorf("top_k should be omitted when zero")
		}
	})

	t.Run("SendsReturnDocumentsFlag", func(t *testing.T) {
		var sent []byte
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, &sent, nil)
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
		srv := newRerankServer(t, http.StatusBadRequest, `{"detail":"bad"}`, nil, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		_, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Query: "q", Documents: docs5(),
		})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.Provider != "voyage" {
			t.Errorf("Provider = %q, want voyage", ue.Provider)
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
		srv := newRerankServer(t, http.StatusOK, voyageRerankFixture, &sent, nil)
		defer srv.Close()
		p := newRerankProvider(t, srv.URL)
		raw := json.RawMessage(`{"truncation":true,"model":"old","query":"old-q"}`)
		_, err := p.Rerank(context.Background(), llmrouter.RerankRequest{
			Model: "rerank-2", Query: "new-q", Documents: docs5(), Raw: raw,
		})
		if err != nil {
			t.Fatalf("Rerank: %v", err)
		}
		var m map[string]any
		if err := json.Unmarshal(sent, &m); err != nil {
			t.Fatalf("unmarshal sent: %v", err)
		}
		if v, ok := m["truncation"].(bool); !ok || !v {
			t.Errorf("truncation = %v, want true (raw passthrough)", m["truncation"])
		}
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
		_, _ = io.WriteString(w, voyageRerankFixture)
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
	body := `{"object":"list","data":[],"model":"rerank-2"}`
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
