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

// cohereEmbedFixture mirrors a real /v2/embed response with three inputs
// and float-only embeddings.
const cohereEmbedFixture = `{
  "id": "embed-1",
  "model": "embed-english-v3.0",
  "embeddings": {
    "float": [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
      [0.7, 0.8, 0.9]
    ]
  },
  "billed_units": {"input_tokens": 12}
}`

// newEmbedServer asserts auth + path then writes the supplied body.
func newEmbedServer(t *testing.T, status int, body string, capturedBody *[]byte, capturedPath *string) *httptest.Server {
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

func newEmbedProvider(t *testing.T, url string) *Provider {
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

func TestEmbed_HappyPath(t *testing.T) {
	t.Run("ReturnsThreeIndexAlignedVectors", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusOK, cohereEmbedFixture, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)

		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:  "embed-english-v3.0",
			Inputs: []string{"a", "b", "c"},
		})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if got, want := len(resp.Embeddings), 3; got != want {
			t.Fatalf("len(embeddings) = %d, want %d", got, want)
		}
		want := [][]float32{
			{0.1, 0.2, 0.3},
			{0.4, 0.5, 0.6},
			{0.7, 0.8, 0.9},
		}
		for i := range want {
			for j := range want[i] {
				if resp.Embeddings[i][j] != want[i][j] {
					t.Errorf("Embeddings[%d][%d] = %v, want %v", i, j, resp.Embeddings[i][j], want[i][j])
				}
			}
		}
	})

	t.Run("ModelEchoedFromResponse", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusOK, cohereEmbedFixture, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:  "embed-english-v3.0",
			Inputs: []string{"a"},
		})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if resp.Model != "embed-english-v3.0" {
			t.Errorf("Model = %q, want %q", resp.Model, "embed-english-v3.0")
		}
	})

	t.Run("RawPopulatedFromWire", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusOK, cohereEmbedFixture, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"a"}})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if len(resp.Raw) == 0 {
			t.Errorf("Raw is empty, want JSON")
		}
	})
}

func TestEmbed_RequestShape(t *testing.T) {
	t.Run("HitsEmbedPath", func(t *testing.T) {
		var path string
		srv := newEmbedServer(t, http.StatusOK, cohereEmbedFixture, nil, &path)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if !strings.HasSuffix(path, "/embed") {
			t.Errorf("path = %q, want suffix /embed", path)
		}
	})

	t.Run("SendsTextsArray", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, cohereEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"hello", "world"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		if err := json.Unmarshal(body, &m); err != nil {
			t.Fatalf("unmarshal sent body: %v", err)
		}
		texts, ok := m["texts"].([]any)
		if !ok {
			t.Fatalf("texts missing or wrong type: %v", m["texts"])
		}
		if len(texts) != 2 || texts[0] != "hello" || texts[1] != "world" {
			t.Errorf("texts = %v, want [hello world]", texts)
		}
	})

	t.Run("SendsEmbeddingTypesFloat", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, cohereEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		et, ok := m["embedding_types"].([]any)
		if !ok || len(et) != 1 || et[0] != "float" {
			t.Errorf("embedding_types = %v, want [float]", m["embedding_types"])
		}
	})

	t.Run("SendsOutputDimensionWhenSet", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, cohereEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}, Dimensions: 256}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		if v, ok := m["output_dimension"].(float64); !ok || int(v) != 256 {
			t.Errorf("output_dimension = %v, want 256", m["output_dimension"])
		}
	})

	t.Run("OmitsOutputDimensionWhenZero", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, cohereEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		if _, ok := m["output_dimension"]; ok {
			t.Errorf("output_dimension should be omitted when zero")
		}
	})
}

func TestEmbed_TaskTypeMapping(t *testing.T) {
	cases := []struct {
		taskType  string
		inputType string
	}{
		{"RETRIEVAL_QUERY", "search_query"},
		{"RETRIEVAL_DOCUMENT", "search_document"},
		{"SEMANTIC_SIMILARITY", "classification"},
		{"CLASSIFICATION", "classification"},
		{"CLUSTERING", "clustering"},
		{"", "search_document"},
		{"GARBAGE", "search_document"},
	}
	for _, tc := range cases {
		t.Run(tc.taskType+"->"+tc.inputType, func(t *testing.T) {
			var body []byte
			srv := newEmbedServer(t, http.StatusOK, cohereEmbedFixture, &body, nil)
			defer srv.Close()
			p := newEmbedProvider(t, srv.URL)
			if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
				Model:    "m",
				Inputs:   []string{"x"},
				TaskType: tc.taskType,
			}); err != nil {
				t.Fatalf("Embed: %v", err)
			}
			var m map[string]any
			_ = json.Unmarshal(body, &m)
			if got := m["input_type"]; got != tc.inputType {
				t.Errorf("input_type for %q = %v, want %q", tc.taskType, got, tc.inputType)
			}
		})
	}
}

func TestEmbed_Usage(t *testing.T) {
	t.Run("PopulatedFromBilledUnits", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusOK, cohereEmbedFixture, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if resp.Usage == nil {
			t.Fatal("Usage is nil")
		}
		if resp.Usage.PromptTokens != 12 {
			t.Errorf("PromptTokens = %d, want 12", resp.Usage.PromptTokens)
		}
		if resp.Usage.TotalTokens != 12 {
			t.Errorf("TotalTokens = %d, want 12", resp.Usage.TotalTokens)
		}
	})

	t.Run("NilWhenMissing", func(t *testing.T) {
		body := `{"id":"e","model":"m","embeddings":{"float":[[0.1]]}}`
		srv := newEmbedServer(t, http.StatusOK, body, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if resp.Usage != nil {
			t.Errorf("Usage = %+v, want nil", resp.Usage)
		}
	})
}

func TestEmbed_ErrUpstream(t *testing.T) {
	t.Run("4xx", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusBadRequest, `{"message":"bad"}`, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
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
		srv := newEmbedServer(t, http.StatusInternalServerError, "boom", nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
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
		srv := newEmbedServer(t, http.StatusBadGateway, big, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if len(ue.Body) > 8*1024 {
			t.Errorf("body length = %d, want <= %d", len(ue.Body), 8*1024)
		}
	})
}

func TestEmbed_RawPassthrough(t *testing.T) {
	t.Run("RawPreservesVendorKeys", func(t *testing.T) {
		var sent []byte
		srv := newEmbedServer(t, http.StatusOK, cohereEmbedFixture, &sent, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		raw := json.RawMessage(`{"truncate":"END","model":"old","texts":["old"]}`)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:  "new",
			Inputs: []string{"new-text"},
			Raw:    raw,
		})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		if err := json.Unmarshal(sent, &m); err != nil {
			t.Fatalf("unmarshal sent: %v", err)
		}
		if m["truncate"] != "END" {
			t.Errorf("truncate = %v, want END", m["truncate"])
		}
		if m["model"] != "new" {
			t.Errorf("model = %v, want new (overlaid)", m["model"])
		}
		texts, _ := m["texts"].([]any)
		if len(texts) != 1 || texts[0] != "new-text" {
			t.Errorf("texts = %v, want [new-text]", texts)
		}
	})

	t.Run("RawInvalidReturnsError", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		_, err = p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model: "m",
			Raw:   json.RawMessage(`not-json`),
		})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestEmbed_EmptyEmbeddingsArray(t *testing.T) {
	body := `{"id":"e","model":"m","embeddings":{"float":[]}}`
	srv := newEmbedServer(t, http.StatusOK, body, nil, nil)
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if resp.Embeddings == nil {
		t.Fatal("Embeddings is nil, want empty slice")
	}
	if len(resp.Embeddings) != 0 {
		t.Errorf("len(Embeddings) = %d, want 0", len(resp.Embeddings))
	}
}

func TestEmbed_AuthHeaderIsBearer(t *testing.T) {
	var seen string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, cohereEmbedFixture)
	}))
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if seen != "Bearer test-key" {
		t.Errorf("Authorization = %q, want %q", seen, "Bearer test-key")
	}
}
