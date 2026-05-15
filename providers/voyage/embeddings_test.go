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

// voyageEmbedFixture mirrors a real Voyage /v1/embeddings response with
// three inputs returned in order.
const voyageEmbedFixture = `{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
    {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
    {"object": "embedding", "embedding": [0.7, 0.8, 0.9], "index": 2}
  ],
  "model": "voyage-3",
  "usage": {"total_tokens": 21}
}`

// voyageEmbedFixtureShuffled returns the three vectors out of order to
// exercise the index-based re-ordering on decode.
const voyageEmbedFixtureShuffled = `{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.7, 0.8, 0.9], "index": 2},
    {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
    {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1}
  ],
  "model": "voyage-3",
  "usage": {"total_tokens": 21}
}`

// newEmbedServer asserts auth + path then writes the supplied body. It
// captures the outgoing request body and path when the corresponding
// pointers are non-nil.
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
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)

		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:  "voyage-3",
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

	t.Run("VectorsSortedByIndex", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixtureShuffled, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:  "voyage-3",
			Inputs: []string{"a", "b", "c"},
		})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if resp.Embeddings[0][0] != 0.1 {
			t.Errorf("first vector first element = %v, want 0.1 (sort by index failed)", resp.Embeddings[0][0])
		}
		if resp.Embeddings[2][0] != 0.7 {
			t.Errorf("last vector first element = %v, want 0.7 (sort by index failed)", resp.Embeddings[2][0])
		}
	})

	t.Run("ModelEchoedFromResponse", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:  "voyage-3",
			Inputs: []string{"a"},
		})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if resp.Model != "voyage-3" {
			t.Errorf("Model = %q, want %q", resp.Model, "voyage-3")
		}
	})

	t.Run("RawPopulatedFromWire", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"a"}})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if len(resp.Raw) == 0 {
			t.Errorf("Raw is empty, want JSON")
		}
	})
}

func TestEmbed_RequestShape(t *testing.T) {
	t.Run("HitsV1EmbeddingsPath", func(t *testing.T) {
		var path string
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, nil, &path)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if !strings.HasSuffix(path, "/v1/embeddings") {
			t.Errorf("path = %q, want suffix /v1/embeddings", path)
		}
	})

	t.Run("SendsInputArray", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"hello", "world"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		if err := json.Unmarshal(body, &m); err != nil {
			t.Fatalf("unmarshal sent body: %v", err)
		}
		input, ok := m["input"].([]any)
		if !ok {
			t.Fatalf("input missing or wrong type: %v", m["input"])
		}
		if len(input) != 2 || input[0] != "hello" || input[1] != "world" {
			t.Errorf("input = %v, want [hello world]", input)
		}
	})

	t.Run("DefaultsModelToVoyage3WhenEmpty", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		if m["model"] != "voyage-3" {
			t.Errorf("model = %v, want voyage-3", m["model"])
		}
	})

	t.Run("ForwardsExplicitModel", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-large-2", Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		if m["model"] != "voyage-large-2" {
			t.Errorf("model = %v, want voyage-large-2", m["model"])
		}
	})

	t.Run("SendsOutputDimensionWhenSet", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"x"}, Dimensions: 1024}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		if v, ok := m["output_dimension"].(float64); !ok || int(v) != 1024 {
			t.Errorf("output_dimension = %v, want 1024", m["output_dimension"])
		}
	})

	t.Run("OmitsOutputDimensionWhenZero", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		if _, ok := m["output_dimension"]; ok {
			t.Errorf("output_dimension should be omitted when zero")
		}
	})

	t.Run("SendsEncodingFormatWhenSet", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"x"}, EncodingFormat: "base64"}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		if m["encoding_format"] != "base64" {
			t.Errorf("encoding_format = %v, want base64", m["encoding_format"])
		}
	})

	t.Run("OmitsEncodingFormatWhenEmpty", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		if _, ok := m["encoding_format"]; ok {
			t.Errorf("encoding_format should be omitted when empty")
		}
	})
}

func TestEmbed_TaskTypeMapping(t *testing.T) {
	cases := []struct {
		taskType   string
		wantField  bool
		wantValue  string
	}{
		{"RETRIEVAL_QUERY", true, "query"},
		{"QUESTION_ANSWERING", true, "query"},
		{"FACT_VERIFICATION", true, "query"},
		{"RETRIEVAL_DOCUMENT", true, "document"},
		{"CLUSTERING", true, "document"},
		{"CLASSIFICATION", true, "document"},
		{"SEMANTIC_SIMILARITY", true, "document"},
		{"", false, ""},
		{"GARBAGE", false, ""},
	}
	for _, tc := range cases {
		tc := tc
		name := tc.taskType
		if name == "" {
			name = "EMPTY"
		}
		t.Run(name, func(t *testing.T) {
			var body []byte
			srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, &body, nil)
			defer srv.Close()
			p := newEmbedProvider(t, srv.URL)
			if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
				Model:    "voyage-3",
				Inputs:   []string{"x"},
				TaskType: tc.taskType,
			}); err != nil {
				t.Fatalf("Embed: %v", err)
			}
			var m map[string]any
			_ = json.Unmarshal(body, &m)
			got, present := m["input_type"]
			if tc.wantField {
				if !present {
					t.Errorf("input_type missing, want %q", tc.wantValue)
				}
				if got != tc.wantValue {
					t.Errorf("input_type = %v, want %q", got, tc.wantValue)
				}
			} else {
				if present {
					t.Errorf("input_type should be omitted for %q, got %v", tc.taskType, got)
				}
			}
		})
	}
}

func TestEmbed_Usage(t *testing.T) {
	t.Run("PopulatedFromTotalTokens", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"x"}})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if resp.Usage == nil {
			t.Fatal("Usage is nil")
		}
		if resp.Usage.PromptTokens != 21 {
			t.Errorf("PromptTokens = %d, want 21", resp.Usage.PromptTokens)
		}
		if resp.Usage.TotalTokens != 21 {
			t.Errorf("TotalTokens = %d, want 21", resp.Usage.TotalTokens)
		}
	})

	t.Run("NilWhenMissing", func(t *testing.T) {
		body := `{"object":"list","model":"voyage-3","data":[{"object":"embedding","embedding":[0.1],"index":0}]}`
		srv := newEmbedServer(t, http.StatusOK, body, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"x"}})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if resp.Usage != nil {
			t.Errorf("Usage = %+v, want nil", resp.Usage)
		}
	})
}

func TestEmbed_ErrUpstream(t *testing.T) {
	t.Run("BadRequest", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusBadRequest, `{"detail":"bad"}`, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"x"}})
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

	t.Run("Unauthorized", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusUnauthorized, `{"detail":"invalid api key"}`, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"x"}})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.StatusCode != http.StatusUnauthorized {
			t.Errorf("StatusCode = %d, want 401", ue.StatusCode)
		}
	})

	t.Run("InternalServerError", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusInternalServerError, "boom", nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"x"}})
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
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"x"}})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if len(ue.Body) > 8*1024 {
			t.Errorf("body length = %d, want <= %d", len(ue.Body), 8*1024)
		}
	})
}

func TestEmbed_AuthHeaderIsBearer(t *testing.T) {
	var seen string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, voyageEmbedFixture)
	}))
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"x"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if seen != "Bearer test-key" {
		t.Errorf("Authorization = %q, want %q", seen, "Bearer test-key")
	}
}

func TestEmbed_RawPassthrough(t *testing.T) {
	t.Run("RawMergesOverComputedFields", func(t *testing.T) {
		var sent []byte
		srv := newEmbedServer(t, http.StatusOK, voyageEmbedFixture, &sent, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		raw := json.RawMessage(`{"truncation":true,"model":"voyage-override"}`)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:  "voyage-3",
			Inputs: []string{"x"},
			Raw:    raw,
		})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		if err := json.Unmarshal(sent, &m); err != nil {
			t.Fatalf("unmarshal sent: %v", err)
		}
		if m["truncation"] != true {
			t.Errorf("truncation = %v, want true (raw preserved)", m["truncation"])
		}
		if m["model"] != "voyage-override" {
			t.Errorf("model = %v, want voyage-override (raw overrides)", m["model"])
		}
	})

	t.Run("RawInvalidReturnsError", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		_, err = p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model: "voyage-3",
			Raw:   json.RawMessage(`not-json`),
		})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestEmbed_EmptyDataArray(t *testing.T) {
	body := `{"object":"list","model":"voyage-3","data":[]}`
	srv := newEmbedServer(t, http.StatusOK, body, nil, nil)
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "voyage-3", Inputs: []string{"x"}})
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

func TestMapTaskTypeToInputType(t *testing.T) {
	cases := map[string]string{
		"RETRIEVAL_QUERY":     "query",
		"QUESTION_ANSWERING":  "query",
		"FACT_VERIFICATION":   "query",
		"RETRIEVAL_DOCUMENT":  "document",
		"CLUSTERING":          "document",
		"CLASSIFICATION":      "document",
		"SEMANTIC_SIMILARITY": "document",
		"":                    "",
		"UNKNOWN":             "",
	}
	for in, want := range cases {
		in, want := in, want
		name := in
		if name == "" {
			name = "EMPTY"
		}
		t.Run(name, func(t *testing.T) {
			if got := mapTaskTypeToInputType(in); got != want {
				t.Errorf("mapTaskTypeToInputType(%q) = %q, want %q", in, got, want)
			}
		})
	}
}
