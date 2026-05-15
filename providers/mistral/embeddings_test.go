package mistral

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

const mistralEmbedFixture = `{
  "object": "list",
  "model": "mistral-embed",
  "data": [
    {"object":"embedding","index":2,"embedding":[0.7,0.8]},
    {"object":"embedding","index":0,"embedding":[0.1,0.2]},
    {"object":"embedding","index":1,"embedding":[0.3,0.4]}
  ],
  "usage": {"prompt_tokens": 11, "total_tokens": 11}
}`

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
	t.Run("ReturnsIndexSortedVectors", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusOK, mistralEmbedFixture, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:  "mistral-embed",
			Inputs: []string{"a", "b", "c"},
		})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		want := [][]float32{
			{0.1, 0.2},
			{0.3, 0.4},
			{0.7, 0.8},
		}
		if len(resp.Embeddings) != 3 {
			t.Fatalf("len = %d, want 3", len(resp.Embeddings))
		}
		for i := range want {
			for j := range want[i] {
				if resp.Embeddings[i][j] != want[i][j] {
					t.Errorf("Embeddings[%d][%d] = %v, want %v", i, j, resp.Embeddings[i][j], want[i][j])
				}
			}
		}
	})

	t.Run("ModelEcho", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusOK, mistralEmbedFixture, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:  "mistral-embed",
			Inputs: []string{"x"},
		})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if resp.Model != "mistral-embed" {
			t.Errorf("Model = %q, want mistral-embed", resp.Model)
		}
	})

	t.Run("RawPopulated", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusOK, mistralEmbedFixture, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if len(resp.Raw) == 0 {
			t.Errorf("Raw is empty")
		}
	})
}

func TestEmbed_RequestShape(t *testing.T) {
	t.Run("HitsEmbeddingsPath", func(t *testing.T) {
		var path string
		srv := newEmbedServer(t, http.StatusOK, mistralEmbedFixture, nil, &path)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if !strings.HasSuffix(path, "/embeddings") {
			t.Errorf("path = %q, want suffix /embeddings", path)
		}
	})

	t.Run("SendsInputArray", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, mistralEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x", "y"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		input, ok := m["input"].([]any)
		if !ok || len(input) != 2 || input[0] != "x" || input[1] != "y" {
			t.Errorf("input = %v, want [x y]", m["input"])
		}
	})

	t.Run("EncodingFormatDefaultsToFloat", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, mistralEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		if m["encoding_format"] != "float" {
			t.Errorf("encoding_format = %v, want float", m["encoding_format"])
		}
	})

	t.Run("EncodingFormatHonouredFromRequest", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, mistralEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}, EncodingFormat: "base64"}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		if m["encoding_format"] != "base64" {
			t.Errorf("encoding_format = %v, want base64", m["encoding_format"])
		}
	})
}

func TestEmbed_DropsUnsupportedFields(t *testing.T) {
	t.Run("TaskTypeIgnored", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, mistralEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:    "m",
			Inputs:   []string{"x"},
			TaskType: "RETRIEVAL_QUERY",
		}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		if _, ok := m["task_type"]; ok {
			t.Errorf("task_type should be stripped, got %v", m["task_type"])
		}
		if _, ok := m["input_type"]; ok {
			t.Errorf("input_type should not appear in outgoing body, got %v", m["input_type"])
		}
	})

	t.Run("DimensionsIgnored", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, mistralEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:      "m",
			Inputs:     []string{"x"},
			Dimensions: 256,
		}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		if _, ok := m["dimensions"]; ok {
			t.Errorf("dimensions should be stripped, got %v", m["dimensions"])
		}
	})

	t.Run("TaskTypeStrippedFromRaw", func(t *testing.T) {
		var body []byte
		srv := newEmbedServer(t, http.StatusOK, mistralEmbedFixture, &body, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:  "m",
			Inputs: []string{"x"},
			Raw:    json.RawMessage(`{"task_type":"RETRIEVAL_QUERY","dimensions":128}`),
		}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(body, &m)
		if _, ok := m["task_type"]; ok {
			t.Errorf("task_type should be stripped from raw, got %v", m["task_type"])
		}
		if _, ok := m["dimensions"]; ok {
			t.Errorf("dimensions should be stripped from raw, got %v", m["dimensions"])
		}
	})
}

func TestEmbed_Usage(t *testing.T) {
	t.Run("Populated", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusOK, mistralEmbedFixture, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if resp.Usage == nil {
			t.Fatal("Usage is nil")
		}
		if resp.Usage.PromptTokens != 11 || resp.Usage.TotalTokens != 11 {
			t.Errorf("Usage = %+v, want {Prompt:11 Total:11}", resp.Usage)
		}
	})

	t.Run("NilWhenAbsent", func(t *testing.T) {
		body := `{"object":"list","model":"m","data":[{"object":"embedding","index":0,"embedding":[0.1]}]}`
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
		srv := newEmbedServer(t, http.StatusUnauthorized, `{"message":"bad key"}`, nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.Provider != "mistral" {
			t.Errorf("Provider = %q, want mistral", ue.Provider)
		}
		if ue.StatusCode != http.StatusUnauthorized {
			t.Errorf("StatusCode = %d, want 401", ue.StatusCode)
		}
	})

	t.Run("5xx", func(t *testing.T) {
		srv := newEmbedServer(t, http.StatusBadGateway, "boom", nil, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.StatusCode != http.StatusBadGateway {
			t.Errorf("StatusCode = %d, want 502", ue.StatusCode)
		}
	})
}

func TestEmbed_RawPassthrough(t *testing.T) {
	t.Run("PreservesVendorKeys", func(t *testing.T) {
		var sent []byte
		srv := newEmbedServer(t, http.StatusOK, mistralEmbedFixture, &sent, nil)
		defer srv.Close()
		p := newEmbedProvider(t, srv.URL)
		raw := json.RawMessage(`{"safe_prompt":true,"model":"old","input":["old"]}`)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:  "new",
			Inputs: []string{"new-text"},
			Raw:    raw,
		}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var m map[string]any
		_ = json.Unmarshal(sent, &m)
		if m["safe_prompt"] != true {
			t.Errorf("safe_prompt = %v, want true", m["safe_prompt"])
		}
		if m["model"] != "new" {
			t.Errorf("model = %v, want new (overlaid)", m["model"])
		}
		input, _ := m["input"].([]any)
		if len(input) != 1 || input[0] != "new-text" {
			t.Errorf("input = %v, want [new-text]", input)
		}
	})

	t.Run("RawInvalidReturnsError", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		_, err = p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Raw: json.RawMessage(`not-json`)})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestEmbed_AuthHeader(t *testing.T) {
	var seen string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, mistralEmbedFixture)
	}))
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if seen != "Bearer test-key" {
		t.Errorf("Authorization = %q, want Bearer test-key", seen)
	}
}
