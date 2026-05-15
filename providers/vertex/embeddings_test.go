package vertex

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/genai"

	"github.com/elloloop/llmrouter"
)

// fakeGenAIServer wraps an httptest.Server that mimics the Vertex AI REST
// surface used by google.golang.org/genai. Tests register a handler and
// receive the inbound request body / path for assertions.
type fakeGenAIServer struct {
	srv      *httptest.Server
	body     []byte
	path     string
	method   string
	respCode int
	respBody string
}

// newFakeGenAIServer builds a server with a permissive default handler.
// Individual tests mutate respCode / respBody before issuing a call.
func newFakeGenAIServer(t *testing.T) *fakeGenAIServer {
	t.Helper()
	f := &fakeGenAIServer{respCode: http.StatusOK, respBody: "{}"}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		f.body = b
		f.path = r.URL.Path
		f.method = r.Method
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(f.respCode)
		_, _ = io.WriteString(w, f.respBody)
	}))
	return f
}

func (f *fakeGenAIServer) close() { f.srv.Close() }

// newTestProvider builds a Provider whose genai client is rooted at the
// fake server. This bypasses New() (which requires ADC) by constructing
// the genai client directly with HTTPOptions.BaseURL — the skipADC code
// path in genai.NewClient.
func (f *fakeGenAIServer) newTestProvider(t *testing.T) *Provider {
	t.Helper()
	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		Backend: genai.BackendVertexAI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL: f.srv.URL,
		},
	})
	if err != nil {
		t.Fatalf("genai client: %v", err)
	}
	return &Provider{cfg: &llmrouter.Config{}, client: client}
}

// --- pure helpers -----------------------------------------------------------

func TestBuildEmbedConfig_EmptyWhenNoFields(t *testing.T) {
	cfg := buildEmbedConfig(llmrouter.EmbedRequest{})
	if cfg.TaskType != "" {
		t.Errorf("TaskType = %q, want empty", cfg.TaskType)
	}
	if cfg.OutputDimensionality != nil {
		t.Errorf("OutputDimensionality should be nil, got %d", *cfg.OutputDimensionality)
	}
}

func TestBuildEmbedConfig_TaskTypePropagates(t *testing.T) {
	cfg := buildEmbedConfig(llmrouter.EmbedRequest{TaskType: "RETRIEVAL_QUERY"})
	if cfg.TaskType != "RETRIEVAL_QUERY" {
		t.Errorf("TaskType = %q", cfg.TaskType)
	}
}

func TestBuildEmbedConfig_DimensionsPropagates(t *testing.T) {
	cfg := buildEmbedConfig(llmrouter.EmbedRequest{Dimensions: 768})
	if cfg.OutputDimensionality == nil || *cfg.OutputDimensionality != 768 {
		t.Errorf("OutputDimensionality = %v", cfg.OutputDimensionality)
	}
}

func TestBuildEmbedConfig_ZeroDimensionsOmitted(t *testing.T) {
	cfg := buildEmbedConfig(llmrouter.EmbedRequest{Dimensions: 0})
	if cfg.OutputDimensionality != nil {
		t.Errorf("zero dims should not set OutputDimensionality")
	}
}

func TestExtractFirstEmbedding_Empty(t *testing.T) {
	_, err := extractFirstEmbedding(&genai.EmbedContentResponse{})
	if err == nil {
		t.Fatal("expected error on empty Embeddings")
	}
}

func TestExtractFirstEmbedding_Nil(t *testing.T) {
	_, err := extractFirstEmbedding(nil)
	if err == nil {
		t.Fatal("expected error on nil response")
	}
}

func TestExtractFirstEmbedding_NilEntry(t *testing.T) {
	_, err := extractFirstEmbedding(&genai.EmbedContentResponse{
		Embeddings: []*genai.ContentEmbedding{nil},
	})
	if err == nil {
		t.Fatal("expected error on nil embedding entry")
	}
}

func TestExtractFirstEmbedding_Values(t *testing.T) {
	resp := &genai.EmbedContentResponse{
		Embeddings: []*genai.ContentEmbedding{{Values: []float32{0.1, 0.2, 0.3}}},
	}
	v, err := extractFirstEmbedding(resp)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(v) != 3 || v[2] != 0.3 {
		t.Errorf("Values = %v", v)
	}
}

// --- Embed (integration via fake genai server) -----------------------------

func TestEmbed_RequiresModel(t *testing.T) {
	p := &Provider{cfg: &llmrouter.Config{}}
	_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Inputs: []string{"x"},
	})
	if err == nil {
		t.Fatal("expected error on empty model")
	}
}

func TestEmbed_RequiresInputs(t *testing.T) {
	p := &Provider{cfg: &llmrouter.Config{}}
	_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model: "m",
	})
	if err == nil {
		t.Fatal("expected error on empty inputs")
	}
}

func TestEmbed_SingleInputAgainstFakeServer(t *testing.T) {
	f := newFakeGenAIServer(t)
	defer f.close()
	f.respBody = `{"predictions":[{"embeddings":{"values":[0.1,0.2]}}],"embeddings":[{"values":[0.1,0.2]}]}`
	p := f.newTestProvider(t)

	resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "text-embedding-004",
		Inputs: []string{"hello"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Embeddings) != 1 || len(resp.Embeddings[0]) != 2 {
		t.Errorf("Embeddings shape = %v", resp.Embeddings)
	}
	if resp.Embeddings[0][0] != 0.1 {
		t.Errorf("first value = %v", resp.Embeddings[0][0])
	}
	if resp.Usage != nil {
		t.Errorf("Usage should be nil for vertex embeddings: %+v", resp.Usage)
	}
}

func TestEmbed_MultipleInputsCallsServerPerInput(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"predictions":[{"embeddings":{"values":[1.0]}}],"embeddings":[{"values":[1.0]}]}`)
	}))
	defer srv.Close()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		Backend:     genai.BackendVertexAI,
		HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL},
	})
	if err != nil {
		t.Fatalf("genai client: %v", err)
	}
	p := &Provider{cfg: &llmrouter.Config{}, client: client}

	resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"a", "b", "c"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if calls != 3 {
		t.Errorf("server calls = %d, want 3", calls)
	}
	if len(resp.Embeddings) != 3 {
		t.Errorf("Embeddings len = %d, want 3", len(resp.Embeddings))
	}
}

func TestEmbed_UpstreamError(t *testing.T) {
	f := newFakeGenAIServer(t)
	defer f.close()
	f.respCode = http.StatusUnauthorized
	f.respBody = `{"error":{"code":401,"message":"unauthorized"}}`
	p := f.newTestProvider(t)
	_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"x"},
	})
	var up *llmrouter.ErrUpstream
	if !errors.As(err, &up) {
		t.Fatalf("err = %v, want ErrUpstream", err)
	}
	if up.Provider != providerName {
		t.Errorf("Provider = %q", up.Provider)
	}
}

func TestEmbed_TaskTypeAndDimsInRequest(t *testing.T) {
	f := newFakeGenAIServer(t)
	defer f.close()
	f.respBody = `{"predictions":[{"embeddings":{"values":[]}}],"embeddings":[{"values":[]}]}`
	p := f.newTestProvider(t)
	_, _ = p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:      "m",
		Inputs:     []string{"x"},
		TaskType:   "RETRIEVAL_DOCUMENT",
		Dimensions: 128,
	})
	// Decode whatever JSON the SDK serialised — accept either Vertex
	// `instances` (predict-style) or genai's modern `content`/`taskType`
	// envelope. We just need taskType + outputDimensionality to be present.
	body := string(f.body)
	if !strings.Contains(body, "RETRIEVAL_DOCUMENT") {
		t.Errorf("body missing taskType: %s", body)
	}
	if !strings.Contains(body, "128") {
		t.Errorf("body missing dimensions=128: %s", body)
	}
}

func TestEmbed_ResponseModelEchoed(t *testing.T) {
	f := newFakeGenAIServer(t)
	defer f.close()
	f.respBody = `{"predictions":[{"embeddings":{"values":[1.0]}}],"embeddings":[{"values":[1.0]}]}`
	p := f.newTestProvider(t)
	resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "my-model-id",
		Inputs: []string{"x"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if resp.Model != "my-model-id" {
		t.Errorf("Model = %q, want my-model-id", resp.Model)
	}
}

func TestEmbed_BodyIsJSON(t *testing.T) {
	f := newFakeGenAIServer(t)
	defer f.close()
	f.respBody = `{"predictions":[{"embeddings":{"values":[1.0]}}],"embeddings":[{"values":[1.0]}]}`
	p := f.newTestProvider(t)
	_, _ = p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"x"},
	})
	var probe map[string]any
	if err := json.Unmarshal(f.body, &probe); err != nil {
		t.Errorf("body is not JSON: %v\n%s", err, string(f.body))
	}
	if f.method != http.MethodPost {
		t.Errorf("method = %q, want POST", f.method)
	}
}
