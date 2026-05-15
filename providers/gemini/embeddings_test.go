package gemini

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

// embedRecorder captures the inbound embedding request body/url so
// individual tests can assert on what the provider sent.
type embedRecorder struct {
	srv      *httptest.Server
	urlPath  string
	body     []byte
	headers  http.Header
	respCode int
	respBody string
}

// newEmbedRecorder wires up an httptest server that replies with the
// supplied JSON body (status 200) and records the inbound request. Use
// withStatus / withResponse for non-default behaviour.
func newEmbedRecorder(t *testing.T) *embedRecorder {
	t.Helper()
	er := &embedRecorder{respCode: http.StatusOK}
	er.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		er.body = b
		er.urlPath = r.URL.Path
		er.headers = r.Header.Clone()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(er.respCode)
		_, _ = io.WriteString(w, er.respBody)
	}))
	return er
}

func (e *embedRecorder) close() { e.srv.Close() }

// newProvider constructs a Provider rooted at the recorder server.
func (e *embedRecorder) newProvider(t *testing.T) *Provider {
	t.Helper()
	p, err := New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(e.srv.URL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// --- :embedContent (single input) ------------------------------------------

func TestEmbed_Single_SuccessShape(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embedding":{"values":[0.1,0.2,0.3]}}`

	p := r.newProvider(t)
	resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "text-embedding-004",
		Inputs: []string{"hello"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if resp.Model != "text-embedding-004" {
		t.Errorf("Model = %q, want text-embedding-004", resp.Model)
	}
	if len(resp.Embeddings) != 1 || len(resp.Embeddings[0]) != 3 {
		t.Fatalf("Embeddings shape = %v", resp.Embeddings)
	}
	if resp.Embeddings[0][0] != 0.1 || resp.Embeddings[0][2] != 0.3 {
		t.Errorf("values not preserved: %v", resp.Embeddings[0])
	}
	if resp.Usage != nil {
		t.Errorf("Usage should be nil (Gemini omits it), got %+v", resp.Usage)
	}
}

func TestEmbed_Single_HitsEmbedContentPath(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embedding":{"values":[]}}`
	p := r.newProvider(t)
	_, _ = p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "models-x",
		Inputs: []string{"hi"},
	})
	if !strings.Contains(r.urlPath, ":embedContent") {
		t.Errorf("path = %q, want :embedContent suffix", r.urlPath)
	}
	if strings.Contains(r.urlPath, ":batchEmbedContents") {
		t.Errorf("path = %q, single input must NOT use batch", r.urlPath)
	}
}

func TestEmbed_Single_SetsAuthHeader(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embedding":{"values":[]}}`
	p := r.newProvider(t)
	_, _ = p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"hi"},
	})
	if got := r.headers.Get(apiKeyHeader); got != "test-key" {
		t.Errorf("%s = %q, want test-key", apiKeyHeader, got)
	}
	if got := r.headers.Get("Authorization"); got != "" {
		t.Errorf("unexpected Authorization header: %q", got)
	}
}

func TestEmbed_Single_BodyShape(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embedding":{"values":[]}}`
	p := r.newProvider(t)
	_, _ = p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"hello world"},
	})
	var body map[string]any
	if err := json.Unmarshal(r.body, &body); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if body["model"] != "models/m" {
		t.Errorf("model = %v, want models/m", body["model"])
	}
	content, ok := body["content"].(map[string]any)
	if !ok {
		t.Fatalf("content not a map: %T", body["content"])
	}
	parts, _ := content["parts"].([]any)
	if len(parts) != 1 {
		t.Fatalf("parts len = %d", len(parts))
	}
	part0, _ := parts[0].(map[string]any)
	if part0["text"] != "hello world" {
		t.Errorf("text = %v", part0["text"])
	}
}

func TestEmbed_Single_TaskTypePropagates(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embedding":{"values":[]}}`
	p := r.newProvider(t)
	_, _ = p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:    "m",
		Inputs:   []string{"x"},
		TaskType: "RETRIEVAL_QUERY",
	})
	var body map[string]any
	_ = json.Unmarshal(r.body, &body)
	if body["taskType"] != "RETRIEVAL_QUERY" {
		t.Errorf("taskType = %v, want RETRIEVAL_QUERY", body["taskType"])
	}
}

func TestEmbed_Single_DimensionsPropagates(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embedding":{"values":[]}}`
	p := r.newProvider(t)
	_, _ = p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:      "m",
		Inputs:     []string{"x"},
		Dimensions: 256,
	})
	var body map[string]any
	_ = json.Unmarshal(r.body, &body)
	if got, _ := body["outputDimensionality"].(float64); int(got) != 256 {
		t.Errorf("outputDimensionality = %v, want 256", body["outputDimensionality"])
	}
}

func TestEmbed_Single_OmitsTaskTypeWhenEmpty(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embedding":{"values":[]}}`
	p := r.newProvider(t)
	_, _ = p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"x"},
	})
	var body map[string]any
	_ = json.Unmarshal(r.body, &body)
	if _, present := body["taskType"]; present {
		t.Errorf("taskType should be omitted, got %v", body["taskType"])
	}
	if _, present := body["outputDimensionality"]; present {
		t.Errorf("outputDimensionality should be omitted, got %v", body["outputDimensionality"])
	}
}

func TestEmbed_Single_RawOverlay(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embedding":{"values":[]}}`
	p := r.newProvider(t)
	_, _ = p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"x"},
		Raw:    json.RawMessage(`{"customKey":"value","model":"should-be-ignored"}`),
	})
	var body map[string]any
	_ = json.Unmarshal(r.body, &body)
	if body["customKey"] != "value" {
		t.Errorf("customKey not propagated: %v", body)
	}
	if body["model"] != "models/m" {
		t.Errorf("Raw must not clobber typed model, got %v", body["model"])
	}
}

// --- :batchEmbedContents (multi input) -------------------------------------

func TestEmbed_Batch_SuccessShape(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embeddings":[{"values":[0.1,0.2]},{"values":[0.3,0.4]},{"values":[0.5,0.6]}]}`
	p := r.newProvider(t)
	resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"a", "b", "c"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Embeddings) != 3 {
		t.Fatalf("len = %d, want 3", len(resp.Embeddings))
	}
	if resp.Embeddings[0][0] != 0.1 || resp.Embeddings[2][1] != 0.6 {
		t.Errorf("values out of order: %v", resp.Embeddings)
	}
}

func TestEmbed_Batch_HitsBatchPath(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embeddings":[{"values":[]},{"values":[]}]}`
	p := r.newProvider(t)
	_, _ = p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"a", "b"},
	})
	if !strings.Contains(r.urlPath, ":batchEmbedContents") {
		t.Errorf("path = %q, want :batchEmbedContents", r.urlPath)
	}
}

func TestEmbed_Batch_IndexAligned(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embeddings":[{"values":[1]},{"values":[2]},{"values":[3]},{"values":[4]}]}`
	p := r.newProvider(t)
	resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"w", "x", "y", "z"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	for i, want := range []float32{1, 2, 3, 4} {
		if resp.Embeddings[i][0] != want {
			t.Errorf("Embeddings[%d] = %v, want [%v]", i, resp.Embeddings[i], want)
		}
	}
}

func TestEmbed_Batch_TaskTypePerRequest(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embeddings":[{"values":[]},{"values":[]}]}`
	p := r.newProvider(t)
	_, _ = p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:    "m",
		Inputs:   []string{"a", "b"},
		TaskType: "SEMANTIC_SIMILARITY",
	})
	var body struct {
		Requests []map[string]any `json:"requests"`
	}
	if err := json.Unmarshal(r.body, &body); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(body.Requests) != 2 {
		t.Fatalf("requests len = %d", len(body.Requests))
	}
	for i, req := range body.Requests {
		if req["taskType"] != "SEMANTIC_SIMILARITY" {
			t.Errorf("requests[%d].taskType = %v", i, req["taskType"])
		}
		if req["model"] != "models/m" {
			t.Errorf("requests[%d].model = %v", i, req["model"])
		}
	}
}

func TestEmbed_Batch_PreservesInputOrder(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embeddings":[{"values":[]},{"values":[]},{"values":[]}]}`
	p := r.newProvider(t)
	_, _ = p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"first", "second", "third"},
	})
	var body struct {
		Requests []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"requests"`
	}
	_ = json.Unmarshal(r.body, &body)
	got := []string{
		body.Requests[0].Content.Parts[0].Text,
		body.Requests[1].Content.Parts[0].Text,
		body.Requests[2].Content.Parts[0].Text,
	}
	want := []string{"first", "second", "third"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("text[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// --- error paths -----------------------------------------------------------

func TestEmbed_UpstreamError(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respCode = http.StatusUnauthorized
	r.respBody = `{"error":{"code":401,"message":"bad key"}}`
	p := r.newProvider(t)
	_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"x"},
	})
	var up *llmrouter.ErrUpstream
	if !errors.As(err, &up) {
		t.Fatalf("err = %v, want ErrUpstream", err)
	}
	if up.StatusCode != 401 {
		t.Errorf("StatusCode = %d", up.StatusCode)
	}
	if up.Provider != providerName {
		t.Errorf("Provider = %q", up.Provider)
	}
	if !strings.Contains(up.Body, "bad key") {
		t.Errorf("Body = %q", up.Body)
	}
}

func TestEmbed_UpstreamError_Batch(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respCode = http.StatusTooManyRequests
	r.respBody = `rate limit`
	p := r.newProvider(t)
	_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"a", "b"},
	})
	var up *llmrouter.ErrUpstream
	if !errors.As(err, &up) {
		t.Fatalf("err = %v", err)
	}
	if up.StatusCode != 429 {
		t.Errorf("StatusCode = %d", up.StatusCode)
	}
}

func TestEmbed_NoInputs(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	p := r.newProvider(t)
	_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: nil,
	})
	if err == nil {
		t.Fatal("expected error on empty inputs")
	}
}

func TestEmbed_NoModel(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	p := r.newProvider(t)
	_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Inputs: []string{"x"},
	})
	if err == nil {
		t.Fatal("expected error on empty model")
	}
}

func TestEmbed_RawOverlayInvalidJSON(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	p := r.newProvider(t)
	_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"x"},
		Raw:    json.RawMessage(`not-json`),
	})
	if err == nil {
		t.Fatal("expected error on invalid raw json")
	}
}

func TestEmbed_RawCapturesNonNumericExtras(t *testing.T) {
	// Ensures Raw passthrough survives marshalling.
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embedding":{"values":[1.0]}}`
	p := r.newProvider(t)
	_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"x"},
		Raw:    json.RawMessage(`{"customNumeric":42,"customString":"hi"}`),
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	var body map[string]any
	_ = json.Unmarshal(r.body, &body)
	if got, _ := body["customNumeric"].(float64); int(got) != 42 {
		t.Errorf("customNumeric = %v", body["customNumeric"])
	}
	if body["customString"] != "hi" {
		t.Errorf("customString = %v", body["customString"])
	}
}

func TestEmbed_RawPreservedInResponse(t *testing.T) {
	r := newEmbedRecorder(t)
	defer r.close()
	r.respBody = `{"embedding":{"values":[0.5]}}`
	p := r.newProvider(t)
	resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
		Model:  "m",
		Inputs: []string{"x"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if !strings.Contains(string(resp.Raw), `"values":[0.5]`) {
		t.Errorf("Raw should mirror wire bytes: %s", string(resp.Raw))
	}
}
