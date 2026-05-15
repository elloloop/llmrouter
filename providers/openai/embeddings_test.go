package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/openai"
)

// fakeEmbedServer returns an httptest server whose handler runs `inspect`
// for every request and responds with the supplied status and body.
func fakeEmbedServer(t *testing.T, status int, body string, inspect func(*testing.T, *http.Request)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if inspect != nil {
			inspect(t, r)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
}

func newEmbedProvider(t *testing.T, baseURL string) *openai.Provider {
	t.Helper()
	p, err := openai.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(baseURL),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func newEmbedCtx(t *testing.T) (context.Context, context.CancelFunc) {
	t.Helper()
	return context.WithTimeout(context.Background(), 5*time.Second)
}

// happyEmbedBody returns a deterministic embeddings response with three
// vectors and Usage populated.
func happyEmbedBody() string {
	return `{
		"object":"list",
		"data":[
			{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]},
			{"object":"embedding","index":1,"embedding":[0.4,0.5,0.6]},
			{"object":"embedding","index":2,"embedding":[0.7,0.8,0.9]}
		],
		"model":"text-embedding-3-small",
		"usage":{"prompt_tokens":9,"total_tokens":9}
	}`
}

func TestEmbed_HappyPath_ThreeInputs(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusOK, happyEmbedBody(), func(t *testing.T, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %q, want /embeddings", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("method = %q, want POST", r.Method)
		}
	})
	defer srv.Close()

	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()

	resp, err := p.Embed(ctx, llmrouter.EmbedRequest{
		Model:  "text-embedding-3-small",
		Inputs: []string{"alpha", "beta", "gamma"},
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if got := len(resp.Embeddings); got != 3 {
		t.Fatalf("len(Embeddings) = %d, want 3", got)
	}
	if resp.Model != "text-embedding-3-small" {
		t.Errorf("Model = %q, want text-embedding-3-small", resp.Model)
	}
	if resp.Usage == nil || resp.Usage.PromptTokens != 9 {
		t.Errorf("Usage = %+v, want PromptTokens=9", resp.Usage)
	}
	if resp.Usage.CompletionTokens != 0 {
		t.Errorf("CompletionTokens = %d, want 0", resp.Usage.CompletionTokens)
	}
	if resp.Usage.TotalTokens != 9 {
		t.Errorf("TotalTokens = %d, want 9", resp.Usage.TotalTokens)
	}
	if len(resp.Raw) == 0 {
		t.Errorf("Raw is empty")
	}
}

func TestEmbed_URLPath(t *testing.T) {
	var seenPath string
	srv := fakeEmbedServer(t, http.StatusOK, happyEmbedBody(), func(t *testing.T, r *http.Request) {
		seenPath = r.URL.Path
	})
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if seenPath != "/embeddings" {
		t.Errorf("path = %q, want /embeddings", seenPath)
	}
}

func TestEmbed_AuthorizationHeader(t *testing.T) {
	var seenAuth string
	srv := fakeEmbedServer(t, http.StatusOK, happyEmbedBody(), func(t *testing.T, r *http.Request) {
		seenAuth = r.Header.Get("Authorization")
	})
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if seenAuth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want Bearer test-key", seenAuth)
	}
}

func TestEmbed_BodyShape(t *testing.T) {
	var seenBody []byte
	srv := fakeEmbedServer(t, http.StatusOK, happyEmbedBody(), func(t *testing.T, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		seenBody = b
	})
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	dims := 256
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{
		Model:          "text-embedding-3-small",
		Inputs:         []string{"alpha", "beta"},
		Dimensions:     dims,
		EncodingFormat: "float",
		User:           "u-1",
	}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(seenBody, &m); err != nil {
		t.Fatalf("body decode: %v\n%s", err, seenBody)
	}
	if m["model"] != "text-embedding-3-small" {
		t.Errorf("model = %v, want text-embedding-3-small", m["model"])
	}
	inputs, ok := m["input"].([]any)
	if !ok || len(inputs) != 2 {
		t.Fatalf("input = %v, want 2-element array", m["input"])
	}
	if inputs[0] != "alpha" || inputs[1] != "beta" {
		t.Errorf("inputs = %v, want [alpha beta]", inputs)
	}
	if m["dimensions"].(float64) != 256 {
		t.Errorf("dimensions = %v, want 256", m["dimensions"])
	}
	if m["encoding_format"] != "float" {
		t.Errorf("encoding_format = %v, want float", m["encoding_format"])
	}
	if m["user"] != "u-1" {
		t.Errorf("user = %v, want u-1", m["user"])
	}
}

func TestEmbed_IndexAlignment(t *testing.T) {
	// Server returns embeddings in reverse order; library must sort by index.
	body := `{
		"data":[
			{"index":2,"embedding":[3.0]},
			{"index":0,"embedding":[1.0]},
			{"index":1,"embedding":[2.0]}
		],
		"model":"m",
		"usage":{"prompt_tokens":3,"total_tokens":3}
	}`
	srv := fakeEmbedServer(t, http.StatusOK, body, nil)
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	resp, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"a", "b", "c"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	want := []float32{1.0, 2.0, 3.0}
	for i, w := range want {
		if len(resp.Embeddings[i]) != 1 || resp.Embeddings[i][0] != w {
			t.Errorf("Embeddings[%d] = %v, want [%v]", i, resp.Embeddings[i], w)
		}
	}
}

func TestEmbed_RawPreserved(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusOK, happyEmbedBody(), nil)
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	resp, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if !strings.Contains(string(resp.Raw), `"text-embedding-3-small"`) {
		t.Errorf("Raw did not preserve upstream JSON: %s", resp.Raw)
	}
}

func TestEmbed_RawPassthroughPreservesVendorExtras(t *testing.T) {
	var seenBody []byte
	srv := fakeEmbedServer(t, http.StatusOK, happyEmbedBody(), func(t *testing.T, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		seenBody = b
	})
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()

	raw := json.RawMessage(`{"vendor_extra":"keep","model":"old","input":["old"]}`)
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{
		Model:  "new-model",
		Inputs: []string{"new-input"},
		Raw:    raw,
	}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(seenBody, &m); err != nil {
		t.Fatalf("body decode: %v", err)
	}
	if m["vendor_extra"] != "keep" {
		t.Errorf("vendor_extra dropped: %v", m)
	}
	if m["model"] != "new-model" {
		t.Errorf("model overlay failed: %v", m["model"])
	}
	inputs := m["input"].([]any)
	if inputs[0] != "new-input" {
		t.Errorf("input overlay failed: %v", inputs)
	}
}

func TestEmbed_401_ErrUpstream(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusUnauthorized, `{"error":{"message":"bad key"}}`, nil)
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	_, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	var upstream *llmrouter.ErrUpstream
	if !errors.As(err, &upstream) {
		t.Fatalf("err = %v, want *ErrUpstream", err)
	}
	if upstream.StatusCode != 401 {
		t.Errorf("StatusCode = %d, want 401", upstream.StatusCode)
	}
	if upstream.Provider != "openai" {
		t.Errorf("Provider = %q, want openai", upstream.Provider)
	}
}

func TestEmbed_429_ErrUpstream(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusTooManyRequests, `{"error":{"message":"slow down"}}`, nil)
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	_, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	var upstream *llmrouter.ErrUpstream
	if !errors.As(err, &upstream) {
		t.Fatalf("err = %v, want *ErrUpstream", err)
	}
	if upstream.StatusCode != 429 {
		t.Errorf("StatusCode = %d, want 429", upstream.StatusCode)
	}
}

func TestEmbed_500_ErrUpstream(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusInternalServerError, `boom`, nil)
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	_, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	var upstream *llmrouter.ErrUpstream
	if !errors.As(err, &upstream) {
		t.Fatalf("err = %v, want *ErrUpstream", err)
	}
	if upstream.StatusCode != 500 {
		t.Errorf("StatusCode = %d, want 500", upstream.StatusCode)
	}
}

func TestEmbed_ErrUpstreamBodyCappedAt1KiB(t *testing.T) {
	large := strings.Repeat("a", 4096)
	srv := fakeEmbedServer(t, http.StatusBadGateway, large, nil)
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	_, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	var upstream *llmrouter.ErrUpstream
	if !errors.As(err, &upstream) {
		t.Fatalf("err = %v, want *ErrUpstream", err)
	}
	if len(upstream.Body) > 1024 {
		t.Errorf("body length = %d, want <= 1024", len(upstream.Body))
	}
}

func TestEmbed_NoUsage(t *testing.T) {
	body := `{"data":[{"index":0,"embedding":[0.1]}],"model":"m"}`
	srv := fakeEmbedServer(t, http.StatusOK, body, nil)
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	resp, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if resp.Usage != nil {
		t.Errorf("Usage = %+v, want nil", resp.Usage)
	}
}

func TestEmbed_EmptyInputs(t *testing.T) {
	body := `{"data":[],"model":"m"}`
	srv := fakeEmbedServer(t, http.StatusOK, body, nil)
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	resp, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Embeddings) != 0 {
		t.Errorf("Embeddings = %v, want empty", resp.Embeddings)
	}
}

func TestEmbed_ContextCancelled(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusOK, happyEmbedBody(), func(t *testing.T, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
	})
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	if err == nil {
		t.Fatalf("expected error from cancelled context")
	}
}

func TestEmbed_DimensionsZeroOmitted(t *testing.T) {
	var seenBody []byte
	srv := fakeEmbedServer(t, http.StatusOK, happyEmbedBody(), func(t *testing.T, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		seenBody = b
	})
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	var m map[string]any
	_ = json.Unmarshal(seenBody, &m)
	if _, ok := m["dimensions"]; ok {
		t.Errorf("dimensions present when zero: %v", m["dimensions"])
	}
}

func TestEmbed_EmptyResponse(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusOK, `not-json`, nil)
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err == nil {
		t.Fatalf("expected decode error")
	}
}

func TestEmbed_ProviderName(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusOK, happyEmbedBody(), nil)
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	if p.Name() != "openai" {
		t.Errorf("Name = %q, want openai", p.Name())
	}
}

func TestEmbed_AcceptHeader(t *testing.T) {
	var seen string
	srv := fakeEmbedServer(t, http.StatusOK, happyEmbedBody(), func(t *testing.T, r *http.Request) {
		seen = r.Header.Get("Accept")
	})
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if seen != "application/json" {
		t.Errorf("Accept = %q, want application/json", seen)
	}
}

func TestEmbed_ContentTypeHeader(t *testing.T) {
	var seen string
	srv := fakeEmbedServer(t, http.StatusOK, happyEmbedBody(), func(t *testing.T, r *http.Request) {
		seen = r.Header.Get("Content-Type")
	})
	defer srv.Close()
	p := newEmbedProvider(t, srv.URL)
	ctx, cancel := newEmbedCtx(t)
	defer cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if seen != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", seen)
	}
}
