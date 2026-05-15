package azureopenai_test

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
	"github.com/elloloop/llmrouter/providers/azureopenai"
)

// fakeEmbedServer responds with the supplied status and body, invoking
// `inspect` for every request.
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

func newAzureEmbedProvider(t *testing.T, baseURL string) *azureopenai.Provider {
	t.Helper()
	p, err := azureopenai.New(
		llmrouter.WithAPIKey(testKey),
		llmrouter.WithBaseURL(baseURL),
		azureopenai.WithDeployment(testDeployment),
		azureopenai.WithAPIVersion(testAPIVersion),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func newAzureEmbedProviderAAD(t *testing.T, baseURL string, tok string) *azureopenai.Provider {
	t.Helper()
	src := azureopenai.AADTokenSource(func(ctx context.Context) (string, error) { return tok, nil })
	p, err := azureopenai.New(
		azureopenai.WithAADToken(src),
		llmrouter.WithBaseURL(baseURL),
		azureopenai.WithDeployment(testDeployment),
		azureopenai.WithAPIVersion(testAPIVersion),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func happyAzureEmbedBody() string {
	return `{
		"object":"list",
		"data":[
			{"object":"embedding","index":0,"embedding":[0.1]},
			{"object":"embedding","index":1,"embedding":[0.2]},
			{"object":"embedding","index":2,"embedding":[0.3]}
		],
		"model":"text-embedding-3-small",
		"usage":{"prompt_tokens":3,"total_tokens":3}
	}`
}

func TestAzureEmbed_HappyPath(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusOK, happyAzureEmbedBody(), nil)
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"a", "b", "c"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Embeddings) != 3 {
		t.Errorf("Embeddings len = %d, want 3", len(resp.Embeddings))
	}
	if resp.Usage == nil || resp.Usage.PromptTokens != 3 {
		t.Errorf("Usage = %+v", resp.Usage)
	}
}

func TestAzureEmbed_URLPath(t *testing.T) {
	var seen string
	srv := fakeEmbedServer(t, http.StatusOK, happyAzureEmbedBody(), func(t *testing.T, r *http.Request) {
		seen = r.URL.Path
	})
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	want := "/openai/deployments/" + testDeployment + "/embeddings"
	if seen != want {
		t.Errorf("path = %q, want %q", seen, want)
	}
}

func TestAzureEmbed_APIVersionQuery(t *testing.T) {
	var seen string
	srv := fakeEmbedServer(t, http.StatusOK, happyAzureEmbedBody(), func(t *testing.T, r *http.Request) {
		seen = r.URL.Query().Get("api-version")
	})
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if seen != testAPIVersion {
		t.Errorf("api-version = %q, want %q", seen, testAPIVersion)
	}
}

func TestAzureEmbed_APIKeyHeader(t *testing.T) {
	var seenKey, seenAuth string
	srv := fakeEmbedServer(t, http.StatusOK, happyAzureEmbedBody(), func(t *testing.T, r *http.Request) {
		seenKey = r.Header.Get("api-key")
		seenAuth = r.Header.Get("Authorization")
	})
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if seenKey != testKey {
		t.Errorf("api-key = %q, want %q", seenKey, testKey)
	}
	if seenAuth != "" {
		t.Errorf("Authorization header set for api-key auth: %q", seenAuth)
	}
}

func TestAzureEmbed_AADBearer(t *testing.T) {
	var seenKey, seenAuth string
	srv := fakeEmbedServer(t, http.StatusOK, happyAzureEmbedBody(), func(t *testing.T, r *http.Request) {
		seenKey = r.Header.Get("api-key")
		seenAuth = r.Header.Get("Authorization")
	})
	defer srv.Close()
	p := newAzureEmbedProviderAAD(t, srv.URL, "aad-token")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if seenKey != "" {
		t.Errorf("api-key set for AAD auth: %q", seenKey)
	}
	if seenAuth != "Bearer aad-token" {
		t.Errorf("Authorization = %q, want Bearer aad-token", seenAuth)
	}
}

func TestAzureEmbed_AADEmptyTokenError(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusOK, happyAzureEmbedBody(), nil)
	defer srv.Close()
	p := newAzureEmbedProviderAAD(t, srv.URL, "")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err == nil {
		t.Fatalf("expected error for empty AAD token")
	}
}

func TestAzureEmbed_RawPassthrough(t *testing.T) {
	var seenBody []byte
	srv := fakeEmbedServer(t, http.StatusOK, happyAzureEmbedBody(), func(t *testing.T, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		seenBody = b
	})
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	raw := json.RawMessage(`{"extra":"keep","model":"old"}`)
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "new", Inputs: []string{"x"}, Raw: raw}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	var m map[string]any
	_ = json.Unmarshal(seenBody, &m)
	if m["extra"] != "keep" {
		t.Errorf("vendor extras dropped: %v", m)
	}
	if m["model"] != "new" {
		t.Errorf("model overlay: %v", m["model"])
	}
}

func TestAzureEmbed_4xx_ErrUpstream(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusUnauthorized, `{"error":"bad key"}`, nil)
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	var upstream *llmrouter.ErrUpstream
	if !errors.As(err, &upstream) {
		t.Fatalf("err = %v, want *ErrUpstream", err)
	}
	if upstream.Provider != "azureopenai" {
		t.Errorf("Provider = %q, want azureopenai", upstream.Provider)
	}
	if upstream.StatusCode != 401 {
		t.Errorf("StatusCode = %d, want 401", upstream.StatusCode)
	}
}

func TestAzureEmbed_500_ErrUpstream(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusInternalServerError, `boom`, nil)
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	var upstream *llmrouter.ErrUpstream
	if !errors.As(err, &upstream) {
		t.Fatalf("err = %v", err)
	}
	if upstream.StatusCode != 500 {
		t.Errorf("StatusCode = %d, want 500", upstream.StatusCode)
	}
}

func TestAzureEmbed_429_ErrUpstream(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusTooManyRequests, `{"error":"slow"}`, nil)
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	var upstream *llmrouter.ErrUpstream
	if !errors.As(err, &upstream) || upstream.StatusCode != 429 {
		t.Fatalf("err = %v, want 429 ErrUpstream", err)
	}
}

func TestAzureEmbed_ErrUpstreamBodyCappedAt1KiB(t *testing.T) {
	large := strings.Repeat("z", 4096)
	srv := fakeEmbedServer(t, http.StatusBadGateway, large, nil)
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	var upstream *llmrouter.ErrUpstream
	if !errors.As(err, &upstream) {
		t.Fatalf("err = %v", err)
	}
	if len(upstream.Body) > 1024 {
		t.Errorf("body length = %d, want <=1024", len(upstream.Body))
	}
}

func TestAzureEmbed_IndexAlignment(t *testing.T) {
	body := `{"data":[{"index":1,"embedding":[2.0]},{"index":0,"embedding":[1.0]}],"model":"m"}`
	srv := fakeEmbedServer(t, http.StatusOK, body, nil)
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"a", "b"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if resp.Embeddings[0][0] != 1.0 || resp.Embeddings[1][0] != 2.0 {
		t.Errorf("Embeddings not sorted by index: %v", resp.Embeddings)
	}
}

func TestAzureEmbed_RawResponsePreserved(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusOK, happyAzureEmbedBody(), nil)
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(resp.Raw) == 0 || !strings.Contains(string(resp.Raw), "text-embedding-3-small") {
		t.Errorf("Raw unexpected: %s", resp.Raw)
	}
}

func TestAzureEmbed_NoUsage(t *testing.T) {
	body := `{"data":[{"index":0,"embedding":[0.1]}],"model":"m"}`
	srv := fakeEmbedServer(t, http.StatusOK, body, nil)
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	resp, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if resp.Usage != nil {
		t.Errorf("Usage = %+v, want nil", resp.Usage)
	}
}

func TestAzureEmbed_DecodeError(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusOK, "not-json", nil)
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err == nil {
		t.Fatalf("expected decode error")
	}
}

func TestAzureEmbed_DimensionsForwarded(t *testing.T) {
	var seenBody []byte
	srv := fakeEmbedServer(t, http.StatusOK, happyAzureEmbedBody(), func(t *testing.T, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		seenBody = b
	})
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}, Dimensions: 512}); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	var m map[string]any
	_ = json.Unmarshal(seenBody, &m)
	if m["dimensions"].(float64) != 512 {
		t.Errorf("dimensions = %v, want 512", m["dimensions"])
	}
}

func TestAzureEmbed_ContextCancelled(t *testing.T) {
	srv := fakeEmbedServer(t, http.StatusOK, happyAzureEmbedBody(), func(t *testing.T, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
	})
	defer srv.Close()
	p := newAzureEmbedProvider(t, srv.URL)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Embed(ctx, llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err == nil {
		t.Fatalf("expected error from cancelled context")
	}
}
