package together_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/together"
)

const togetherEmbedFixture = `{
  "object": "list",
  "model": "togethercomputer/m2-bert-80M-8k-retrieval",
  "data": [
    {"object":"embedding","index":0,"embedding":[0.1,0.2]},
    {"object":"embedding","index":1,"embedding":[0.3,0.4]}
  ],
  "usage": {"prompt_tokens": 7, "total_tokens": 7}
}`

type embedRecorded struct {
	path string
	auth string
}

func newTogetherEmbedServer(t *testing.T, status int, body string, rec *embedRecorded) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rec != nil {
			rec.path = r.URL.Path
			rec.auth = r.Header.Get("Authorization")
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
}

func newTogetherEmbedProvider(t *testing.T, url string) *together.Provider {
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

func TestEmbed_DelegatesToOpenAI(t *testing.T) {
	t.Run("ReturnsIndexAlignedVectors", func(t *testing.T) {
		srv := newTogetherEmbedServer(t, http.StatusOK, togetherEmbedFixture, nil)
		defer srv.Close()
		p := newTogetherEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:  "togethercomputer/m2-bert-80M-8k-retrieval",
			Inputs: []string{"a", "b"},
		})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if len(resp.Embeddings) != 2 {
			t.Fatalf("len = %d, want 2", len(resp.Embeddings))
		}
		if resp.Embeddings[0][0] != 0.1 || resp.Embeddings[1][1] != 0.4 {
			t.Errorf("unexpected vectors: %v", resp.Embeddings)
		}
	})

	t.Run("UsagePropagated", func(t *testing.T) {
		srv := newTogetherEmbedServer(t, http.StatusOK, togetherEmbedFixture, nil)
		defer srv.Close()
		p := newTogetherEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if resp.Usage == nil || resp.Usage.PromptTokens != 7 {
			t.Errorf("Usage = %+v, want PromptTokens=7", resp.Usage)
		}
	})

	t.Run("HitsEmbeddingsPath", func(t *testing.T) {
		rec := &embedRecorded{}
		srv := newTogetherEmbedServer(t, http.StatusOK, togetherEmbedFixture, rec)
		defer srv.Close()
		p := newTogetherEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if !strings.HasSuffix(rec.path, "/embeddings") {
			t.Errorf("path = %q, want suffix /embeddings", rec.path)
		}
	})

	t.Run("ForwardsAuthHeader", func(t *testing.T) {
		rec := &embedRecorded{}
		srv := newTogetherEmbedServer(t, http.StatusOK, togetherEmbedFixture, rec)
		defer srv.Close()
		p := newTogetherEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if rec.auth != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", rec.auth)
		}
	})
}

func TestEmbed_DefaultBaseURL(t *testing.T) {
	t.Run("UsesTogetherDefault", func(t *testing.T) {
		// Build a provider without WithBaseURL — it should pick up the
		// together default endpoint. We can't dial it; we just check that
		// New succeeds and the provider is non-nil.
		p, err := together.New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if p == nil {
			t.Fatal("provider is nil")
		}
		if together.DefaultBaseURL != "https://api.together.xyz/v1" {
			t.Errorf("DefaultBaseURL = %q, want https://api.together.xyz/v1", together.DefaultBaseURL)
		}
	})
}

func TestEmbed_ErrorRewrite(t *testing.T) {
	t.Run("4xx_ProviderRewrittenToTogether", func(t *testing.T) {
		srv := newTogetherEmbedServer(t, http.StatusBadRequest, `{"error":"bad"}`, nil)
		defer srv.Close()
		p := newTogetherEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.Provider != "together" {
			t.Errorf("Provider = %q, want together (was openai)", ue.Provider)
		}
		if ue.StatusCode != http.StatusBadRequest {
			t.Errorf("StatusCode = %d, want 400", ue.StatusCode)
		}
	})

	t.Run("5xx_ProviderRewrittenToTogether", func(t *testing.T) {
		srv := newTogetherEmbedServer(t, http.StatusInternalServerError, "boom", nil)
		defer srv.Close()
		p := newTogetherEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.Provider != "together" {
			t.Errorf("Provider = %q, want together", ue.Provider)
		}
	})

	t.Run("BodyPropagated", func(t *testing.T) {
		srv := newTogetherEmbedServer(t, http.StatusTooManyRequests, "rate limit hit", nil)
		defer srv.Close()
		p := newTogetherEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if !strings.Contains(ue.Body, "rate limit") {
			t.Errorf("Body = %q, want to contain 'rate limit'", ue.Body)
		}
	})
}
