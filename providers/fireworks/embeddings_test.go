package fireworks_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/fireworks"
)

const fireworksEmbedFixture = `{
  "object": "list",
  "model": "nomic-ai/nomic-embed-text-v1.5",
  "data": [
    {"object":"embedding","index":0,"embedding":[0.11,0.22]},
    {"object":"embedding","index":1,"embedding":[0.33,0.44]}
  ],
  "usage": {"prompt_tokens": 9, "total_tokens": 9}
}`

type fireworksEmbedRecorded struct {
	path string
	auth string
}

func newFireworksEmbedServer(t *testing.T, status int, body string, rec *fireworksEmbedRecorded) *httptest.Server {
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

func newFireworksEmbedProvider(t *testing.T, url string) *fireworks.Provider {
	t.Helper()
	p, err := fireworks.New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(url),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

// Compile-time assertion that *Provider satisfies llmrouter.Embedder.
var _ llmrouter.Embedder = (*fireworks.Provider)(nil)

func TestEmbed_DelegatesToOpenAI(t *testing.T) {
	t.Run("ReturnsIndexAlignedVectors", func(t *testing.T) {
		srv := newFireworksEmbedServer(t, http.StatusOK, fireworksEmbedFixture, nil)
		defer srv.Close()
		p := newFireworksEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{
			Model:  "nomic-ai/nomic-embed-text-v1.5",
			Inputs: []string{"a", "b"},
		})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if len(resp.Embeddings) != 2 {
			t.Fatalf("len = %d, want 2", len(resp.Embeddings))
		}
		if resp.Embeddings[0][0] != 0.11 || resp.Embeddings[1][1] != 0.44 {
			t.Errorf("unexpected vectors: %v", resp.Embeddings)
		}
	})

	t.Run("UsagePropagated", func(t *testing.T) {
		srv := newFireworksEmbedServer(t, http.StatusOK, fireworksEmbedFixture, nil)
		defer srv.Close()
		p := newFireworksEmbedProvider(t, srv.URL)
		resp, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if resp.Usage == nil || resp.Usage.PromptTokens != 9 {
			t.Errorf("Usage = %+v, want PromptTokens=9", resp.Usage)
		}
	})

	t.Run("HitsEmbeddingsPath", func(t *testing.T) {
		rec := &fireworksEmbedRecorded{}
		srv := newFireworksEmbedServer(t, http.StatusOK, fireworksEmbedFixture, rec)
		defer srv.Close()
		p := newFireworksEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if !strings.HasSuffix(rec.path, "/embeddings") {
			t.Errorf("path = %q, want suffix /embeddings", rec.path)
		}
	})

	t.Run("ForwardsAuthHeader", func(t *testing.T) {
		rec := &fireworksEmbedRecorded{}
		srv := newFireworksEmbedServer(t, http.StatusOK, fireworksEmbedFixture, rec)
		defer srv.Close()
		p := newFireworksEmbedProvider(t, srv.URL)
		if _, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if rec.auth != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", rec.auth)
		}
	})
}

func TestEmbed_DefaultBaseURL(t *testing.T) {
	t.Run("UsesFireworksDefault", func(t *testing.T) {
		// Build a provider without WithBaseURL — it should pick up the
		// fireworks default endpoint. We can't dial it; we just verify
		// construction succeeds and the exported default constant is the
		// expected Fireworks AI URL.
		p, err := fireworks.New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if p == nil {
			t.Fatal("provider is nil")
		}
		if fireworks.DefaultBaseURL != "https://api.fireworks.ai/inference/v1" {
			t.Errorf("DefaultBaseURL = %q, want https://api.fireworks.ai/inference/v1", fireworks.DefaultBaseURL)
		}
	})
}

func TestEmbed_ErrorRewrite(t *testing.T) {
	t.Run("4xx_ProviderRewrittenToFireworks", func(t *testing.T) {
		srv := newFireworksEmbedServer(t, http.StatusBadRequest, `{"error":"bad"}`, nil)
		defer srv.Close()
		p := newFireworksEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.Provider != "fireworks" {
			t.Errorf("Provider = %q, want fireworks (was openai)", ue.Provider)
		}
		if ue.StatusCode != http.StatusBadRequest {
			t.Errorf("StatusCode = %d, want 400", ue.StatusCode)
		}
	})

	t.Run("5xx_ProviderRewrittenToFireworks", func(t *testing.T) {
		srv := newFireworksEmbedServer(t, http.StatusInternalServerError, "boom", nil)
		defer srv.Close()
		p := newFireworksEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.Provider != "fireworks" {
			t.Errorf("Provider = %q, want fireworks", ue.Provider)
		}
	})

	t.Run("BodyPropagated", func(t *testing.T) {
		srv := newFireworksEmbedServer(t, http.StatusTooManyRequests, "rate limit hit", nil)
		defer srv.Close()
		p := newFireworksEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if !strings.Contains(ue.Body, "rate limit") {
			t.Errorf("Body = %q, want to contain 'rate limit'", ue.Body)
		}
	})

	t.Run("ErrorMessageMentionsFireworksNotOpenAI", func(t *testing.T) {
		srv := newFireworksEmbedServer(t, http.StatusBadRequest, `{"error":"bad"}`, nil)
		defer srv.Close()
		p := newFireworksEmbedProvider(t, srv.URL)
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		if err == nil {
			t.Fatal("err = nil, want error")
		}
		msg := err.Error()
		if !strings.Contains(msg, "fireworks") {
			t.Errorf("err message %q should mention fireworks", msg)
		}
		if strings.Contains(msg, "openai") {
			t.Errorf("err message %q must not mention openai", msg)
		}
	})

	t.Run("TransportErrorNotWrappedAsErrUpstream", func(t *testing.T) {
		// Point provider at a dead address so the transport itself fails
		// before any HTTP response is produced. The resulting error must
		// pass through untouched — only ErrUpstream values get rewritten.
		p := newFireworksEmbedProvider(t, "http://127.0.0.1:1")
		_, err := p.Embed(context.Background(), llmrouter.EmbedRequest{Model: "m", Inputs: []string{"x"}})
		if err == nil {
			t.Fatal("err = nil, want transport error")
		}
		var ue *llmrouter.ErrUpstream
		if errors.As(err, &ue) {
			t.Errorf("transport error wrapped as *ErrUpstream: %+v", ue)
		}
	})
}
