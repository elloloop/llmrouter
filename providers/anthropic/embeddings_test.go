package anthropic_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/anthropic"
)

// voyageOKFixture mirrors a real Voyage /v1/embeddings response so the
// recommended embedder can be exercised end-to-end against an httptest
// server standing in for api.voyageai.com.
const voyageOKFixture = `{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.11, 0.22, 0.33], "index": 0}
  ],
  "model": "voyage-3",
  "usage": {"total_tokens": 7}
}`

// fakeVoyageServer stands in for api.voyageai.com and captures request
// metadata so individual subtests can assert on path, body, and auth.
type fakeVoyageServer struct {
	srv     *httptest.Server
	path    string
	body    []byte
	auth    string
	calls   int
	respond func(w http.ResponseWriter)
}

func newFakeVoyageServer(t *testing.T) *fakeVoyageServer {
	t.Helper()
	f := &fakeVoyageServer{}
	f.respond = func(w http.ResponseWriter) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, voyageOKFixture)
	}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.calls++
		f.path = r.URL.Path
		f.auth = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		f.body = b
		f.respond(w)
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func TestNewRecommendedEmbedder(t *testing.T) {
	t.Run("ReturnsNonNilEmbedderForValidKey", func(t *testing.T) {
		emb, err := anthropic.NewRecommendedEmbedder("voyage-key")
		if err != nil {
			t.Fatalf("NewRecommendedEmbedder: %v", err)
		}
		if emb == nil {
			t.Fatal("embedder is nil")
		}
	})

	t.Run("EmptyAPIKeyReturnsError", func(t *testing.T) {
		emb, err := anthropic.NewRecommendedEmbedder("")
		if err == nil {
			t.Fatal("expected error for empty api key, got nil")
		}
		if emb != nil {
			t.Errorf("embedder = %v, want nil on error", emb)
		}
	})

	t.Run("WhitespaceOnlyAPIKeyReturnsError", func(t *testing.T) {
		emb, err := anthropic.NewRecommendedEmbedder("   ")
		if err == nil {
			t.Fatal("expected error for whitespace api key, got nil")
		}
		if emb != nil {
			t.Errorf("embedder = %v, want nil on error", emb)
		}
	})

	t.Run("SatisfiesEmbedderInterface", func(t *testing.T) {
		emb, err := anthropic.NewRecommendedEmbedder("voyage-key")
		if err != nil {
			t.Fatalf("NewRecommendedEmbedder: %v", err)
		}
		// Compile-time assertion: the returned value implements Embedder.
		var _ llmrouter.Embedder = emb
	})

	t.Run("EndToEndAgainstFakeVoyage", func(t *testing.T) {
		fake := newFakeVoyageServer(t)
		emb, err := anthropic.NewRecommendedEmbedder(
			"voyage-key",
			llmrouter.WithBaseURL(fake.srv.URL),
		)
		if err != nil {
			t.Fatalf("NewRecommendedEmbedder: %v", err)
		}

		resp, err := emb.Embed(context.Background(), llmrouter.EmbedRequest{
			Inputs: []string{"hi"},
		})
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if got, want := len(resp.Embeddings), 1; got != want {
			t.Fatalf("len(embeddings) = %d, want %d", got, want)
		}
		if got, want := resp.Embeddings[0][0], float32(0.11); got != want {
			t.Errorf("Embeddings[0][0] = %v, want %v", got, want)
		}
		if resp.Usage == nil {
			t.Fatal("Usage is nil, want populated")
		}
		if got, want := resp.Usage.PromptTokens, 7; got != want {
			t.Errorf("Usage.PromptTokens = %d, want %d", got, want)
		}
	})

	t.Run("CallerSuppliedBaseURLOverridesDefault", func(t *testing.T) {
		fake := newFakeVoyageServer(t)
		emb, err := anthropic.NewRecommendedEmbedder(
			"voyage-key",
			llmrouter.WithBaseURL(fake.srv.URL),
		)
		if err != nil {
			t.Fatalf("NewRecommendedEmbedder: %v", err)
		}
		if _, err := emb.Embed(context.Background(), llmrouter.EmbedRequest{
			Inputs: []string{"hi"},
		}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		// If the override worked, the fake (and not the real api.voyageai.com)
		// received the call.
		if fake.calls != 1 {
			t.Errorf("fake.calls = %d, want 1 (caller-supplied base URL was ignored)", fake.calls)
		}
	})

	t.Run("DefaultModelVoyage3WhenUnset", func(t *testing.T) {
		fake := newFakeVoyageServer(t)
		emb, err := anthropic.NewRecommendedEmbedder(
			"voyage-key",
			llmrouter.WithBaseURL(fake.srv.URL),
		)
		if err != nil {
			t.Fatalf("NewRecommendedEmbedder: %v", err)
		}
		if _, err := emb.Embed(context.Background(), llmrouter.EmbedRequest{
			Inputs: []string{"hi"},
		}); err != nil {
			t.Fatalf("Embed: %v", err)
		}

		var body map[string]any
		if err := json.Unmarshal(fake.body, &body); err != nil {
			t.Fatalf("unmarshal body: %v", err)
		}
		if got, want := body["model"], "voyage-3"; got != want {
			t.Errorf("model = %v, want %v (default not applied)", got, want)
		}
	})

	t.Run("VariadicOptsForwardedToVoyage", func(t *testing.T) {
		// WithTimeout is a benign option that should be accepted by voyage.New
		// without error when forwarded through the variadic.
		emb, err := anthropic.NewRecommendedEmbedder(
			"voyage-key",
			llmrouter.WithTimeout(5*time.Second),
		)
		if err != nil {
			t.Fatalf("NewRecommendedEmbedder: %v", err)
		}
		if emb == nil {
			t.Fatal("embedder is nil")
		}
	})

	t.Run("InvalidOptionPropagatesError", func(t *testing.T) {
		// WithTimeout rejects non-positive durations; the error must surface
		// through the shim unchanged rather than being swallowed.
		emb, err := anthropic.NewRecommendedEmbedder(
			"voyage-key",
			llmrouter.WithTimeout(0),
		)
		if err == nil {
			t.Fatal("expected error from invalid WithTimeout, got nil")
		}
		if emb != nil {
			t.Errorf("embedder = %v, want nil on option error", emb)
		}
	})

	t.Run("VoyageKeySentAsBearerNotAnthropicHeader", func(t *testing.T) {
		fake := newFakeVoyageServer(t)
		emb, err := anthropic.NewRecommendedEmbedder(
			"voyage-key",
			llmrouter.WithBaseURL(fake.srv.URL),
		)
		if err != nil {
			t.Fatalf("NewRecommendedEmbedder: %v", err)
		}
		if _, err := emb.Embed(context.Background(), llmrouter.EmbedRequest{
			Inputs: []string{"hi"},
		}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if got, want := fake.auth, "Bearer voyage-key"; got != want {
			t.Errorf("Authorization = %q, want %q", got, want)
		}
	})

	t.Run("HitsVoyageEmbeddingsPathNotAnthropicMessages", func(t *testing.T) {
		// The shim must route to Voyage's /v1/embeddings — never to Anthropic's
		// /v1/messages. This guards against accidental future regressions
		// where someone wires the Anthropic provider into the shim.
		fake := newFakeVoyageServer(t)
		emb, err := anthropic.NewRecommendedEmbedder(
			"voyage-key",
			llmrouter.WithBaseURL(fake.srv.URL),
		)
		if err != nil {
			t.Fatalf("NewRecommendedEmbedder: %v", err)
		}
		if _, err := emb.Embed(context.Background(), llmrouter.EmbedRequest{
			Inputs: []string{"hi"},
		}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		if got, want := fake.path, "/v1/embeddings"; got != want {
			t.Errorf("path = %q, want %q", got, want)
		}
		if strings.Contains(fake.path, "messages") {
			t.Errorf("path = %q hit an Anthropic messages endpoint", fake.path)
		}
	})

	t.Run("InputsForwardedInRequestBody", func(t *testing.T) {
		fake := newFakeVoyageServer(t)
		emb, err := anthropic.NewRecommendedEmbedder(
			"voyage-key",
			llmrouter.WithBaseURL(fake.srv.URL),
		)
		if err != nil {
			t.Fatalf("NewRecommendedEmbedder: %v", err)
		}
		if _, err := emb.Embed(context.Background(), llmrouter.EmbedRequest{
			Inputs: []string{"alpha", "beta"},
		}); err != nil {
			t.Fatalf("Embed: %v", err)
		}
		var body map[string]any
		if err := json.Unmarshal(fake.body, &body); err != nil {
			t.Fatalf("unmarshal body: %v", err)
		}
		raw, ok := body["input"].([]any)
		if !ok {
			t.Fatalf("input field missing or wrong type: %v", body["input"])
		}
		if len(raw) != 2 || raw[0] != "alpha" || raw[1] != "beta" {
			t.Errorf("input = %v, want [alpha beta]", raw)
		}
	})
}
