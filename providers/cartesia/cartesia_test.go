package cartesia

import (
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
)

// Compile-time assertion that *Provider satisfies llmrouter.Speaker.
var _ llmrouter.Speaker = (*Provider)(nil)

func TestNew(t *testing.T) {
	t.Run("requires api key", func(t *testing.T) {
		_, err := New()
		if err == nil {
			t.Fatal("expected error when api key missing")
		}
		if !strings.Contains(err.Error(), "api key") {
			t.Fatalf("error should mention api key, got: %v", err)
		}
	})

	t.Run("defaults base url", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if p.cfg.BaseURL != DefaultBaseURL {
			t.Fatalf("expected default base url %q, got %q", DefaultBaseURL, p.cfg.BaseURL)
		}
	})

	t.Run("overrides base url", func(t *testing.T) {
		custom := "https://proxy.example.com"
		p, err := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL(custom))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if p.cfg.BaseURL != custom {
			t.Fatalf("expected base url %q, got %q", custom, p.cfg.BaseURL)
		}
	})

	t.Run("name returns cartesia", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := p.Name(); got != "cartesia" {
			t.Fatalf("expected provider name 'cartesia', got %q", got)
		}
	})

	t.Run("rejects empty api key", func(t *testing.T) {
		_, err := New(llmrouter.WithAPIKey("   "))
		if err == nil {
			t.Fatal("expected error for whitespace-only api key")
		}
	})

	t.Run("constants exposed", func(t *testing.T) {
		if DefaultBaseURL == "" {
			t.Fatal("DefaultBaseURL must not be empty")
		}
		if cartesiaVersion == "" {
			t.Fatal("cartesiaVersion must not be empty")
		}
		if providerName != "cartesia" {
			t.Fatalf("providerName should be 'cartesia', got %q", providerName)
		}
	})
}
