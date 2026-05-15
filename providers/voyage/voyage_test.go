package voyage

import (
	"errors"
	"testing"

	"github.com/elloloop/llmrouter"
)

// Compile-time assertion that *Provider satisfies the Embedder interface.
var _ llmrouter.Embedder = (*Provider)(nil)

func TestNew(t *testing.T) {
	t.Run("ErrorWithoutAPIKey", func(t *testing.T) {
		_, err := New()
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !errors.Is(err, llmrouter.ErrInvalidConfig) {
			t.Errorf("err = %v, want wrap of ErrInvalidConfig", err)
		}
	})

	t.Run("DefaultsBaseURL", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if p.cfg.BaseURL != DefaultBaseURL {
			t.Errorf("BaseURL = %q, want %q", p.cfg.BaseURL, DefaultBaseURL)
		}
	})

	t.Run("DefaultBaseURLConstantValue", func(t *testing.T) {
		if DefaultBaseURL != "https://api.voyageai.com" {
			t.Errorf("DefaultBaseURL = %q, want https://api.voyageai.com", DefaultBaseURL)
		}
	})

	t.Run("OverridesBaseURL", func(t *testing.T) {
		p, err := New(
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL("https://example.com"),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if p.cfg.BaseURL != "https://example.com" {
			t.Errorf("BaseURL = %q", p.cfg.BaseURL)
		}
	})

	t.Run("NameReturnsVoyage", func(t *testing.T) {
		p, _ := New(llmrouter.WithAPIKey("k"))
		if got := p.Name(); got != "voyage" {
			t.Errorf("Name() = %q, want voyage", got)
		}
	})

	t.Run("APIKeyIsStored", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("secret-key"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if p.cfg.APIKey != "secret-key" {
			t.Errorf("APIKey = %q, want secret-key", p.cfg.APIKey)
		}
	})
}
