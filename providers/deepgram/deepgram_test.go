package deepgram

import (
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
)

// Compile-time guarantee that *Provider implements Transcriber. Mirrors
// the assertion in deepgram.go but kept here so a refactor that breaks
// the contract fails inside the test binary too.
var _ llmrouter.Transcriber = (*Provider)(nil)

func TestNew(t *testing.T) {
	t.Run("requires_api_key", func(t *testing.T) {
		_, err := New()
		if err == nil {
			t.Fatal("expected error when api key is missing")
		}
	})

	t.Run("rejects_empty_api_key", func(t *testing.T) {
		// llmrouter.WithAPIKey rejects empty strings at the option layer.
		_, err := New(llmrouter.WithAPIKey("   "))
		if err == nil {
			t.Fatal("expected error for blank api key")
		}
	})

	t.Run("accepts_api_key_and_sets_default_base_url", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("dg-test"))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if p.cfg.BaseURL != DefaultBaseURL {
			t.Fatalf("expected default base url %q, got %q", DefaultBaseURL, p.cfg.BaseURL)
		}
	})

	t.Run("custom_base_url_preserved", func(t *testing.T) {
		const custom = "https://eu.example.deepgram.com"
		p, err := New(
			llmrouter.WithAPIKey("dg-test"),
			llmrouter.WithBaseURL(custom),
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if p.cfg.BaseURL != custom {
			t.Fatalf("expected base url %q, got %q", custom, p.cfg.BaseURL)
		}
	})

	t.Run("name_is_deepgram", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("dg-test"))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got := p.Name(); got != "deepgram" {
			t.Fatalf("expected name deepgram, got %q", got)
		}
	})

	t.Run("missing_api_key_error_wraps_invalid_config", func(t *testing.T) {
		_, err := New()
		if err == nil {
			t.Fatal("expected error")
		}
		if !strings.Contains(err.Error(), "deepgram") {
			t.Fatalf("expected error to mention deepgram, got %q", err.Error())
		}
	})
}
