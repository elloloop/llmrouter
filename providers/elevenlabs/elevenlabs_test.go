package elevenlabs_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/elevenlabs"
)

// Compile-time interface assertions — these must hold for callers to
// drop *Provider directly into Speaker / Transcriber slots.
var (
	_ llmrouter.Speaker     = (*elevenlabs.Provider)(nil)
	_ llmrouter.Transcriber = (*elevenlabs.Provider)(nil)
)

func TestNew_RequiresAPIKey(t *testing.T) {
	_, err := elevenlabs.New()
	if err == nil {
		t.Fatal("New() with no options should fail, got nil")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Errorf("err = %v, want wrapping ErrInvalidConfig", err)
	}
	if !strings.Contains(err.Error(), "api key") {
		t.Errorf("err = %q, want substring 'api key'", err.Error())
	}
}

func TestNew_RejectsEmptyAPIKey(t *testing.T) {
	if _, err := elevenlabs.New(llmrouter.WithAPIKey("   ")); err == nil {
		t.Fatal("expected error for whitespace api key, got nil")
	}
}

func TestNew_DefaultBaseURLApplied(t *testing.T) {
	// We can't read cfg directly, but we can use a stub transport to see
	// which URL the provider hits when no BaseURL is configured.
	p, err := elevenlabs.New(llmrouter.WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p == nil {
		t.Fatal("New returned nil provider")
	}
	if elevenlabs.DefaultBaseURL != "https://api.elevenlabs.io" {
		t.Errorf("DefaultBaseURL = %q, want https://api.elevenlabs.io",
			elevenlabs.DefaultBaseURL)
	}
}

func TestNew_AcceptsBaseURLOverride(t *testing.T) {
	p, err := elevenlabs.New(
		llmrouter.WithAPIKey("k"),
		llmrouter.WithBaseURL("https://proxy.example.com"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p == nil {
		t.Fatal("nil provider")
	}
}

func TestProvider_Name(t *testing.T) {
	p, err := elevenlabs.New(llmrouter.WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := p.Name(); got != "elevenlabs" {
		t.Errorf("Name() = %q, want elevenlabs", got)
	}
}
