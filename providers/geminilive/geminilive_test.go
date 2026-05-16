package geminilive_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/geminilive"
)

func TestNew_RequiresAPIKey(t *testing.T) {
	_, err := geminilive.New()
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
	if _, err := geminilive.New(llmrouter.WithAPIKey("   ")); err == nil {
		t.Fatal("expected error for whitespace api key, got nil")
	}
}

func TestNew_DefaultBaseURL(t *testing.T) {
	p, err := geminilive.New(llmrouter.WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p == nil {
		t.Fatal("New returned nil provider")
	}
	if geminilive.DefaultBaseURL != "wss://generativelanguage.googleapis.com" {
		t.Errorf("DefaultBaseURL = %q, want wss://generativelanguage.googleapis.com",
			geminilive.DefaultBaseURL)
	}
}

func TestNew_AcceptsBaseURLOverride(t *testing.T) {
	p, err := geminilive.New(
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
	p, err := geminilive.New(llmrouter.WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := p.Name(); got != "geminilive" {
		t.Errorf("Name() = %q, want geminilive", got)
	}
}

// Compile-time API surface assertions — these fail to compile if a
// future refactor accidentally drops or renames a public symbol that
// callers depend on.
var (
	_ = (*geminilive.Provider)(nil)
	_ = (*geminilive.Session)(nil)
	_ = geminilive.SessionConfig{}
	_ = geminilive.SessionEvent{}
)
