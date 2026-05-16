package openairealtime_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/openairealtime"
)

func TestNew_RequiresAPIKey(t *testing.T) {
	_, err := openairealtime.New()
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
	if _, err := openairealtime.New(llmrouter.WithAPIKey("   ")); err == nil {
		t.Fatal("expected error for whitespace api key, got nil")
	}
}

func TestNew_DefaultBaseURL(t *testing.T) {
	p, err := openairealtime.New(llmrouter.WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p == nil {
		t.Fatal("New returned nil provider")
	}
	if openairealtime.DefaultBaseURL != "wss://api.openai.com/v1" {
		t.Errorf("DefaultBaseURL = %q, want wss://api.openai.com/v1",
			openairealtime.DefaultBaseURL)
	}
}

func TestNew_AcceptsBaseURLOverride(t *testing.T) {
	p, err := openairealtime.New(
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
	p, err := openairealtime.New(llmrouter.WithAPIKey("k"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := p.Name(); got != "openairealtime" {
		t.Errorf("Name() = %q, want openairealtime", got)
	}
}
