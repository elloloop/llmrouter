package llmrouter_test

import (
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
)

// ---------------------------------------------------------------------------
// NewConfig
// ---------------------------------------------------------------------------

func TestNewConfig_NoOpts_HasDefaults(t *testing.T) {
	c, err := llmrouter.NewConfig()
	if err != nil {
		t.Fatalf("NewConfig: %v", err)
	}
	if c.Timeout != 120*time.Second {
		t.Errorf("Timeout = %v, want 120s", c.Timeout)
	}
	if c.APIKey != "" {
		t.Errorf("APIKey should default empty, got %q", c.APIKey)
	}
	if c.BaseURL != "" {
		t.Errorf("BaseURL should default empty, got %q", c.BaseURL)
	}
	if c.HTTPClient != nil {
		t.Errorf("HTTPClient should default nil")
	}
	if c.Extra != nil {
		t.Errorf("Extra should default nil")
	}
}

func TestNewConfig_NilOption_Skipped(t *testing.T) {
	c, err := llmrouter.NewConfig(nil, llmrouter.WithAPIKey("k"), nil)
	if err != nil {
		t.Fatalf("NewConfig: %v", err)
	}
	if c.APIKey != "k" {
		t.Errorf("APIKey = %q", c.APIKey)
	}
}

func TestNewConfig_OptionError_Propagated(t *testing.T) {
	_, err := llmrouter.NewConfig(llmrouter.WithAPIKey(""))
	if err == nil {
		t.Fatal("expected error from empty api key")
	}
}

func TestNewConfig_ManyOptionsApplied(t *testing.T) {
	c, err := llmrouter.NewConfig(
		llmrouter.WithAPIKey("key"),
		llmrouter.WithBaseURL("https://example.com/v1"),
		llmrouter.WithTimeout(5*time.Second),
		llmrouter.WithExtra("region", "us-east-1"),
		llmrouter.WithExtra("api-version", "2024-10-21"),
	)
	if err != nil {
		t.Fatalf("NewConfig: %v", err)
	}
	if c.APIKey != "key" {
		t.Errorf("APIKey")
	}
	if c.BaseURL != "https://example.com/v1" {
		t.Errorf("BaseURL = %q", c.BaseURL)
	}
	if c.Timeout != 5*time.Second {
		t.Errorf("Timeout = %v", c.Timeout)
	}
	if c.Extra["region"] != "us-east-1" {
		t.Errorf("Extra[region]")
	}
	if c.Extra["api-version"] != "2024-10-21" {
		t.Errorf("Extra[api-version]")
	}
}

// ---------------------------------------------------------------------------
// WithAPIKey
// ---------------------------------------------------------------------------

func TestWithAPIKey_Valid(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"simple", "abc", "abc"},
		{"sk-prefix", "sk-test", "sk-test"},
		{"long", strings.Repeat("a", 256), strings.Repeat("a", 256)},
		{"leading-space-trimmed", "  k", "k"},
		{"trailing-space-trimmed", "k  ", "k"},
		{"both-trimmed", "  k  ", "k"},
		{"tab-newline-trimmed", "\t\nkey\t\n", "key"},
		{"contains-special-chars", "sk-!@#$%^&*()", "sk-!@#$%^&*()"},
		{"contains-spaces-inside", "a b c", "a b c"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, err := llmrouter.NewConfig(llmrouter.WithAPIKey(tc.in))
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			if c.APIKey != tc.want {
				t.Fatalf("APIKey = %q, want %q", c.APIKey, tc.want)
			}
		})
	}
}

func TestWithAPIKey_Invalid(t *testing.T) {
	cases := []string{"", "   ", "\t\t", "\n", "\r\n"}
	for _, in := range cases {
		t.Run("len="+strings.TrimSpace(in)+"_pad", func(t *testing.T) {
			_, err := llmrouter.NewConfig(llmrouter.WithAPIKey(in))
			if err == nil {
				t.Fatalf("expected error for %q", in)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// WithBaseURL
// ---------------------------------------------------------------------------

func TestWithBaseURL_Valid(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"openai", "https://api.openai.com/v1", "https://api.openai.com/v1"},
		{"openrouter", "https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1"},
		{"together", "https://api.together.xyz/v1", "https://api.together.xyz/v1"},
		{"localhost", "http://localhost:11434/v1", "http://localhost:11434/v1"},
		{"trailing-slash-trimmed", "https://api.openai.com/v1/", "https://api.openai.com/v1"},
		{"multi-trailing-slash-trimmed", "https://x.test/y///", "https://x.test/y"},
		{"with-port", "http://127.0.0.1:8080", "http://127.0.0.1:8080"},
		{"with-path", "https://example.com/llm/v2", "https://example.com/llm/v2"},
		{"leading-trailing-space", "  https://x.test  ", "https://x.test"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, err := llmrouter.NewConfig(
				llmrouter.WithAPIKey("k"),
				llmrouter.WithBaseURL(tc.in),
			)
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			if c.BaseURL != tc.want {
				t.Fatalf("BaseURL = %q, want %q", c.BaseURL, tc.want)
			}
		})
	}
}

func TestWithBaseURL_Invalid(t *testing.T) {
	cases := []struct {
		name string
		in   string
	}{
		{"empty", ""},
		{"whitespace-only", "   "},
		{"tab", "\t"},
		{"control-char", "http://\x7f\x00bad"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := llmrouter.NewConfig(
				llmrouter.WithAPIKey("k"),
				llmrouter.WithBaseURL(tc.in),
			)
			if err == nil {
				t.Fatalf("expected error for %q", tc.in)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// WithHTTPClient
// ---------------------------------------------------------------------------

func TestWithHTTPClient_Custom(t *testing.T) {
	custom := &http.Client{Timeout: 7 * time.Second}
	c, err := llmrouter.NewConfig(
		llmrouter.WithAPIKey("k"),
		llmrouter.WithHTTPClient(custom),
	)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if c.HTTPClient != custom {
		t.Fatal("HTTPClient not stored")
	}
	if c.HTTP() != custom {
		t.Fatal("HTTP() did not return supplied client")
	}
}

func TestWithHTTPClient_NilRejected(t *testing.T) {
	_, err := llmrouter.NewConfig(
		llmrouter.WithAPIKey("k"),
		llmrouter.WithHTTPClient(nil),
	)
	if err == nil {
		t.Fatal("expected error for nil http client")
	}
}

// ---------------------------------------------------------------------------
// WithTimeout
// ---------------------------------------------------------------------------

func TestWithTimeout_Valid(t *testing.T) {
	cases := []time.Duration{
		1 * time.Millisecond,
		1 * time.Second,
		30 * time.Second,
		2 * time.Minute,
		1 * time.Hour,
	}
	for _, d := range cases {
		t.Run(d.String(), func(t *testing.T) {
			c, err := llmrouter.NewConfig(
				llmrouter.WithAPIKey("k"),
				llmrouter.WithTimeout(d),
			)
			if err != nil {
				t.Fatalf("err = %v", err)
			}
			if c.Timeout != d {
				t.Fatalf("Timeout = %v, want %v", c.Timeout, d)
			}
		})
	}
}

func TestWithTimeout_Invalid(t *testing.T) {
	cases := []time.Duration{0, -1, -1 * time.Second, -1 * time.Hour}
	for _, d := range cases {
		t.Run(d.String(), func(t *testing.T) {
			_, err := llmrouter.NewConfig(llmrouter.WithAPIKey("k"), llmrouter.WithTimeout(d))
			if err == nil {
				t.Fatalf("expected error for %v", d)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// WithExtra
// ---------------------------------------------------------------------------

func TestWithExtra_StoresVariousTypes(t *testing.T) {
	c, err := llmrouter.NewConfig(
		llmrouter.WithAPIKey("k"),
		llmrouter.WithExtra("string", "v"),
		llmrouter.WithExtra("int", 42),
		llmrouter.WithExtra("bool", true),
		llmrouter.WithExtra("float", 3.14),
		llmrouter.WithExtra("slice", []string{"a", "b"}),
		llmrouter.WithExtra("nil-value", nil),
	)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if c.Extra["string"] != "v" {
		t.Errorf("string")
	}
	if c.Extra["int"] != 42 {
		t.Errorf("int")
	}
	if c.Extra["bool"] != true {
		t.Errorf("bool")
	}
	if c.Extra["float"] != 3.14 {
		t.Errorf("float")
	}
	if _, ok := c.Extra["slice"]; !ok {
		t.Errorf("slice missing")
	}
	if v, ok := c.Extra["nil-value"]; !ok || v != nil {
		t.Errorf("nil-value = %v ok=%v", v, ok)
	}
}

func TestWithExtra_OverrideExistingKey(t *testing.T) {
	c, err := llmrouter.NewConfig(
		llmrouter.WithAPIKey("k"),
		llmrouter.WithExtra("k", "first"),
		llmrouter.WithExtra("k", "second"),
	)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if c.Extra["k"] != "second" {
		t.Fatalf("Extra[k] = %v, want 'second'", c.Extra["k"])
	}
}

func TestWithExtra_EmptyKeyRejected(t *testing.T) {
	cases := []string{"", "   ", "\t", "\n"}
	for _, in := range cases {
		t.Run("k="+strings.TrimSpace(in)+"_pad", func(t *testing.T) {
			_, err := llmrouter.NewConfig(
				llmrouter.WithAPIKey("k"),
				llmrouter.WithExtra(in, "v"),
			)
			if err == nil {
				t.Fatalf("expected error for empty key %q", in)
			}
		})
	}
}

func TestWithExtra_KeyTrimmed(t *testing.T) {
	c, err := llmrouter.NewConfig(
		llmrouter.WithAPIKey("k"),
		llmrouter.WithExtra("  region  ", "us-east-1"),
	)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if c.Extra["region"] != "us-east-1" {
		t.Fatalf("Extra[region] = %v", c.Extra["region"])
	}
}

// ---------------------------------------------------------------------------
// Config.HTTP()
// ---------------------------------------------------------------------------

func TestHTTP_LazyDefaultUsesTimeout(t *testing.T) {
	c, err := llmrouter.NewConfig(
		llmrouter.WithAPIKey("k"),
		llmrouter.WithTimeout(9*time.Second),
	)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	client := c.HTTP()
	if client == nil {
		t.Fatal("HTTP() returned nil")
	}
	if client.Timeout != 9*time.Second {
		t.Fatalf("Timeout = %v, want 9s", client.Timeout)
	}
}

func TestHTTP_LazyClientIsCached(t *testing.T) {
	c, err := llmrouter.NewConfig(llmrouter.WithAPIKey("k"))
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	first := c.HTTP()
	second := c.HTTP()
	if first != second {
		t.Fatal("HTTP() returned different instances on repeat calls")
	}
}

func TestHTTP_CustomClientWins(t *testing.T) {
	custom := &http.Client{Timeout: 1 * time.Minute}
	c, err := llmrouter.NewConfig(
		llmrouter.WithAPIKey("k"),
		llmrouter.WithHTTPClient(custom),
		llmrouter.WithTimeout(1*time.Second), // should be ignored
	)
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if c.HTTP() != custom {
		t.Fatal("custom client should win")
	}
}

// ---------------------------------------------------------------------------
// ErrInvalidConfig sentinel
// ---------------------------------------------------------------------------

func TestErrInvalidConfig_Sentinel(t *testing.T) {
	if llmrouter.ErrInvalidConfig == nil {
		t.Fatal("ErrInvalidConfig is nil")
	}
	if llmrouter.ErrInvalidConfig.Error() == "" {
		t.Fatal("ErrInvalidConfig has empty message")
	}
	wrapped := errors.New("not it")
	if errors.Is(wrapped, llmrouter.ErrInvalidConfig) {
		t.Fatal("unrelated error should not match sentinel")
	}
}
