package llmrouter

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Config is the per-provider configuration. Providers expose New(opts...)
// which builds a Config internally; consumers don't usually touch Config
// directly.
type Config struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
	Timeout    time.Duration
	// Extra holds provider-specific options (e.g. Azure api-version,
	// AWS region, GCP project). Providers document the keys they read.
	Extra map[string]any
}

// Option mutates a Config. Use the With* constructors below.
type Option func(*Config) error

// NewConfig applies the given options to a fresh Config. Each provider
// calls this internally from its New constructor.
func NewConfig(opts ...Option) (*Config, error) {
	c := &Config{Timeout: 120 * time.Second}
	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt(c); err != nil {
			return nil, err
		}
	}
	return c, nil
}

// HTTP returns the configured HTTP client, constructing a default one
// lazily if the caller did not supply one. Provider implementations
// should always go through this method.
func (c *Config) HTTP() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	c.HTTPClient = &http.Client{Timeout: c.Timeout}
	return c.HTTPClient
}

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(c *Config) error {
		key = strings.TrimSpace(key)
		if key == "" {
			return errors.New("api key cannot be empty")
		}
		c.APIKey = key
		return nil
	}
}

// WithBaseURL overrides the provider's default base URL. Useful for
// pointing the OpenAI provider at OpenRouter, Together, Groq, or a
// self-hosted endpoint; or the Azure OpenAI provider at a specific
// resource's *.openai.azure.com hostname.
func WithBaseURL(u string) Option {
	return func(c *Config) error {
		u = strings.TrimSpace(u)
		if u == "" {
			return errors.New("base url cannot be empty")
		}
		if _, err := url.Parse(u); err != nil {
			return fmt.Errorf("invalid base url: %w", err)
		}
		c.BaseURL = strings.TrimRight(u, "/")
		return nil
	}
}

// WithHTTPClient supplies a custom HTTP client. When provided, Timeout
// is ignored — the client's own transport governs timeouts.
func WithHTTPClient(client *http.Client) Option {
	return func(c *Config) error {
		if client == nil {
			return errors.New("http client cannot be nil")
		}
		c.HTTPClient = client
		return nil
	}
}

// WithTimeout sets the request timeout used by the default HTTP client.
// Ignored if WithHTTPClient is also supplied.
func WithTimeout(d time.Duration) Option {
	return func(c *Config) error {
		if d <= 0 {
			return errors.New("timeout must be positive")
		}
		c.Timeout = d
		return nil
	}
}

// WithExtra attaches a provider-specific config value. Examples:
//   - Azure: llmrouter.WithExtra("api-version", "2024-10-21")
//   - AWS Bedrock: llmrouter.WithExtra("region", "us-east-1")
//   - Vertex: llmrouter.WithExtra("project", "my-gcp-proj"), WithExtra("region", "us-central1")
func WithExtra(key string, value any) Option {
	return func(c *Config) error {
		key = strings.TrimSpace(key)
		if key == "" {
			return errors.New("extra key cannot be empty")
		}
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[key] = value
		return nil
	}
}
