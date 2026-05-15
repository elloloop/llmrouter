// Package deepgram implements llmrouter.Transcriber against Deepgram's
// batch speech-to-text API (/v1/listen).
//
// The provider is STT only — it does not implement chat, TTS, or
// embeddings. Streaming (/v1/listen over WebSocket) is deferred to a
// future version: for v0.3 we always issue a synchronous batch request
// and emit a single Final TranscriptSegment.
package deepgram

import (
	"fmt"

	"github.com/elloloop/llmrouter"
)

const (
	// DefaultBaseURL is the public Deepgram API root used when the
	// caller does not override it via llmrouter.WithBaseURL.
	DefaultBaseURL = "https://api.deepgram.com"
	providerName   = "deepgram"
)

// Provider talks to Deepgram /v1/listen. Construct with New.
type Provider struct {
	cfg *llmrouter.Config
}

// New builds a Provider from llmrouter options. WithAPIKey is required;
// BaseURL defaults to DefaultBaseURL when not supplied.
func New(opts ...llmrouter.Option) (*Provider, error) {
	cfg, err := llmrouter.NewConfig(opts...)
	if err != nil {
		return nil, err
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("%w: deepgram requires an api key", llmrouter.ErrInvalidConfig)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	return &Provider{cfg: cfg}, nil
}

// Name returns the provider id.
func (p *Provider) Name() string { return providerName }

// Compile-time assertion: *Provider satisfies llmrouter.Transcriber.
var _ llmrouter.Transcriber = (*Provider)(nil)
