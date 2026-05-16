// Package openairealtime implements OpenAI's Realtime API
// (gpt-4o-realtime), a WebSocket-based bidirectional protocol that
// carries text + audio events between client and model.
//
// Unlike llmrouter's REST providers, Realtime is not a request/response
// API. Callers open a Session via Provider.Connect, then concurrently
// push events (text, audio buffers, response.create) and consume server
// events (response.text.delta, response.audio.delta, response.done,
// error) from Session.Events().
//
// Scope of v0.4.0 is intentionally limited to three patterns:
//   - text-in / audio-out
//   - audio-in / audio-out
//   - audio-in / text-out
//
// Tool use and function calls are deferred. The package therefore does
// NOT implement llmrouter.Speaker, llmrouter.Transcriber, or
// llmrouter.Provider — Realtime is its own thing.
//
// Authentication uses an `Authorization: Bearer <key>` header alongside
// the required `OpenAI-Beta: realtime=v1` header on the WebSocket
// upgrade. Configure with llmrouter.WithAPIKey and optionally
// llmrouter.WithBaseURL (use a wss:// or https:// scheme — both work).
package openairealtime

import (
	"fmt"

	"github.com/elloloop/llmrouter"
)

// DefaultBaseURL is the production OpenAI Realtime root. The Realtime
// path lives under /realtime; the scheme is normalised to wss:// at
// dial time so callers may also pass https://api.openai.com/v1.
const DefaultBaseURL = "wss://api.openai.com/v1"

// providerName is the canonical id used in ErrUpstream and Name().
const providerName = "openairealtime"

// betaHeader is the value required on the `OpenAI-Beta` request header
// for any /realtime upgrade. Without it the server rejects the dial.
const betaHeader = "realtime=v1"

// defaultModel is the model used when SessionConfig.Model is empty.
const defaultModel = "gpt-4o-realtime-preview"

// errorBodySnippetLimit caps how many bytes of an upstream error body
// we surface in ErrUpstream.Body when the WebSocket handshake fails
// with a non-2xx HTTP status.
const errorBodySnippetLimit = 1024

// Provider talks to the OpenAI Realtime endpoint. Construct with New.
// Provider does NOT satisfy llmrouter.Speaker, llmrouter.Transcriber,
// or llmrouter.Provider — Realtime is exposed via its own Connect API.
type Provider struct {
	cfg *llmrouter.Config
}

// New builds a Provider from llmrouter options. WithAPIKey is required;
// WithBaseURL defaults to DefaultBaseURL when omitted.
func New(opts ...llmrouter.Option) (*Provider, error) {
	cfg, err := llmrouter.NewConfig(opts...)
	if err != nil {
		return nil, err
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("%w: openairealtime requires an api key", llmrouter.ErrInvalidConfig)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	return &Provider{cfg: cfg}, nil
}

// Name returns the provider id used in error wrapping and telemetry.
func (p *Provider) Name() string { return providerName }
