// Package geminilive implements Google's Gemini Live API, a
// WebSocket-based bidirectional protocol that carries text + audio +
// tool-call events between client and a Gemini model.
//
// Unlike llmrouter's REST providers, Live is not a request/response
// API. Callers open a Session via Provider.Connect, then concurrently
// push events (text, audio buffers, tool results) and consume server
// events (server.text, server.audio, server.tool_call,
// server.turn_complete, error) from Session.Events().
//
// The public surface intentionally mirrors providers/openairealtime so
// callers can swap providers with minimal code changes; only the
// SessionConfig fields and SessionEvent.Type values that the two
// vendors model differently diverge.
//
// Authentication uses the API key in the URL query string (this is
// Google's choice for this endpoint — there is no Authorization
// header). Configure with llmrouter.WithAPIKey and optionally
// llmrouter.WithBaseURL.
package geminilive

import (
	"fmt"

	"github.com/elloloop/llmrouter"
)

// DefaultBaseURL is the production Gemini Live root. The Live path
// lives under
// /ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent.
// The scheme is normalised to wss:// at dial time so callers may also
// pass https://generativelanguage.googleapis.com.
const DefaultBaseURL = "wss://generativelanguage.googleapis.com"

// providerName is the canonical id used in ErrUpstream and Name().
const providerName = "geminilive"

// defaultModel is the model used when SessionConfig.Model is empty.
const defaultModel = "models/gemini-2.0-flash-exp"

// errorBodySnippetLimit caps how many bytes of an upstream error body
// we surface in ErrUpstream.Body when the WebSocket handshake fails
// with a non-2xx HTTP status.
const errorBodySnippetLimit = 1024

// Provider talks to the Gemini Live endpoint. Construct with New.
// Provider does NOT satisfy llmrouter.Speaker, llmrouter.Transcriber,
// or llmrouter.Provider — Live is exposed via its own Connect API.
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
		return nil, fmt.Errorf("%w: geminilive requires an api key", llmrouter.ErrInvalidConfig)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	return &Provider{cfg: cfg}, nil
}

// Name returns the provider id used in error wrapping and telemetry.
func (p *Provider) Name() string { return providerName }
