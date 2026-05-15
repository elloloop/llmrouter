// Package cartesia implements the llmrouter.Speaker interface against
// the Cartesia API.
//
// Cartesia is popular for real-time conversational voice agents with very
// low latency. This package exposes only the TTS path: batch synthesis
// via POST /tts/bytes and streaming synthesis via POST /tts/sse (Server-
// Sent Events). The websocket endpoint is intentionally not implemented
// in v0.3.
//
// Cartesia does not offer an LLM chat completion API, so this provider
// intentionally does NOT implement llmrouter.Provider — only Speaker.
//
// Authentication uses two headers — `X-API-Key` and `Cartesia-Version` —
// configured via llmrouter.WithAPIKey and the package-level
// cartesiaVersion constant.
package cartesia

import (
	"fmt"

	"github.com/elloloop/llmrouter"
)

// DefaultBaseURL is the production Cartesia API root.
const DefaultBaseURL = "https://api.cartesia.ai"

// cartesiaVersion pins the Cartesia API contract version. Sent as the
// `Cartesia-Version` header on every request.
const cartesiaVersion = "2024-11-13"

// providerName is the canonical id used in ErrUpstream and Name().
const providerName = "cartesia"

// defaultVoiceID is Cartesia's reference English voice — used when
// SpeechRequest.Voice is empty.
//
// TODO(cartesia): expose this as a provider-level option (e.g. via
// llmrouter.WithExtra("voice_id", "...")) so callers can change the
// fallback without having to set Voice on every SpeechRequest.
const defaultVoiceID = "79a125e8-cd45-4c13-8a67-188112f4dd22"

// defaultTTSModel is the Cartesia model used when SpeechRequest.Model is
// empty. Sonic-2 is Cartesia's current production voice model.
const defaultTTSModel = "sonic-2"

// Provider talks to Cartesia. Construct with New.
//
// Provider satisfies llmrouter.Speaker but NOT llmrouter.Provider —
// Cartesia has no chat-completion endpoint.
type Provider struct {
	cfg *llmrouter.Config
}

// New builds a Provider from llmrouter options. WithAPIKey is required.
func New(opts ...llmrouter.Option) (*Provider, error) {
	cfg, err := llmrouter.NewConfig(opts...)
	if err != nil {
		return nil, err
	}
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("%w: cartesia requires an api key", llmrouter.ErrInvalidConfig)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	return &Provider{cfg: cfg}, nil
}

// Name returns the provider id used in error wrapping and telemetry.
func (p *Provider) Name() string { return providerName }
