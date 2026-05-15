// Package elevenlabs implements the llmrouter.Speaker (TTS) and
// llmrouter.Transcriber (STT) interfaces against the ElevenLabs API.
//
// ElevenLabs is best-in-class for text-to-speech; their Scribe model
// powers the speech-to-text path. ElevenLabs does not offer an LLM chat
// completion API, so this provider intentionally does NOT implement
// llmrouter.Provider — only the audio interfaces.
//
// Authentication uses the `xi-api-key` header (not `Authorization:
// Bearer`). Configure the provider with llmrouter.WithAPIKey and,
// optionally, llmrouter.WithBaseURL to point at a proxy.
package elevenlabs

import (
	"fmt"

	"github.com/elloloop/llmrouter"
)

// DefaultBaseURL is the production ElevenLabs API root.
const DefaultBaseURL = "https://api.elevenlabs.io"

// providerName is the canonical id used in ErrUpstream and Name().
const providerName = "elevenlabs"

// defaultVoiceID is ElevenLabs' "Rachel" voice — used when
// SpeechRequest.Voice is empty.
const defaultVoiceID = "21m00Tcm4TlvDq8ikWAM"

// defaultTTSModel is the model used when SpeechRequest.Model is empty.
const defaultTTSModel = "eleven_turbo_v2_5"

// defaultSTTModel is the Scribe model used when TranscribeRequest.Model
// is empty.
const defaultSTTModel = "scribe_v1"

// Provider talks to ElevenLabs. Construct with New.
//
// Provider satisfies llmrouter.Speaker and llmrouter.Transcriber but NOT
// llmrouter.Provider — ElevenLabs has no chat-completion endpoint.
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
		return nil, fmt.Errorf("%w: elevenlabs requires an api key", llmrouter.ErrInvalidConfig)
	}
	if cfg.BaseURL == "" {
		cfg.BaseURL = DefaultBaseURL
	}
	return &Provider{cfg: cfg}, nil
}

// Name returns the provider id used in error wrapping and telemetry.
func (p *Provider) Name() string { return providerName }
