package groq

import (
	"context"

	"github.com/elloloop/llmrouter"
)

// Transcribe delegates to the inner openai provider's STT implementation
// against Groq's OpenAI-compatible /audio/transcriptions endpoint. Any
// *llmrouter.ErrUpstream surfaced by the inner provider has its Provider
// field rewritten from "openai" to "groq" so callers, logs, and metrics
// attribute the failure correctly.
//
// Groq does not host TTS, so no Speak method is implemented here.
func (p *Provider) Transcribe(ctx context.Context, req llmrouter.TranscribeRequest) (*llmrouter.TranscriptStream, error) {
	stream, err := p.inner.Transcribe(ctx, req)
	if err != nil {
		return nil, rewriteUpstreamProvider(err)
	}
	return stream, nil
}
