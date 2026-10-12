package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"mime"
	"net/http"
	"strings"
	"unicode/utf8"

	"github.com/elloloop/llmrouter"
)

// CompletionLimits supplies explicit bounds on serialized native request and
// response bodies, including metadata and opaque thinking blocks.
type CompletionLimits struct {
	MaxRequestBytes  int64
	MaxResponseBytes int64
}

// MessageResponse preserves Anthropic's native model, stop reason and content.
// Complete does not decide whether a stop reason or tool is acceptable to the
// caller. In particular, a parsed response is not evidence of a successful turn.
// Raw retains the entire body, including fields unknown to this version.
type MessageResponse struct {
	ID           string            `json:"id"`
	Type         string            `json:"type"`
	Role         string            `json:"role"`
	Model        string            `json:"model"`
	StopReason   string            `json:"stop_reason"`
	Content      []json.RawMessage `json:"content"`
	StopSequence *string           `json:"stop_sequence"`
	Usage        json.RawMessage   `json:"usage"`
	Raw          json.RawMessage   `json:"-"`
}

var (
	// ErrRequestTooLarge means the serialized request exceeded the caller's bound.
	ErrRequestTooLarge = errors.New("anthropic: request exceeds byte limit")
	// ErrInvalidResponse means the body lacks a valid native message envelope.
	ErrInvalidResponse = errors.New("anthropic: invalid native message response")
	// ErrResponseTooLarge means the response exceeded the caller's byte bound.
	ErrResponseTooLarge = errors.New("anthropic: response exceeds byte limit")
	// ErrTransport means dispatch or response reading failed. Remote completion
	// may be unknown; Complete never retries. Diagnostics are not exposed.
	ErrTransport = errors.New("anthropic: request transport failed; remote completion may be unknown")
)

// HTTPStatusError exposes only the HTTP status, never provider diagnostics.
type HTTPStatusError struct {
	StatusCode int
}

func (e *HTTPStatusError) Error() string {
	return fmt.Sprintf("anthropic: unexpected HTTP status %d", e.StatusCode)
}

// Complete issues one non-streaming Messages request with a required positive
// token budget and request/response byte limits. Native assistant content is replayed
// without translation. The response model is never replaced with the requested
// model, and unknown stop reasons and content types are preserved for callers.
//
// Complete never follows redirects, sends cookies, retries, logs provider data,
// or exposes diagnostic bodies/URLs in errors. An injected transport must honor
// context cancellation and must not retry. Other provider methods are unchanged.
func (p *Provider) Complete(ctx context.Context, request llmrouter.ChatRequest, limits CompletionLimits) (*MessageResponse, error) {
	if ctx == nil || strings.TrimSpace(request.Model) == "" || request.MaxTokens <= 0 || request.Stream || limits.MaxRequestBytes <= 0 || limits.MaxResponseBytes <= 0 || limits.MaxResponseBytes == math.MaxInt64 {
		return nil, fmt.Errorf("%w: native completion needs a context, model, positive token/byte limits and stream disabled", llmrouter.ErrInvalidConfig)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	body, err := buildMessagesBody(request, false)
	if err != nil {
		return nil, fmt.Errorf("%w: invalid native completion request", llmrouter.ErrInvalidConfig)
	}
	if int64(len(body)) > limits.MaxRequestBytes {
		return nil, ErrRequestTooLarge
	}
	hreq, err := p.messageRequest(ctx, body, "application/json")
	if err != nil {
		return nil, fmt.Errorf("%w: invalid message endpoint", llmrouter.ErrInvalidConfig)
	}
	hreq.GetBody = nil
	client := http.Client{Timeout: p.cfg.Timeout}
	if p.cfg.HTTPClient != nil {
		client = *p.cfg.HTTPClient
	}
	client.Jar = nil
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	response, err := client.Do(hreq)
	if err != nil {
		return nil, completionTransportError(ctx)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, &HTTPStatusError{StatusCode: response.StatusCode}
	}
	mediaType, _, err := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if err != nil || mediaType != "application/json" {
		return nil, ErrInvalidResponse
	}
	raw, err := io.ReadAll(io.LimitReader(response.Body, limits.MaxResponseBytes+1))
	if err != nil {
		return nil, completionTransportError(ctx)
	}
	if int64(len(raw)) > limits.MaxResponseBytes {
		return nil, ErrResponseTooLarge
	}
	return decodeMessageResponse(raw)
}

func completionTransportError(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return errors.Join(ErrTransport, err)
	}
	return ErrTransport
}

func decodeMessageResponse(raw []byte) (*MessageResponse, error) {
	var message MessageResponse
	if !utf8.Valid(raw) || json.Unmarshal(raw, &message) != nil || strings.TrimSpace(message.ID) == "" || message.Type != "message" || message.Role != "assistant" || strings.TrimSpace(message.Model) == "" || strings.TrimSpace(message.StopReason) == "" || message.Content == nil {
		return nil, ErrInvalidResponse
	}
	for _, content := range message.Content {
		var block struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(content, &block) != nil || strings.TrimSpace(block.Type) == "" {
			return nil, ErrInvalidResponse
		}
	}
	message.Raw = raw
	return &message, nil
}
