package llmrouter

import (
	"errors"
	"fmt"
)

// ErrUpstream wraps an error response from the upstream provider.
//
// A non-zero StatusCode indicates the upstream returned that HTTP status
// before any streaming began (the request never reached the SSE phase).
//
// A StatusCode of 0 indicates a mid-stream error: the upstream HTTP
// response was 200 and an SSE stream began, but an error envelope was
// emitted in-band as part of the event stream (for example Anthropic's
// `event: error` overload event, or OpenAI's `data: {"error": ...}`
// chunk used by some compatibility proxies). Callers can distinguish
// connect-time errors from mid-stream errors by inspecting StatusCode.
type ErrUpstream struct {
	Provider   string
	StatusCode int
	Body       string
}

func (e *ErrUpstream) Error() string {
	if e.StatusCode == 0 {
		return fmt.Sprintf("%s upstream mid-stream error: %s", e.Provider, e.Body)
	}
	return fmt.Sprintf("%s upstream %d: %s", e.Provider, e.StatusCode, e.Body)
}

// ErrInvalidConfig is returned by provider New() for malformed config.
var ErrInvalidConfig = errors.New("invalid provider config")
