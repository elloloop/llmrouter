package llmrouter

import (
	"errors"
	"fmt"
)

// ErrUpstream wraps a non-2xx response from the upstream provider.
type ErrUpstream struct {
	Provider   string
	StatusCode int
	Body       string
}

func (e *ErrUpstream) Error() string {
	return fmt.Sprintf("%s upstream %d: %s", e.Provider, e.StatusCode, e.Body)
}

// ErrInvalidConfig is returned by provider New() for malformed config.
var ErrInvalidConfig = errors.New("invalid provider config")
