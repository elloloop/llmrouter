package llmrouter_test

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/elloloop/llmrouter"
)

func TestErrUpstream_ImplementsError(t *testing.T) {
	var _ error = (*llmrouter.ErrUpstream)(nil)
}

func TestErrUpstream_Message_ContainsProviderStatusBody(t *testing.T) {
	cases := []struct {
		name string
		e    llmrouter.ErrUpstream
	}{
		{"openai-401", llmrouter.ErrUpstream{Provider: "openai", StatusCode: 401, Body: "invalid api key"}},
		{"anthropic-429", llmrouter.ErrUpstream{Provider: "anthropic", StatusCode: 429, Body: "rate limited"}},
		{"unknown-500", llmrouter.ErrUpstream{Provider: "x", StatusCode: 500, Body: "boom"}},
		{"empty-body", llmrouter.ErrUpstream{Provider: "openai", StatusCode: 503, Body: ""}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			msg := tc.e.Error()
			if !strings.Contains(msg, tc.e.Provider) {
				t.Errorf("missing provider: %q", msg)
			}
			if !strings.Contains(msg, fmt.Sprintf("%d", tc.e.StatusCode)) {
				t.Errorf("missing status: %q", msg)
			}
			if tc.e.Body != "" && !strings.Contains(msg, tc.e.Body) {
				t.Errorf("missing body: %q", msg)
			}
		})
	}
}

func TestErrUpstream_WorksWithErrorsAs(t *testing.T) {
	src := &llmrouter.ErrUpstream{Provider: "openai", StatusCode: http.StatusBadGateway, Body: "x"}
	wrapped := fmt.Errorf("calling upstream: %w", src)
	var got *llmrouter.ErrUpstream
	if !errors.As(wrapped, &got) {
		t.Fatalf("errors.As did not unwrap")
	}
	if got.StatusCode != http.StatusBadGateway {
		t.Errorf("StatusCode = %d", got.StatusCode)
	}
}

func TestErrUpstream_StatusCodeRangeCommonValues(t *testing.T) {
	codes := []int{400, 401, 403, 404, 408, 409, 422, 429, 500, 502, 503, 504}
	for _, code := range codes {
		t.Run(fmt.Sprintf("status-%d", code), func(t *testing.T) {
			e := &llmrouter.ErrUpstream{Provider: "openai", StatusCode: code, Body: "x"}
			if !strings.Contains(e.Error(), fmt.Sprintf("%d", code)) {
				t.Errorf("error message missing status %d: %q", code, e.Error())
			}
		})
	}
}

func TestErrInvalidConfig_IsExported(t *testing.T) {
	if llmrouter.ErrInvalidConfig == nil {
		t.Fatal("ErrInvalidConfig nil")
	}
	if !strings.Contains(llmrouter.ErrInvalidConfig.Error(), "invalid") {
		t.Errorf("message should mention 'invalid': %q", llmrouter.ErrInvalidConfig.Error())
	}
}

func TestErrInvalidConfig_WrappingIsRespectedByErrorsIs(t *testing.T) {
	wrapped := fmt.Errorf("openai: %w: missing api key", llmrouter.ErrInvalidConfig)
	if !errors.Is(wrapped, llmrouter.ErrInvalidConfig) {
		t.Fatal("wrapped error not matched by errors.Is")
	}
}
