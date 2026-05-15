package openrouter_test

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/openrouter"
)

// capturedRequest records the headers and path the fake upstream saw on
// the most recent inbound request.
type capturedRequest struct {
	mu      sync.Mutex
	headers http.Header
	path    string
	hits    int32
}

func (c *capturedRequest) record(r *http.Request) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.headers = r.Header.Clone()
	c.path = r.URL.Path
	atomic.AddInt32(&c.hits, 1)
}

func (c *capturedRequest) get(key string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.headers == nil {
		return ""
	}
	return c.headers.Get(key)
}

func (c *capturedRequest) urlPath() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.path
}

func (c *capturedRequest) count() int {
	return int(atomic.LoadInt32(&c.hits))
}

// fakeSSEServer returns an httptest server that records the inbound
// request and emits one trivial SSE event followed by [DONE].
func fakeSSEServer(t *testing.T, cap *capturedRequest) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.record(r)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		payload := `{"id":"id-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}`
		fmt.Fprintf(w, "data: %s\n\n", payload)
		if flusher != nil {
			flusher.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
	}))
}

// fakeErrorServer returns an httptest server that always responds with
// the given status code and body.
func fakeErrorServer(t *testing.T, status int, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		_, _ = w.Write([]byte(body))
	}))
}

// drainStream consumes all chunks and returns the terminal stream
// error (if any).
func drainStream(s *llmrouter.Stream) error {
	for range s.Chunks() {
	}
	return s.Err()
}

// newAndStream is a small helper that constructs the provider, opens a
// stream, drains it, and returns the error from CompletionStream.
func newAndStream(t *testing.T, opts ...llmrouter.Option) error {
	t.Helper()
	p, err := openrouter.New(opts...)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "openrouter/auto",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		return err
	}
	return drainStream(stream)
}

func TestNew(t *testing.T) {
	t.Run("requires api key", func(t *testing.T) {
		if _, err := openrouter.New(); err == nil {
			t.Fatal("expected error when API key is missing")
		}
	})

	t.Run("succeeds with api key", func(t *testing.T) {
		p, err := openrouter.New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if p == nil {
			t.Fatal("provider is nil")
		}
	})

	t.Run("returns error from WithAPIKey empty", func(t *testing.T) {
		if _, err := openrouter.New(llmrouter.WithAPIKey("")); err == nil {
			t.Fatal("expected empty api key error")
		}
	})

	t.Run("returns error from WithBaseURL empty", func(t *testing.T) {
		if _, err := openrouter.New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL("")); err == nil {
			t.Fatal("expected empty base url error")
		}
	})

	t.Run("nil option is ignored", func(t *testing.T) {
		if _, err := openrouter.New(llmrouter.WithAPIKey("k"), nil); err != nil {
			t.Fatalf("nil option should be tolerated: %v", err)
		}
	})
}

func TestName(t *testing.T) {
	t.Run("returns openrouter", func(t *testing.T) {
		p, err := openrouter.New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if got := p.Name(); got != "openrouter" {
			t.Fatalf("Name = %q, want openrouter", got)
		}
	})
}

func TestDefaultBaseURL(t *testing.T) {
	t.Run("constant value", func(t *testing.T) {
		want := "https://openrouter.ai/api/v1"
		if openrouter.DefaultBaseURL != want {
			t.Fatalf("DefaultBaseURL = %q, want %q", openrouter.DefaultBaseURL, want)
		}
	})
}

func TestBaseURL(t *testing.T) {
	t.Run("defaults to OpenRouter when not set", func(t *testing.T) {
		// Use an httptest server only so we can observe the request
		// path; the host portion comes from WithBaseURL.
		// To verify the default URL is used, intercept via a custom
		// RoundTripper.
		var captured string
		client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			captured = req.URL.String()
			return &http.Response{
				StatusCode: 200,
				Body:       http.NoBody,
				Header:     make(http.Header),
			}, nil
		})}
		p, err := openrouter.New(
			llmrouter.WithAPIKey("k"),
			llmrouter.WithHTTPClient(client),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_, _ = p.CompletionStream(ctx, llmrouter.ChatRequest{Model: "m"})
		if !strings.HasPrefix(captured, openrouter.DefaultBaseURL+"/chat/completions") {
			t.Fatalf("default base URL not used: %q", captured)
		}
	})

	t.Run("user override wins", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()

		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if cap.urlPath() != "/chat/completions" {
			t.Fatalf("path = %q, want /chat/completions", cap.urlPath())
		}
	})

	t.Run("override base URL with trailing slash trimmed", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()

		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL+"/"),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if cap.urlPath() != "/chat/completions" {
			t.Fatalf("path = %q, want /chat/completions", cap.urlPath())
		}
	})
}

func TestAttributionHeaders(t *testing.T) {
	t.Run("WithReferer sets HTTP-Referer", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
			openrouter.WithReferer("https://example.com"),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := cap.get("Http-Referer"); got != "https://example.com" {
			t.Fatalf("HTTP-Referer = %q, want https://example.com", got)
		}
	})

	t.Run("WithTitle sets X-Title", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
			openrouter.WithTitle("My App"),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := cap.get("X-Title"); got != "My App" {
			t.Fatalf("X-Title = %q, want My App", got)
		}
	})

	t.Run("both headers together", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
			openrouter.WithReferer("https://a.test"),
			openrouter.WithTitle("Title"),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := cap.get("Http-Referer"); got != "https://a.test" {
			t.Fatalf("HTTP-Referer = %q", got)
		}
		if got := cap.get("X-Title"); got != "Title" {
			t.Fatalf("X-Title = %q", got)
		}
	})

	t.Run("absent when neither option supplied", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := cap.get("Http-Referer"); got != "" {
			t.Fatalf("HTTP-Referer should be empty, got %q", got)
		}
		if got := cap.get("X-Title"); got != "" {
			t.Fatalf("X-Title should be empty, got %q", got)
		}
	})

	t.Run("only referer set", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
			openrouter.WithReferer("https://only.test"),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := cap.get("Http-Referer"); got != "https://only.test" {
			t.Fatalf("HTTP-Referer = %q", got)
		}
		if got := cap.get("X-Title"); got != "" {
			t.Fatalf("X-Title should be empty, got %q", got)
		}
	})

	t.Run("only title set", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
			openrouter.WithTitle("Just Title"),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := cap.get("X-Title"); got != "Just Title" {
			t.Fatalf("X-Title = %q", got)
		}
		if got := cap.get("Http-Referer"); got != "" {
			t.Fatalf("HTTP-Referer should be empty, got %q", got)
		}
	})

	t.Run("empty referer string is skipped", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
			openrouter.WithReferer(""),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := cap.get("Http-Referer"); got != "" {
			t.Fatalf("HTTP-Referer should be empty, got %q", got)
		}
	})

	t.Run("empty title string is skipped", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
			openrouter.WithTitle(""),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := cap.get("X-Title"); got != "" {
			t.Fatalf("X-Title should be empty, got %q", got)
		}
	})

	t.Run("headers applied on every request", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		p, err := openrouter.New(
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
			openrouter.WithReferer("https://r.test"),
			openrouter.WithTitle("T"),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		for i := 0; i < 3; i++ {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{Model: "m"})
			if err != nil {
				cancel()
				t.Fatalf("CompletionStream: %v", err)
			}
			if err := drainStream(stream); err != nil {
				cancel()
				t.Fatalf("drain: %v", err)
			}
			cancel()
		}
		if cap.count() != 3 {
			t.Fatalf("server hits = %d, want 3", cap.count())
		}
		if got := cap.get("Http-Referer"); got != "https://r.test" {
			t.Fatalf("HTTP-Referer = %q on last request", got)
		}
	})
}

func TestAuthorizationHeader(t *testing.T) {
	t.Run("bearer token propagated", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		if err := newAndStream(t,
			llmrouter.WithAPIKey("sk-or-secret"),
			llmrouter.WithBaseURL(srv.URL),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := cap.get("Authorization"); got != "Bearer sk-or-secret" {
			t.Fatalf("Authorization = %q, want Bearer sk-or-secret", got)
		}
	})

	t.Run("bearer coexists with attribution headers", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		if err := newAndStream(t,
			llmrouter.WithAPIKey("sk-x"),
			llmrouter.WithBaseURL(srv.URL),
			openrouter.WithReferer("https://r.test"),
			openrouter.WithTitle("T"),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := cap.get("Authorization"); got != "Bearer sk-x" {
			t.Fatalf("Authorization = %q", got)
		}
		if got := cap.get("Http-Referer"); got != "https://r.test" {
			t.Fatalf("HTTP-Referer = %q", got)
		}
		if got := cap.get("X-Title"); got != "T" {
			t.Fatalf("X-Title = %q", got)
		}
	})
}

func TestUpstreamError(t *testing.T) {
	t.Run("4xx maps to ErrUpstream with openrouter provider", func(t *testing.T) {
		srv := fakeErrorServer(t, http.StatusBadRequest, "bad model")
		defer srv.Close()

		err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
		)
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.Provider != "openrouter" {
			t.Fatalf("Provider = %q, want openrouter", ue.Provider)
		}
		if ue.StatusCode != http.StatusBadRequest {
			t.Fatalf("StatusCode = %d, want 400", ue.StatusCode)
		}
		if !strings.Contains(ue.Body, "bad model") {
			t.Fatalf("Body = %q, want to contain bad model", ue.Body)
		}
	})

	t.Run("401 maps to ErrUpstream", func(t *testing.T) {
		srv := fakeErrorServer(t, http.StatusUnauthorized, "unauthorized")
		defer srv.Close()
		err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
		)
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.Provider != "openrouter" || ue.StatusCode != 401 {
			t.Fatalf("got provider=%q status=%d", ue.Provider, ue.StatusCode)
		}
	})

	t.Run("429 maps to ErrUpstream", func(t *testing.T) {
		srv := fakeErrorServer(t, http.StatusTooManyRequests, "rate limited")
		defer srv.Close()
		err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
		)
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.Provider != "openrouter" {
			t.Fatalf("Provider = %q", ue.Provider)
		}
	})

	t.Run("500 maps to ErrUpstream", func(t *testing.T) {
		srv := fakeErrorServer(t, http.StatusInternalServerError, "boom")
		defer srv.Close()
		err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
		)
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("err = %v, want *ErrUpstream", err)
		}
		if ue.Provider != "openrouter" {
			t.Fatalf("Provider = %q", ue.Provider)
		}
	})

	t.Run("error message contains provider name", func(t *testing.T) {
		srv := fakeErrorServer(t, 418, "teapot")
		defer srv.Close()
		err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err == nil || !strings.Contains(err.Error(), "openrouter") {
			t.Fatalf("err = %v, want to contain openrouter", err)
		}
	})
}

func TestSuccessfulStream(t *testing.T) {
	t.Run("chunks are forwarded", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		p, err := openrouter.New(
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
			Model:    "m",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}
		var got string
		for chunk := range stream.Chunks() {
			for _, c := range chunk.Choices {
				got += c.Delta.Content
			}
		}
		if err := stream.Err(); err != nil {
			t.Fatalf("stream err: %v", err)
		}
		if got != "hi" {
			t.Fatalf("content = %q, want hi", got)
		}
	})

	t.Run("path is chat/completions", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if cap.urlPath() != "/chat/completions" {
			t.Fatalf("path = %q", cap.urlPath())
		}
	})
}

func TestHTTPClientOverride(t *testing.T) {
	t.Run("default client receives wrapped attribution headers", func(t *testing.T) {
		// When no custom HTTPClient is supplied, the wrapping
		// roundtripper must inject the headers before the default
		// transport sends the request. Use a real httptest server so
		// the default transport actually carries the request through.
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
			openrouter.WithReferer("https://app.test"),
			openrouter.WithTitle("App"),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if got := cap.get("Http-Referer"); got != "https://app.test" {
			t.Fatalf("HTTP-Referer = %q", got)
		}
		if got := cap.get("X-Title"); got != "App" {
			t.Fatalf("X-Title = %q", got)
		}
	})

	t.Run("timeout preserved from user http client", func(t *testing.T) {
		// We can't observe the wrapped client's Timeout directly, but
		// we can at least confirm provider construction does not strip
		// it. Use a near-zero deadline against a slow server to verify
		// the timeout actually fires.
		slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			time.Sleep(500 * time.Millisecond)
			w.WriteHeader(200)
		}))
		defer slow.Close()
		client := &http.Client{Timeout: 50 * time.Millisecond}
		err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(slow.URL),
			llmrouter.WithHTTPClient(client),
		)
		if err == nil {
			t.Fatal("expected timeout error")
		}
	})
}

func TestOptionOrdering(t *testing.T) {
	t.Run("user WithBaseURL overrides default", func(t *testing.T) {
		cap := &capturedRequest{}
		srv := fakeSSEServer(t, cap)
		defer srv.Close()
		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		// We hit srv.URL, not the OpenRouter default. The capturedRequest
		// being populated at all confirms the override worked.
		if cap.count() != 1 {
			t.Fatalf("hits = %d, want 1", cap.count())
		}
	})

	t.Run("user WithHTTPClient overrides our wrapped client", func(t *testing.T) {
		// When the user supplies their own HTTPClient after our
		// wrapped one, the wrapping is lost — confirm this is the
		// documented behaviour by observing that the attribution
		// headers are NOT injected through the caller-supplied client.
		var seenReferer string
		rt := roundTripFunc(func(req *http.Request) (*http.Response, error) {
			seenReferer = req.Header.Get("HTTP-Referer")
			body := "data: " + `{"id":"i","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"x"}}]}` + "\n\n" +
				"data: [DONE]\n\n"
			return &http.Response{
				StatusCode: 200,
				Body:       newStringReadCloser(body),
				Header:     make(http.Header),
			}, nil
		})
		// User-supplied client comes LAST, so it overrides our
		// wrapped one entirely.
		if err := newAndStream(t,
			llmrouter.WithAPIKey("k"),
			openrouter.WithReferer("https://lost.test"),
			llmrouter.WithHTTPClient(&http.Client{Transport: rt}),
		); err != nil {
			t.Fatalf("stream: %v", err)
		}
		if seenReferer != "" {
			t.Fatalf("expected referer to be dropped when user overrides HTTPClient, got %q", seenReferer)
		}
	})
}

func TestContextCancellation(t *testing.T) {
	t.Run("cancelled context aborts request", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			<-r.Context().Done()
		}))
		defer srv.Close()
		p, err := openrouter.New(
			llmrouter.WithAPIKey("k"),
			llmrouter.WithBaseURL(srv.URL),
		)
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		_, err = p.CompletionStream(ctx, llmrouter.ChatRequest{Model: "m"})
		if err == nil {
			t.Fatal("expected error from cancelled context")
		}
	})
}

// --- helpers -----------------------------------------------------------------

// roundTripFunc adapts a function value into an http.RoundTripper.
type roundTripFunc func(req *http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

// newStringReadCloser is a tiny helper that turns a string into an
// io.ReadCloser without pulling in extra deps.
func newStringReadCloser(s string) *stringReadCloser {
	return &stringReadCloser{r: strings.NewReader(s)}
}

type stringReadCloser struct {
	r *strings.Reader
}

func (s *stringReadCloser) Read(p []byte) (int, error) { return s.r.Read(p) }
func (s *stringReadCloser) Close() error               { return nil }
