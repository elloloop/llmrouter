package vertexanthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
)

// ---------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------

// fullSSE is a complete Vertex-on-Anthropic SSE response covering all
// event types: message_start, content_block_delta x2, message_delta
// (with stop_reason + usage) and message_stop.
const fullSSE = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"claude-3-5-sonnet-v2@20241022","usage":{"input_tokens":12,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}

event: message_stop
data: {"type":"message_stop"}

`

// staticToken is the access token used across happy-path tests.
const staticToken = "ya29.test-access-token"

// staticProject is the GCP project used across happy-path tests.
const staticProject = "tinykite-vertex"

// staticRegion is the Vertex region used across happy-path tests.
const staticRegion = "us-central1"

// staticModel is the Vertex-flavored Claude model id used in tests.
const staticModel = "claude-3-5-sonnet-v2@20241022"

// requireOpts returns the minimum required option set with a static
// access token, pointed at the supplied base URL.
func requireOpts(baseURL string) []llmrouter.Option {
	return []llmrouter.Option{
		llmrouter.WithBaseURL(baseURL),
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithAccessToken(staticToken),
	}
}

// echoSSEServer returns a server that writes the supplied SSE body and
// captures the inbound request via the given hook.
func echoSSEServer(t *testing.T, body string, inspect func(r *http.Request)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if inspect != nil {
			inspect(r)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, body)
	}))
}

// drain reads all chunks from a stream and returns them with any
// terminal error.
func drain(t *testing.T, s *llmrouter.Stream) ([]llmrouter.Chunk, error) {
	t.Helper()
	var chunks []llmrouter.Chunk
	for c := range s.Chunks() {
		chunks = append(chunks, c)
	}
	return chunks, s.Err()
}

// ---------------------------------------------------------------
// Construction validation
// ---------------------------------------------------------------

func TestNew_RejectsMissingProject(t *testing.T) {
	_, err := New(
		WithRegion(staticRegion),
		WithAccessToken(staticToken),
	)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Errorf("err not ErrInvalidConfig: %v", err)
	}
	if !strings.Contains(err.Error(), "project") {
		t.Errorf("err = %v, want 'project' in message", err)
	}
}

func TestNew_RejectsMissingRegion(t *testing.T) {
	_, err := New(
		WithProject(staticProject),
		WithAccessToken(staticToken),
	)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Errorf("err not ErrInvalidConfig: %v", err)
	}
	if !strings.Contains(err.Error(), "region") {
		t.Errorf("err = %v, want 'region' in message", err)
	}
}

func TestNew_RejectsMissingProjectAndRegion(t *testing.T) {
	_, err := New(WithAccessToken(staticToken))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Errorf("err not ErrInvalidConfig: %v", err)
	}
}

func TestNew_RejectsAPIKey(t *testing.T) {
	_, err := New(
		WithProject(staticProject),
		WithRegion(staticRegion),
		llmrouter.WithAPIKey("not-allowed"),
		WithAccessToken(staticToken),
	)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Errorf("err not ErrInvalidConfig: %v", err)
	}
	if !strings.Contains(err.Error(), "api key") {
		t.Errorf("err = %v, want 'api key' in message", err)
	}
}

func TestNew_RejectsMissingAuth(t *testing.T) {
	_, err := New(
		WithProject(staticProject),
		WithRegion(staticRegion),
	)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Errorf("err not ErrInvalidConfig: %v", err)
	}
	if !strings.Contains(err.Error(), "token") {
		t.Errorf("err = %v, want 'token' in message", err)
	}
}

func TestNew_RejectsBothAuth(t *testing.T) {
	src := func(ctx context.Context) (string, error) { return "t", nil }
	_, err := New(
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithAccessToken(staticToken),
		WithTokenSource(src),
	)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Errorf("err not ErrInvalidConfig: %v", err)
	}
	if !strings.Contains(err.Error(), "both") {
		t.Errorf("err = %v, want 'both'", err)
	}
}

func TestNew_SucceedsWithAccessToken(t *testing.T) {
	p, err := New(
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithAccessToken(staticToken),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p.Name() != "vertexanthropic" {
		t.Errorf("Name = %q", p.Name())
	}
	if p.accessToken != staticToken {
		t.Errorf("accessToken = %q", p.accessToken)
	}
	if p.tokenSource != nil {
		t.Errorf("tokenSource unexpectedly set")
	}
}

func TestNew_SucceedsWithTokenSource(t *testing.T) {
	src := func(ctx context.Context) (string, error) { return "tok", nil }
	p, err := New(
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithTokenSource(src),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if p.tokenSource == nil {
		t.Errorf("tokenSource not set")
	}
	if p.accessToken != "" {
		t.Errorf("accessToken unexpectedly set: %q", p.accessToken)
	}
}

func TestNew_RejectsEmptyProject(t *testing.T) {
	_, err := New(
		WithProject("   "),
		WithRegion(staticRegion),
		WithAccessToken(staticToken),
	)
	if err == nil {
		t.Fatal("expected error for blank project, got nil")
	}
}

func TestNew_RejectsEmptyRegion(t *testing.T) {
	_, err := New(
		WithProject(staticProject),
		WithRegion("   "),
		WithAccessToken(staticToken),
	)
	if err == nil {
		t.Fatal("expected error for blank region, got nil")
	}
}

func TestNew_RejectsNilTokenSource(t *testing.T) {
	_, err := New(
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithTokenSource(nil),
	)
	if err == nil {
		t.Fatal("expected error for nil token source, got nil")
	}
}

func TestNew_RejectsEmptyAccessToken(t *testing.T) {
	_, err := New(
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithAccessToken("   "),
	)
	if err == nil {
		t.Fatal("expected error for blank access token, got nil")
	}
}

// ---------------------------------------------------------------
// URL construction
// ---------------------------------------------------------------

func TestURL_DefaultRegionalHost(t *testing.T) {
	// No WithBaseURL provided — endpointURL must derive the regional
	// host. Use endpointURL directly because we don't have a server at
	// that derived host.
	p, err := New(
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithAccessToken(staticToken),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got := p.endpointURL(staticModel)
	want := "https://us-central1-aiplatform.googleapis.com/v1/projects/tinykite-vertex/locations/us-central1/publishers/anthropic/models/claude-3-5-sonnet-v2@20241022:streamRawPredict"
	if got != want {
		t.Errorf("endpointURL = %q\nwant %q", got, want)
	}
}

func TestURL_CustomBaseURLOverridesRegionalDefault(t *testing.T) {
	var seenURL string
	srv := echoSSEServer(t, fullSSE, func(r *http.Request) {
		seenURL = r.URL.String()
	})
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	_, _ = drain(t, stream)
	// httptest server URL is captured only as the path portion in r.URL.String().
	if !strings.Contains(seenURL, "/v1/projects/"+staticProject+"/locations/"+staticRegion+
		"/publishers/anthropic/models/"+staticModel+":streamRawPredict") {
		t.Errorf("seenURL = %q, want vertex raw-predict path", seenURL)
	}
}

func TestURL_AtVersionedModelFlowsVerbatim(t *testing.T) {
	var seenURL string
	srv := echoSSEServer(t, fullSSE, func(r *http.Request) {
		seenURL = r.URL.String()
	})
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    "claude-3-5-sonnet-v2@20241022",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	_, _ = drain(t, stream)
	// The @ may be URL-encoded by some routers but http.Request keeps
	// it verbatim in our flow.
	if !strings.Contains(seenURL, "claude-3-5-sonnet-v2") {
		t.Errorf("seenURL = %q, want model substring", seenURL)
	}
	if !strings.Contains(seenURL, "20241022:streamRawPredict") {
		t.Errorf("seenURL = %q, want @-version suffix preserved", seenURL)
	}
}

func TestURL_BaseURLTrailingSlashTrimmed(t *testing.T) {
	var seenURL string
	srv := echoSSEServer(t, fullSSE, func(r *http.Request) {
		seenURL = r.URL.String()
	})
	defer srv.Close()
	p, err := New(
		llmrouter.WithBaseURL(srv.URL+"/"),
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithAccessToken(staticToken),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	_, _ = drain(t, stream)
	if strings.Contains(seenURL, "//v1/") {
		t.Errorf("seenURL = %q has double slash", seenURL)
	}
}

func TestURL_ProjectAndRegionFlowInto(t *testing.T) {
	var seenURL string
	srv := echoSSEServer(t, fullSSE, func(r *http.Request) {
		seenURL = r.URL.String()
	})
	defer srv.Close()
	p, _ := New(
		llmrouter.WithBaseURL(srv.URL),
		WithProject("my-proj"),
		WithRegion("europe-west4"),
		WithAccessToken(staticToken),
	)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	_, _ = drain(t, stream)
	if !strings.Contains(seenURL, "/projects/my-proj/") {
		t.Errorf("seenURL = %q, want project in path", seenURL)
	}
	if !strings.Contains(seenURL, "/locations/europe-west4/") {
		t.Errorf("seenURL = %q, want region in path", seenURL)
	}
}

func TestURL_RawPredictSuffixPresent(t *testing.T) {
	var seenURL string
	srv := echoSSEServer(t, fullSSE, func(r *http.Request) {
		seenURL = r.URL.String()
	})
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	_, _ = drain(t, stream)
	if !strings.HasSuffix(seenURL, ":streamRawPredict") {
		t.Errorf("seenURL = %q, want :streamRawPredict suffix", seenURL)
	}
}

// ---------------------------------------------------------------
// Auth headers
// ---------------------------------------------------------------

func TestAuth_StaticAccessToken_HeadersCorrect(t *testing.T) {
	var captured http.Header
	srv := echoSSEServer(t, fullSSE, func(r *http.Request) {
		captured = r.Header.Clone()
	})
	defer srv.Close()

	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	_, _ = drain(t, stream)

	if got := captured.Get("Authorization"); got != "Bearer "+staticToken {
		t.Errorf("Authorization = %q, want Bearer %s", got, staticToken)
	}
	if got := captured.Get("api-key"); got != "" {
		t.Errorf("api-key header should be absent, got %q", got)
	}
	if got := captured.Get("x-api-key"); got != "" {
		t.Errorf("x-api-key header should be absent, got %q", got)
	}
	if got := captured.Get("anthropic-version"); got != "" {
		t.Errorf("anthropic-version header should be absent (Vertex uses body field), got %q", got)
	}
}

func TestAuth_TokenSource_HeadersCorrect(t *testing.T) {
	var captured http.Header
	srv := echoSSEServer(t, fullSSE, func(r *http.Request) {
		captured = r.Header.Clone()
	})
	defer srv.Close()

	src := func(ctx context.Context) (string, error) { return "tok-xyz", nil }
	p, _ := New(
		llmrouter.WithBaseURL(srv.URL),
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithTokenSource(src),
	)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	_, _ = drain(t, stream)

	if got := captured.Get("Authorization"); got != "Bearer tok-xyz" {
		t.Errorf("Authorization = %q, want Bearer tok-xyz", got)
	}
}

func TestAuth_TokenSourceCalledPerRequest(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()

	var calls atomic.Int32
	src := func(ctx context.Context) (string, error) {
		calls.Add(1)
		return "tok", nil
	}
	p, _ := New(
		llmrouter.WithBaseURL(srv.URL),
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithTokenSource(src),
	)
	for i := 0; i < 2; i++ {
		stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
			Model:    staticModel,
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("CompletionStream: %v", err)
		}
		_, _ = drain(t, stream)
	}

	if got := calls.Load(); got < 2 {
		t.Errorf("token source called %d times, want >= 2", got)
	}
}

func TestAuth_TokenSourceEmptyTokenError(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()

	src := func(ctx context.Context) (string, error) { return "  ", nil }
	p, _ := New(
		llmrouter.WithBaseURL(srv.URL),
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithTokenSource(src),
	)
	_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("expected error for empty token, got nil")
	}
	if !strings.Contains(err.Error(), "empty token") {
		t.Errorf("err = %v, want 'empty token'", err)
	}
}

func TestAuth_TokenSourceError(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()

	bad := errors.New("oauth boom")
	src := func(ctx context.Context) (string, error) { return "", bad }
	p, _ := New(
		llmrouter.WithBaseURL(srv.URL),
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithTokenSource(src),
	)
	_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, bad) {
		t.Errorf("err = %v, want wraps %v", err, bad)
	}
}

func TestAuth_StandardHeadersPresent(t *testing.T) {
	var captured http.Header
	srv := echoSSEServer(t, fullSSE, func(r *http.Request) {
		captured = r.Header.Clone()
	})
	defer srv.Close()

	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	_, _ = drain(t, stream)

	if got := captured.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q", got)
	}
	if got := captured.Get("Accept"); got != "text/event-stream" {
		t.Errorf("Accept = %q", got)
	}
}

// ---------------------------------------------------------------
// Body assembly
// ---------------------------------------------------------------

func TestBody_AnthropicVersionInjected(t *testing.T) {
	// The key Vertex-specific assertion: anthropic_version is in the
	// body with the Vertex sentinel value.
	body, err := buildBody(llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("buildBody: %v", err)
	}
	if !strings.Contains(string(body), `"anthropic_version":"vertex-2023-10-16"`) {
		t.Errorf("anthropic_version missing or wrong: %s", body)
	}
}

func TestBody_AnthropicVersionPresentOnWire(t *testing.T) {
	// End-to-end: server inspects the request body and asserts the
	// anthropic_version field is present.
	var seenBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, fullSSE)
	}))
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	_, _ = drain(t, stream)
	if !strings.Contains(string(seenBody), `"anthropic_version":"vertex-2023-10-16"`) {
		t.Errorf("wire body missing anthropic_version: %s", seenBody)
	}
}

func TestBody_LiftsSingleSystemMessage(t *testing.T) {
	body, err := buildBody(llmrouter.ChatRequest{
		Model: "claude",
		Messages: []llmrouter.Message{
			llmrouter.TextMessage("system", "be brief"),
			llmrouter.TextMessage("user", "hi"),
		},
	})
	if err != nil {
		t.Fatalf("buildBody: %v", err)
	}
	s := string(body)
	if !strings.Contains(s, `"system":"be brief"`) {
		t.Errorf("system not lifted: %s", s)
	}
	if strings.Contains(s, `"role":"system"`) {
		t.Errorf("system role leaked: %s", s)
	}
}

func TestBody_JoinsMultipleSystemMessages(t *testing.T) {
	body, _ := buildBody(llmrouter.ChatRequest{
		Model: "claude",
		Messages: []llmrouter.Message{
			llmrouter.TextMessage("system", "you are nice"),
			llmrouter.TextMessage("system", "be concise"),
			llmrouter.TextMessage("user", "hi"),
		},
	})
	if !strings.Contains(string(body), `"system":"you are nice\n\nbe concise"`) {
		t.Errorf("system not joined: %s", body)
	}
}

func TestBody_DefaultMaxTokens(t *testing.T) {
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if !strings.Contains(string(body), `"max_tokens":4096`) {
		t.Errorf("default max_tokens missing: %s", body)
	}
}

func TestBody_MaxTokensOverride(t *testing.T) {
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:     "claude",
		MaxTokens: 512,
		Messages:  []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if !strings.Contains(string(body), `"max_tokens":512`) {
		t.Errorf("max_tokens not overridden: %s", body)
	}
}

func TestBody_StreamTrueAlways(t *testing.T) {
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if !strings.Contains(string(body), `"stream":true`) {
		t.Errorf("stream flag missing: %s", body)
	}
}

func TestBody_TemperaturePassedThrough(t *testing.T) {
	temp := 0.7
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:       "claude",
		Temperature: &temp,
		Messages:    []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if !strings.Contains(string(body), `"temperature":0.7`) {
		t.Errorf("temperature missing: %s", body)
	}
}

func TestBody_TopPPassedThrough(t *testing.T) {
	tp := 0.9
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:    "claude",
		TopP:     &tp,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if !strings.Contains(string(body), `"top_p":0.9`) {
		t.Errorf("top_p missing: %s", body)
	}
}

func TestBody_StopBecomesStopSequences(t *testing.T) {
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:    "claude",
		Stop:     []string{"END"},
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	s := string(body)
	if !strings.Contains(s, `"stop_sequences":["END"]`) {
		t.Errorf("stop_sequences missing: %s", s)
	}
}

func TestBody_RawPassthroughOverlaysModel(t *testing.T) {
	raw := json.RawMessage(`{"model":"raw-claude","temperature":0.25}`)
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:    "ignored-claude",
		Raw:      raw,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	s := string(body)
	if !strings.Contains(s, `"model":"raw-claude"`) {
		t.Errorf("raw model not applied: %s", s)
	}
	if !strings.Contains(s, `"temperature":0.25`) {
		t.Errorf("raw temperature not applied: %s", s)
	}
}

func TestBody_ToolsEmitFlatAnthropicShape(t *testing.T) {
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		Tools: []llmrouter.Tool{{
			Type: "function",
			Function: llmrouter.ToolFunction{
				Name:        "get_weather",
				Description: "look up weather",
				Parameters:  json.RawMessage(`{"type":"object"}`),
			},
		}},
	})
	s := string(body)
	if !strings.Contains(s, `"name":"get_weather"`) {
		t.Errorf("tool name missing: %s", s)
	}
	if !strings.Contains(s, `"input_schema":{"type":"object"}`) {
		t.Errorf("input_schema missing/wrong: %s", s)
	}
	if strings.Contains(s, `"function":`) {
		t.Errorf("tool wrapped in function field (openai shape leaked): %s", s)
	}
}

func TestBody_ToolMessageBecomesToolResult(t *testing.T) {
	body, _ := buildBody(llmrouter.ChatRequest{
		Model: "claude",
		Messages: []llmrouter.Message{
			llmrouter.ToolResultMessage("toolu_1", "{\"ok\":true}"),
		},
	})
	s := string(body)
	if !strings.Contains(s, `"tool_use_id":"toolu_1"`) {
		t.Errorf("tool_use_id missing: %s", s)
	}
	if !strings.Contains(s, `"type":"tool_result"`) {
		t.Errorf("tool_result missing: %s", s)
	}
}

func TestBody_NoSystemNoSystemField(t *testing.T) {
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if strings.Contains(string(body), `"system":`) {
		t.Errorf("system field should not appear when no system messages: %s", body)
	}
}

func TestBody_NoToolsNoToolsField(t *testing.T) {
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if strings.Contains(string(body), `"tools":`) {
		t.Errorf("tools field should not appear when no tools: %s", body)
	}
}

func TestBody_RawStopBecomesStopSequences(t *testing.T) {
	raw := json.RawMessage(`{"stop":["END"]}`)
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:    "claude",
		Raw:      raw,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if !strings.Contains(string(body), `"stop_sequences":["END"]`) {
		t.Errorf("raw stop not translated: %s", body)
	}
}

func TestBody_RawTopK(t *testing.T) {
	raw := json.RawMessage(`{"top_k":40}`)
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:    "claude",
		Raw:      raw,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if !strings.Contains(string(body), `"top_k":40`) {
		t.Errorf("top_k missing: %s", body)
	}
}

func TestBody_ResponseSchemaInjectsTool(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`)
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:    "claude",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		ResponseSchema: &llmrouter.ResponseSchema{
			Name:   "extract",
			Schema: schema,
		},
	})
	s := string(body)
	if !strings.Contains(s, `"name":"extract"`) {
		t.Errorf("schema tool name missing: %s", s)
	}
	if !strings.Contains(s, `"tool_choice":{"name":"extract","type":"tool"}`) &&
		!strings.Contains(s, `"tool_choice":{"type":"tool","name":"extract"}`) {
		t.Errorf("forced tool_choice missing: %s", s)
	}
}

func TestBody_TextMultipartContentTranslated(t *testing.T) {
	raw := json.RawMessage(`[{"type":"text","text":"hello"}]`)
	body, _ := buildBody(llmrouter.ChatRequest{
		Model: "claude",
		Messages: []llmrouter.Message{{
			Role:    "user",
			Content: raw,
		}},
	})
	s := string(body)
	if !strings.Contains(s, `"type":"text"`) || !strings.Contains(s, `"text":"hello"`) {
		t.Errorf("multipart text not translated: %s", s)
	}
}

func TestBody_ImageURLDataURLTranslatedToBase64(t *testing.T) {
	raw := json.RawMessage(`[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]`)
	body, _ := buildBody(llmrouter.ChatRequest{
		Model: "claude",
		Messages: []llmrouter.Message{{
			Role:    "user",
			Content: raw,
		}},
	})
	s := string(body)
	if !strings.Contains(s, `"media_type":"image/png"`) {
		t.Errorf("media_type missing: %s", s)
	}
	if !strings.Contains(s, `"data":"AAAA"`) {
		t.Errorf("data missing: %s", s)
	}
}

func TestBody_ImageURLHTTPTranslatedToURLSource(t *testing.T) {
	raw := json.RawMessage(`[{"type":"image_url","image_url":{"url":"https://example.com/x.png"}}]`)
	body, _ := buildBody(llmrouter.ChatRequest{
		Model: "claude",
		Messages: []llmrouter.Message{{
			Role:    "user",
			Content: raw,
		}},
	})
	s := string(body)
	if !strings.Contains(s, `"type":"url"`) {
		t.Errorf("url source missing: %s", s)
	}
	if !strings.Contains(s, `"url":"https://example.com/x.png"`) {
		t.Errorf("url value missing: %s", s)
	}
}

// ---------------------------------------------------------------
// Tool choice modes
// ---------------------------------------------------------------

func TestBody_ToolChoiceAuto(t *testing.T) {
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:      "claude",
		Messages:   []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		ToolChoice: &llmrouter.ToolChoice{Mode: "auto"},
	})
	if !strings.Contains(string(body), `"tool_choice":{"type":"auto"}`) {
		t.Errorf("tool_choice auto missing: %s", body)
	}
}

func TestBody_ToolChoiceNone(t *testing.T) {
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:      "claude",
		Messages:   []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		ToolChoice: &llmrouter.ToolChoice{Mode: "none"},
	})
	if !strings.Contains(string(body), `"tool_choice":{"type":"none"}`) {
		t.Errorf("tool_choice none missing: %s", body)
	}
}

func TestBody_ToolChoiceRequiredMappedToAny(t *testing.T) {
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:      "claude",
		Messages:   []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		ToolChoice: &llmrouter.ToolChoice{Mode: "required"},
	})
	if !strings.Contains(string(body), `"tool_choice":{"type":"any"}`) {
		t.Errorf("tool_choice required must become 'any': %s", body)
	}
}

func TestBody_ToolChoiceSpecific(t *testing.T) {
	body, _ := buildBody(llmrouter.ChatRequest{
		Model:      "claude",
		Messages:   []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		ToolChoice: &llmrouter.ToolChoice{Mode: "specific", Function: "get_weather"},
	})
	s := string(body)
	if !strings.Contains(s, `"type":"tool"`) || !strings.Contains(s, `"name":"get_weather"`) {
		t.Errorf("tool_choice specific malformed: %s", s)
	}
}

// ---------------------------------------------------------------
// End-to-end happy path
// ---------------------------------------------------------------

func TestE2E_DeliversFourChunks(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)

	stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	chunks, err := drain(t, stream)
	if err != nil {
		t.Fatalf("stream.Err: %v", err)
	}
	if len(chunks) != 4 {
		t.Fatalf("got %d chunks, want 4", len(chunks))
	}
}

func TestE2E_ConcatContentMatches(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)

	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	chunks, _ := drain(t, stream)
	var out string
	for _, c := range chunks {
		out += c.Choices[0].Delta.Content
	}
	if out != "Hello world" {
		t.Errorf("concat = %q, want 'Hello world'", out)
	}
}

func TestE2E_RolePrimerOnFirstChunk(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	chunks, _ := drain(t, stream)
	if chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Errorf("first chunk role = %q", chunks[0].Choices[0].Delta.Role)
	}
}

func TestE2E_FinishReasonMappedStop(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	chunks, _ := drain(t, stream)
	final := chunks[len(chunks)-1]
	if final.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", final.Choices[0].FinishReason)
	}
}

func TestE2E_UsagePopulated(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	chunks, _ := drain(t, stream)
	final := chunks[len(chunks)-1]
	if final.Usage == nil {
		t.Fatal("Usage nil")
	}
	if final.Usage.PromptTokens != 12 {
		t.Errorf("prompt = %d", final.Usage.PromptTokens)
	}
	if final.Usage.CompletionTokens != 7 {
		t.Errorf("completion = %d", final.Usage.CompletionTokens)
	}
	if final.Usage.TotalTokens != 19 {
		t.Errorf("total = %d", final.Usage.TotalTokens)
	}
}

func TestE2E_StableChunkIDs(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	chunks, _ := drain(t, stream)
	id := chunks[0].ID
	if !strings.HasPrefix(id, "chatcmpl-") {
		t.Errorf("ID prefix wrong: %q", id)
	}
	for i, c := range chunks {
		if c.ID != id {
			t.Errorf("chunk[%d].ID = %q, want %q", i, c.ID, id)
		}
	}
}

func TestE2E_ChunkObjectFieldSet(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	chunks, _ := drain(t, stream)
	for i, c := range chunks {
		if c.Object != "chat.completion.chunk" {
			t.Errorf("chunk[%d].Object = %q", i, c.Object)
		}
	}
}

func TestE2E_RawPopulated(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	chunks, _ := drain(t, stream)
	for i, c := range chunks {
		if len(c.Raw) == 0 {
			t.Errorf("chunk[%d].Raw empty", i)
		}
	}
}

func TestE2E_ModelEchoedFromMessageStart(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	chunks, _ := drain(t, stream)
	// message_start sets the model to upstream-reported value.
	for i, c := range chunks {
		if c.Model != "claude-3-5-sonnet-v2@20241022" {
			t.Errorf("chunk[%d].Model = %q", i, c.Model)
		}
	}
}

func TestE2E_FinishReasonOnLastOnly(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	chunks, _ := drain(t, stream)
	for i, c := range chunks[:len(chunks)-1] {
		if c.Choices[0].FinishReason != "" {
			t.Errorf("chunk[%d].FinishReason = %q, want empty", i, c.Choices[0].FinishReason)
		}
	}
}

func TestE2E_NoTerminalError(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	_, err := drain(t, stream)
	if err != nil {
		t.Errorf("stream.Err = %v, want nil", err)
	}
}

// ---------------------------------------------------------------
// Tool use + thinking + cache + mid-stream error
// ---------------------------------------------------------------

const toolUseSSE = `event: message_start
data: {"type":"message_start","message":{"id":"msg_t","model":"claude","usage":{"input_tokens":5}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_x","name":"get_weather"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"SF\"}"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}

event: message_stop
data: {"type":"message_stop"}

`

func TestE2E_ToolCallDeltasPopulated(t *testing.T) {
	srv := echoSSEServer(t, toolUseSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	chunks, err := drain(t, stream)
	if err != nil {
		t.Fatalf("stream.Err: %v", err)
	}
	var sawName, sawArgsCity, sawArgsValue bool
	for _, c := range chunks {
		for _, tc := range c.Choices[0].Delta.ToolCalls {
			if tc.Function != nil {
				if tc.Function.Name == "get_weather" {
					sawName = true
				}
				if strings.Contains(tc.Function.Arguments, `"city":`) {
					sawArgsCity = true
				}
				if strings.Contains(tc.Function.Arguments, `"SF"`) {
					sawArgsValue = true
				}
			}
		}
	}
	if !sawName {
		t.Errorf("tool name fragment not delivered")
	}
	if !sawArgsCity || !sawArgsValue {
		t.Errorf("tool argument fragments not delivered: city=%v val=%v", sawArgsCity, sawArgsValue)
	}
	final := chunks[len(chunks)-1]
	if final.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("final finish = %q, want tool_calls", final.Choices[0].FinishReason)
	}
}

const thinkingSSE = `event: message_start
data: {"type":"message_start","message":{"id":"msg_th","model":"claude","usage":{"input_tokens":3}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","text":"let me think..."}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}

`

func TestE2E_ThinkingDeltaPopulated(t *testing.T) {
	srv := echoSSEServer(t, thinkingSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	chunks, err := drain(t, stream)
	if err != nil {
		t.Fatalf("stream.Err: %v", err)
	}
	var thinking string
	for _, c := range chunks {
		thinking += c.Choices[0].Delta.Thinking
	}
	if thinking != "let me think..." {
		t.Errorf("thinking = %q", thinking)
	}
}

const cacheSSE = `event: message_start
data: {"type":"message_start","message":{"id":"msg_c","model":"claude","usage":{"input_tokens":100,"output_tokens":0,"cache_read_input_tokens":80,"cache_creation_input_tokens":20}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`

func TestE2E_CacheTokensCaptured(t *testing.T) {
	srv := echoSSEServer(t, cacheSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	chunks, _ := drain(t, stream)
	final := chunks[len(chunks)-1]
	if final.Usage == nil {
		t.Fatal("usage nil")
	}
	if final.Usage.CachedPromptTokens != 80 {
		t.Errorf("CachedPromptTokens = %d", final.Usage.CachedPromptTokens)
	}
	if final.Usage.CacheCreationTokens != 20 {
		t.Errorf("CacheCreationTokens = %d", final.Usage.CacheCreationTokens)
	}
}

const midStreamErrorSSE = `event: message_start
data: {"type":"message_start","message":{"id":"m","model":"claude","usage":{"input_tokens":1}}}

event: error
data: {"type":"error","error":{"type":"overloaded_error","message":"server is overloaded"}}

`

func TestE2E_MidStreamErrorEmitsErrUpstream(t *testing.T) {
	srv := echoSSEServer(t, midStreamErrorSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	_, err := drain(t, stream)
	if err == nil {
		t.Fatal("expected mid-stream error")
	}
	var ue *llmrouter.ErrUpstream
	if !errors.As(err, &ue) {
		t.Fatalf("err = %v, want *llmrouter.ErrUpstream", err)
	}
	if ue.Provider != "vertexanthropic" {
		t.Errorf("Provider = %q", ue.Provider)
	}
	if ue.StatusCode != 0 {
		t.Errorf("StatusCode = %d, want 0 (mid-stream)", ue.StatusCode)
	}
	if !strings.Contains(ue.Body, "overloaded") {
		t.Errorf("Body = %q", ue.Body)
	}
}

// ---------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------

func errorServer(t *testing.T, status int, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
}

func runErrorTest(t *testing.T, status int, body string) {
	t.Helper()
	srv := errorServer(t, status, body)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatalf("expected error for status %d", status)
	}
	var ue *llmrouter.ErrUpstream
	if !errors.As(err, &ue) {
		t.Fatalf("err = %v, want *llmrouter.ErrUpstream", err)
	}
	if ue.Provider != "vertexanthropic" {
		t.Errorf("Provider = %q", ue.Provider)
	}
	if ue.StatusCode != status {
		t.Errorf("StatusCode = %d, want %d", ue.StatusCode, status)
	}
	if !strings.Contains(ue.Body, body) {
		t.Errorf("Body = %q, want contains %q", ue.Body, body)
	}
}

func TestError_401(t *testing.T) { runErrorTest(t, 401, `{"error":"unauthorized"}`) }
func TestError_403(t *testing.T) { runErrorTest(t, 403, `{"error":"forbidden"}`) }
func TestError_404(t *testing.T) { runErrorTest(t, 404, `{"error":"model not found"}`) }
func TestError_429(t *testing.T) { runErrorTest(t, 429, `{"error":"rate limited"}`) }
func TestError_500(t *testing.T) { runErrorTest(t, 500, `{"error":"internal"}`) }
func TestError_502(t *testing.T) { runErrorTest(t, 502, `{"error":"bad gateway"}`) }
func TestError_503(t *testing.T) { runErrorTest(t, 503, `{"error":"unavailable"}`) }
func TestError_504(t *testing.T) { runErrorTest(t, 504, `{"error":"timeout"}`) }

func TestError_BodyCappedAt8KiB(t *testing.T) {
	big := strings.Repeat("X", 16*1024) // 16 KiB
	srv := errorServer(t, 500, big)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	var ue *llmrouter.ErrUpstream
	if !errors.As(err, &ue) {
		t.Fatalf("err = %v, want ErrUpstream", err)
	}
	if len(ue.Body) > 8*1024 {
		t.Errorf("Body len = %d, want <= %d", len(ue.Body), 8*1024)
	}
	if len(ue.Body) != 8*1024 {
		t.Errorf("Body len = %d, want exactly 8KiB", len(ue.Body))
	}
}

func TestError_NetworkErrorNotWrappedAsErrUpstream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := srv.URL
	srv.Close()

	p, _ := New(
		llmrouter.WithBaseURL(deadURL),
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithAccessToken(staticToken),
		llmrouter.WithTimeout(500*time.Millisecond),
	)
	_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err == nil {
		t.Fatal("expected network error")
	}
	var ue *llmrouter.ErrUpstream
	if errors.As(err, &ue) {
		t.Errorf("network err should not be ErrUpstream, got %v", err)
	}
	if !strings.Contains(err.Error(), "vertexanthropic") {
		t.Errorf("err = %v, want 'vertexanthropic' prefix", err)
	}
}

// ---------------------------------------------------------------
// mapStopReason
// ---------------------------------------------------------------

func TestMapStopReason(t *testing.T) {
	cases := map[string]string{
		"end_turn":      "stop",
		"stop_sequence": "stop",
		"max_tokens":    "length",
		"tool_use":      "tool_calls",
		"unknown":       "stop",
	}
	for in, want := range cases {
		if got := mapStopReason(in); got != want {
			t.Errorf("mapStopReason(%q) = %q, want %q", in, got, want)
		}
	}
}

// ---------------------------------------------------------------
// Context cancellation
// ---------------------------------------------------------------

func TestE2E_ContextCancelStopsStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		_, _ = io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"c\",\"usage\":{\"input_tokens\":1}}}\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		<-r.Context().Done()
	}))
	defer srv.Close()

	p, _ := New(requireOpts(srv.URL)...)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}

	select {
	case _, ok := <-stream.Chunks():
		if !ok {
			t.Fatal("stream closed before primer")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("primer not received within 2s")
	}
	cancel()
	deadline := time.After(3 * time.Second)
	for {
		select {
		case _, ok := <-stream.Chunks():
			if !ok {
				return
			}
		case <-deadline:
			t.Fatal("stream did not close after context cancel")
		}
	}
}

// ---------------------------------------------------------------
// Provider sanity
// ---------------------------------------------------------------

func TestProvider_Name(t *testing.T) {
	p, _ := New(
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithAccessToken(staticToken),
	)
	if p.Name() != "vertexanthropic" {
		t.Errorf("Name = %q", p.Name())
	}
}

func TestNew_OptionPropagationError(t *testing.T) {
	bad := errors.New("bad option")
	_, err := New(
		WithProject(staticProject),
		WithRegion(staticRegion),
		WithAccessToken(staticToken),
		func(*llmrouter.Config) error { return bad },
	)
	if err == nil {
		t.Fatal("expected option error to propagate")
	}
	if !errors.Is(err, bad) {
		t.Errorf("err = %v, want wraps %v", err, bad)
	}
}

func TestBody_EmptyMessages(t *testing.T) {
	body, err := buildBody(llmrouter.ChatRequest{Model: "claude"})
	if err != nil {
		t.Fatalf("buildBody: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("not valid JSON: %v", err)
	}
	if parsed["model"] != "claude" {
		t.Errorf("model = %v", parsed["model"])
	}
	if parsed["anthropic_version"] != vertexAnthropicVer {
		t.Errorf("anthropic_version = %v, want %s", parsed["anthropic_version"], vertexAnthropicVer)
	}
}

func TestE2E_ChunkIDFormat(t *testing.T) {
	srv := echoSSEServer(t, fullSSE, nil)
	defer srv.Close()
	p, _ := New(requireOpts(srv.URL)...)
	stream, _ := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
		Model:    staticModel,
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	chunks, _ := drain(t, stream)
	id := chunks[0].ID
	if !strings.HasPrefix(id, "chatcmpl-") {
		t.Errorf("id = %q, want prefix chatcmpl-", id)
	}
	if len(id) <= len("chatcmpl-") {
		t.Errorf("id = %q, want suffix after prefix", id)
	}
}

func TestProviderName_Constant(t *testing.T) {
	if providerName != "vertexanthropic" {
		t.Errorf("providerName = %q", providerName)
	}
	if fmt.Sprintf("%s", providerName) != "vertexanthropic" {
		t.Errorf("sprintf failed")
	}
}

func TestVertexAnthropicVersion_Constant(t *testing.T) {
	if vertexAnthropicVer != "vertex-2023-10-16" {
		t.Errorf("vertexAnthropicVer = %q, want vertex-2023-10-16", vertexAnthropicVer)
	}
}
