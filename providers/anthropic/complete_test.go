package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
)

const nativeFixture = `{"id":"message-fixture","type":"message","role":"assistant","model":"returned-snapshot","stop_reason":"end_turn","stop_sequence":null,"content":[{"type":"thinking","thinking":"opaque-private","signature":"signed-fixture"},{"type":"redacted_thinking","data":"redacted-fixture"},{"type":"text","text":"answer","citations":[{"type":"future-citation","evidence":"source"}],"future_metadata":{"keep":true}}],"usage":{"input_tokens":2,"output_tokens":3,"future_count":4},"future_envelope":{"keep":true}}`

func nativeRequest() llmrouter.ChatRequest {
	return llmrouter.ChatRequest{Model: "requested-model", MaxTokens: 64, Messages: []llmrouter.Message{llmrouter.TextMessage("user", "fixture")}, ToolChoice: &llmrouter.ToolChoice{Mode: "none"}}
}
func nativeLimits() CompletionLimits {
	return CompletionLimits{MaxRequestBytes: 64 << 10, MaxResponseBytes: 64 << 10}
}
func nativeProvider(t *testing.T, srv *httptest.Server) *Provider {
	t.Helper()
	p, err := New(llmrouter.WithAPIKey("fixture-secret"), llmrouter.WithBaseURL(srv.URL+"/v1"), llmrouter.WithHTTPClient(srv.Client()))
	if err != nil {
		t.Fatal(err)
	}
	return p
}
func nativeJSON(w http.ResponseWriter, body string) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(w, body)
}
func sameJSON(t *testing.T, a, b []byte) {
	t.Helper()
	var left, right any
	if json.Unmarshal(a, &left) != nil || json.Unmarshal(b, &right) != nil || !reflect.DeepEqual(left, right) {
		t.Fatalf("native JSON changed:\n%s\n%s", a, b)
	}
}

type nativeRoundTripper func(*http.Request) (*http.Response, error)

func (f nativeRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestCompletePreservesNativeEnvelopeAndAssistantReplay(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := calls.Add(1)
		if r.Method != "POST" || r.URL.Path != "/v1/messages" || r.Header.Get("X-Api-Key") != "fixture-secret" || r.Header.Get("Anthropic-Version") != anthropicVersion || r.Header.Get("Accept") != "application/json" {
			t.Errorf("wrong native request: %s %s", r.Method, r.URL)
		}
		var request struct {
			Model      string              `json:"model"`
			MaxTokens  int                 `json:"max_tokens"`
			Stream     bool                `json:"stream"`
			Messages   []llmrouter.Message `json:"messages"`
			ToolChoice struct {
				Type string `json:"type"`
			} `json:"tool_choice"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		if request.Stream || request.Model != "requested-model" || request.MaxTokens != 64 || request.ToolChoice.Type != "none" {
			t.Errorf("request changed: %#v", request)
		}
		if count == 2 {
			if len(request.Messages) != 3 {
				t.Errorf("history length %d", len(request.Messages))
			} else {
				var original MessageResponse
				_ = json.Unmarshal([]byte(nativeFixture), &original)
				expected, _ := json.Marshal(original.Content)
				sameJSON(t, request.Messages[1].Content, expected)
			}
		}
		nativeJSON(w, nativeFixture)
	}))
	defer srv.Close()
	p := nativeProvider(t, srv)
	response, err := p.Complete(context.Background(), nativeRequest(), nativeLimits())
	if err != nil {
		t.Fatal(err)
	}
	if response.Model != "returned-snapshot" || response.StopReason != "end_turn" || string(response.Raw) != nativeFixture || len(response.Content) != 3 || response.StopSequence != nil {
		t.Fatalf("native envelope changed: %#v", response)
	}
	if !strings.Contains(string(response.Usage), "future_count") {
		t.Fatal("usage metadata lost")
	}
	request := nativeRequest()
	content, _ := json.Marshal(response.Content)
	request.Messages = append(request.Messages, llmrouter.Message{Role: "assistant", Content: content}, llmrouter.TextMessage("user", "continue"))
	if _, err = p.Complete(context.Background(), request, nativeLimits()); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Fatalf("unexpected retry count %d", calls.Load())
	}
}

func TestCompletePreservesUnknownStopAndContent(t *testing.T) {
	for _, reason := range []string{"refusal", "pause_turn", "model_context_window_exceeded", "future_stop"} {
		t.Run(reason, func(t *testing.T) {
			body := strings.Replace(nativeFixture, `"stop_reason":"end_turn"`, `"stop_reason":"`+reason+`"`, 1)
			body = strings.Replace(body, `"type":"text"`, `"type":"server_tool_use"`, 1)
			srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { nativeJSON(w, body) }))
			defer srv.Close()
			response, err := nativeProvider(t, srv).Complete(context.Background(), nativeRequest(), nativeLimits())
			if err != nil || response.StopReason != reason || !strings.Contains(string(response.Content[2]), "server_tool_use") {
				t.Fatalf("lost caller policy evidence: %#v %v", response, err)
			}
		})
	}
}

func TestCompleteRequiresValidEnvelopeAndOneJSONValue(t *testing.T) {
	tests := map[string]string{
		"truncated": nativeFixture[:len(nativeFixture)-1], "trailing": nativeFixture + `{}`, "malformed": `{"bad":`, "null": `null`,
		"missing model":      strings.Replace(nativeFixture, `"model":"returned-snapshot",`, "", 1),
		"missing stop":       strings.Replace(nativeFixture, `"stop_reason":"end_turn",`, "", 1),
		"null stop":          strings.Replace(nativeFixture, `"stop_reason":"end_turn"`, `"stop_reason":null`, 1),
		"blank id":           strings.Replace(nativeFixture, `"id":"message-fixture"`, `"id":" "`, 1),
		"wrong type":         strings.Replace(nativeFixture, `"type":"message"`, `"type":"error"`, 1),
		"wrong role":         strings.Replace(nativeFixture, `"role":"assistant"`, `"role":"user"`, 1),
		"null content":       `{"id":"id","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","content":null}`,
		"bad block":          `{"id":"id","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","content":[null]}`,
		"scalar block":       `{"id":"id","type":"message","role":"assistant","model":"m","stop_reason":"end_turn","content":[5]}`,
		"missing block type": strings.Replace(nativeFixture, `"type":"thinking",`, "", 1),
		"invalid utf8":       strings.Replace(nativeFixture, "answer", string([]byte{0xff}), 1),
	}
	for name, body := range tests {
		t.Run(name, func(t *testing.T) {
			srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { nativeJSON(w, body) }))
			defer srv.Close()
			response, err := nativeProvider(t, srv).Complete(context.Background(), nativeRequest(), nativeLimits())
			if response != nil || !errors.Is(err, ErrInvalidResponse) {
				t.Fatalf("accepted malformed envelope: %#v %v", response, err)
			}
		})
	}
}

func TestCompleteBoundsSerializedRequestAndResponse(t *testing.T) {
	var count atomic.Int32
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { count.Add(1); nativeJSON(w, nativeFixture) }))
	defer srv.Close()
	p := nativeProvider(t, srv)
	request := nativeRequest()
	body, err := buildMessagesBody(request, false)
	if err != nil {
		t.Fatal(err)
	}
	limits := CompletionLimits{MaxRequestBytes: int64(len(body)), MaxResponseBytes: int64(len(nativeFixture))}
	if _, err = p.Complete(context.Background(), request, limits); err != nil {
		t.Fatalf("exact bounds rejected: %v", err)
	}
	limits.MaxRequestBytes--
	if _, err = p.Complete(context.Background(), request, limits); !errors.Is(err, ErrRequestTooLarge) {
		t.Fatalf("request bound missed: %v", err)
	}
	if count.Load() != 1 {
		t.Fatal("oversized request dispatched")
	}
	limits.MaxRequestBytes++
	limits.MaxResponseBytes--
	if _, err = p.Complete(context.Background(), request, limits); !errors.Is(err, ErrResponseTooLarge) {
		t.Fatalf("response bound missed: %v", err)
	}
	if count.Load() != 2 {
		t.Fatal("size failure retried")
	}
}

func TestCompleteRejectsInvalidArgumentsBeforeDispatch(t *testing.T) {
	var count atomic.Int32
	p, err := New(llmrouter.WithAPIKey("fixture"), llmrouter.WithHTTPClient(&http.Client{Transport: nativeRoundTripper(func(*http.Request) (*http.Response, error) {
		count.Add(1)
		return nil, errors.New("should not dispatch")
	})}))
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		mutate func(*llmrouter.ChatRequest, *CompletionLimits)
	}{
		{"missing model", func(r *llmrouter.ChatRequest, l *CompletionLimits) { r.Model = " " }},
		{"missing token budget", func(r *llmrouter.ChatRequest, l *CompletionLimits) { r.MaxTokens = 0 }},
		{"stream true", func(r *llmrouter.ChatRequest, l *CompletionLimits) { r.Stream = true }},
		{"missing request limit", func(r *llmrouter.ChatRequest, l *CompletionLimits) { l.MaxRequestBytes = 0 }},
		{"missing response limit", func(r *llmrouter.ChatRequest, l *CompletionLimits) { l.MaxResponseBytes = 0 }},
		{"overflow response limit", func(r *llmrouter.ChatRequest, l *CompletionLimits) { l.MaxResponseBytes = math.MaxInt64 }},
		{"invalid content", func(r *llmrouter.ChatRequest, l *CompletionLimits) {
			r.Messages = []llmrouter.Message{{Role: "assistant", Content: json.RawMessage(`{invalid`)}}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			request, limits := nativeRequest(), nativeLimits()
			test.mutate(&request, &limits)
			if _, err := p.Complete(context.Background(), request, limits); !errors.Is(err, llmrouter.ErrInvalidConfig) {
				t.Fatal(err)
			}
		})
	}
	//lint:ignore SA1012 Deliberately test the public boundary's nil-context rejection.
	if _, err = p.Complete(nil, nativeRequest(), nativeLimits()); !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = p.Complete(ctx, nativeRequest(), nativeLimits()); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	p.cfg.BaseURL = ":invalid-fixture-secret"
	if _, err = p.Complete(context.Background(), nativeRequest(), nativeLimits()); !errors.Is(err, llmrouter.ErrInvalidConfig) || strings.Contains(err.Error(), "fixture-secret") {
		t.Fatal(err)
	}
	if count.Load() != 0 {
		t.Fatal("invalid request dispatched")
	}
}

func TestCompleteDeniesRedirectsCookiesAndDiagnosticLeaks(t *testing.T) {
	var destinations atomic.Int32
	destination := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { destinations.Add(1) }))
	defer destination.Close()
	for _, status := range []int{301, 302, 303, 307, 308, 401, 429, 500} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var calls atomic.Int32
			srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if r.Header.Get("Cookie") != "" {
					t.Error("ambient cookie sent")
				}
				w.Header().Set("Location", destination.URL+"/fixture-secret")
				w.WriteHeader(status)
				_, _ = io.WriteString(w, "fixture-secret diagnostic")
			}))
			defer srv.Close()
			client := srv.Client()
			client.Jar, _ = cookiejar.New(nil)
			u, _ := url.Parse(srv.URL)
			client.Jar.SetCookies(u, []*http.Cookie{{Name: "ambient", Value: "fixture-cookie"}})
			client.CheckRedirect = func(*http.Request, []*http.Request) error { t.Error("permissive redirect policy used"); return nil }
			p, err := New(llmrouter.WithAPIKey("fixture-secret"), llmrouter.WithBaseURL(srv.URL), llmrouter.WithHTTPClient(client))
			if err != nil {
				t.Fatal(err)
			}
			_, err = p.Complete(context.Background(), nativeRequest(), nativeLimits())
			var statusError *HTTPStatusError
			if !errors.As(err, &statusError) || statusError.StatusCode != status || strings.Contains(err.Error(), "fixture-secret") {
				t.Fatalf("unsafe HTTP error %v", err)
			}
			if calls.Load() != 1 {
				t.Fatal("HTTP failure retried")
			}
			if client.Jar == nil || len(client.Jar.Cookies(u)) != 1 {
				t.Fatal("caller client mutated")
			}
		})
	}
	if destinations.Load() != 0 {
		t.Fatal("credential redirected")
	}
}

func TestCompleteTransportAndContentTypeErrorsAreSafe(t *testing.T) {
	for _, kind := range []string{"transport", "body-read", "wrong-content-type", "malformed-content-type"} {
		t.Run(kind, func(t *testing.T) {
			var count atomic.Int32
			client := &http.Client{Transport: nativeRoundTripper(func(r *http.Request) (*http.Response, error) {
				count.Add(1)
				if r.GetBody != nil {
					t.Error("request permits replay")
				}
				if kind == "transport" {
					return nil, errors.New("fixture-secret URL diagnostic")
				}
				response := &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}}, Body: io.NopCloser(strings.NewReader(nativeFixture)), Request: r}
				switch kind {
				case "body-read":
					response.Body = nativeErrorReader{}
				case "wrong-content-type":
					response.Header.Set("Content-Type", "text/event-stream")
				case "malformed-content-type":
					response.Header.Set("Content-Type", "invalid/")
				}
				return response, nil
			})}
			p, err := New(llmrouter.WithAPIKey("fixture"), llmrouter.WithHTTPClient(client))
			if err != nil {
				t.Fatal(err)
			}
			_, err = p.Complete(context.Background(), nativeRequest(), nativeLimits())
			want := ErrInvalidResponse
			if kind == "transport" || kind == "body-read" {
				want = ErrTransport
			}
			if !errors.Is(err, want) || strings.Contains(err.Error(), "fixture-secret") {
				t.Fatalf("unsafe transport error %v", err)
			}
			if count.Load() != 1 {
				t.Fatal("transport failure retried")
			}
		})
	}
}

type nativeErrorReader struct{}

func (nativeErrorReader) Read([]byte) (int, error) {
	return 0, errors.New("fixture-secret body diagnostic")
}
func (nativeErrorReader) Close() error { return nil }

func TestCompleteContextCancellationStopsTheRequest(t *testing.T) {
	for _, timeout := range []bool{false, true} {
		t.Run(map[bool]string{false: "cancel", true: "deadline"}[timeout], func(t *testing.T) {
			entered := make(chan struct{})
			var count atomic.Int32
			p, err := New(llmrouter.WithAPIKey("fixture"), llmrouter.WithHTTPClient(&http.Client{Transport: nativeRoundTripper(func(r *http.Request) (*http.Response, error) {
				count.Add(1)
				close(entered)
				<-r.Context().Done()
				return nil, r.Context().Err()
			})}))
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			if timeout {
				ctx, cancel = context.WithTimeout(context.Background(), 30*time.Millisecond)
			}
			defer cancel()
			done := make(chan error, 1)
			go func() { _, err := p.Complete(ctx, nativeRequest(), nativeLimits()); done <- err }()
			select {
			case <-entered:
			case <-time.After(time.Second):
				t.Fatal("no request")
			}
			if !timeout {
				cancel()
			}
			select {
			case err := <-done:
				want := error(context.Canceled)
				if timeout {
					want = context.DeadlineExceeded
				}
				if !errors.Is(err, want) || !errors.Is(err, ErrTransport) {
					t.Fatalf("lost cancellation identity %v", err)
				}
			case <-time.After(time.Second):
				t.Fatal("cancellation did not stop request")
			}
			if count.Load() != 1 {
				t.Fatal("cancelled request retried")
			}
		})
	}
}
