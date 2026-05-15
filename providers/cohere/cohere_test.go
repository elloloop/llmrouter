package cohere

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/elloloop/llmrouter"
)

// cohereSSEFixture is a complete /v2/chat SSE response covering
// message-start, two content-delta events, a skipped tool-call event,
// and message-end with COMPLETE + usage.
const cohereSSEFixture = `event: message-start
data: {"type":"message-start","id":"msg_1","delta":{"message":{"role":"assistant"}}}

event: content-start
data: {"type":"content-start","index":0,"delta":{"message":{"content":{"type":"text","text":""}}}}

event: content-delta
data: {"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"Hello"}}}}

event: content-delta
data: {"type":"content-delta","index":0,"delta":{"message":{"content":{"text":" world"}}}}

event: tool-call-start
data: {"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":{"id":"tc1","type":"function","function":{"name":"search","arguments":""}}}}}

event: tool-call-delta
data: {"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":{"function":{"arguments":"{}"}}}}}

event: tool-call-end
data: {"type":"tool-call-end","index":0}

event: content-end
data: {"type":"content-end","index":0}

event: message-end
data: {"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"billed_units":{"input_tokens":12,"output_tokens":7}}}}

`

// newFixtureServer asserts headers + path then writes the supplied body.
func newFixtureServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); !strings.HasPrefix(got, "Bearer ") {
			t.Errorf("Authorization = %q, want Bearer ...", got)
		}
		if got := r.Header.Get("Accept"); got != "text/event-stream" {
			t.Errorf("Accept = %q, want text/event-stream", got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("Content-Type = %q, want application/json", got)
		}
		if !strings.HasSuffix(r.URL.Path, "/chat") {
			t.Errorf("path = %q, want .../chat", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, body)
	}))
}

func newProvider(t *testing.T, url string) *Provider {
	t.Helper()
	p, err := New(
		llmrouter.WithAPIKey("test-key"),
		llmrouter.WithBaseURL(url),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return p
}

func TestNew(t *testing.T) {
	t.Run("ErrorWithoutAPIKey", func(t *testing.T) {
		_, err := New()
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !errors.Is(err, llmrouter.ErrInvalidConfig) {
			t.Errorf("err = %v, want wrap of ErrInvalidConfig", err)
		}
	})

	t.Run("DefaultsBaseURL", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("k"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if p.cfg.BaseURL != defaultBaseURL {
			t.Errorf("BaseURL = %q, want %q", p.cfg.BaseURL, defaultBaseURL)
		}
	})

	t.Run("OverridesBaseURL", func(t *testing.T) {
		p, err := New(llmrouter.WithAPIKey("k"), llmrouter.WithBaseURL("https://example.com"))
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		if p.cfg.BaseURL != "https://example.com" {
			t.Errorf("BaseURL = %q", p.cfg.BaseURL)
		}
	})

	t.Run("NameReturnsCohere", func(t *testing.T) {
		p, _ := New(llmrouter.WithAPIKey("k"))
		if got := p.Name(); got != "cohere" {
			t.Errorf("Name() = %q, want cohere", got)
		}
	})
}

func TestCompletionStream_TranslatesCohereSSEToOpenAIChunks(t *testing.T) {
	srv := newFixtureServer(t, cohereSSEFixture)
	defer srv.Close()

	p := newProvider(t, srv.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "command-r-plus",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}

	var chunks []llmrouter.Chunk
	for c := range stream.Chunks() {
		chunks = append(chunks, c)
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err: %v", err)
	}

	t.Run("FourChunksEmitted", func(t *testing.T) {
		// primer + 2 content deltas + finish (tool-call-* events skipped).
		if len(chunks) != 4 {
			t.Fatalf("got %d chunks, want 4: %+v", len(chunks), chunks)
		}
	})

	t.Run("PrimerHasAssistantRole", func(t *testing.T) {
		if chunks[0].Choices[0].Delta.Role != "assistant" {
			t.Errorf("primer role = %q, want assistant", chunks[0].Choices[0].Delta.Role)
		}
	})

	t.Run("PrimerHasEmptyContent", func(t *testing.T) {
		if chunks[0].Choices[0].Delta.Content != "" {
			t.Errorf("primer content = %q, want empty", chunks[0].Choices[0].Delta.Content)
		}
	})

	t.Run("FirstContentDelta", func(t *testing.T) {
		if chunks[1].Choices[0].Delta.Content != "Hello" {
			t.Errorf("chunk[1].content = %q, want Hello", chunks[1].Choices[0].Delta.Content)
		}
	})

	t.Run("SecondContentDelta", func(t *testing.T) {
		if chunks[2].Choices[0].Delta.Content != " world" {
			t.Errorf("chunk[2].content = %q, want ' world'", chunks[2].Choices[0].Delta.Content)
		}
	})

	t.Run("FinishReasonMapsToStop", func(t *testing.T) {
		if chunks[3].Choices[0].FinishReason != "stop" {
			t.Errorf("finish_reason = %q, want stop", chunks[3].Choices[0].FinishReason)
		}
	})

	t.Run("UsagePopulatedOnFinal", func(t *testing.T) {
		if chunks[3].Usage == nil {
			t.Fatalf("final.Usage = nil")
		}
		u := chunks[3].Usage
		if u.PromptTokens != 12 || u.CompletionTokens != 7 || u.TotalTokens != 19 {
			t.Errorf("usage = %+v, want {12,7,19}", *u)
		}
	})

	t.Run("ChunkIDsArePrefixed", func(t *testing.T) {
		for i, c := range chunks {
			if !strings.HasPrefix(c.ID, "chatcmpl-") {
				t.Errorf("chunk[%d].ID = %q, want chatcmpl-...", i, c.ID)
			}
		}
	})

	t.Run("ChunkIDsAreStableAcross", func(t *testing.T) {
		for i, c := range chunks {
			if c.ID != chunks[0].ID {
				t.Errorf("chunk[%d].ID = %q, want %q", i, c.ID, chunks[0].ID)
			}
		}
	})

	t.Run("ChunkObjectField", func(t *testing.T) {
		for i, c := range chunks {
			if c.Object != "chat.completion.chunk" {
				t.Errorf("chunk[%d].Object = %q", i, c.Object)
			}
		}
	})

	t.Run("ChunkModelMatchesRequest", func(t *testing.T) {
		for i, c := range chunks {
			if c.Model != "command-r-plus" {
				t.Errorf("chunk[%d].Model = %q, want command-r-plus", i, c.Model)
			}
		}
	})

	t.Run("ChunkCreatedNonZero", func(t *testing.T) {
		for i, c := range chunks {
			if c.Created == 0 {
				t.Errorf("chunk[%d].Created = 0", i)
			}
		}
	})

	t.Run("RawIsPopulated", func(t *testing.T) {
		for i, c := range chunks {
			if len(c.Raw) == 0 {
				t.Errorf("chunk[%d].Raw is empty", i)
			}
		}
	})

	t.Run("RawIsValidJSON", func(t *testing.T) {
		for i, c := range chunks {
			var v map[string]any
			if err := json.Unmarshal(c.Raw, &v); err != nil {
				t.Errorf("chunk[%d].Raw not valid JSON: %v", i, err)
			}
		}
	})

	t.Run("NoContentInPrimer", func(t *testing.T) {
		if chunks[0].Choices[0].FinishReason != "" {
			t.Errorf("primer finish_reason = %q, want empty", chunks[0].Choices[0].FinishReason)
		}
	})

	t.Run("OnlyFinalChunkHasFinish", func(t *testing.T) {
		for i := 0; i < 3; i++ {
			if chunks[i].Choices[0].FinishReason != "" {
				t.Errorf("chunk[%d] has finish_reason = %q, want empty", i, chunks[i].Choices[0].FinishReason)
			}
		}
	})

	t.Run("ToolCallEventsSkipped", func(t *testing.T) {
		for i, c := range chunks {
			// Tool-call events would have empty content + no finish_reason
			// but we already asserted len(chunks) == 4 which is primer + 2
			// content + finish — meaning the 3 tool-call events were skipped.
			if i > 0 && i < 3 && c.Choices[0].Delta.Content == "" {
				t.Errorf("unexpected empty-content chunk[%d]", i)
			}
		}
	})
}

func TestBuildCohereBody(t *testing.T) {
	t.Run("DefaultMaxTokens", func(t *testing.T) {
		body, err := buildCohereBody(llmrouter.ChatRequest{
			Model:    "command-r-plus",
			Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
		})
		if err != nil {
			t.Fatalf("buildCohereBody: %v", err)
		}
		if !strings.Contains(string(body), `"max_tokens":4096`) {
			t.Errorf("missing default max_tokens=4096: %s", body)
		}
	})

	t.Run("StreamFlagSet", func(t *testing.T) {
		body, _ := buildCohereBody(llmrouter.ChatRequest{Model: "m"})
		if !strings.Contains(string(body), `"stream":true`) {
			t.Errorf("stream flag missing: %s", body)
		}
	})

	t.Run("ModelPresent", func(t *testing.T) {
		body, _ := buildCohereBody(llmrouter.ChatRequest{Model: "command-r"})
		if !strings.Contains(string(body), `"model":"command-r"`) {
			t.Errorf("model missing: %s", body)
		}
	})

	t.Run("SystemMessagePassesThroughUnchanged", func(t *testing.T) {
		body, err := buildCohereBody(llmrouter.ChatRequest{
			Model: "command-r-plus",
			Messages: []llmrouter.Message{
				llmrouter.TextMessage("system", "you are nice"),
				llmrouter.TextMessage("user", "hi"),
			},
		})
		if err != nil {
			t.Fatalf("buildCohereBody: %v", err)
		}
		s := string(body)
		if !strings.Contains(s, `"role":"system"`) {
			t.Errorf("system message should remain in messages array: %s", s)
		}
		if strings.Contains(s, `"system":"you are nice"`) {
			t.Errorf("system must NOT be lifted to top level: %s", s)
		}
	})

	t.Run("MultipleSystemMessagesAllRetained", func(t *testing.T) {
		body, _ := buildCohereBody(llmrouter.ChatRequest{
			Model: "command-r-plus",
			Messages: []llmrouter.Message{
				llmrouter.TextMessage("system", "be nice"),
				llmrouter.TextMessage("system", "be concise"),
				llmrouter.TextMessage("user", "hi"),
			},
		})
		s := string(body)
		if strings.Count(s, `"role":"system"`) != 2 {
			t.Errorf("expected 2 system messages, got: %s", s)
		}
	})

	t.Run("TypedTemperatureKnob", func(t *testing.T) {
		temp := 0.5
		body, _ := buildCohereBody(llmrouter.ChatRequest{
			Model:       "m",
			Temperature: &temp,
		})
		if !strings.Contains(string(body), `"temperature":0.5`) {
			t.Errorf("temperature missing: %s", body)
		}
	})

	t.Run("TypedTopPMapsToP", func(t *testing.T) {
		topP := 0.9
		body, _ := buildCohereBody(llmrouter.ChatRequest{
			Model: "m",
			TopP:  &topP,
		})
		s := string(body)
		if !strings.Contains(s, `"p":0.9`) {
			t.Errorf("top_p should map to 'p': %s", s)
		}
	})

	t.Run("TypedStopMapsToStopSequences", func(t *testing.T) {
		body, _ := buildCohereBody(llmrouter.ChatRequest{
			Model: "m",
			Stop:  []string{"END"},
		})
		if !strings.Contains(string(body), `"stop_sequences":["END"]`) {
			t.Errorf("stop_sequences missing: %s", body)
		}
	})

	t.Run("RawKnobOverridesTyped", func(t *testing.T) {
		body, _ := buildCohereBody(llmrouter.ChatRequest{
			Model: "m",
			Raw:   json.RawMessage(`{"temperature":0.7,"p":0.8,"k":40,"stop_sequences":["X"]}`),
		})
		s := string(body)
		if !strings.Contains(s, `"temperature":0.7`) {
			t.Errorf("raw temperature missing: %s", s)
		}
		if !strings.Contains(s, `"p":0.8`) {
			t.Errorf("raw p missing: %s", s)
		}
		if !strings.Contains(s, `"k":40`) {
			t.Errorf("raw k missing: %s", s)
		}
		if !strings.Contains(s, `"stop_sequences":["X"]`) {
			t.Errorf("raw stop_sequences missing: %s", s)
		}
	})

	t.Run("RawOpenAIStyleKeysTranslated", func(t *testing.T) {
		body, _ := buildCohereBody(llmrouter.ChatRequest{
			Model: "m",
			Raw:   json.RawMessage(`{"top_p":0.85,"top_k":50,"stop":["STOP"]}`),
		})
		s := string(body)
		if !strings.Contains(s, `"p":0.85`) {
			t.Errorf("top_p should map to 'p': %s", s)
		}
		if !strings.Contains(s, `"k":50`) {
			t.Errorf("top_k should map to 'k': %s", s)
		}
		if !strings.Contains(s, `"stop_sequences":["STOP"]`) {
			t.Errorf("stop should map to 'stop_sequences': %s", s)
		}
	})

	t.Run("RawMaxTokensOverridesDefault", func(t *testing.T) {
		body, _ := buildCohereBody(llmrouter.ChatRequest{
			Model: "m",
			Raw:   json.RawMessage(`{"max_tokens":256}`),
		})
		if !strings.Contains(string(body), `"max_tokens":256`) {
			t.Errorf("max_tokens override missing: %s", body)
		}
	})

	t.Run("ExplicitMaxTokensOverridesDefault", func(t *testing.T) {
		body, _ := buildCohereBody(llmrouter.ChatRequest{Model: "m", MaxTokens: 128})
		if !strings.Contains(string(body), `"max_tokens":128`) {
			t.Errorf("typed max_tokens missing: %s", body)
		}
	})
}

func TestMapFinishReason(t *testing.T) {
	cases := map[string]string{
		"COMPLETE":      "stop",
		"MAX_TOKENS":    "length",
		"STOP_SEQUENCE": "stop",
		"TOOL_CALL":     "tool_calls",
		"UNKNOWN":       "stop",
	}
	for in, want := range cases {
		t.Run("Map_"+in, func(t *testing.T) {
			if got := mapFinishReason(in); got != want {
				t.Errorf("mapFinishReason(%q) = %q, want %q", in, got, want)
			}
		})
	}
}

func TestCompletionStream_UpstreamErrors(t *testing.T) {
	cases := []struct {
		name string
		code int
		body string
	}{
		{"BadRequest", http.StatusBadRequest, `{"message":"bad"}`},
		{"Unauthorized", http.StatusUnauthorized, `{"message":"invalid api key"}`},
		{"TooManyRequests", http.StatusTooManyRequests, `{"message":"rate limited"}`},
		{"InternalServerError", http.StatusInternalServerError, `{"message":"oops"}`},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.code)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer srv.Close()

			p := newProvider(t, srv.URL)
			_, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
				Model:    "command-r-plus",
				Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
			})
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			var ue *llmrouter.ErrUpstream
			if !errors.As(err, &ue) {
				t.Fatalf("err = %v, want *llmrouter.ErrUpstream", err)
			}
			if ue.StatusCode != tc.code {
				t.Errorf("StatusCode = %d, want %d", ue.StatusCode, tc.code)
			}
			if ue.Provider != "cohere" {
				t.Errorf("Provider = %q, want cohere", ue.Provider)
			}
			if !strings.Contains(ue.Body, "message") {
				t.Errorf("Body = %q, want to contain 'message'", ue.Body)
			}
		})
	}
}

func TestCompletionStream_RequestBodyShape(t *testing.T) {
	var captured []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, cohereSSEFixture)
	}))
	defer srv.Close()

	p := newProvider(t, srv.URL)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model: "command-r-plus",
		Messages: []llmrouter.Message{
			llmrouter.TextMessage("system", "you are nice"),
			llmrouter.TextMessage("user", "hi"),
		},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	for range stream.Chunks() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream.Err: %v", err)
	}

	var body map[string]any
	if err := json.Unmarshal(captured, &body); err != nil {
		t.Fatalf("body not JSON: %v", err)
	}

	t.Run("HasModelField", func(t *testing.T) {
		if body["model"] != "command-r-plus" {
			t.Errorf("model = %v", body["model"])
		}
	})
	t.Run("HasStreamTrue", func(t *testing.T) {
		if body["stream"] != true {
			t.Errorf("stream = %v", body["stream"])
		}
	})
	t.Run("MessagesArrayPresent", func(t *testing.T) {
		msgs, ok := body["messages"].([]any)
		if !ok {
			t.Fatalf("messages not an array: %v", body["messages"])
		}
		if len(msgs) != 2 {
			t.Errorf("messages len = %d, want 2", len(msgs))
		}
	})
	t.Run("NoTopLevelSystemKey", func(t *testing.T) {
		if _, ok := body["system"]; ok {
			t.Errorf("body has top-level system key, should not")
		}
	})
}

func TestCompletionStream_ContextCancellation(t *testing.T) {
	// Server hangs forever — we cancel ctx to ensure the stream cleans up.
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		w.WriteHeader(http.StatusOK)
		if flusher != nil {
			flusher.Flush()
		}
		<-block
	}))
	defer srv.Close()
	defer close(block)

	p := newProvider(t, srv.URL)

	ctx, cancel := context.WithCancel(context.Background())
	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{
		Model:    "command-r-plus",
		Messages: []llmrouter.Message{llmrouter.TextMessage("user", "hi")},
	})
	if err != nil {
		t.Fatalf("CompletionStream: %v", err)
	}
	cancel()
	// Drain — should terminate.
	done := make(chan struct{})
	go func() {
		for range stream.Chunks() {
		}
		_ = stream.Err()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("stream did not terminate after ctx cancel")
	}
}

func TestCurrentUsage(t *testing.T) {
	t.Run("NilWhenEmpty", func(t *testing.T) {
		if u := currentUsage(&pumpState{}); u != nil {
			t.Errorf("got %+v, want nil", u)
		}
	})
	t.Run("PopulatedWhenAvailable", func(t *testing.T) {
		u := currentUsage(&pumpState{inputTokens: 3, outputTokens: 5})
		if u == nil || u.PromptTokens != 3 || u.CompletionTokens != 5 || u.TotalTokens != 8 {
			t.Errorf("got %+v", u)
		}
	})
}
