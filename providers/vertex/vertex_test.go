package vertex

import (
	"errors"
	"strings"
	"testing"

	"google.golang.org/genai"

	"github.com/elloloop/llmrouter"
)

// --- New() validation -------------------------------------------------------

func TestNew_MissingProject(t *testing.T) {
	_, err := New(WithRegion("us-central1"))
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Fatalf("want ErrInvalidConfig, got %v", err)
	}
	if !strings.Contains(strings.ToLower(err.Error()), "project") {
		t.Fatalf("want 'project' in message, got %q", err.Error())
	}
}

func TestNew_MissingRegion(t *testing.T) {
	_, err := New(WithProject("my-proj"))
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Fatalf("want ErrInvalidConfig, got %v", err)
	}
	if !strings.Contains(strings.ToLower(err.Error()), "region") {
		t.Fatalf("want 'region' in message, got %q", err.Error())
	}
}

func TestNew_APIKeyRejected(t *testing.T) {
	_, err := New(
		llmrouter.WithAPIKey("sk-xxx"),
		WithProject("my-proj"),
		WithRegion("us-central1"),
	)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Fatalf("want ErrInvalidConfig, got %v", err)
	}
	if !strings.Contains(err.Error(), "WithAPIKey") {
		t.Fatalf("want WithAPIKey reason, got %q", err.Error())
	}
}

func TestNew_EmptyProjectOption(t *testing.T) {
	if err := WithProject("   ")(&llmrouter.Config{}); err == nil {
		t.Fatalf("expected error on empty project")
	}
}

func TestNew_EmptyRegionOption(t *testing.T) {
	if err := WithRegion("")(&llmrouter.Config{}); err == nil {
		t.Fatalf("expected error on empty region")
	}
}

func TestNew_EmptyCredentialsJSONOption(t *testing.T) {
	if err := WithCredentialsJSON(nil)(&llmrouter.Config{}); err == nil {
		t.Fatalf("expected error on nil credentials")
	}
	if err := WithCredentialsJSON([]byte{})(&llmrouter.Config{}); err == nil {
		t.Fatalf("expected error on empty credentials")
	}
}

func TestNew_InvalidCredentialsJSON(t *testing.T) {
	_, err := New(
		WithProject("p"),
		WithRegion("r"),
		WithCredentialsJSON([]byte("not-json")),
	)
	if err == nil {
		t.Fatalf("expected error on bad credentials json")
	}
}

func TestNew_Succeeds(t *testing.T) {
	// Without WithCredentialsJSON, genai's vertex backend won't try to
	// fetch tokens until a request is made, so construction should succeed
	// even in test envs without ADC. (genai.NewClient itself does not
	// short-circuit; it accepts the project/location and defers auth.)
	p, err := New(WithProject("my-proj"), WithRegion("us-central1"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p == nil {
		t.Fatalf("expected non-nil provider")
	}
	if p.Name() != "vertex" {
		t.Fatalf("want name 'vertex', got %q", p.Name())
	}
}

// --- Option storage ---------------------------------------------------------

func TestWithProject_StoresInExtra(t *testing.T) {
	c := &llmrouter.Config{}
	if err := WithProject("alpha")(c); err != nil {
		t.Fatalf("err: %v", err)
	}
	got, _ := stringExtra(c, extraProjectKey)
	if got != "alpha" {
		t.Fatalf("want alpha, got %q", got)
	}
}

func TestWithRegion_StoresInExtra(t *testing.T) {
	c := &llmrouter.Config{}
	if err := WithRegion("us-central1")(c); err != nil {
		t.Fatalf("err: %v", err)
	}
	got, _ := stringExtra(c, extraRegionKey)
	if got != "us-central1" {
		t.Fatalf("want us-central1, got %q", got)
	}
}

func TestWithCredentialsJSON_StoresInExtra(t *testing.T) {
	c := &llmrouter.Config{}
	payload := []byte(`{"type":"service_account"}`)
	if err := WithCredentialsJSON(payload)(c); err != nil {
		t.Fatalf("err: %v", err)
	}
	got, ok := bytesExtra(c, extraCredentialsJSONKey)
	if !ok || string(got) != string(payload) {
		t.Fatalf("creds not stored correctly")
	}
}

func TestExtraHelpers_Absent(t *testing.T) {
	c := &llmrouter.Config{}
	if _, ok := stringExtra(c, "missing"); ok {
		t.Fatalf("expected missing string")
	}
	if _, ok := bytesExtra(c, "missing"); ok {
		t.Fatalf("expected missing bytes")
	}
	if _, ok := stringExtra(nil, "missing"); ok {
		t.Fatalf("expected nil-safe stringExtra")
	}
	if _, ok := bytesExtra(nil, "missing"); ok {
		t.Fatalf("expected nil-safe bytesExtra")
	}
}

func TestExtraHelpers_WrongType(t *testing.T) {
	c := &llmrouter.Config{Extra: map[string]any{"k": 42}}
	if _, ok := stringExtra(c, "k"); ok {
		t.Fatalf("int should not coerce to string")
	}
	if _, ok := bytesExtra(c, "k"); ok {
		t.Fatalf("int should not coerce to bytes")
	}
}

// --- Role mapping -----------------------------------------------------------

func TestMapRequestRole(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"user", "user"},
		{"assistant", "model"},
		{"tool", "tool"},
		{"", ""},
		{"function", "function"},
	}
	for _, tc := range cases {
		t.Run("req/"+tc.in, func(t *testing.T) {
			if got := mapRequestRole(tc.in); got != tc.want {
				t.Fatalf("want %q got %q", tc.want, got)
			}
		})
	}
}

func TestMapResponseRole(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"model", "assistant"},
		{"", "assistant"},
		{"user", "user"},
		{"custom", "custom"},
	}
	for _, tc := range cases {
		t.Run("resp/"+tc.in, func(t *testing.T) {
			if got := mapResponseRole(tc.in); got != tc.want {
				t.Fatalf("want %q got %q", tc.want, got)
			}
		})
	}
}

// --- Finish-reason mapping --------------------------------------------------

func TestMapFinishReason(t *testing.T) {
	cases := []struct {
		in   genai.FinishReason
		want string
	}{
		{genai.FinishReasonStop, "stop"},
		{genai.FinishReasonMaxTokens, "length"},
		{genai.FinishReasonSafety, "content_filter"},
		{genai.FinishReasonRecitation, "content_filter"},
		{genai.FinishReasonBlocklist, "content_filter"},
		{genai.FinishReasonProhibitedContent, "content_filter"},
		{genai.FinishReasonSPII, "content_filter"},
		{genai.FinishReasonImageSafety, "content_filter"},
		{genai.FinishReasonImageProhibitedContent, "content_filter"},
		{genai.FinishReasonOther, "stop"},
		{genai.FinishReasonLanguage, "stop"},
		{genai.FinishReasonMalformedFunctionCall, "stop"},
		{"", ""},
		{"NOVEL", "stop"},
	}
	for _, tc := range cases {
		t.Run("finish/"+string(tc.in), func(t *testing.T) {
			if got := mapFinishReason(tc.in); got != tc.want {
				t.Fatalf("want %q got %q", tc.want, got)
			}
		})
	}
}

// --- Message translation ----------------------------------------------------

func TestTranslateMessages_UserOnly(t *testing.T) {
	contents, sys := translateMessages([]llmrouter.Message{
		llmrouter.TextMessage("user", "hi"),
	})
	if sys != nil {
		t.Fatalf("did not expect system instruction, got %v", sys)
	}
	if len(contents) != 1 {
		t.Fatalf("want 1 content, got %d", len(contents))
	}
	if contents[0].Role != "user" {
		t.Fatalf("want role user, got %q", contents[0].Role)
	}
	if len(contents[0].Parts) != 1 || contents[0].Parts[0].Text != "hi" {
		t.Fatalf("want one text part 'hi'")
	}
}

func TestTranslateMessages_AssistantMapped(t *testing.T) {
	contents, _ := translateMessages([]llmrouter.Message{
		llmrouter.TextMessage("assistant", "hello"),
	})
	if len(contents) != 1 || contents[0].Role != "model" {
		t.Fatalf("assistant must map to model, got %+v", contents)
	}
}

func TestTranslateMessages_SystemLifted(t *testing.T) {
	contents, sys := translateMessages([]llmrouter.Message{
		llmrouter.TextMessage("system", "be terse"),
		llmrouter.TextMessage("user", "hi"),
	})
	if sys == nil {
		t.Fatalf("want system instruction")
	}
	if len(sys.Parts) != 1 || sys.Parts[0].Text != "be terse" {
		t.Fatalf("system content mismatch: %+v", sys)
	}
	if sys.Role != "" {
		t.Fatalf("system instruction should not carry a role, got %q", sys.Role)
	}
	if len(contents) != 1 || contents[0].Role != "user" {
		t.Fatalf("want only the user message in contents")
	}
}

func TestTranslateMessages_MultipleSystemsConcatenated(t *testing.T) {
	contents, sys := translateMessages([]llmrouter.Message{
		llmrouter.TextMessage("system", "rule one"),
		llmrouter.TextMessage("system", "rule two"),
		llmrouter.TextMessage("user", "go"),
	})
	if sys == nil {
		t.Fatalf("want system instruction")
	}
	if sys.Parts[0].Text != "rule one\n\nrule two" {
		t.Fatalf("system not concatenated: %q", sys.Parts[0].Text)
	}
	if len(contents) != 1 {
		t.Fatalf("want 1 content, got %d", len(contents))
	}
}

func TestTranslateMessages_EmptyInput(t *testing.T) {
	contents, sys := translateMessages(nil)
	if sys != nil {
		t.Fatalf("expected nil system")
	}
	if len(contents) != 0 {
		t.Fatalf("expected zero contents")
	}
}

func TestTranslateMessages_OrderPreserved(t *testing.T) {
	contents, _ := translateMessages([]llmrouter.Message{
		llmrouter.TextMessage("user", "one"),
		llmrouter.TextMessage("assistant", "two"),
		llmrouter.TextMessage("user", "three"),
	})
	if len(contents) != 3 {
		t.Fatalf("want 3 contents, got %d", len(contents))
	}
	wantRoles := []string{"user", "model", "user"}
	wantTexts := []string{"one", "two", "three"}
	for i, c := range contents {
		if c.Role != wantRoles[i] {
			t.Fatalf("idx %d: role %q want %q", i, c.Role, wantRoles[i])
		}
		if c.Parts[0].Text != wantTexts[i] {
			t.Fatalf("idx %d: text %q want %q", i, c.Parts[0].Text, wantTexts[i])
		}
	}
}

// --- GenerationConfig building ---------------------------------------------

func TestBuildGenerateContentConfig_Defaults(t *testing.T) {
	cfg := buildGenerateContentConfig(llmrouter.ChatRequest{Model: "gemini-1.5-flash"}, nil)
	if cfg.MaxOutputTokens != defaultMaxTokens {
		t.Fatalf("want default max %d, got %d", defaultMaxTokens, cfg.MaxOutputTokens)
	}
	if cfg.Temperature != nil {
		t.Fatalf("temperature must be unset by default")
	}
	if cfg.TopP != nil {
		t.Fatalf("topP must be unset by default")
	}
	if cfg.StopSequences != nil {
		t.Fatalf("stop must be unset by default")
	}
	if cfg.SystemInstruction != nil {
		t.Fatalf("system instruction must be unset")
	}
}

func TestBuildGenerateContentConfig_RespectsMaxTokens(t *testing.T) {
	cfg := buildGenerateContentConfig(llmrouter.ChatRequest{MaxTokens: 256}, nil)
	if cfg.MaxOutputTokens != 256 {
		t.Fatalf("want 256, got %d", cfg.MaxOutputTokens)
	}
}

func TestBuildGenerateContentConfig_NegativeMaxTokensUsesDefault(t *testing.T) {
	cfg := buildGenerateContentConfig(llmrouter.ChatRequest{MaxTokens: -5}, nil)
	if cfg.MaxOutputTokens != defaultMaxTokens {
		t.Fatalf("want default, got %d", cfg.MaxOutputTokens)
	}
}

func TestBuildGenerateContentConfig_Temperature(t *testing.T) {
	temp := 0.7
	cfg := buildGenerateContentConfig(llmrouter.ChatRequest{Temperature: &temp}, nil)
	if cfg.Temperature == nil || *cfg.Temperature < 0.69 || *cfg.Temperature > 0.71 {
		t.Fatalf("temperature mismatch: %+v", cfg.Temperature)
	}
}

func TestBuildGenerateContentConfig_TopP(t *testing.T) {
	tp := 0.9
	cfg := buildGenerateContentConfig(llmrouter.ChatRequest{TopP: &tp}, nil)
	if cfg.TopP == nil || *cfg.TopP < 0.89 || *cfg.TopP > 0.91 {
		t.Fatalf("topP mismatch: %+v", cfg.TopP)
	}
}

func TestBuildGenerateContentConfig_Stop(t *testing.T) {
	cfg := buildGenerateContentConfig(llmrouter.ChatRequest{Stop: []string{"END", "STOP"}}, nil)
	if len(cfg.StopSequences) != 2 || cfg.StopSequences[0] != "END" || cfg.StopSequences[1] != "STOP" {
		t.Fatalf("stop sequences mismatch: %v", cfg.StopSequences)
	}
}

func TestBuildGenerateContentConfig_SystemInstruction(t *testing.T) {
	sys := &genai.Content{Parts: []*genai.Part{{Text: "tone: friendly"}}}
	cfg := buildGenerateContentConfig(llmrouter.ChatRequest{}, sys)
	if cfg.SystemInstruction == nil || cfg.SystemInstruction.Parts[0].Text != "tone: friendly" {
		t.Fatalf("system instruction not threaded: %+v", cfg.SystemInstruction)
	}
}

// --- Chunk construction -----------------------------------------------------

func TestNewChunk_FieldsPopulated(t *testing.T) {
	st := &pumpState{chatID: "chatcmpl-abc", created: 1700000000, model: "gemini-1.5-flash"}
	c := newChunk(st, llmrouter.Delta{Role: "assistant"}, "")
	if c.ID != "chatcmpl-abc" || c.Object != objectChunk || c.Created != 1700000000 || c.Model != "gemini-1.5-flash" {
		t.Fatalf("chunk header mismatch: %+v", c)
	}
	if len(c.Choices) != 1 {
		t.Fatalf("want 1 choice, got %d", len(c.Choices))
	}
	if c.Choices[0].Delta.Role != "assistant" {
		t.Fatalf("role primer not propagated")
	}
	if c.Choices[0].FinishReason != "" {
		t.Fatalf("finish reason should be empty")
	}
}

func TestNewChunk_FinishReasonPropagates(t *testing.T) {
	st := &pumpState{chatID: "x", created: 1, model: "m"}
	c := newChunk(st, llmrouter.Delta{}, "stop")
	if c.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish not propagated")
	}
}

// --- Usage accumulation -----------------------------------------------------

func TestCurrentUsage_NilWhenEmpty(t *testing.T) {
	if currentUsage(&pumpState{}) != nil {
		t.Fatalf("want nil when no tokens recorded")
	}
}

func TestCurrentUsage_Sums(t *testing.T) {
	u := currentUsage(&pumpState{promptTokens: 10, candTokens: 25})
	if u == nil {
		t.Fatalf("want non-nil usage")
	}
	if u.PromptTokens != 10 || u.CompletionTokens != 25 || u.TotalTokens != 35 {
		t.Fatalf("usage math wrong: %+v", u)
	}
}

func TestCurrentUsage_OnlyPrompt(t *testing.T) {
	u := currentUsage(&pumpState{promptTokens: 7})
	if u == nil || u.TotalTokens != 7 {
		t.Fatalf("partial usage wrong: %+v", u)
	}
}

// --- Error wrapping ---------------------------------------------------------

func TestWrapSDKError_Nil(t *testing.T) {
	if wrapSDKError(nil) != nil {
		t.Fatalf("nil in must produce nil out")
	}
}

func TestWrapSDKError_GenericError(t *testing.T) {
	err := wrapSDKError(errors.New("transport boom"))
	var up *llmrouter.ErrUpstream
	if !errors.As(err, &up) {
		t.Fatalf("want ErrUpstream, got %T", err)
	}
	if up.Provider != "vertex" {
		t.Fatalf("want provider vertex, got %q", up.Provider)
	}
	if up.StatusCode != 0 {
		t.Fatalf("want status 0 for generic error, got %d", up.StatusCode)
	}
	if !strings.Contains(up.Body, "transport boom") {
		t.Fatalf("body lost: %q", up.Body)
	}
}

func TestWrapSDKError_APIError(t *testing.T) {
	api := genai.APIError{Code: 429, Message: "rate limited", Status: "RESOURCE_EXHAUSTED"}
	err := wrapSDKError(api)
	var up *llmrouter.ErrUpstream
	if !errors.As(err, &up) {
		t.Fatalf("want ErrUpstream, got %T", err)
	}
	if up.StatusCode != 429 {
		t.Fatalf("want 429, got %d", up.StatusCode)
	}
	if up.Body != "rate limited" {
		t.Fatalf("want body 'rate limited', got %q", up.Body)
	}
}

// --- ensureExtra ------------------------------------------------------------

func TestEnsureExtra_Lazy(t *testing.T) {
	c := &llmrouter.Config{}
	ensureExtra(c)
	if c.Extra == nil {
		t.Fatalf("Extra should be allocated")
	}
	// Should not replace existing.
	c.Extra["sentinel"] = "x"
	ensureExtra(c)
	if c.Extra["sentinel"] != "x" {
		t.Fatalf("ensureExtra wiped existing map")
	}
}
