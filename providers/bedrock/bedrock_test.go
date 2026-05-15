package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	smithyhttp "github.com/aws/smithy-go/transport/http"

	"github.com/elloloop/llmrouter"
)

// --- New / option validation -------------------------------------------------

func TestNew_RequiresRegion(t *testing.T) {
	t.Parallel()
	if _, err := New(); err == nil {
		t.Fatal("expected error when WithRegion not supplied")
	} else if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Fatalf("want ErrInvalidConfig, got %v", err)
	}
}

func TestNew_RejectsAPIKey(t *testing.T) {
	t.Parallel()
	_, err := New(WithRegion("us-east-1"), llmrouter.WithAPIKey("sk-bad"))
	if err == nil {
		t.Fatal("expected error when WithAPIKey supplied")
	}
	if !errors.Is(err, llmrouter.ErrInvalidConfig) {
		t.Fatalf("want ErrInvalidConfig, got %v", err)
	}
}

func TestNew_WithRegion_Succeeds(t *testing.T) {
	t.Parallel()
	p, err := New(WithRegion("us-west-2"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.Name() != "bedrock" {
		t.Fatalf("Name() = %q, want bedrock", p.Name())
	}
	if p.region != "us-west-2" {
		t.Fatalf("region = %q, want us-west-2", p.region)
	}
}

func TestNew_WithAWSConfig_Stored(t *testing.T) {
	t.Parallel()
	awsCfg := aws.Config{Region: "eu-west-1"}
	p, err := New(WithRegion("us-east-1"), WithAWSConfig(awsCfg))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.awsCfg == nil {
		t.Fatal("awsCfg not stored")
	}
	if p.awsCfg.Region != "eu-west-1" {
		t.Fatalf("awsCfg.Region = %q, want eu-west-1", p.awsCfg.Region)
	}
}

func TestWithRegion_EmptyRejected(t *testing.T) {
	t.Parallel()
	_, err := New(WithRegion(""))
	if err == nil {
		t.Fatal("expected error for empty region")
	}
}

// --- mapRole -----------------------------------------------------------------

func TestMapRole(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		in   string
		want types.ConversationRole
	}{
		{"user lowercase", "user", types.ConversationRoleUser},
		{"assistant lowercase", "assistant", types.ConversationRoleAssistant},
		{"system falls back to user", "system", types.ConversationRoleUser},
		{"tool falls back to user", "tool", types.ConversationRoleUser},
		{"empty falls back to user", "", types.ConversationRoleUser},
		{"unknown falls back to user", "developer", types.ConversationRoleUser},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := mapRole(tc.in); got != tc.want {
				t.Fatalf("mapRole(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// --- mapStopReason -----------------------------------------------------------

func TestMapStopReason(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		in   types.StopReason
		want string
	}{
		{"end_turn → stop", types.StopReasonEndTurn, "stop"},
		{"stop_sequence → stop", types.StopReasonStopSequence, "stop"},
		{"max_tokens → length", types.StopReasonMaxTokens, "length"},
		{"tool_use → tool_calls", types.StopReasonToolUse, "tool_calls"},
		{"guardrail intervened → stop (default)", types.StopReasonGuardrailIntervened, "stop"},
		{"content_filtered → stop (default)", types.StopReasonContentFiltered, "stop"},
		{"unknown → stop (default)", types.StopReason("totally_unknown"), "stop"},
		{"empty → stop (default)", types.StopReason(""), "stop"},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := mapStopReason(tc.in); got != tc.want {
				t.Fatalf("mapStopReason(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// --- liftSystem --------------------------------------------------------------

func TestLiftSystem(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		messages []llmrouter.Message
		want     []string // expected text values in order
	}{
		{
			name:     "empty",
			messages: nil,
			want:     nil,
		},
		{
			name: "no system messages",
			messages: []llmrouter.Message{
				llmrouter.TextMessage("user", "hi"),
				llmrouter.TextMessage("assistant", "yo"),
			},
			want: nil,
		},
		{
			name: "single system",
			messages: []llmrouter.Message{
				llmrouter.TextMessage("system", "be brief"),
				llmrouter.TextMessage("user", "hi"),
			},
			want: []string{"be brief"},
		},
		{
			name: "two systems both kept",
			messages: []llmrouter.Message{
				llmrouter.TextMessage("system", "first"),
				llmrouter.TextMessage("user", "hi"),
				llmrouter.TextMessage("system", "second"),
			},
			want: []string{"first", "second"},
		},
		{
			name: "empty system message dropped",
			messages: []llmrouter.Message{
				llmrouter.TextMessage("system", ""),
				llmrouter.TextMessage("user", "hi"),
			},
			want: nil,
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := liftSystem(tc.messages)
			if len(got) != len(tc.want) {
				t.Fatalf("len = %d, want %d (%v)", len(got), len(tc.want), got)
			}
			for i, blk := range got {
				txt, ok := blk.(*types.SystemContentBlockMemberText)
				if !ok {
					t.Fatalf("block %d not SystemContentBlockMemberText: %T", i, blk)
				}
				if txt.Value != tc.want[i] {
					t.Fatalf("block %d = %q, want %q", i, txt.Value, tc.want[i])
				}
			}
		})
	}
}

// --- toConverseMessages ------------------------------------------------------

func TestToConverseMessages(t *testing.T) {
	t.Parallel()

	t.Run("drops system messages", func(t *testing.T) {
		t.Parallel()
		got := toConverseMessages([]llmrouter.Message{
			llmrouter.TextMessage("system", "rules"),
			llmrouter.TextMessage("user", "hi"),
			llmrouter.TextMessage("assistant", "ok"),
		})
		if len(got) != 2 {
			t.Fatalf("len = %d, want 2", len(got))
		}
		if got[0].Role != types.ConversationRoleUser {
			t.Fatalf("got[0].Role = %q", got[0].Role)
		}
		if got[1].Role != types.ConversationRoleAssistant {
			t.Fatalf("got[1].Role = %q", got[1].Role)
		}
	})

	t.Run("text content preserved", func(t *testing.T) {
		t.Parallel()
		got := toConverseMessages([]llmrouter.Message{
			llmrouter.TextMessage("user", "hello world"),
		})
		if len(got) != 1 || len(got[0].Content) != 1 {
			t.Fatalf("unexpected shape: %+v", got)
		}
		txt, ok := got[0].Content[0].(*types.ContentBlockMemberText)
		if !ok {
			t.Fatalf("content not text block: %T", got[0].Content[0])
		}
		if txt.Value != "hello world" {
			t.Fatalf("text = %q, want %q", txt.Value, "hello world")
		}
	})

	t.Run("empty input → empty output", func(t *testing.T) {
		t.Parallel()
		got := toConverseMessages(nil)
		if len(got) != 0 {
			t.Fatalf("len = %d, want 0", len(got))
		}
	})

	t.Run("unknown role mapped to user", func(t *testing.T) {
		t.Parallel()
		got := toConverseMessages([]llmrouter.Message{
			llmrouter.TextMessage("developer", "x"),
		})
		if len(got) != 1 {
			t.Fatalf("len = %d", len(got))
		}
		if got[0].Role != types.ConversationRoleUser {
			t.Fatalf("role = %q, want user", got[0].Role)
		}
	})

	t.Run("preserves order", func(t *testing.T) {
		t.Parallel()
		got := toConverseMessages([]llmrouter.Message{
			llmrouter.TextMessage("user", "1"),
			llmrouter.TextMessage("assistant", "2"),
			llmrouter.TextMessage("user", "3"),
		})
		if len(got) != 3 {
			t.Fatalf("len = %d", len(got))
		}
		for i, want := range []string{"1", "2", "3"} {
			txt := got[i].Content[0].(*types.ContentBlockMemberText)
			if txt.Value != want {
				t.Fatalf("[%d] = %q, want %q", i, txt.Value, want)
			}
		}
	})
}

// --- buildInferenceConfig ----------------------------------------------------

func float64Ptr(v float64) *float64 { return &v }

func TestBuildInferenceConfig(t *testing.T) {
	t.Parallel()

	t.Run("defaults max tokens to 4096", func(t *testing.T) {
		t.Parallel()
		c := buildInferenceConfig(llmrouter.ChatRequest{})
		if c.MaxTokens == nil || *c.MaxTokens != 4096 {
			t.Fatalf("MaxTokens = %v, want 4096", c.MaxTokens)
		}
	})

	t.Run("honors explicit max tokens", func(t *testing.T) {
		t.Parallel()
		c := buildInferenceConfig(llmrouter.ChatRequest{MaxTokens: 100})
		if c.MaxTokens == nil || *c.MaxTokens != 100 {
			t.Fatalf("MaxTokens = %v, want 100", c.MaxTokens)
		}
	})

	t.Run("temperature passed when set", func(t *testing.T) {
		t.Parallel()
		c := buildInferenceConfig(llmrouter.ChatRequest{Temperature: float64Ptr(0.42)})
		if c.Temperature == nil || *c.Temperature < 0.41 || *c.Temperature > 0.43 {
			t.Fatalf("Temperature = %v, want ~0.42", c.Temperature)
		}
	})

	t.Run("temperature nil when unset", func(t *testing.T) {
		t.Parallel()
		c := buildInferenceConfig(llmrouter.ChatRequest{})
		if c.Temperature != nil {
			t.Fatalf("Temperature = %v, want nil", c.Temperature)
		}
	})

	t.Run("top_p passed when set", func(t *testing.T) {
		t.Parallel()
		c := buildInferenceConfig(llmrouter.ChatRequest{TopP: float64Ptr(0.9)})
		if c.TopP == nil || *c.TopP < 0.89 || *c.TopP > 0.91 {
			t.Fatalf("TopP = %v, want ~0.9", c.TopP)
		}
	})

	t.Run("top_p nil when unset", func(t *testing.T) {
		t.Parallel()
		c := buildInferenceConfig(llmrouter.ChatRequest{})
		if c.TopP != nil {
			t.Fatalf("TopP = %v, want nil", c.TopP)
		}
	})

	t.Run("stop sequences passed through", func(t *testing.T) {
		t.Parallel()
		c := buildInferenceConfig(llmrouter.ChatRequest{Stop: []string{"a", "b"}})
		if len(c.StopSequences) != 2 || c.StopSequences[0] != "a" || c.StopSequences[1] != "b" {
			t.Fatalf("StopSequences = %v, want [a b]", c.StopSequences)
		}
	})

	t.Run("stop sequences empty when not provided", func(t *testing.T) {
		t.Parallel()
		c := buildInferenceConfig(llmrouter.ChatRequest{})
		if len(c.StopSequences) != 0 {
			t.Fatalf("StopSequences = %v, want empty", c.StopSequences)
		}
	})

	t.Run("negative max tokens uses default", func(t *testing.T) {
		t.Parallel()
		c := buildInferenceConfig(llmrouter.ChatRequest{MaxTokens: -5})
		if c.MaxTokens == nil || *c.MaxTokens != 4096 {
			t.Fatalf("MaxTokens = %v, want 4096", c.MaxTokens)
		}
	})
}

// --- buildConverseStreamInput ------------------------------------------------

func TestBuildConverseStreamInput(t *testing.T) {
	t.Parallel()

	t.Run("model id set", func(t *testing.T) {
		t.Parallel()
		in := buildConverseStreamInput(llmrouter.ChatRequest{Model: "anthropic.claude-3"})
		if in.ModelId == nil || *in.ModelId != "anthropic.claude-3" {
			t.Fatalf("ModelId = %v", in.ModelId)
		}
	})

	t.Run("system lifted, others in messages", func(t *testing.T) {
		t.Parallel()
		in := buildConverseStreamInput(llmrouter.ChatRequest{
			Model: "m",
			Messages: []llmrouter.Message{
				llmrouter.TextMessage("system", "rules"),
				llmrouter.TextMessage("user", "hi"),
			},
		})
		if len(in.System) != 1 {
			t.Fatalf("System len = %d, want 1", len(in.System))
		}
		if len(in.Messages) != 1 {
			t.Fatalf("Messages len = %d, want 1", len(in.Messages))
		}
	})

	t.Run("no system → nil System", func(t *testing.T) {
		t.Parallel()
		in := buildConverseStreamInput(llmrouter.ChatRequest{
			Model: "m",
			Messages: []llmrouter.Message{
				llmrouter.TextMessage("user", "hi"),
			},
		})
		if in.System != nil {
			t.Fatalf("System = %v, want nil", in.System)
		}
	})
}

// --- wrapUpstream ------------------------------------------------------------

func TestWrapUpstream(t *testing.T) {
	t.Parallel()

	t.Run("nil → nil", func(t *testing.T) {
		t.Parallel()
		if err := wrapUpstream(nil); err != nil {
			t.Fatalf("got %v, want nil", err)
		}
	})

	t.Run("plain error → status 0", func(t *testing.T) {
		t.Parallel()
		err := wrapUpstream(errors.New("boom"))
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("not ErrUpstream: %v", err)
		}
		if ue.Provider != "bedrock" {
			t.Fatalf("Provider = %q", ue.Provider)
		}
		if ue.StatusCode != 0 {
			t.Fatalf("StatusCode = %d, want 0", ue.StatusCode)
		}
		if !strings.Contains(ue.Body, "boom") {
			t.Fatalf("Body = %q", ue.Body)
		}
	})

	t.Run("smithy response error → status preserved", func(t *testing.T) {
		t.Parallel()
		resp := &http.Response{StatusCode: 403}
		raw := &smithyhttp.ResponseError{
			Response: &smithyhttp.Response{Response: resp},
			Err:      errors.New("denied"),
		}
		err := wrapUpstream(raw)
		var ue *llmrouter.ErrUpstream
		if !errors.As(err, &ue) {
			t.Fatalf("not ErrUpstream: %v", err)
		}
		if ue.StatusCode != 403 {
			t.Fatalf("StatusCode = %d, want 403", ue.StatusCode)
		}
	})
}

// --- extractTextDelta --------------------------------------------------------

func TestExtractTextDelta(t *testing.T) {
	t.Parallel()

	t.Run("text variant", func(t *testing.T) {
		t.Parallel()
		txt, ok := extractTextDelta(&types.ContentBlockDeltaMemberText{Value: "hi"})
		if !ok || txt != "hi" {
			t.Fatalf("got (%q, %v), want (hi, true)", txt, ok)
		}
	})

	t.Run("non-text variant rejected", func(t *testing.T) {
		t.Parallel()
		_, ok := extractTextDelta(&types.ContentBlockDeltaMemberToolUse{})
		if ok {
			t.Fatal("expected ok=false for non-text delta")
		}
	})

	t.Run("nil delta rejected", func(t *testing.T) {
		t.Parallel()
		_, ok := extractTextDelta(nil)
		if ok {
			t.Fatal("expected ok=false for nil delta")
		}
	})
}

// --- currentUsage / applyMetadata --------------------------------------------

func TestCurrentUsage(t *testing.T) {
	t.Parallel()

	t.Run("zero → nil", func(t *testing.T) {
		t.Parallel()
		if u := currentUsage(&pumpState{}); u != nil {
			t.Fatalf("usage = %+v, want nil", u)
		}
	})

	t.Run("input only", func(t *testing.T) {
		t.Parallel()
		u := currentUsage(&pumpState{inputTokens: 10})
		if u == nil || u.PromptTokens != 10 || u.TotalTokens != 10 {
			t.Fatalf("usage = %+v", u)
		}
	})

	t.Run("both populated", func(t *testing.T) {
		t.Parallel()
		u := currentUsage(&pumpState{inputTokens: 7, outputTokens: 3})
		if u.PromptTokens != 7 || u.CompletionTokens != 3 || u.TotalTokens != 10 {
			t.Fatalf("usage = %+v", u)
		}
	})
}

func TestApplyMetadata(t *testing.T) {
	t.Parallel()

	t.Run("nil usage tolerated", func(t *testing.T) {
		t.Parallel()
		st := &pumpState{}
		applyMetadata(types.ConverseStreamMetadataEvent{}, st)
		if st.inputTokens != 0 || st.outputTokens != 0 {
			t.Fatalf("state mutated: %+v", st)
		}
	})

	t.Run("copies token counts", func(t *testing.T) {
		t.Parallel()
		st := &pumpState{}
		in, out := int32(11), int32(22)
		applyMetadata(types.ConverseStreamMetadataEvent{Usage: &types.TokenUsage{
			InputTokens:  &in,
			OutputTokens: &out,
		}}, st)
		if st.inputTokens != 11 || st.outputTokens != 22 {
			t.Fatalf("state = %+v", st)
		}
	})

	t.Run("partial usage", func(t *testing.T) {
		t.Parallel()
		st := &pumpState{}
		in := int32(5)
		applyMetadata(types.ConverseStreamMetadataEvent{Usage: &types.TokenUsage{InputTokens: &in}}, st)
		if st.inputTokens != 5 || st.outputTokens != 0 {
			t.Fatalf("state = %+v", st)
		}
	})
}

// --- newChunk ----------------------------------------------------------------

func TestNewChunk(t *testing.T) {
	t.Parallel()
	st := &pumpState{chatID: "id-1", created: 42, model: "m"}

	t.Run("basic shape", func(t *testing.T) {
		t.Parallel()
		c := newChunk(st, llmrouter.Delta{Content: "x"}, "")
		if c.ID != "id-1" || c.Object != "chat.completion.chunk" || c.Created != 42 || c.Model != "m" {
			t.Fatalf("chunk = %+v", c)
		}
		if len(c.Choices) != 1 {
			t.Fatalf("choices = %d", len(c.Choices))
		}
		if c.Choices[0].Delta.Content != "x" {
			t.Fatalf("delta = %q", c.Choices[0].Delta.Content)
		}
	})

	t.Run("raw is json", func(t *testing.T) {
		t.Parallel()
		c := newChunk(st, llmrouter.Delta{Content: "y"}, "stop")
		var decoded map[string]any
		if err := json.Unmarshal(c.Raw, &decoded); err != nil {
			t.Fatalf("Raw not JSON: %v", err)
		}
		if decoded["object"] != "chat.completion.chunk" {
			t.Fatalf("decoded object = %v", decoded["object"])
		}
	})

	t.Run("finish_reason set", func(t *testing.T) {
		t.Parallel()
		c := newChunk(st, llmrouter.Delta{}, "length")
		if c.Choices[0].FinishReason != "length" {
			t.Fatalf("FinishReason = %q", c.Choices[0].FinishReason)
		}
	})
}

// --- handleEvent: pump-state driven ------------------------------------------

// captureHooks records chunks routed through ProducerHooks for assertions.
type captureHooks struct {
	chunks []llmrouter.Chunk
	stop   bool // when true, Send returns false (consumer cancelled)
}

func (c *captureHooks) hooks() llmrouter.ProducerHooks {
	return llmrouter.ProducerHooks{
		Send: func(ch llmrouter.Chunk) bool {
			c.chunks = append(c.chunks, ch)
			return !c.stop
		},
		Finish: func(error) {},
	}
}

func TestHandleEvent_MessageStart(t *testing.T) {
	t.Parallel()
	cap := &captureHooks{}
	st := &pumpState{chatID: "i", model: "m"}
	terminate := handleEvent(&types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}}, st, cap.hooks())
	if terminate {
		t.Fatal("unexpected terminate=true")
	}
	if len(cap.chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(cap.chunks))
	}
	if cap.chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Fatalf("Delta.Role = %q", cap.chunks[0].Choices[0].Delta.Role)
	}
}

func TestHandleEvent_ContentBlockDelta_Text(t *testing.T) {
	t.Parallel()
	cap := &captureHooks{}
	st := &pumpState{}
	ev := &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		Delta: &types.ContentBlockDeltaMemberText{Value: "hello"},
	}}
	terminate := handleEvent(ev, st, cap.hooks())
	if terminate {
		t.Fatal("unexpected terminate=true")
	}
	if len(cap.chunks) != 1 || cap.chunks[0].Choices[0].Delta.Content != "hello" {
		t.Fatalf("chunks = %+v", cap.chunks)
	}
}

func TestHandleEvent_ContentBlockDelta_EmptyText(t *testing.T) {
	t.Parallel()
	cap := &captureHooks{}
	st := &pumpState{}
	ev := &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		Delta: &types.ContentBlockDeltaMemberText{Value: ""},
	}}
	if handleEvent(ev, st, cap.hooks()) {
		t.Fatal("unexpected terminate=true")
	}
	if len(cap.chunks) != 0 {
		t.Fatalf("expected no chunks, got %d", len(cap.chunks))
	}
}

func TestHandleEvent_ContentBlockDelta_NonText(t *testing.T) {
	t.Parallel()
	cap := &captureHooks{}
	st := &pumpState{}
	ev := &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		Delta: &types.ContentBlockDeltaMemberToolUse{},
	}}
	if handleEvent(ev, st, cap.hooks()) {
		t.Fatal("unexpected terminate=true")
	}
	if len(cap.chunks) != 0 {
		t.Fatalf("expected no chunks, got %d", len(cap.chunks))
	}
}

func TestHandleEvent_MessageStop(t *testing.T) {
	t.Parallel()
	cap := &captureHooks{}
	st := &pumpState{inputTokens: 4, outputTokens: 5}
	ev := &types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonMaxTokens}}
	handleEvent(ev, st, cap.hooks())
	if len(cap.chunks) != 1 {
		t.Fatalf("chunks = %d", len(cap.chunks))
	}
	ch := cap.chunks[0]
	if ch.Choices[0].FinishReason != "length" {
		t.Fatalf("FinishReason = %q", ch.Choices[0].FinishReason)
	}
	if ch.Usage == nil || ch.Usage.TotalTokens != 9 {
		t.Fatalf("Usage = %+v", ch.Usage)
	}
	if !st.stopEmitted {
		t.Fatal("stopEmitted not set")
	}
}

func TestHandleEvent_MetadataBeforeStop(t *testing.T) {
	t.Parallel()
	cap := &captureHooks{}
	st := &pumpState{}
	in, out := int32(2), int32(3)
	ev := &types.ConverseStreamOutputMemberMetadata{Value: types.ConverseStreamMetadataEvent{
		Usage: &types.TokenUsage{InputTokens: &in, OutputTokens: &out},
	}}
	handleEvent(ev, st, cap.hooks())
	if len(cap.chunks) != 0 {
		t.Fatalf("expected no chunks before stop, got %d", len(cap.chunks))
	}
	if st.inputTokens != 2 || st.outputTokens != 3 {
		t.Fatalf("state = %+v", st)
	}
}

func TestHandleEvent_MetadataAfterStop_EmitsUsageChunk(t *testing.T) {
	t.Parallel()
	cap := &captureHooks{}
	st := &pumpState{stopEmitted: true, finishReason: "stop"}
	in, out := int32(2), int32(3)
	ev := &types.ConverseStreamOutputMemberMetadata{Value: types.ConverseStreamMetadataEvent{
		Usage: &types.TokenUsage{InputTokens: &in, OutputTokens: &out},
	}}
	handleEvent(ev, st, cap.hooks())
	if len(cap.chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(cap.chunks))
	}
	if cap.chunks[0].Usage == nil || cap.chunks[0].Usage.TotalTokens != 5 {
		t.Fatalf("Usage = %+v", cap.chunks[0].Usage)
	}
}

func TestHandleEvent_Cancellation(t *testing.T) {
	t.Parallel()
	cap := &captureHooks{stop: true}
	st := &pumpState{}
	ev := &types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
		Delta: &types.ContentBlockDeltaMemberText{Value: "hi"},
	}}
	if !handleEvent(ev, st, cap.hooks()) {
		t.Fatal("expected terminate=true after consumer cancelled")
	}
}

func TestHandleEvent_UnknownIgnored(t *testing.T) {
	t.Parallel()
	cap := &captureHooks{}
	st := &pumpState{}
	if handleEvent(&types.ConverseStreamOutputMemberContentBlockStart{}, st, cap.hooks()) {
		t.Fatal("unexpected terminate=true")
	}
	if len(cap.chunks) != 0 {
		t.Fatalf("expected no chunks, got %d", len(cap.chunks))
	}
}

// --- pump end-to-end (mock reader) -------------------------------------------

// mockReader implements bedrockruntime.ConverseStreamOutputReader so we can
// feed canned events through pump without spinning up an AWS server.
type mockReader struct {
	events chan types.ConverseStreamOutput
	err    error
}

func newMockReader(evs []types.ConverseStreamOutput) *mockReader {
	ch := make(chan types.ConverseStreamOutput, len(evs))
	for _, e := range evs {
		ch <- e
	}
	close(ch)
	return &mockReader{events: ch}
}

func (m *mockReader) Events() <-chan types.ConverseStreamOutput { return m.events }
func (m *mockReader) Close() error                              { return nil }
func (m *mockReader) Err() error                                { return m.err }

func TestPumpFrom_EndToEnd(t *testing.T) {
	t.Parallel()
	in, out := int32(3), int32(4)
	events := []types.ConverseStreamOutput{
		&types.ConverseStreamOutputMemberMessageStart{Value: types.MessageStartEvent{Role: types.ConversationRoleAssistant}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			Delta: &types.ContentBlockDeltaMemberText{Value: "hello "},
		}},
		&types.ConverseStreamOutputMemberContentBlockDelta{Value: types.ContentBlockDeltaEvent{
			Delta: &types.ContentBlockDeltaMemberText{Value: "world"},
		}},
		&types.ConverseStreamOutputMemberMetadata{Value: types.ConverseStreamMetadataEvent{
			Usage: &types.TokenUsage{InputTokens: &in, OutputTokens: &out},
		}},
		&types.ConverseStreamOutputMemberMessageStop{Value: types.MessageStopEvent{StopReason: types.StopReasonEndTurn}},
	}

	stream, _, hooks := llmrouter.NewStream(context.Background())
	collected := make(chan []llmrouter.Chunk, 1)
	go func() {
		var got []llmrouter.Chunk
		for c := range stream.Chunks() {
			got = append(got, c)
		}
		collected <- got
	}()

	pumpFrom(context.Background(), newMockReader(events), "m", hooks)
	got := <-collected
	if err := stream.Err(); err != nil {
		t.Fatalf("pump err: %v", err)
	}

	// Expect: start chunk + 2 content chunks + stop chunk with usage.
	if len(got) != 4 {
		t.Fatalf("chunks = %d, want 4 (%+v)", len(got), got)
	}
	if got[0].Choices[0].Delta.Role != "assistant" {
		t.Fatalf("first chunk delta role = %q", got[0].Choices[0].Delta.Role)
	}
	if got[1].Choices[0].Delta.Content+got[2].Choices[0].Delta.Content != "hello world" {
		t.Fatalf("concat = %q", got[1].Choices[0].Delta.Content+got[2].Choices[0].Delta.Content)
	}
	last := got[3]
	if last.Choices[0].FinishReason != "stop" {
		t.Fatalf("FinishReason = %q", last.Choices[0].FinishReason)
	}
	if last.Usage == nil || last.Usage.TotalTokens != 7 {
		t.Fatalf("Usage = %+v", last.Usage)
	}
	// IDs should be uuid-prefixed and stable across chunks of a stream.
	if !strings.HasPrefix(got[0].ID, "chatcmpl-") {
		t.Fatalf("ID = %q", got[0].ID)
	}
	for _, c := range got {
		if c.ID != got[0].ID {
			t.Fatalf("ID mismatch: %q vs %q", c.ID, got[0].ID)
		}
		if c.Object != "chat.completion.chunk" {
			t.Fatalf("Object = %q", c.Object)
		}
	}
}

func TestPumpFrom_ReaderError(t *testing.T) {
	t.Parallel()
	r := newMockReader(nil)
	r.err = errors.New("boom")

	stream, _, hooks := llmrouter.NewStream(context.Background())
	done := make(chan struct{})
	go func() {
		for range stream.Chunks() {
		}
		close(done)
	}()
	pumpFrom(context.Background(), r, "m", hooks)
	<-done
	err := stream.Err()
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("err = %v, want wrapped boom", err)
	}
}

func TestPumpFrom_ContextCancel(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(context.Background())
	r := &blockingReader{ch: make(chan types.ConverseStreamOutput)}

	stream, _, hooks := llmrouter.NewStream(ctx)
	done := make(chan struct{})
	go func() {
		for range stream.Chunks() {
		}
		close(done)
	}()

	pumpDone := make(chan struct{})
	go func() {
		pumpFrom(ctx, r, "m", hooks)
		close(pumpDone)
	}()

	cancel()
	<-pumpDone
	<-done
	if err := stream.Err(); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

type blockingReader struct {
	ch chan types.ConverseStreamOutput
}

func (b *blockingReader) Events() <-chan types.ConverseStreamOutput { return b.ch }
func (b *blockingReader) Close() error                              { return nil }
func (b *blockingReader) Err() error                                { return nil }

// Compile-time check: bedrockruntime.ConverseStreamEventStream
// satisfies the eventSource abstraction pump consumes.
var _ eventSource = (*bedrockruntime.ConverseStreamEventStream)(nil)
