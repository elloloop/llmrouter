// Package bedrock implements the llmrouter.Provider interface against
// AWS Bedrock Runtime's ConverseStream API. The provider accepts
// OpenAI-shaped requests (llmrouter.ChatRequest), translates them to
// the Bedrock Converse wire format, then re-encodes the streaming
// event channel as OpenAI-shaped streaming chunks (llmrouter.Chunk).
//
// Bedrock authenticates via the standard AWS credential chain
// (environment, profile, IAM role, etc.), not API keys. Use
// WithRegion to select the Bedrock region; optionally pass
// WithAWSConfig to inject a pre-built aws.Config (useful for tests).
//
// Only the unified Converse / ConverseStream API is supported in v0.1
// of this provider — it lets us send a single body shape across model
// families (Anthropic, Llama, Cohere, Titan, Mistral).
package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	smithyhttp "github.com/aws/smithy-go/transport/http"
	"github.com/google/uuid"

	"github.com/elloloop/llmrouter"
)

const (
	providerName     = "bedrock"
	defaultMaxTokens = 4096
	// regionKey is the llmrouter.Config.Extra key used to store the AWS region.
	regionKey = "bedrock.region"
	// awsConfigKey is the llmrouter.Config.Extra key used to store an
	// optional pre-built aws.Config supplied via WithAWSConfig.
	awsConfigKey = "bedrock.aws_config"
)

// Provider talks to AWS Bedrock Runtime's ConverseStream API. Construct
// with New. The underlying *bedrockruntime.Client is built lazily on
// the first request so that loading the default AWS credential chain
// can use the caller's request-scoped context.
type Provider struct {
	cfg    *llmrouter.Config
	region string
	// awsCfg holds the AWS SDK config used to construct the bedrock
	// client. nil means "load default config on first request".
	awsCfg *aws.Config
	client *bedrockruntime.Client
}

// WithRegion sets the AWS region (e.g. "us-east-1"). Required.
//
// Stored under llmrouter.Config.Extra["bedrock.region"].
func WithRegion(region string) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		if region == "" {
			return errors.New("bedrock region cannot be empty")
		}
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[regionKey] = region
		return nil
	}
}

// WithAWSConfig supplies a pre-built aws.Config. Optional. When
// absent, New defers credential loading until the first request and
// then calls awsconfig.LoadDefaultConfig with the request context.
//
// Stored under llmrouter.Config.Extra["bedrock.aws_config"].
func WithAWSConfig(cfg aws.Config) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		if c.Extra == nil {
			c.Extra = make(map[string]any)
		}
		c.Extra[awsConfigKey] = cfg
		return nil
	}
}

// New builds a Provider from llmrouter options. WithRegion is required;
// WithAPIKey is rejected because Bedrock uses AWS credentials, not API
// keys.
func New(opts ...llmrouter.Option) (*Provider, error) {
	cfg, err := llmrouter.NewConfig(opts...)
	if err != nil {
		return nil, err
	}
	if cfg.APIKey != "" {
		return nil, fmt.Errorf("%w: bedrock does not accept WithAPIKey — use AWS credentials", llmrouter.ErrInvalidConfig)
	}
	region, _ := cfg.Extra[regionKey].(string)
	if region == "" {
		return nil, fmt.Errorf("%w: bedrock requires WithRegion", llmrouter.ErrInvalidConfig)
	}
	p := &Provider{cfg: cfg, region: region}
	if v, ok := cfg.Extra[awsConfigKey]; ok {
		if ac, ok := v.(aws.Config); ok {
			p.awsCfg = &ac
		}
	}
	return p, nil
}

// Name returns the provider id.
func (p *Provider) Name() string { return providerName }

// CompletionStream issues a streaming Bedrock ConverseStream request and
// returns a Stream that yields OpenAI-shaped chunks.
func (p *Provider) CompletionStream(ctx context.Context, req llmrouter.ChatRequest) (*llmrouter.Stream, error) {
	client, err := p.resolveClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("bedrock: aws config: %w", err)
	}
	input := buildConverseStreamInput(req)
	output, err := client.ConverseStream(ctx, input)
	if err != nil {
		return nil, wrapUpstream(err)
	}

	stream, sctx, hooks := llmrouter.NewStream(ctx)
	go pump(sctx, output, req.Model, hooks)
	return stream, nil
}

// resolveClient returns the bedrockruntime client, constructing it
// lazily from the default credential chain on first use.
func (p *Provider) resolveClient(ctx context.Context) (*bedrockruntime.Client, error) {
	if p.client != nil {
		return p.client, nil
	}
	if p.awsCfg != nil {
		ac := *p.awsCfg
		if ac.Region == "" {
			ac.Region = p.region
		}
		p.client = bedrockruntime.NewFromConfig(ac)
		return p.client, nil
	}
	ac, err := awsconfig.LoadDefaultConfig(ctx, awsconfig.WithRegion(p.region))
	if err != nil {
		return nil, err
	}
	p.client = bedrockruntime.NewFromConfig(ac)
	return p.client, nil
}

// buildConverseStreamInput translates an OpenAI-shaped ChatRequest into
// a Bedrock ConverseStreamInput. System messages are lifted to the
// top-level System field; user/assistant messages become Converse
// messages with text content blocks.
func buildConverseStreamInput(req llmrouter.ChatRequest) *bedrockruntime.ConverseStreamInput {
	sys := liftSystem(req.Messages)
	msgs := toConverseMessages(req.Messages)
	modelID := req.Model
	in := &bedrockruntime.ConverseStreamInput{
		ModelId:         &modelID,
		Messages:        msgs,
		InferenceConfig: buildInferenceConfig(req),
	}
	if len(sys) > 0 {
		in.System = sys
	}
	return in
}

// buildInferenceConfig assembles the Converse inference configuration
// from the typed fields on the request, defaulting MaxTokens.
func buildInferenceConfig(req llmrouter.ChatRequest) *types.InferenceConfiguration {
	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	mt := int32(maxTokens)
	cfg := &types.InferenceConfiguration{MaxTokens: &mt}
	if req.Temperature != nil {
		t := float32(*req.Temperature)
		cfg.Temperature = &t
	}
	if req.TopP != nil {
		tp := float32(*req.TopP)
		cfg.TopP = &tp
	}
	if len(req.Stop) > 0 {
		cfg.StopSequences = append(cfg.StopSequences, req.Stop...)
	}
	return cfg
}

// liftSystem extracts system-role messages into Converse system
// content blocks. Empty system messages are ignored.
func liftSystem(messages []llmrouter.Message) []types.SystemContentBlock {
	var out []types.SystemContentBlock
	for _, m := range messages {
		if m.Role != "system" {
			continue
		}
		text := m.PlainText()
		if text == "" {
			continue
		}
		out = append(out, &types.SystemContentBlockMemberText{Value: text})
	}
	return out
}

// toConverseMessages translates non-system messages into Converse
// Message values. Unknown roles fall back to "user". Empty content
// produces a message with an empty text block so the API still sees
// the turn.
func toConverseMessages(messages []llmrouter.Message) []types.Message {
	out := make([]types.Message, 0, len(messages))
	for _, m := range messages {
		if m.Role == "system" {
			continue
		}
		role := mapRole(m.Role)
		text := m.PlainText()
		out = append(out, types.Message{
			Role:    role,
			Content: []types.ContentBlock{&types.ContentBlockMemberText{Value: text}},
		})
	}
	return out
}

// mapRole converts an OpenAI role string to a Bedrock ConversationRole.
// Anything other than "assistant" is treated as user input.
func mapRole(role string) types.ConversationRole {
	if role == "assistant" {
		return types.ConversationRoleAssistant
	}
	return types.ConversationRoleUser
}

// mapStopReason converts a Bedrock StopReason to the OpenAI
// finish_reason vocabulary.
func mapStopReason(r types.StopReason) string {
	switch r {
	case types.StopReasonEndTurn, types.StopReasonStopSequence:
		return "stop"
	case types.StopReasonMaxTokens:
		return "length"
	case types.StopReasonToolUse:
		return "tool_calls"
	default:
		return "stop"
	}
}

// wrapUpstream converts an AWS SDK error into the llmrouter ErrUpstream
// shape. If the underlying error carries an HTTP response (via the
// smithyhttp.ResponseError wrapper) the status code is preserved;
// otherwise StatusCode is 0.
func wrapUpstream(err error) error {
	if err == nil {
		return nil
	}
	out := &llmrouter.ErrUpstream{Provider: providerName, Body: err.Error()}
	var rerr *smithyhttp.ResponseError
	if errors.As(err, &rerr) && rerr.Response != nil {
		out.StatusCode = rerr.Response.StatusCode
	}
	return out
}

// eventSource is the minimal abstraction pump consumes — satisfied by
// *bedrockruntime.ConverseStreamEventStream and by test doubles.
type eventSource interface {
	Events() <-chan types.ConverseStreamOutput
	Close() error
	Err() error
}

// pump reads ConverseStream events and emits OpenAI-shaped chunks via
// hooks until the upstream channel closes, the context cancels, or an
// error occurs. Always calls hooks.Finish.
func pump(ctx context.Context, output *bedrockruntime.ConverseStreamOutput, model string, hooks llmrouter.ProducerHooks) {
	pumpFrom(ctx, output.GetStream(), model, hooks)
}

// pumpFrom is the testable core of pump. Exported only to the package.
func pumpFrom(ctx context.Context, stream eventSource, model string, hooks llmrouter.ProducerHooks) {
	defer stream.Close()

	st := &pumpState{
		chatID:  "chatcmpl-" + uuid.NewString(),
		created: time.Now().Unix(),
		model:   model,
	}

	events := stream.Events()
	for {
		select {
		case <-ctx.Done():
			hooks.Finish(ctx.Err())
			return
		case ev, ok := <-events:
			if !ok {
				if err := stream.Err(); err != nil {
					hooks.Finish(fmt.Errorf("bedrock: stream: %w", err))
					return
				}
				hooks.Finish(nil)
				return
			}
			if done := handleEvent(ev, st, hooks); done {
				if err := stream.Err(); err != nil {
					hooks.Finish(fmt.Errorf("bedrock: stream: %w", err))
					return
				}
				hooks.Finish(nil)
				return
			}
		}
	}
}

// pumpState carries identifiers and usage accumulated across events.
type pumpState struct {
	chatID       string
	created      int64
	model        string
	inputTokens  int
	outputTokens int
	finishReason string
	// stopEmitted records whether the message_stop finish chunk has
	// already been sent; if metadata arrives later with token usage we
	// emit a second, usage-only chunk so the consumer still sees it.
	stopEmitted bool
}

// handleEvent processes a single ConverseStream event and emits the
// corresponding OpenAI-shaped chunk(s). Returns true when the stream
// should terminate (i.e. the consumer has cancelled).
func handleEvent(ev types.ConverseStreamOutput, st *pumpState, hooks llmrouter.ProducerHooks) (terminate bool) {
	switch e := ev.(type) {
	case *types.ConverseStreamOutputMemberMessageStart:
		chunk := newChunk(st, llmrouter.Delta{Role: "assistant"}, "")
		return !hooks.Send(chunk)

	case *types.ConverseStreamOutputMemberContentBlockDelta:
		text, ok := extractTextDelta(e.Value.Delta)
		if !ok || text == "" {
			return false
		}
		chunk := newChunk(st, llmrouter.Delta{Content: text}, "")
		return !hooks.Send(chunk)

	case *types.ConverseStreamOutputMemberMessageStop:
		st.finishReason = mapStopReason(e.Value.StopReason)
		chunk := newChunk(st, llmrouter.Delta{}, st.finishReason)
		if usage := currentUsage(st); usage != nil {
			chunk.Usage = usage
			if raw, err := json.Marshal(chunk); err == nil {
				chunk.Raw = raw
			}
		}
		st.stopEmitted = true
		return !hooks.Send(chunk)

	case *types.ConverseStreamOutputMemberMetadata:
		applyMetadata(e.Value, st)
		// If the finish chunk was already emitted, follow up with a
		// usage-only chunk so the caller still sees token counts.
		if st.stopEmitted {
			usage := currentUsage(st)
			if usage == nil {
				return false
			}
			chunk := newChunk(st, llmrouter.Delta{}, st.finishReason)
			chunk.Usage = usage
			if raw, err := json.Marshal(chunk); err == nil {
				chunk.Raw = raw
			}
			return !hooks.Send(chunk)
		}
		return false

	default:
		// content_block_start, content_block_stop, and unknown event
		// variants are intentionally ignored.
		return false
	}
}

// extractTextDelta pulls the text payload out of a content-block delta,
// returning false for non-text variants.
func extractTextDelta(d types.ContentBlockDelta) (string, bool) {
	td, ok := d.(*types.ContentBlockDeltaMemberText)
	if !ok {
		return "", false
	}
	return td.Value, true
}

// applyMetadata copies token usage from a Bedrock metadata event onto
// the pump state.
func applyMetadata(meta types.ConverseStreamMetadataEvent, st *pumpState) {
	if meta.Usage == nil {
		return
	}
	if meta.Usage.InputTokens != nil {
		st.inputTokens = int(*meta.Usage.InputTokens)
	}
	if meta.Usage.OutputTokens != nil {
		st.outputTokens = int(*meta.Usage.OutputTokens)
	}
}

// newChunk builds a normalized OpenAI-shaped Chunk and pre-populates
// Raw with its JSON form.
func newChunk(st *pumpState, delta llmrouter.Delta, finish string) llmrouter.Chunk {
	c := llmrouter.Chunk{
		ID:      st.chatID,
		Object:  "chat.completion.chunk",
		Created: st.created,
		Model:   st.model,
		Choices: []llmrouter.Choice{{
			Index:        0,
			Delta:        delta,
			FinishReason: finish,
		}},
	}
	if raw, err := json.Marshal(c); err == nil {
		c.Raw = raw
	}
	return c
}

// currentUsage returns the accumulated token usage, or nil if nothing
// is known yet.
func currentUsage(st *pumpState) *llmrouter.Usage {
	if st.inputTokens == 0 && st.outputTokens == 0 {
		return nil
	}
	return &llmrouter.Usage{
		PromptTokens:     st.inputTokens,
		CompletionTokens: st.outputTokens,
		TotalTokens:      st.inputTokens + st.outputTokens,
	}
}
