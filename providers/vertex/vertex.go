// Package vertex implements the llmrouter.Provider interface against
// Google Vertex AI's Gemini streaming generateContent endpoint. The
// provider accepts OpenAI-shaped requests (llmrouter.ChatRequest),
// translates them to Vertex's GenerateContent request shape via the
// official google.golang.org/genai SDK, then re-encodes the streamed
// GenerateContentResponse events as OpenAI-shaped streaming chunks
// (llmrouter.Chunk).
//
// Authentication uses Application Default Credentials (ADC) by default;
// service-account JSON bytes may be supplied via WithCredentialsJSON.
package vertex

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"strings"
	"time"

	"cloud.google.com/go/auth/credentials"
	"github.com/google/uuid"
	"google.golang.org/genai"

	"github.com/elloloop/llmrouter"
)

// streamSeq is the iterator type returned by genai.Models.GenerateContentStream.
type streamSeq = iter.Seq2[*genai.GenerateContentResponse, error]

const (
	providerName     = "vertex"
	defaultMaxTokens = 4096

	// extraProjectKey is the llmrouter.Config.Extra key for the GCP project id.
	extraProjectKey = "vertex.project"
	// extraRegionKey is the llmrouter.Config.Extra key for the GCP region.
	extraRegionKey = "vertex.region"
	// extraCredentialsJSONKey is the Config.Extra key for an optional ADC override.
	extraCredentialsJSONKey = "vertex.credentialsJSON"

	roleUser      = "user"
	roleAssistant = "assistant"
	roleSystem    = "system"
	roleModel     = "model"

	finishStop          = "stop"
	finishLength        = "length"
	finishContentFilter = "content_filter"

	objectChunk = "chat.completion.chunk"

	// cloudPlatformScope is the OAuth scope required by Vertex AI.
	cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"
)

// Provider talks to Vertex AI Gemini streaming generateContent.
// Construct with New.
type Provider struct {
	cfg    *llmrouter.Config
	client *genai.Client
}

// New builds a Provider from llmrouter options. WithProject and WithRegion
// are required. WithAPIKey is rejected — Vertex authenticates via ADC.
func New(opts ...llmrouter.Option) (*Provider, error) {
	cfg, err := llmrouter.NewConfig(opts...)
	if err != nil {
		return nil, err
	}
	if cfg.APIKey != "" {
		return nil, fmt.Errorf("%w: vertex does not accept WithAPIKey; use ADC or WithCredentialsJSON", llmrouter.ErrInvalidConfig)
	}
	project, _ := stringExtra(cfg, extraProjectKey)
	if project == "" {
		return nil, fmt.Errorf("%w: vertex requires WithProject", llmrouter.ErrInvalidConfig)
	}
	region, _ := stringExtra(cfg, extraRegionKey)
	if region == "" {
		return nil, fmt.Errorf("%w: vertex requires WithRegion", llmrouter.ErrInvalidConfig)
	}

	clientCfg := &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Project:  project,
		Location: region,
	}
	if cfg.HTTPClient != nil {
		clientCfg.HTTPClient = cfg.HTTPClient
	}
	if credsJSON, ok := bytesExtra(cfg, extraCredentialsJSONKey); ok && len(credsJSON) > 0 {
		creds, derr := credentials.DetectDefault(&credentials.DetectOptions{
			CredentialsJSON: credsJSON,
			Scopes:          []string{cloudPlatformScope},
		})
		if derr != nil {
			return nil, fmt.Errorf("vertex: parse credentials json: %w", derr)
		}
		clientCfg.Credentials = creds
	}

	client, err := genai.NewClient(context.Background(), clientCfg)
	if err != nil {
		return nil, fmt.Errorf("vertex: create genai client: %w", err)
	}
	return &Provider{cfg: cfg, client: client}, nil
}

// Name returns the provider id.
func (p *Provider) Name() string { return providerName }

// WithProject sets the GCP project id (required). The value is stored in
// llmrouter.Config.Extra under the key "vertex.project".
func WithProject(project string) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		project = strings.TrimSpace(project)
		if project == "" {
			return errors.New("vertex project cannot be empty")
		}
		ensureExtra(c)
		c.Extra[extraProjectKey] = project
		return nil
	}
}

// WithRegion sets the GCP region/location (required). The value is stored
// in llmrouter.Config.Extra under the key "vertex.region".
func WithRegion(region string) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		region = strings.TrimSpace(region)
		if region == "" {
			return errors.New("vertex region cannot be empty")
		}
		ensureExtra(c)
		c.Extra[extraRegionKey] = region
		return nil
	}
}

// WithCredentialsJSON supplies service-account credentials JSON bytes. If
// omitted, Application Default Credentials are used. The bytes are stored
// in llmrouter.Config.Extra under "vertex.credentialsJSON".
func WithCredentialsJSON(jsonBytes []byte) llmrouter.Option {
	return func(c *llmrouter.Config) error {
		if len(jsonBytes) == 0 {
			return errors.New("vertex credentials json cannot be empty")
		}
		ensureExtra(c)
		c.Extra[extraCredentialsJSONKey] = jsonBytes
		return nil
	}
}

// ensureExtra lazily allocates Config.Extra so With* options can write to it.
func ensureExtra(c *llmrouter.Config) {
	if c.Extra == nil {
		c.Extra = make(map[string]any)
	}
}

// stringExtra retrieves a string from Config.Extra by key, returning ok=false
// when missing or wrong type.
func stringExtra(c *llmrouter.Config, key string) (string, bool) {
	if c == nil || c.Extra == nil {
		return "", false
	}
	v, ok := c.Extra[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// bytesExtra retrieves a []byte from Config.Extra by key.
func bytesExtra(c *llmrouter.Config, key string) ([]byte, bool) {
	if c == nil || c.Extra == nil {
		return nil, false
	}
	v, ok := c.Extra[key]
	if !ok {
		return nil, false
	}
	b, ok := v.([]byte)
	return b, ok
}

// CompletionStream issues a streaming Vertex generateContent request and
// returns a Stream that yields OpenAI-shaped chunks.
func (p *Provider) CompletionStream(ctx context.Context, req llmrouter.ChatRequest) (*llmrouter.Stream, error) {
	contents, systemInstruction := translateMessages(req.Messages)
	genCfg := buildGenerateContentConfig(req, systemInstruction)

	stream, sctx, hooks := llmrouter.NewStream(ctx)
	seq := p.client.Models.GenerateContentStream(sctx, req.Model, contents, genCfg)
	go pump(sctx, seq, req.Model, hooks)
	return stream, nil
}

// translateMessages converts OpenAI-shaped messages into Vertex Content
// objects, lifting consecutive "system" messages out into a single
// SystemInstruction. Non-system roles map: user→user, assistant→model.
// Other roles are passed through verbatim.
func translateMessages(msgs []llmrouter.Message) ([]*genai.Content, *genai.Content) {
	contents := make([]*genai.Content, 0, len(msgs))
	var systemBuf string
	for _, m := range msgs {
		if m.Role == roleSystem {
			if systemBuf != "" {
				systemBuf += "\n\n"
			}
			systemBuf += m.PlainText()
			continue
		}
		role := mapRequestRole(m.Role)
		text := m.PlainText()
		contents = append(contents, &genai.Content{
			Role:  role,
			Parts: []*genai.Part{{Text: text}},
		})
	}
	var systemInstruction *genai.Content
	if systemBuf != "" {
		systemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: systemBuf}},
		}
	}
	return contents, systemInstruction
}

// mapRequestRole converts an OpenAI role to a Vertex Content.Role value.
func mapRequestRole(role string) string {
	switch role {
	case roleAssistant:
		return roleModel
	case roleUser:
		return roleUser
	default:
		return role
	}
}

// mapResponseRole converts a Vertex content role (e.g. "model") to the
// OpenAI role vocabulary used in Delta.Role.
func mapResponseRole(role string) string {
	switch role {
	case roleModel:
		return roleAssistant
	case "":
		return roleAssistant
	default:
		return role
	}
}

// buildGenerateContentConfig assembles GenerationConfig from the inbound
// ChatRequest, applying defaults where the caller didn't specify a value.
func buildGenerateContentConfig(req llmrouter.ChatRequest, systemInstruction *genai.Content) *genai.GenerateContentConfig {
	cfg := &genai.GenerateContentConfig{}
	cfg.MaxOutputTokens = int32(req.MaxTokens)
	if cfg.MaxOutputTokens <= 0 {
		cfg.MaxOutputTokens = defaultMaxTokens
	}
	if req.Temperature != nil {
		t := float32(*req.Temperature)
		cfg.Temperature = &t
	}
	if req.TopP != nil {
		p := float32(*req.TopP)
		cfg.TopP = &p
	}
	if len(req.Stop) > 0 {
		cfg.StopSequences = append(cfg.StopSequences, req.Stop...)
	}
	if systemInstruction != nil {
		cfg.SystemInstruction = systemInstruction
	}
	applyResponseSchema(cfg, req.ResponseSchema)
	return cfg
}

// applyResponseSchema wires an llmrouter.ResponseSchema into the Vertex
// GenerateContentConfig. Vertex supports schema-coerced output via
// ResponseMIMEType="application/json" + a typed ResponseSchema. We parse
// the caller's raw JSON Schema bytes into *genai.Schema; if the parse
// fails we still set ResponseMIMEType so the model emits JSON, but leave
// ResponseSchema nil (best-effort fallback — the model may produce JSON
// without strict shape constraints).
func applyResponseSchema(cfg *genai.GenerateContentConfig, schema *llmrouter.ResponseSchema) {
	if schema == nil {
		return
	}
	cfg.ResponseMIMEType = "application/json"
	if len(schema.Schema) == 0 {
		return
	}
	var parsed genai.Schema
	if err := json.Unmarshal(schema.Schema, &parsed); err != nil {
		// Best-effort fallback: keep MIME type so output is JSON-ish.
		return
	}
	cfg.ResponseSchema = &parsed
}

// mapFinishReason translates a Vertex FinishReason to the OpenAI
// finish_reason vocabulary. Unknown reasons map to "stop" so consumers
// always see a terminal value.
func mapFinishReason(r genai.FinishReason) string {
	switch r {
	case genai.FinishReasonStop:
		return finishStop
	case genai.FinishReasonMaxTokens:
		return finishLength
	case genai.FinishReasonSafety,
		genai.FinishReasonRecitation,
		genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII,
		genai.FinishReasonImageSafety,
		genai.FinishReasonImageProhibitedContent:
		return finishContentFilter
	case "":
		return ""
	default:
		return finishStop
	}
}

// pumpState carries identifiers and accumulators across iterator events.
type pumpState struct {
	chatID       string
	created      int64
	model        string
	rolePrimerOK bool
	promptTokens int
	candTokens   int
}

// pump drains the GenerateContentStream iterator and emits OpenAI-shaped
// chunks via hooks until the upstream terminates, the iterator errors, or
// the context cancels. Always calls hooks.Finish exactly once.
func pump(ctx context.Context, seq streamSeq, model string, hooks llmrouter.ProducerHooks) {
	st := &pumpState{
		chatID:  "chatcmpl-" + uuid.NewString(),
		created: time.Now().Unix(),
		model:   model,
	}
	var terminalErr error
	seq(func(resp *genai.GenerateContentResponse, err error) bool {
		if ctx.Err() != nil {
			terminalErr = ctx.Err()
			return false
		}
		if err != nil {
			terminalErr = wrapSDKError(err)
			return false
		}
		if resp == nil {
			return true
		}
		if !handleResponse(ctx, resp, st, hooks) {
			return false
		}
		return true
	})
	hooks.Finish(terminalErr)
}

// handleResponse emits the chunk(s) produced by one streamed response.
// Returns false if downstream cancelled.
func handleResponse(ctx context.Context, resp *genai.GenerateContentResponse, st *pumpState, hooks llmrouter.ProducerHooks) bool {
	// Update token counters whenever usage is reported (Vertex may emit
	// usage on the final event only, but we update opportunistically).
	if resp.UsageMetadata != nil {
		if resp.UsageMetadata.PromptTokenCount > 0 {
			st.promptTokens = int(resp.UsageMetadata.PromptTokenCount)
		}
		if resp.UsageMetadata.CandidatesTokenCount > 0 {
			st.candTokens = int(resp.UsageMetadata.CandidatesTokenCount)
		}
	}

	if len(resp.Candidates) == 0 {
		return true
	}
	cand := resp.Candidates[0]

	// Emit a role primer once, on the first candidate-bearing response.
	if !st.rolePrimerOK {
		role := ""
		if cand.Content != nil {
			role = mapResponseRole(cand.Content.Role)
		}
		if role == "" {
			role = roleAssistant
		}
		primer := newChunk(st, llmrouter.Delta{Role: role}, "")
		if !sendWithRaw(ctx, hooks, primer) {
			return false
		}
		st.rolePrimerOK = true
	}

	// Emit one content chunk per non-empty text part.
	if cand.Content != nil {
		for _, part := range cand.Content.Parts {
			if part == nil || part.Text == "" {
				continue
			}
			chunk := newChunk(st, llmrouter.Delta{Content: part.Text}, "")
			if !sendWithRaw(ctx, hooks, chunk) {
				return false
			}
		}
	}

	// Emit a finish chunk when the candidate carries a FinishReason.
	finish := mapFinishReason(cand.FinishReason)
	if finish != "" {
		chunk := newChunk(st, llmrouter.Delta{}, finish)
		chunk.Usage = currentUsage(st)
		if !sendWithRaw(ctx, hooks, chunk) {
			return false
		}
	}
	return true
}

// newChunk constructs an OpenAI-shaped Chunk skeleton.
func newChunk(st *pumpState, delta llmrouter.Delta, finish string) llmrouter.Chunk {
	return llmrouter.Chunk{
		ID:      st.chatID,
		Object:  objectChunk,
		Created: st.created,
		Model:   st.model,
		Choices: []llmrouter.Choice{{
			Index:        0,
			Delta:        delta,
			FinishReason: finish,
		}},
	}
}

// sendWithRaw marshals the chunk into Chunk.Raw (so consumers can forward
// bytes verbatim) and pushes it to the consumer.
func sendWithRaw(_ context.Context, hooks llmrouter.ProducerHooks, c llmrouter.Chunk) bool {
	if raw, err := json.Marshal(c); err == nil {
		c.Raw = raw
	}
	return hooks.Send(c)
}

// currentUsage returns accumulated token counts, or nil if neither side
// has reported anything yet.
func currentUsage(st *pumpState) *llmrouter.Usage {
	if st.promptTokens == 0 && st.candTokens == 0 {
		return nil
	}
	return &llmrouter.Usage{
		PromptTokens:     st.promptTokens,
		CompletionTokens: st.candTokens,
		TotalTokens:      st.promptTokens + st.candTokens,
	}
}

// wrapSDKError converts a genai SDK error into an llmrouter.ErrUpstream.
// If the underlying type carries an HTTP status code, it is propagated.
func wrapSDKError(err error) error {
	if err == nil {
		return nil
	}
	var apiErr genai.APIError
	if errors.As(err, &apiErr) {
		return &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: apiErr.Code,
			Body:       apiErr.Message,
		}
	}
	return &llmrouter.ErrUpstream{
		Provider:   providerName,
		StatusCode: 0,
		Body:       err.Error(),
	}
}
