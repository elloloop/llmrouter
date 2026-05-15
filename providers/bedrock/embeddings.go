// Embeddings for AWS Bedrock.
//
// Bedrock embedding models are invoked through the InvokeModel API
// (not Converse). Each model family has its own request/response wire
// shape; we currently support two families:
//
//   - Amazon Titan Embeddings: amazon.titan-embed-text-v1,
//     amazon.titan-embed-text-v2:0. Single input per call — we loop
//     sequentially when callers pass batches.
//   - Cohere on Bedrock: cohere.embed-english-v3,
//     cohere.embed-multilingual-v3. Native batch support (up to 96).
//
// Amazon Titan Multimodal Embeddings (amazon.titan-embed-image-v1)
// are intentionally not supported in v0.3 — callers receive an
// ErrUpstream noting the limitation.
package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/elloloop/llmrouter"
)

const (
	familyTitan  = "titan"
	familyCohere = "cohere"

	contentTypeJSON = "application/json"
)

// Embed produces vector embeddings for the inputs in req. The model id
// determines the wire format: Titan models accept one input per
// InvokeModel call (we batch client-side), Cohere models accept up to
// 96 in a single call.
func (p *Provider) Embed(ctx context.Context, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	client, err := p.resolveClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("bedrock: aws config: %w", err)
	}

	family := bedrockFamily(req.Model)
	switch family {
	case familyTitan:
		if strings.Contains(req.Model, "image") {
			return nil, &llmrouter.ErrUpstream{
				Provider:   providerName,
				StatusCode: 0,
				Body:       "multimodal embeddings not supported in v0.3",
			}
		}
		return p.embedTitan(ctx, client, req)
	case familyCohere:
		return p.embedCohere(ctx, client, req)
	default:
		return nil, &llmrouter.ErrUpstream{
			Provider:   providerName,
			StatusCode: 0,
			Body:       "unknown embedding model family: " + req.Model,
		}
	}
}

// embedTitan calls InvokeModel once per input string (Titan accepts a
// single inputText per call) and concatenates the resulting vectors.
func (p *Provider) embedTitan(ctx context.Context, client *bedrockruntime.Client, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	vectors := make([][]float32, 0, len(req.Inputs))
	totalPromptTokens := 0
	for _, in := range req.Inputs {
		body, err := buildTitanRequest(in, req.Dimensions, req.Model)
		if err != nil {
			return nil, fmt.Errorf("bedrock: encode titan request: %w", err)
		}
		out, err := client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
			ModelId:     aws.String(req.Model),
			ContentType: aws.String(contentTypeJSON),
			Accept:      aws.String(contentTypeJSON),
			Body:        body,
		})
		if err != nil {
			return nil, wrapUpstream(err)
		}
		vec, tokens, err := parseTitanResponse(out.Body)
		if err != nil {
			return nil, err
		}
		vectors = append(vectors, vec)
		totalPromptTokens += tokens
	}
	resp := &llmrouter.EmbedResponse{
		Model:      req.Model,
		Embeddings: vectors,
	}
	if totalPromptTokens > 0 {
		resp.Usage = &llmrouter.Usage{
			PromptTokens: totalPromptTokens,
			TotalTokens:  totalPromptTokens,
		}
	}
	return resp, nil
}

// embedCohere issues a single InvokeModel call with all inputs.
// Cohere on Bedrock does not return token counts, so Usage is nil.
func (p *Provider) embedCohere(ctx context.Context, client *bedrockruntime.Client, req llmrouter.EmbedRequest) (*llmrouter.EmbedResponse, error) {
	body, err := buildCohereRequest(req.Inputs, req.TaskType)
	if err != nil {
		return nil, fmt.Errorf("bedrock: encode cohere request: %w", err)
	}
	out, err := client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(req.Model),
		ContentType: aws.String(contentTypeJSON),
		Accept:      aws.String(contentTypeJSON),
		Body:        body,
	})
	if err != nil {
		return nil, wrapUpstream(err)
	}
	vectors, err := parseCohereResponse(out.Body)
	if err != nil {
		return nil, err
	}
	return &llmrouter.EmbedResponse{
		Model:      req.Model,
		Embeddings: vectors,
	}, nil
}

// bedrockFamily detects the embedding-model family from a model id
// prefix. Returns "" when no family matches.
func bedrockFamily(modelID string) string {
	switch {
	case strings.HasPrefix(modelID, "amazon.titan-embed"):
		return familyTitan
	case strings.HasPrefix(modelID, "cohere.embed"):
		return familyCohere
	default:
		return ""
	}
}

// titanRequest mirrors Amazon Titan's embedding request body. The
// dimensions and normalize fields are honoured only by the v2 model;
// older v1 models silently ignore them but accept them in the body.
type titanRequest struct {
	InputText  string `json:"inputText"`
	Dimensions int    `json:"dimensions,omitempty"`
	Normalize  *bool  `json:"normalize,omitempty"`
}

// buildTitanRequest encodes a single Titan InvokeModel body. The
// Dimensions field is only emitted when non-zero AND the model is v2
// (v1 doesn't recognise the parameter — we keep its body minimal).
func buildTitanRequest(text string, dimensions int, modelID string) ([]byte, error) {
	body := titanRequest{InputText: text}
	if isTitanV2(modelID) {
		t := true
		body.Normalize = &t
		if dimensions > 0 {
			body.Dimensions = dimensions
		}
	}
	return json.Marshal(body)
}

// isTitanV2 reports whether modelID is the v2 family, which supports
// the dimensions/normalize parameters.
func isTitanV2(modelID string) bool {
	return strings.HasPrefix(modelID, "amazon.titan-embed-text-v2")
}

// titanResponse is Amazon Titan's embedding response body.
type titanResponse struct {
	Embedding           []float32 `json:"embedding"`
	InputTextTokenCount int       `json:"inputTextTokenCount"`
}

// parseTitanResponse decodes a Titan InvokeModel response body and
// returns the vector plus the input token count.
func parseTitanResponse(body []byte) ([]float32, int, error) {
	var resp titanResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, 0, fmt.Errorf("bedrock: decode titan response: %w", err)
	}
	if len(resp.Embedding) == 0 {
		return nil, 0, fmt.Errorf("bedrock: titan response missing embedding")
	}
	return resp.Embedding, resp.InputTextTokenCount, nil
}

// cohereRequest mirrors Cohere on Bedrock's embedding request body.
type cohereRequest struct {
	Texts          []string `json:"texts"`
	InputType      string   `json:"input_type"`
	EmbeddingTypes []string `json:"embedding_types"`
	Truncate       string   `json:"truncate,omitempty"`
}

// buildCohereRequest encodes a single Cohere InvokeModel body
// covering all inputs in the batch.
func buildCohereRequest(inputs []string, taskType string) ([]byte, error) {
	body := cohereRequest{
		Texts:          inputs,
		InputType:      mapCohereInputType(taskType),
		EmbeddingTypes: []string{"float"},
		Truncate:       "END",
	}
	return json.Marshal(body)
}

// cohereResponse is Cohere on Bedrock's embedding response body. The
// embeddings field can be either a top-level []float32 array (legacy)
// or, when embedding_types is set, an object keyed by type. We
// request "float" and decode the object form first, then fall back.
type cohereResponse struct {
	// Embeddings holds either the legacy array form or the typed form;
	// we decode into json.RawMessage and dispatch in parseCohereResponse.
	Embeddings json.RawMessage `json:"embeddings"`
}

// parseCohereResponse decodes a Cohere InvokeModel response body and
// returns the per-input vectors.
func parseCohereResponse(body []byte) ([][]float32, error) {
	var resp cohereResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("bedrock: decode cohere response: %w", err)
	}
	if len(resp.Embeddings) == 0 {
		return nil, fmt.Errorf("bedrock: cohere response missing embeddings")
	}

	// Try the typed (object) form first: {"float": [[...],[...]]}.
	var typed struct {
		Float [][]float32 `json:"float"`
	}
	if err := json.Unmarshal(resp.Embeddings, &typed); err == nil && len(typed.Float) > 0 {
		return typed.Float, nil
	}

	// Fall back to the legacy array form: [[...],[...]].
	var legacy [][]float32
	if err := json.Unmarshal(resp.Embeddings, &legacy); err != nil {
		return nil, fmt.Errorf("bedrock: decode cohere embeddings: %w", err)
	}
	if len(legacy) == 0 {
		return nil, fmt.Errorf("bedrock: cohere response has no vectors")
	}
	return legacy, nil
}

// mapCohereInputType translates the OpenAI/Vertex TaskType vocabulary
// to Cohere's "input_type" enum. Mirrors the standalone Cohere
// provider's mapping so behaviour is consistent across surfaces.
func mapCohereInputType(taskType string) string {
	switch taskType {
	case "RETRIEVAL_QUERY":
		return "search_query"
	case "RETRIEVAL_DOCUMENT":
		return "search_document"
	case "SEMANTIC_SIMILARITY":
		return "classification"
	case "CLASSIFICATION":
		return "classification"
	case "CLUSTERING":
		return "clustering"
	default:
		return "search_document"
	}
}
