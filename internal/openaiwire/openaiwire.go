// Package openaiwire holds the translations shared by the providers that send
// a marshaled llmrouter.ChatRequest as an OpenAI-shaped chat-completions
// body. The typed request marshals to OpenAI's field names except for the
// fields below, so each rule lives here once rather than in every provider.
package openaiwire

import (
	"encoding/json"

	"github.com/elloloop/llmrouter"
)

const (
	// keyResponseSchema is ChatRequest.ResponseSchema's JSON name. It exists
	// so a ChatRequest round-trips through JSON; no upstream accepts it.
	keyResponseSchema      = "response_schema"
	keyResponseFormat      = "response_format"
	keyMaxTokens           = "max_tokens"
	keyMaxCompletionTokens = "max_completion_tokens"
)

// DropResponseSchema removes the typed request's response_schema field from
// a body marshaled from a ChatRequest. Every provider that sends such a body
// calls it; Azure OpenAI answers the field with 400 unknown_parameter.
func DropResponseSchema(body map[string]json.RawMessage) {
	delete(body, keyResponseSchema)
}

// SetResponseFormat drops the typed response_schema field and renders
// schema into body as OpenAI's
// response_format={"type":"json_schema","json_schema":{...}} envelope. A
// body that already carries a response_format (a caller's Raw passthrough)
// keeps it: manual JSON-mode setups are not overwritten. A nil schema adds
// nothing.
func SetResponseFormat(body map[string]json.RawMessage, schema *llmrouter.ResponseSchema) error {
	DropResponseSchema(body)
	if schema == nil {
		return nil
	}
	if _, ok := body[keyResponseFormat]; ok {
		return nil
	}
	rf, err := ResponseFormat(schema)
	if err != nil {
		return err
	}
	body[keyResponseFormat] = rf
	return nil
}

// ResponseFormat renders an llmrouter.ResponseSchema into OpenAI's
// response_format envelope. Description is omitted when empty; Schema is
// forwarded as-is.
func ResponseFormat(s *llmrouter.ResponseSchema) (json.RawMessage, error) {
	jsonSchema := map[string]any{
		"name":   s.Name,
		"strict": s.Strict,
	}
	if s.Description != "" {
		jsonSchema["description"] = s.Description
	}
	if len(s.Schema) > 0 {
		jsonSchema["schema"] = json.RawMessage(s.Schema)
	}
	return json.Marshal(map[string]any{
		"type":        "json_schema",
		"json_schema": jsonSchema,
	})
}

// UseMaxCompletionTokens sends the typed MaxTokens as max_completion_tokens.
// OpenAI deprecated max_tokens for chat completions, and its reasoning
// models (the o-series, GPT-5) reject it outright; max_completion_tokens is
// accepted by every current chat model. A body that already names
// max_completion_tokens keeps it and drops the deprecated field.
func UseMaxCompletionTokens(body map[string]json.RawMessage) {
	limit, ok := body[keyMaxTokens]
	if !ok {
		return
	}
	delete(body, keyMaxTokens)
	if _, set := body[keyMaxCompletionTokens]; !set {
		body[keyMaxCompletionTokens] = limit
	}
}
