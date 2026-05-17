package router

import "strings"

// bedrockModelID translates a vendor-neutral model id into the prefixed
// form AWS Bedrock expects. When the input already carries a known
// Bedrock vendor prefix (anthropic., meta., mistral., cohere.) it is
// returned unchanged. Unknown ids fall through unchanged so Bedrock can
// surface its own model-not-found error.
//
// The translation table covers the common cases; this is intentionally
// not exhaustive because Bedrock adds new model versions frequently and
// we prefer pass-through over stale hard-coded mappings.
func bedrockModelID(family ModelFamily, model string) string {
	id := strings.TrimSpace(model)
	if id == "" {
		return id
	}
	// If the id already has any of the well-known Bedrock vendor prefixes,
	// trust the caller and pass through.
	for _, p := range []string{"anthropic.", "meta.", "mistral.", "cohere.", "amazon.", "ai21.", "stability."} {
		if strings.HasPrefix(id, p) {
			return id
		}
	}
	switch family {
	case FamilyAnthropic:
		return translateAnthropicBedrock(id)
	case FamilyLlama:
		return translateLlamaBedrock(id)
	case FamilyMistral:
		return translateMistralBedrock(id)
	case FamilyCohere:
		return translateCohereBedrock(id)
	}
	return id
}

// translateAnthropicBedrock maps a vendor-neutral Claude id to its
// Bedrock equivalent. Unknown ids return unchanged.
func translateAnthropicBedrock(id string) string {
	switch id {
	case "claude-3-5-sonnet", "claude-3-5-sonnet-latest", "claude-3-5-sonnet-20241022":
		return "anthropic.claude-3-5-sonnet-20241022-v2:0"
	case "claude-3-5-sonnet-20240620":
		return "anthropic.claude-3-5-sonnet-20240620-v1:0"
	case "claude-3-5-haiku", "claude-3-5-haiku-latest", "claude-3-5-haiku-20241022":
		return "anthropic.claude-3-5-haiku-20241022-v1:0"
	case "claude-3-opus", "claude-3-opus-latest", "claude-3-opus-20240229":
		return "anthropic.claude-3-opus-20240229-v1:0"
	case "claude-3-sonnet", "claude-3-sonnet-20240229":
		return "anthropic.claude-3-sonnet-20240229-v1:0"
	case "claude-3-haiku", "claude-3-haiku-20240307":
		return "anthropic.claude-3-haiku-20240307-v1:0"
	}
	return id
}

// translateLlamaBedrock maps a vendor-neutral Llama id to its Bedrock
// equivalent. Unknown ids return unchanged.
func translateLlamaBedrock(id string) string {
	switch id {
	case "llama-3-1-70b-instruct", "llama-3.1-70b-instruct":
		return "meta.llama3-1-70b-instruct-v1:0"
	case "llama-3-1-8b-instruct", "llama-3.1-8b-instruct":
		return "meta.llama3-1-8b-instruct-v1:0"
	case "llama-3-1-405b-instruct", "llama-3.1-405b-instruct":
		return "meta.llama3-1-405b-instruct-v1:0"
	case "llama-3-2-1b-instruct", "llama-3.2-1b-instruct":
		return "meta.llama3-2-1b-instruct-v1:0"
	case "llama-3-2-3b-instruct", "llama-3.2-3b-instruct":
		return "meta.llama3-2-3b-instruct-v1:0"
	}
	return id
}

// translateMistralBedrock maps a vendor-neutral Mistral id to its
// Bedrock equivalent. Unknown ids return unchanged.
func translateMistralBedrock(id string) string {
	switch id {
	case "mistral-large", "mistral-large-latest", "mistral-large-2407":
		return "mistral.mistral-large-2407-v1:0"
	case "mistral-large-2402":
		return "mistral.mistral-large-2402-v1:0"
	case "mistral-small", "mistral-small-latest":
		return "mistral.mistral-small-2402-v1:0"
	case "mixtral-8x7b-instruct":
		return "mistral.mixtral-8x7b-instruct-v0:1"
	}
	return id
}

// translateCohereBedrock maps a vendor-neutral Cohere id to its Bedrock
// equivalent. Unknown ids return unchanged.
func translateCohereBedrock(id string) string {
	switch id {
	case "command-r-plus":
		return "cohere.command-r-plus-v1:0"
	case "command-r":
		return "cohere.command-r-v1:0"
	}
	return id
}

// vertexAnthropicModelID translates a vendor-neutral Claude id into the
// @-versioned form that Vertex AI's Model Garden expects (e.g.
// "claude-3-5-sonnet-v2@20241022"). Unknown ids pass through.
func vertexAnthropicModelID(model string) string {
	id := strings.TrimSpace(model)
	if id == "" || strings.Contains(id, "@") {
		// Already versioned, trust the caller.
		return id
	}
	switch id {
	case "claude-3-5-sonnet", "claude-3-5-sonnet-latest", "claude-3-5-sonnet-20241022":
		return "claude-3-5-sonnet-v2@20241022"
	case "claude-3-5-sonnet-20240620":
		return "claude-3-5-sonnet@20240620"
	case "claude-3-5-haiku", "claude-3-5-haiku-latest", "claude-3-5-haiku-20241022":
		return "claude-3-5-haiku@20241022"
	case "claude-3-opus", "claude-3-opus-latest", "claude-3-opus-20240229":
		return "claude-3-opus@20240229"
	case "claude-3-sonnet", "claude-3-sonnet-20240229":
		return "claude-3-sonnet@20240229"
	case "claude-3-haiku", "claude-3-haiku-20240307":
		return "claude-3-haiku@20240307"
	}
	return id
}
