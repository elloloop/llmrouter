package router

import "strings"

// ModelFamily identifies the vendor of the model itself, independent of
// where it's hosted. Used by the router to pick the right underlying
// provider package for a given (family, platform) combination.
type ModelFamily string

// Known model families. Anything not matched here resolves to FamilyOther,
// which is still routable via platforms that accept arbitrary model ids
// (OpenRouter, Together).
const (
	FamilyOpenAI    ModelFamily = "openai"    // gpt-*, o1-*, o3-*, o4-*, chatgpt-*
	FamilyAnthropic ModelFamily = "anthropic" // claude-*
	FamilyLlama     ModelFamily = "llama"     // llama-*, meta.llama-*, *llama* (case-insensitive)
	FamilyMistral   ModelFamily = "mistral"   // mistral-*, mixtral-*, ministral-*, codestral-*, magistral-*
	FamilyCohere    ModelFamily = "cohere"    // command-*, c4ai-*
	FamilyGemini    ModelFamily = "gemini"    // gemini-*
	FamilyGrok      ModelFamily = "grok"      // grok-*
	FamilyDeepSeek  ModelFamily = "deepseek"  // deepseek-*
	FamilyOther     ModelFamily = "other"     // unknown / opaque
)

// InferFamily returns the ModelFamily implied by the model id. The router
// uses simple prefix matching after stripping any vendor prefix (e.g.
// "anthropic." or "meta.") that the hosting platform might attach.
//
// Matching is case-insensitive. Unknown ids return FamilyOther — the
// router will still resolve them on platforms that accept arbitrary
// model ids (OpenRouter, Together).
func InferFamily(model string) ModelFamily {
	id := strings.ToLower(strings.TrimSpace(model))
	if id == "" {
		return FamilyOther
	}
	// Strip a leading vendor prefix used by hosting platforms.
	// e.g. "anthropic.claude-...", "meta.llama-...", "mistral.mistral-...",
	// "cohere.command-...". Only strip when the prefix matches a known
	// hosting-platform vendor name — model ids like "gemini-1.5-pro"
	// must not be split on their internal dots.
	stripped := id
	for _, vendor := range []string{"anthropic.", "meta.", "mistral.", "cohere.", "amazon.", "ai21.", "stability."} {
		if strings.HasPrefix(id, vendor) {
			stripped = id[len(vendor):]
			break
		}
	}

	switch {
	case hasAnyPrefix(stripped, "gpt-", "gpt4", "o1-", "o1.", "o3-", "o3.", "o4-", "o4.", "chatgpt-", "text-davinci", "text-embedding-3"):
		return FamilyOpenAI
	case strings.HasPrefix(stripped, "claude-") || strings.HasPrefix(stripped, "claude."):
		return FamilyAnthropic
	case strings.HasPrefix(stripped, "llama-") ||
		strings.HasPrefix(stripped, "llama3") ||
		strings.HasPrefix(stripped, "llama2") ||
		strings.HasPrefix(stripped, "llama4") ||
		strings.HasPrefix(id, "meta.llama") ||
		strings.Contains(id, "llama"):
		return FamilyLlama
	case hasAnyPrefix(stripped, "mistral-", "mixtral-", "ministral-", "codestral-", "magistral-", "open-mistral", "open-mixtral", "pixtral-"):
		return FamilyMistral
	case hasAnyPrefix(stripped, "command-", "c4ai-"):
		return FamilyCohere
	case strings.HasPrefix(stripped, "gemini-") || strings.HasPrefix(stripped, "gemini."):
		return FamilyGemini
	case strings.HasPrefix(stripped, "grok-") || strings.HasPrefix(stripped, "grok."):
		return FamilyGrok
	case strings.HasPrefix(stripped, "deepseek-") || strings.HasPrefix(stripped, "deepseek."):
		return FamilyDeepSeek
	}
	return FamilyOther
}

// hasAnyPrefix reports whether s starts with any of the given prefixes.
func hasAnyPrefix(s string, prefixes ...string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}
