package router

// Platform identifies a hosting target. Pass PlatformAuto to let the
// router pick the best available based on Credentials + env vars.
type Platform string

// Supported hosting platforms.
const (
	// PlatformAuto picks the first platform from the family's preference
	// list for which credentials are present.
	PlatformAuto Platform = ""
	// PlatformDirect uses the model vendor's native API
	// (api.openai.com, api.anthropic.com, api.mistral.ai, ...).
	PlatformDirect Platform = "direct"
	// PlatformBedrock uses AWS Bedrock Runtime ConverseStream.
	PlatformBedrock Platform = "bedrock"
	// PlatformVertex uses Google Vertex AI / Model Garden.
	PlatformVertex Platform = "vertex"
	// PlatformAzure uses Azure AI Foundry; the router picks the right
	// sub-provider (azureopenai / azureanthropic / azureserverless) based
	// on the model family.
	PlatformAzure Platform = "azure"
	// PlatformOpenRouter uses OpenRouter as a fan-out proxy.
	PlatformOpenRouter Platform = "openrouter"
	// PlatformTogether uses Together AI's hosted-OSS-models offering.
	PlatformTogether Platform = "together"
	// PlatformGroq uses Groq's low-latency hosted inference.
	PlatformGroq Platform = "groq"
	// PlatformFireworks uses Fireworks AI.
	PlatformFireworks Platform = "fireworks"
	// PlatformCerebras uses Cerebras Cloud Inference.
	PlatformCerebras Platform = "cerebras"
	// PlatformDeepSeek uses DeepSeek's first-party API.
	PlatformDeepSeek Platform = "deepseek"
	// PlatformPerplexity uses Perplexity's sonar-family API.
	PlatformPerplexity Platform = "perplexity"
	// PlatformxAI uses xAI's Grok API.
	PlatformxAI Platform = "xai"
)

// autoPreference is the order in which PlatformAuto considers platforms.
// The router picks the first one in this list that (a) supports the
// inferred family and (b) has the required credentials.
var autoPreference = []Platform{
	PlatformDirect,
	PlatformBedrock,
	PlatformVertex,
	PlatformAzure,
	PlatformOpenRouter,
	PlatformTogether,
	PlatformGroq,
	PlatformFireworks,
	PlatformCerebras,
	PlatformDeepSeek,
	PlatformPerplexity,
	PlatformxAI,
}

// SupportedPlatforms returns the platforms that can serve the given
// family. The first element is the default for PlatformAuto when all
// credentials are present.
func SupportedPlatforms(family ModelFamily) []Platform {
	switch family {
	case FamilyOpenAI:
		return []Platform{PlatformDirect, PlatformAzure, PlatformOpenRouter}
	case FamilyAnthropic:
		return []Platform{PlatformDirect, PlatformBedrock, PlatformVertex, PlatformAzure, PlatformOpenRouter}
	case FamilyLlama:
		return []Platform{PlatformBedrock, PlatformVertex, PlatformAzure, PlatformOpenRouter, PlatformTogether, PlatformGroq, PlatformFireworks, PlatformCerebras}
	case FamilyMistral:
		return []Platform{PlatformDirect, PlatformBedrock, PlatformAzure, PlatformOpenRouter, PlatformTogether, PlatformGroq, PlatformFireworks}
	case FamilyCohere:
		return []Platform{PlatformDirect, PlatformBedrock, PlatformAzure}
	case FamilyGemini:
		return []Platform{PlatformDirect, PlatformVertex, PlatformOpenRouter}
	case FamilyGrok:
		return []Platform{PlatformDirect, PlatformxAI, PlatformOpenRouter}
	case FamilyDeepSeek:
		return []Platform{PlatformDirect, PlatformDeepSeek, PlatformOpenRouter, PlatformTogether, PlatformFireworks}
	case FamilyOther:
		return []Platform{PlatformAzure, PlatformOpenRouter, PlatformTogether}
	}
	return nil
}

// isSupported reports whether the given platform can serve the family.
func isSupported(family ModelFamily, platform Platform) bool {
	for _, p := range SupportedPlatforms(family) {
		if p == platform {
			return true
		}
	}
	return false
}
