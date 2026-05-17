package router

import (
	"net/http"
	"os"
	"strings"
)

// Credentials carries per-platform authentication + config. The router
// picks the relevant fields based on the chosen Platform. Empty fields
// fall back to environment variables (see DefaultEnvVars).
type Credentials struct {
	// APIKey is the generic API key used by direct platforms and most
	// OpenAI-compatible proxies (OpenRouter, Together, Groq, Fireworks,
	// Cerebras, DeepSeek, xAI, Perplexity). When empty, the router falls
	// back to a family- or platform-specific environment variable.
	APIKey string

	// Azure-specific fields. AzureBaseURL is the resource hostname
	// (https://<resource>.openai.azure.com for azureopenai,
	// https://<resource>.services.ai.azure.com for azureanthropic /
	// azureserverless). AzureDeployment is the deployment name from the
	// Foundry portal; AzureAPIVersion is the api-version query param.
	// AzureAPIKey overrides APIKey for Azure-only setups.
	AzureBaseURL    string
	AzureDeployment string
	AzureAPIVersion string
	AzureAPIKey     string

	// AWSRegion is the AWS region used by AWS Bedrock. AWS credentials
	// themselves are loaded by the AWS SDK from the standard credential
	// chain (env vars, profile, IAM role); the router does not handle
	// access-key wiring directly.
	AWSRegion string

	// GCP-specific. GCPProject + GCPRegion select the Vertex AI tenant.
	// GCPAccessToken is optional; when empty, Application Default
	// Credentials are used by the underlying provider.
	GCPProject     string
	GCPRegion      string
	GCPAccessToken string
}

// EnvVars is the lookup table the router uses to resolve missing
// Credentials fields from the process environment. Callers can override
// this global to introduce custom env var names (e.g. for multi-tenant
// CI runners).
type EnvVars struct {
	// Family-keyed API keys for direct platforms.
	OpenAIAPIKey    string
	AnthropicAPIKey string
	MistralAPIKey   string
	CohereAPIKey    string
	GoogleAPIKey    string // Gemini direct
	GrokAPIKey      string // xAI direct
	DeepSeekAPIKey  string

	// Platform-keyed API keys for fan-out platforms.
	OpenRouterAPIKey string
	TogetherAPIKey   string
	GroqAPIKey       string
	FireworksAPIKey  string
	CerebrasAPIKey   string
	PerplexityAPIKey string

	// Azure / AWS / GCP.
	AzureBaseURL string
	AzureAPIKey  string
	AWSRegion    string // primary
	AWSRegionAlt string // fallback
	GCPProject   string
}

// DefaultEnvVars is the env-var lookup table used by ResolveFromEnv and
// by Resolve when a Credentials field is empty. Override this to swap
// in custom env var names — see EnvVars documentation.
var DefaultEnvVars = EnvVars{
	OpenAIAPIKey:    "OPENAI_API_KEY",
	AnthropicAPIKey: "ANTHROPIC_API_KEY",
	MistralAPIKey:   "MISTRAL_API_KEY",
	CohereAPIKey:    "COHERE_API_KEY",
	GoogleAPIKey:    "GOOGLE_API_KEY",
	GrokAPIKey:      "GROK_API_KEY",
	DeepSeekAPIKey:  "DEEPSEEK_API_KEY",

	OpenRouterAPIKey: "OPENROUTER_API_KEY",
	TogetherAPIKey:   "TOGETHER_API_KEY",
	GroqAPIKey:       "GROQ_API_KEY",
	FireworksAPIKey:  "FIREWORKS_API_KEY",
	CerebrasAPIKey:   "CEREBRAS_API_KEY",
	PerplexityAPIKey: "PERPLEXITY_API_KEY",

	AzureBaseURL: "AZURE_OPENAI_ENDPOINT",
	AzureAPIKey:  "AZURE_OPENAI_API_KEY",
	AWSRegion:    "AWS_REGION",
	AWSRegionAlt: "AWS_DEFAULT_REGION",
	GCPProject:   "GOOGLE_CLOUD_PROJECT",
}

// Request is the full resolution input. Model and Platform are required
// (PlatformAuto picks one); Credentials are platform-specific. HTTPClient
// is optional and forwarded to the underlying provider when supported.
type Request struct {
	Model       string
	Platform    Platform
	Credentials Credentials
	HTTPClient  *http.Client
}

// directAPIKeyForFamily returns the env var name that holds the direct
// API key for the given family. Empty when no canonical env var exists.
func directAPIKeyForFamily(family ModelFamily, env EnvVars) string {
	switch family {
	case FamilyOpenAI:
		return env.OpenAIAPIKey
	case FamilyAnthropic:
		return env.AnthropicAPIKey
	case FamilyMistral:
		return env.MistralAPIKey
	case FamilyCohere:
		return env.CohereAPIKey
	case FamilyGemini:
		return env.GoogleAPIKey
	case FamilyGrok:
		return env.GrokAPIKey
	case FamilyDeepSeek:
		return env.DeepSeekAPIKey
	}
	return ""
}

// platformAPIKeyEnv returns the env var name that holds the API key for
// the given platform (when distinct from the family default).
func platformAPIKeyEnv(platform Platform, env EnvVars) string {
	switch platform {
	case PlatformOpenRouter:
		return env.OpenRouterAPIKey
	case PlatformTogether:
		return env.TogetherAPIKey
	case PlatformGroq:
		return env.GroqAPIKey
	case PlatformFireworks:
		return env.FireworksAPIKey
	case PlatformCerebras:
		return env.CerebrasAPIKey
	case PlatformPerplexity:
		return env.PerplexityAPIKey
	}
	return ""
}

// resolveAPIKey picks the API key for a given (family, platform) call.
// Priority: explicit Credentials.APIKey > platform-specific env var >
// family-specific env var. Returns empty when nothing is available.
func resolveAPIKey(family ModelFamily, platform Platform, creds Credentials, env EnvVars) string {
	if v := strings.TrimSpace(creds.APIKey); v != "" {
		return v
	}
	if name := platformAPIKeyEnv(platform, env); name != "" {
		if v := strings.TrimSpace(os.Getenv(name)); v != "" {
			return v
		}
	}
	if name := directAPIKeyForFamily(family, env); name != "" {
		if v := strings.TrimSpace(os.Getenv(name)); v != "" {
			return v
		}
	}
	return ""
}

// resolveAzureBaseURL picks the Azure resource URL from explicit creds or
// the canonical env var. Returns empty when neither is set.
func resolveAzureBaseURL(creds Credentials, env EnvVars) string {
	if v := strings.TrimSpace(creds.AzureBaseURL); v != "" {
		return v
	}
	if env.AzureBaseURL == "" {
		return ""
	}
	return strings.TrimSpace(os.Getenv(env.AzureBaseURL))
}

// resolveAzureAPIKey picks the Azure API key. AzureAPIKey wins over the
// generic APIKey field; both fall back to env vars.
func resolveAzureAPIKey(creds Credentials, env EnvVars) string {
	if v := strings.TrimSpace(creds.AzureAPIKey); v != "" {
		return v
	}
	if v := strings.TrimSpace(creds.APIKey); v != "" {
		return v
	}
	if env.AzureAPIKey == "" {
		return ""
	}
	return strings.TrimSpace(os.Getenv(env.AzureAPIKey))
}

// resolveAWSRegion picks the AWS region from explicit creds or one of the
// AWS env vars. Returns empty when nothing is set.
func resolveAWSRegion(creds Credentials, env EnvVars) string {
	if v := strings.TrimSpace(creds.AWSRegion); v != "" {
		return v
	}
	if env.AWSRegion != "" {
		if v := strings.TrimSpace(os.Getenv(env.AWSRegion)); v != "" {
			return v
		}
	}
	if env.AWSRegionAlt != "" {
		if v := strings.TrimSpace(os.Getenv(env.AWSRegionAlt)); v != "" {
			return v
		}
	}
	return ""
}

// resolveGCPProject picks the GCP project id from explicit creds or env.
func resolveGCPProject(creds Credentials, env EnvVars) string {
	if v := strings.TrimSpace(creds.GCPProject); v != "" {
		return v
	}
	if env.GCPProject == "" {
		return ""
	}
	return strings.TrimSpace(os.Getenv(env.GCPProject))
}

// hasCredentialsFor reports whether the caller has enough info (either in
// creds or in the env) to build a Provider for the (family, platform)
// pair. Used by PlatformAuto to skip platforms that would fail New().
func hasCredentialsFor(family ModelFamily, platform Platform, creds Credentials, env EnvVars) bool {
	switch platform {
	case PlatformDirect,
		PlatformOpenRouter, PlatformTogether, PlatformGroq, PlatformFireworks,
		PlatformCerebras, PlatformDeepSeek, PlatformPerplexity, PlatformxAI:
		return resolveAPIKey(family, platform, creds, env) != ""
	case PlatformBedrock:
		return resolveAWSRegion(creds, env) != ""
	case PlatformVertex:
		return resolveGCPProject(creds, env) != "" && strings.TrimSpace(creds.GCPRegion) != ""
	case PlatformAzure:
		return resolveAzureBaseURL(creds, env) != "" && resolveAzureAPIKey(creds, env) != ""
	}
	return false
}
