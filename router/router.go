package router

import (
	"errors"
	"fmt"

	"github.com/elloloop/llmrouter"
	"github.com/elloloop/llmrouter/providers/anthropic"
	"github.com/elloloop/llmrouter/providers/azureanthropic"
	"github.com/elloloop/llmrouter/providers/azureopenai"
	"github.com/elloloop/llmrouter/providers/bedrock"
	"github.com/elloloop/llmrouter/providers/cerebras"
	"github.com/elloloop/llmrouter/providers/cohere"
	"github.com/elloloop/llmrouter/providers/deepseek"
	"github.com/elloloop/llmrouter/providers/fireworks"
	"github.com/elloloop/llmrouter/providers/gemini"
	"github.com/elloloop/llmrouter/providers/groq"
	"github.com/elloloop/llmrouter/providers/mistral"
	"github.com/elloloop/llmrouter/providers/openai"
	"github.com/elloloop/llmrouter/providers/openrouter"
	"github.com/elloloop/llmrouter/providers/perplexity"
	"github.com/elloloop/llmrouter/providers/together"
	"github.com/elloloop/llmrouter/providers/vertex"
	"github.com/elloloop/llmrouter/providers/xai"
)

// ErrUnsupportedRoute is returned by Resolve when the inferred family
// cannot be served by the requested platform.
var ErrUnsupportedRoute = errors.New("router: unsupported (family, platform) combination")

// ErrMissingCredentials is returned by Resolve when the chosen platform
// requires credentials the caller did not supply (neither in Credentials
// nor in the environment).
var ErrMissingCredentials = errors.New("router: missing required credentials")

// ErrEmptyModel is returned by Resolve when Request.Model is empty.
var ErrEmptyModel = errors.New("router: model id cannot be empty")

// ErrNoAutoPlatform is returned by Resolve when PlatformAuto cannot find
// any supported platform with credentials available.
var ErrNoAutoPlatform = errors.New("router: no platform with credentials available for the inferred family")

// Resolve builds an llmrouter.Provider for the requested model+platform.
// It infers the model family from Request.Model, validates that the
// platform can serve that family, gathers credentials (from Request or
// env), and constructs the underlying provider.
//
// Errors:
//   - ErrEmptyModel — Request.Model is empty.
//   - ErrUnsupportedRoute — platform cannot serve the inferred family.
//   - ErrMissingCredentials — required credentials are missing.
//   - ErrNoAutoPlatform — PlatformAuto found no usable platform.
//   - Wrapped llmrouter.ErrInvalidConfig from the underlying provider's New().
func Resolve(req Request) (llmrouter.Provider, error) {
	if req.Model == "" {
		return nil, ErrEmptyModel
	}
	family := InferFamily(req.Model)
	platform := req.Platform
	if platform == PlatformAuto {
		picked, ok := pickAutoPlatform(family, req.Credentials, DefaultEnvVars)
		if !ok {
			return nil, fmt.Errorf("%w: family=%s", ErrNoAutoPlatform, family)
		}
		platform = picked
	}
	if !isSupported(family, platform) {
		return nil, fmt.Errorf("%w: family=%s platform=%s", ErrUnsupportedRoute, family, platform)
	}
	return build(family, platform, req)
}

// ResolveFromEnv is a convenience for the common case of reading creds
// from env vars and using PlatformAuto.
func ResolveFromEnv(model string) (llmrouter.Provider, error) {
	return Resolve(Request{Model: model, Platform: PlatformAuto})
}

// pickAutoPlatform iterates the global preference list, returning the
// first platform that both supports the family and has credentials.
func pickAutoPlatform(family ModelFamily, creds Credentials, env EnvVars) (Platform, bool) {
	supported := SupportedPlatforms(family)
	supportedSet := make(map[Platform]bool, len(supported))
	for _, p := range supported {
		supportedSet[p] = true
	}
	for _, p := range autoPreference {
		if !supportedSet[p] {
			continue
		}
		if hasCredentialsFor(family, p, creds, env) {
			return p, true
		}
	}
	return "", false
}

// build dispatches to the right provider constructor for the given
// (family, platform) combination. The matrix encoded here is the
// authoritative mapping of "what package handles this".
//
// Routing matrix (rows = family, cols = platform):
//
//	Family       Direct          Bedrock          Vertex                 Azure                       OpenRouter   Together   Groq   Fireworks   Other
//	openai       openai          —                —                      azureopenai                 openrouter   —          —      —           —
//	anthropic    anthropic       bedrock          vertexanthropic*       azureanthropic              openrouter   —          —      —           —
//	llama        —               bedrock          (vertexllama, n/a)     azureserverless*            openrouter   together   groq   fireworks   cerebras
//	mistral      mistral         bedrock          —                      azureserverless*            openrouter   together   groq   fireworks   —
//	cohere       cohere          bedrock          —                      azureserverless*            —            —          —      —           —
//	gemini       gemini          —                vertex                 —                           openrouter   —          —      —           —
//	grok         xai             —                —                      —                           openrouter   —          —      —           xai
//	deepseek     deepseek        —                —                      —                           openrouter   together   —      fireworks   deepseek
//	other        (error)         —                —                      azureserverless*            openrouter   together   —      —           —
//
// (*) vertexanthropic and azureserverless are being built by parallel
// agents and may not compile yet. Those routes return ErrUnsupportedRoute
// wrapped with a TODO message until the upstream packages stabilize.
func build(family ModelFamily, platform Platform, req Request) (llmrouter.Provider, error) {
	switch platform {
	case PlatformDirect:
		return buildDirect(family, req)
	case PlatformBedrock:
		return buildBedrock(family, req)
	case PlatformVertex:
		return buildVertex(family, req)
	case PlatformAzure:
		return buildAzure(family, req)
	case PlatformOpenRouter:
		return buildOpenAICompat(family, platform, req, func(o ...llmrouter.Option) (llmrouter.Provider, error) { return openrouter.New(o...) })
	case PlatformTogether:
		return buildOpenAICompat(family, platform, req, func(o ...llmrouter.Option) (llmrouter.Provider, error) { return together.New(o...) })
	case PlatformGroq:
		return buildOpenAICompat(family, platform, req, func(o ...llmrouter.Option) (llmrouter.Provider, error) { return groq.New(o...) })
	case PlatformFireworks:
		return buildOpenAICompat(family, platform, req, func(o ...llmrouter.Option) (llmrouter.Provider, error) { return fireworks.New(o...) })
	case PlatformCerebras:
		return buildOpenAICompat(family, platform, req, func(o ...llmrouter.Option) (llmrouter.Provider, error) { return cerebras.New(o...) })
	case PlatformDeepSeek:
		return buildOpenAICompat(family, platform, req, func(o ...llmrouter.Option) (llmrouter.Provider, error) { return deepseek.New(o...) })
	case PlatformPerplexity:
		return buildOpenAICompat(family, platform, req, func(o ...llmrouter.Option) (llmrouter.Provider, error) { return perplexity.New(o...) })
	case PlatformxAI:
		return buildOpenAICompat(family, platform, req, func(o ...llmrouter.Option) (llmrouter.Provider, error) { return xai.New(o...) })
	}
	return nil, fmt.Errorf("%w: family=%s platform=%s", ErrUnsupportedRoute, family, platform)
}

// asProvider promotes any *concreteProvider that implements
// llmrouter.Provider into the interface, surfacing any constructor
// error unchanged.
func asProvider(p llmrouter.Provider, err error) (llmrouter.Provider, error) {
	if err != nil {
		return nil, err
	}
	return p, nil
}

// buildDirect dispatches to the model vendor's native API package.
func buildDirect(family ModelFamily, req Request) (llmrouter.Provider, error) {
	apiKey := resolveAPIKey(family, PlatformDirect, req.Credentials, DefaultEnvVars)
	if apiKey == "" && family != FamilyOther {
		return nil, missingCreds(family, PlatformDirect, "API key (set Credentials.APIKey or "+directAPIKeyForFamily(family, DefaultEnvVars)+")")
	}
	opts := baseOpts(apiKey, req)
	switch family {
	case FamilyOpenAI:
		return asProvider(openai.New(opts...))
	case FamilyAnthropic:
		return asProvider(anthropic.New(opts...))
	case FamilyMistral:
		return asProvider(mistral.New(opts...))
	case FamilyCohere:
		return asProvider(cohere.New(opts...))
	case FamilyGemini:
		return asProvider(gemini.New(opts...))
	case FamilyGrok:
		return asProvider(xai.New(opts...))
	case FamilyDeepSeek:
		return asProvider(deepseek.New(opts...))
	}
	return nil, fmt.Errorf("%w: family=%s has no Direct provider", ErrUnsupportedRoute, family)
}

// buildBedrock dispatches to providers/bedrock for all families it
// supports. Bedrock takes AWS credentials from the standard chain.
func buildBedrock(family ModelFamily, req Request) (llmrouter.Provider, error) {
	region := resolveAWSRegion(req.Credentials, DefaultEnvVars)
	if region == "" {
		return nil, missingCreds(family, PlatformBedrock, "AWS region (set Credentials.AWSRegion or AWS_REGION env var)")
	}
	opts := []llmrouter.Option{bedrock.WithRegion(region)}
	if req.HTTPClient != nil {
		opts = append(opts, llmrouter.WithHTTPClient(req.HTTPClient))
	}
	return asProvider(bedrock.New(opts...))
}

// buildVertex dispatches to providers/vertex (for Gemini) or to the
// not-yet-built providers/vertexanthropic (for Claude). Llama on Vertex
// is not implemented.
func buildVertex(family ModelFamily, req Request) (llmrouter.Provider, error) {
	project := resolveGCPProject(req.Credentials, DefaultEnvVars)
	if project == "" {
		return nil, missingCreds(family, PlatformVertex, "GCP project (set Credentials.GCPProject or GOOGLE_CLOUD_PROJECT env var)")
	}
	if req.Credentials.GCPRegion == "" {
		return nil, missingCreds(family, PlatformVertex, "GCP region (set Credentials.GCPRegion)")
	}
	switch family {
	case FamilyGemini:
		opts := []llmrouter.Option{
			vertex.WithProject(project),
			vertex.WithRegion(req.Credentials.GCPRegion),
		}
		if req.HTTPClient != nil {
			opts = append(opts, llmrouter.WithHTTPClient(req.HTTPClient))
		}
		return asProvider(vertex.New(opts...))
	case FamilyAnthropic:
		// TODO: enable once providers/vertexanthropic lands and compiles.
		// When wired, translate req.Model via vertexAnthropicModelID first
		// and construct vertexanthropic.New(project, region, ...).
		return nil, fmt.Errorf("%w: providers/vertexanthropic not yet wired into router (TODO)", ErrUnsupportedRoute)
	case FamilyLlama:
		// TODO: providers/vertexllama doesn't exist yet — fall back to a
		// platform like Bedrock, OpenRouter, Together, Groq, or Fireworks.
		return nil, fmt.Errorf("%w: Llama on Vertex is not implemented; use Bedrock/Together/Groq/Fireworks", ErrUnsupportedRoute)
	}
	return nil, fmt.Errorf("%w: family=%s has no Vertex provider", ErrUnsupportedRoute, family)
}

// buildAzure dispatches to the right Azure sub-provider based on family:
// OpenAI → azureopenai, Anthropic → azureanthropic, anything else →
// azureserverless (not yet wired).
func buildAzure(family ModelFamily, req Request) (llmrouter.Provider, error) {
	baseURL := resolveAzureBaseURL(req.Credentials, DefaultEnvVars)
	apiKey := resolveAzureAPIKey(req.Credentials, DefaultEnvVars)
	apiVersion := req.Credentials.AzureAPIVersion
	if baseURL == "" {
		return nil, missingCreds(family, PlatformAzure, "Azure base URL (set Credentials.AzureBaseURL or AZURE_OPENAI_ENDPOINT)")
	}
	if apiKey == "" {
		return nil, missingCreds(family, PlatformAzure, "Azure API key (set Credentials.AzureAPIKey or AZURE_OPENAI_API_KEY)")
	}
	if apiVersion == "" {
		return nil, missingCreds(family, PlatformAzure, "Azure api-version (set Credentials.AzureAPIVersion, e.g. 2024-10-21)")
	}
	switch family {
	case FamilyOpenAI:
		if req.Credentials.AzureDeployment == "" {
			return nil, missingCreds(family, PlatformAzure, "Azure deployment name (set Credentials.AzureDeployment)")
		}
		opts := []llmrouter.Option{
			llmrouter.WithAPIKey(apiKey),
			llmrouter.WithBaseURL(baseURL),
			azureopenai.WithDeployment(req.Credentials.AzureDeployment),
			azureopenai.WithAPIVersion(apiVersion),
		}
		if req.HTTPClient != nil {
			opts = append(opts, llmrouter.WithHTTPClient(req.HTTPClient))
		}
		return asProvider(azureopenai.New(opts...))
	case FamilyAnthropic:
		opts := []llmrouter.Option{
			llmrouter.WithAPIKey(apiKey),
			llmrouter.WithBaseURL(baseURL),
			azureanthropic.WithAPIVersion(apiVersion),
		}
		if req.Credentials.AzureDeployment != "" {
			opts = append(opts, azureanthropic.WithDeployment(req.Credentials.AzureDeployment))
		}
		if req.HTTPClient != nil {
			opts = append(opts, llmrouter.WithHTTPClient(req.HTTPClient))
		}
		return asProvider(azureanthropic.New(opts...))
	case FamilyLlama, FamilyMistral, FamilyCohere, FamilyOther:
		// TODO: enable once providers/azureserverless lands and compiles.
		// Expected wiring: azureserverless.New(WithBaseURL(baseURL),
		// WithAPIKey(apiKey), WithDeployment(req.Credentials.AzureDeployment),
		// WithAPIVersion(apiVersion)).
		return nil, fmt.Errorf("%w: providers/azureserverless not yet wired into router (TODO)", ErrUnsupportedRoute)
	}
	return nil, fmt.Errorf("%w: family=%s has no Azure provider", ErrUnsupportedRoute, family)
}

// openAICompatCtor is the shared signature of all OpenAI-compatible
// provider constructors (groq.New, together.New, openrouter.New, ...).
type openAICompatCtor func(opts ...llmrouter.Option) (llmrouter.Provider, error)

// buildOpenAICompat dispatches to any OpenAI-compatible provider package.
// They all share the same constructor shape: accept WithAPIKey + standard
// llmrouter options, return *Provider implementing llmrouter.Provider.
// Each call site wraps the concrete *Provider return into the interface
// via a small adapter closure (Go does not auto-convert []func returns
// across pointer types).
func buildOpenAICompat(family ModelFamily, platform Platform, req Request, ctor openAICompatCtor) (llmrouter.Provider, error) {
	apiKey := resolveAPIKey(family, platform, req.Credentials, DefaultEnvVars)
	if apiKey == "" {
		envName := platformAPIKeyEnv(platform, DefaultEnvVars)
		return nil, missingCreds(family, platform, "API key (set Credentials.APIKey or "+envName+" env var)")
	}
	opts := baseOpts(apiKey, req)
	return asProvider(ctor(opts...))
}

// baseOpts is the standard option slice for any provider that accepts
// WithAPIKey + an optional HTTPClient.
func baseOpts(apiKey string, req Request) []llmrouter.Option {
	opts := []llmrouter.Option{llmrouter.WithAPIKey(apiKey)}
	if req.HTTPClient != nil {
		opts = append(opts, llmrouter.WithHTTPClient(req.HTTPClient))
	}
	return opts
}

// missingCreds wraps ErrMissingCredentials with a clear, debuggable
// message naming the field the caller forgot.
func missingCreds(family ModelFamily, platform Platform, detail string) error {
	return fmt.Errorf("%w: family=%s platform=%s — %s", ErrMissingCredentials, family, platform, detail)
}

// ApplyModelTranslation rewrites a Bedrock/Vertex model id in a
// ChatRequest so callers can pass vendor-neutral ids (e.g.
// "claude-3-5-sonnet") and have the right platform-specific id flow
// through. It returns a copy of the request with the rewritten Model;
// the original is unchanged.
//
// Use this when you want the convenience of vendor-neutral ids in
// application code while still letting the router decide the platform.
func ApplyModelTranslation(req llmrouter.ChatRequest, platform Platform) llmrouter.ChatRequest {
	out := req
	family := InferFamily(req.Model)
	switch platform {
	case PlatformBedrock:
		out.Model = bedrockModelID(family, req.Model)
	case PlatformVertex:
		if family == FamilyAnthropic {
			out.Model = vertexAnthropicModelID(req.Model)
		}
	}
	return out
}
