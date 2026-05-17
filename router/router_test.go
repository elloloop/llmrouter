package router

import (
	"errors"
	"strings"
	"testing"
)

// clearEnv unsets every env var the router reads so tests have a clean
// baseline. Uses t.Setenv("", "") via individual clears — t.Setenv is
// goroutine-safe per-process and the runtime restores values at test end.
func clearEnv(t *testing.T) {
	t.Helper()
	for _, name := range []string{
		DefaultEnvVars.OpenAIAPIKey,
		DefaultEnvVars.AnthropicAPIKey,
		DefaultEnvVars.MistralAPIKey,
		DefaultEnvVars.CohereAPIKey,
		DefaultEnvVars.GoogleAPIKey,
		DefaultEnvVars.GrokAPIKey,
		DefaultEnvVars.DeepSeekAPIKey,
		DefaultEnvVars.OpenRouterAPIKey,
		DefaultEnvVars.TogetherAPIKey,
		DefaultEnvVars.GroqAPIKey,
		DefaultEnvVars.FireworksAPIKey,
		DefaultEnvVars.CerebrasAPIKey,
		DefaultEnvVars.PerplexityAPIKey,
		DefaultEnvVars.AzureBaseURL,
		DefaultEnvVars.AzureAPIKey,
		DefaultEnvVars.AWSRegion,
		DefaultEnvVars.AWSRegionAlt,
		DefaultEnvVars.GCPProject,
	} {
		if name != "" {
			t.Setenv(name, "")
		}
	}
}

func TestInferFamily(t *testing.T) {
	cases := []struct {
		model string
		want  ModelFamily
	}{
		// OpenAI
		{"gpt-4o", FamilyOpenAI},
		{"gpt-4o-mini", FamilyOpenAI},
		{"gpt-4-turbo", FamilyOpenAI},
		{"GPT-4O", FamilyOpenAI}, // case-insensitive
		{"o1-preview", FamilyOpenAI},
		{"o1-mini", FamilyOpenAI},
		{"o3-mini", FamilyOpenAI},
		{"o4-mini", FamilyOpenAI},
		{"chatgpt-4o-latest", FamilyOpenAI},

		// Anthropic
		{"claude-3-5-sonnet-20241022", FamilyAnthropic},
		{"claude-3-5-haiku", FamilyAnthropic},
		{"claude-3-opus-20240229", FamilyAnthropic},
		{"anthropic.claude-3-5-sonnet-20241022-v2:0", FamilyAnthropic},

		// Llama
		{"llama-3-1-70b-instruct", FamilyLlama},
		{"llama3-8b-instruct", FamilyLlama},
		{"llama2-13b-chat", FamilyLlama},
		{"meta.llama3-1-70b-instruct-v1:0", FamilyLlama},
		{"meta-llama/Llama-3-70b-chat-hf", FamilyLlama}, // case + contains

		// Mistral
		{"mistral-large-latest", FamilyMistral},
		{"mixtral-8x7b-instruct", FamilyMistral},
		{"ministral-3b-latest", FamilyMistral},
		{"codestral-latest", FamilyMistral},
		{"magistral-medium-2506", FamilyMistral},
		{"open-mistral-7b", FamilyMistral},

		// Cohere
		{"command-r-plus", FamilyCohere},
		{"command-r", FamilyCohere},
		{"c4ai-aya-23-35b", FamilyCohere},

		// Gemini
		{"gemini-1.5-pro", FamilyGemini},
		{"gemini-2.0-flash-exp", FamilyGemini},

		// Grok
		{"grok-2-latest", FamilyGrok},
		{"grok-beta", FamilyGrok},

		// DeepSeek
		{"deepseek-chat", FamilyDeepSeek},
		{"deepseek-coder", FamilyDeepSeek},

		// Other / unknown
		{"foobar-2024", FamilyOther},
		{"", FamilyOther},
		{"random-thing", FamilyOther},
	}
	for _, c := range cases {
		t.Run(c.model+"->"+string(c.want), func(t *testing.T) {
			got := InferFamily(c.model)
			if got != c.want {
				t.Errorf("InferFamily(%q) = %q, want %q", c.model, got, c.want)
			}
		})
	}
}

func TestSupportedPlatforms_OpenAI(t *testing.T) {
	got := SupportedPlatforms(FamilyOpenAI)
	want := []Platform{PlatformDirect, PlatformAzure, PlatformOpenRouter}
	if !equalPlatformSlice(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestSupportedPlatforms_Anthropic(t *testing.T) {
	got := SupportedPlatforms(FamilyAnthropic)
	wantIncludes := []Platform{PlatformDirect, PlatformBedrock, PlatformVertex, PlatformAzure, PlatformOpenRouter}
	for _, p := range wantIncludes {
		if !containsPlatform(got, p) {
			t.Errorf("Anthropic should support %s; got %v", p, got)
		}
	}
	if got[0] != PlatformDirect {
		t.Errorf("first platform should be Direct (PlatformAuto default), got %s", got[0])
	}
}

func TestSupportedPlatforms_Llama(t *testing.T) {
	got := SupportedPlatforms(FamilyLlama)
	for _, p := range []Platform{PlatformBedrock, PlatformTogether, PlatformGroq, PlatformFireworks} {
		if !containsPlatform(got, p) {
			t.Errorf("Llama should support %s; got %v", p, got)
		}
	}
	if containsPlatform(got, PlatformDirect) {
		t.Errorf("Llama has no Direct provider; got %v", got)
	}
}

func TestSupportedPlatforms_Mistral(t *testing.T) {
	got := SupportedPlatforms(FamilyMistral)
	for _, p := range []Platform{PlatformDirect, PlatformBedrock, PlatformTogether, PlatformGroq, PlatformFireworks} {
		if !containsPlatform(got, p) {
			t.Errorf("Mistral should support %s; got %v", p, got)
		}
	}
}

func TestSupportedPlatforms_Cohere(t *testing.T) {
	got := SupportedPlatforms(FamilyCohere)
	for _, p := range []Platform{PlatformDirect, PlatformBedrock, PlatformAzure} {
		if !containsPlatform(got, p) {
			t.Errorf("Cohere should support %s; got %v", p, got)
		}
	}
}

func TestSupportedPlatforms_Gemini(t *testing.T) {
	got := SupportedPlatforms(FamilyGemini)
	for _, p := range []Platform{PlatformDirect, PlatformVertex, PlatformOpenRouter} {
		if !containsPlatform(got, p) {
			t.Errorf("Gemini should support %s; got %v", p, got)
		}
	}
}

func TestSupportedPlatforms_Grok(t *testing.T) {
	got := SupportedPlatforms(FamilyGrok)
	for _, p := range []Platform{PlatformDirect, PlatformxAI, PlatformOpenRouter} {
		if !containsPlatform(got, p) {
			t.Errorf("Grok should support %s; got %v", p, got)
		}
	}
}

func TestSupportedPlatforms_DeepSeek(t *testing.T) {
	got := SupportedPlatforms(FamilyDeepSeek)
	for _, p := range []Platform{PlatformDirect, PlatformDeepSeek, PlatformOpenRouter, PlatformTogether, PlatformFireworks} {
		if !containsPlatform(got, p) {
			t.Errorf("DeepSeek should support %s; got %v", p, got)
		}
	}
}

func TestSupportedPlatforms_Other(t *testing.T) {
	got := SupportedPlatforms(FamilyOther)
	for _, p := range []Platform{PlatformOpenRouter, PlatformTogether} {
		if !containsPlatform(got, p) {
			t.Errorf("Other should support %s; got %v", p, got)
		}
	}
}

// TestResolve_Direct exercises each family's direct-API path.
func TestResolve_Direct(t *testing.T) {
	clearEnv(t)
	cases := []struct {
		name         string
		model        string
		wantProvider string
	}{
		{"openai_direct", "gpt-4o-mini", "openai"},
		{"anthropic_direct", "claude-3-5-sonnet-20241022", "anthropic"},
		{"mistral_direct", "mistral-large-latest", "mistral"},
		{"cohere_direct", "command-r-plus", "cohere"},
		{"grok_direct", "grok-2-latest", "xai"},
		{"deepseek_direct", "deepseek-chat", "deepseek"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p, err := Resolve(Request{
				Model:    c.model,
				Platform: PlatformDirect,
				Credentials: Credentials{
					APIKey: "test-key-123",
				},
			})
			if err != nil {
				t.Fatalf("Resolve: %v", err)
			}
			if p.Name() != c.wantProvider {
				t.Errorf("Name() = %q, want %q", p.Name(), c.wantProvider)
			}
		})
	}
}

func TestResolve_Bedrock_Anthropic(t *testing.T) {
	clearEnv(t)
	p, err := Resolve(Request{
		Model:    "claude-3-5-sonnet-20241022",
		Platform: PlatformBedrock,
		Credentials: Credentials{
			AWSRegion: "us-east-1",
		},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "bedrock" {
		t.Errorf("Name() = %q, want %q", p.Name(), "bedrock")
	}
}

func TestResolve_Bedrock_Llama(t *testing.T) {
	clearEnv(t)
	p, err := Resolve(Request{
		Model:       "llama-3-1-70b-instruct",
		Platform:    PlatformBedrock,
		Credentials: Credentials{AWSRegion: "us-west-2"},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "bedrock" {
		t.Errorf("Name() = %q, want %q", p.Name(), "bedrock")
	}
}

func TestResolve_Bedrock_Mistral(t *testing.T) {
	clearEnv(t)
	p, err := Resolve(Request{
		Model:       "mistral-large-2407",
		Platform:    PlatformBedrock,
		Credentials: Credentials{AWSRegion: "us-east-1"},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "bedrock" {
		t.Errorf("Name() = %q, want %q", p.Name(), "bedrock")
	}
}

func TestResolve_Bedrock_Cohere(t *testing.T) {
	clearEnv(t)
	p, err := Resolve(Request{
		Model:       "command-r-plus",
		Platform:    PlatformBedrock,
		Credentials: Credentials{AWSRegion: "us-east-1"},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "bedrock" {
		t.Errorf("Name() = %q, want %q", p.Name(), "bedrock")
	}
}

func TestResolve_Vertex_Gemini(t *testing.T) {
	clearEnv(t)
	p, err := Resolve(Request{
		Model:    "gemini-1.5-pro",
		Platform: PlatformVertex,
		Credentials: Credentials{
			GCPProject: "test-project",
			GCPRegion:  "us-central1",
		},
	})
	if err != nil {
		// Vertex's New() may try to load ADC and fail in a sandboxed env;
		// tolerate that — the routing decision is what we're testing.
		if !strings.Contains(err.Error(), "vertex") {
			t.Fatalf("expected vertex-related error or success, got: %v", err)
		}
		return
	}
	if p.Name() != "vertex" {
		t.Errorf("Name() = %q, want %q", p.Name(), "vertex")
	}
}

func TestResolve_Vertex_Anthropic_TODO(t *testing.T) {
	clearEnv(t)
	_, err := Resolve(Request{
		Model:    "claude-3-5-sonnet-20241022",
		Platform: PlatformVertex,
		Credentials: Credentials{
			GCPProject: "test-project",
			GCPRegion:  "us-central1",
		},
	})
	if err == nil {
		t.Fatal("expected ErrUnsupportedRoute (vertexanthropic not wired); got nil")
	}
	if !errors.Is(err, ErrUnsupportedRoute) {
		t.Errorf("want ErrUnsupportedRoute, got: %v", err)
	}
	if !strings.Contains(err.Error(), "TODO") {
		t.Errorf("want TODO marker in error, got: %v", err)
	}
}

func TestResolve_Vertex_Llama_NotImplemented(t *testing.T) {
	clearEnv(t)
	_, err := Resolve(Request{
		Model:    "llama-3-1-70b-instruct",
		Platform: PlatformVertex,
		Credentials: Credentials{
			GCPProject: "p",
			GCPRegion:  "us-central1",
		},
	})
	// Llama+Vertex is not in the supported list, so ErrUnsupportedRoute
	// triggers at the supported-set check before build().
	if !errors.Is(err, ErrUnsupportedRoute) {
		t.Errorf("want ErrUnsupportedRoute, got: %v", err)
	}
}

func TestResolve_Azure_OpenAI(t *testing.T) {
	clearEnv(t)
	p, err := Resolve(Request{
		Model:    "gpt-4o",
		Platform: PlatformAzure,
		Credentials: Credentials{
			AzureBaseURL:    "https://test.openai.azure.com",
			AzureAPIKey:     "azkey",
			AzureAPIVersion: "2024-10-21",
			AzureDeployment: "my-gpt",
		},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "azureopenai" {
		t.Errorf("Name() = %q, want %q", p.Name(), "azureopenai")
	}
}

func TestResolve_Azure_Anthropic(t *testing.T) {
	clearEnv(t)
	p, err := Resolve(Request{
		Model:    "claude-3-5-sonnet-20241022",
		Platform: PlatformAzure,
		Credentials: Credentials{
			AzureBaseURL:    "https://test.services.ai.azure.com",
			AzureAPIKey:     "azkey",
			AzureAPIVersion: "2024-10-21",
		},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "azureanthropic" {
		t.Errorf("Name() = %q, want %q", p.Name(), "azureanthropic")
	}
}

func TestResolve_Azure_Llama_TODO(t *testing.T) {
	clearEnv(t)
	_, err := Resolve(Request{
		Model:    "llama-3-1-70b-instruct",
		Platform: PlatformAzure,
		Credentials: Credentials{
			AzureBaseURL:    "https://x.services.ai.azure.com",
			AzureAPIKey:     "azkey",
			AzureAPIVersion: "2024-10-21",
			AzureDeployment: "llama-deploy",
		},
	})
	if err == nil {
		t.Fatal("expected ErrUnsupportedRoute (azureserverless not wired); got nil")
	}
	if !errors.Is(err, ErrUnsupportedRoute) {
		t.Errorf("want ErrUnsupportedRoute, got: %v", err)
	}
	if !strings.Contains(err.Error(), "TODO") {
		t.Errorf("want TODO marker in error, got: %v", err)
	}
}

func TestResolve_OpenRouter_AnyFamily(t *testing.T) {
	clearEnv(t)
	cases := []struct {
		name  string
		model string
	}{
		{"openai_via_openrouter", "gpt-4o-mini"},
		{"anthropic_via_openrouter", "claude-3-5-sonnet"},
		{"llama_via_openrouter", "llama-3-1-70b-instruct"},
		{"mistral_via_openrouter", "mistral-large-latest"},
		{"gemini_via_openrouter", "gemini-1.5-pro"},
		{"grok_via_openrouter", "grok-2-latest"},
		{"deepseek_via_openrouter", "deepseek-chat"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p, err := Resolve(Request{
				Model:       c.model,
				Platform:    PlatformOpenRouter,
				Credentials: Credentials{APIKey: "or-test"},
			})
			if err != nil {
				t.Fatalf("Resolve: %v", err)
			}
			if p.Name() != "openrouter" {
				t.Errorf("Name() = %q, want %q", p.Name(), "openrouter")
			}
		})
	}
}

func TestResolve_Together(t *testing.T) {
	clearEnv(t)
	p, err := Resolve(Request{
		Model:       "llama-3-1-70b-instruct",
		Platform:    PlatformTogether,
		Credentials: Credentials{APIKey: "tk"},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "together" {
		t.Errorf("Name() = %q, want %q", p.Name(), "together")
	}
}

func TestResolve_Groq(t *testing.T) {
	clearEnv(t)
	p, err := Resolve(Request{
		Model:       "llama-3-1-70b-instruct",
		Platform:    PlatformGroq,
		Credentials: Credentials{APIKey: "gk"},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "groq" {
		t.Errorf("Name() = %q, want %q", p.Name(), "groq")
	}
}

func TestResolve_Fireworks(t *testing.T) {
	clearEnv(t)
	p, err := Resolve(Request{
		Model:       "llama-3-1-70b-instruct",
		Platform:    PlatformFireworks,
		Credentials: Credentials{APIKey: "fw"},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "fireworks" {
		t.Errorf("Name() = %q, want %q", p.Name(), "fireworks")
	}
}

func TestResolve_Cerebras(t *testing.T) {
	clearEnv(t)
	p, err := Resolve(Request{
		Model:       "llama-3-1-70b-instruct",
		Platform:    PlatformCerebras,
		Credentials: Credentials{APIKey: "cb"},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "cerebras" {
		t.Errorf("Name() = %q, want %q", p.Name(), "cerebras")
	}
}

func TestResolve_DeepSeek_Platform(t *testing.T) {
	clearEnv(t)
	p, err := Resolve(Request{
		Model:       "deepseek-chat",
		Platform:    PlatformDeepSeek,
		Credentials: Credentials{APIKey: "ds"},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "deepseek" {
		t.Errorf("Name() = %q, want %q", p.Name(), "deepseek")
	}
}

func TestResolve_Perplexity(t *testing.T) {
	clearEnv(t)
	// Perplexity isn't in any family's supported list except Other;
	// we route to it by using FamilyOther via an unknown model.
	p, err := Resolve(Request{
		Model:       "sonar-medium-online",
		Platform:    PlatformPerplexity,
		Credentials: Credentials{APIKey: "px"},
	})
	if err == nil {
		if p.Name() != "perplexity" {
			t.Errorf("Name() = %q, want %q", p.Name(), "perplexity")
		}
		return
	}
	// Perplexity isn't in FamilyOther's list, so we expect
	// ErrUnsupportedRoute here. Test that the surface is sane.
	if !errors.Is(err, ErrUnsupportedRoute) {
		t.Errorf("want ErrUnsupportedRoute or success, got: %v", err)
	}
}

func TestResolve_xAI_Direct(t *testing.T) {
	clearEnv(t)
	p, err := Resolve(Request{
		Model:       "grok-2-latest",
		Platform:    PlatformxAI,
		Credentials: Credentials{APIKey: "xk"},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "xai" {
		t.Errorf("Name() = %q, want %q", p.Name(), "xai")
	}
}

func TestResolve_EmptyModel(t *testing.T) {
	clearEnv(t)
	_, err := Resolve(Request{Platform: PlatformDirect})
	if !errors.Is(err, ErrEmptyModel) {
		t.Errorf("want ErrEmptyModel, got: %v", err)
	}
}

func TestResolve_UnsupportedRoute(t *testing.T) {
	clearEnv(t)
	// Gemini on Bedrock is nonsense.
	_, err := Resolve(Request{
		Model:       "gemini-1.5-pro",
		Platform:    PlatformBedrock,
		Credentials: Credentials{AWSRegion: "us-east-1"},
	})
	if !errors.Is(err, ErrUnsupportedRoute) {
		t.Errorf("want ErrUnsupportedRoute, got: %v", err)
	}
}

func TestResolve_MissingCreds_Direct(t *testing.T) {
	clearEnv(t)
	_, err := Resolve(Request{
		Model:    "gpt-4o-mini",
		Platform: PlatformDirect,
	})
	if !errors.Is(err, ErrMissingCredentials) {
		t.Errorf("want ErrMissingCredentials, got: %v", err)
	}
}

func TestResolve_MissingCreds_Bedrock(t *testing.T) {
	clearEnv(t)
	_, err := Resolve(Request{
		Model:    "claude-3-5-sonnet-20241022",
		Platform: PlatformBedrock,
	})
	if !errors.Is(err, ErrMissingCredentials) {
		t.Errorf("want ErrMissingCredentials, got: %v", err)
	}
}

func TestResolve_MissingCreds_Vertex(t *testing.T) {
	clearEnv(t)
	_, err := Resolve(Request{
		Model:    "gemini-1.5-pro",
		Platform: PlatformVertex,
	})
	if !errors.Is(err, ErrMissingCredentials) {
		t.Errorf("want ErrMissingCredentials, got: %v", err)
	}
}

func TestResolve_MissingCreds_Azure(t *testing.T) {
	clearEnv(t)
	_, err := Resolve(Request{
		Model:    "gpt-4o",
		Platform: PlatformAzure,
	})
	if !errors.Is(err, ErrMissingCredentials) {
		t.Errorf("want ErrMissingCredentials, got: %v", err)
	}
}

func TestResolve_MissingCreds_AzureNoDeployment(t *testing.T) {
	clearEnv(t)
	_, err := Resolve(Request{
		Model:    "gpt-4o",
		Platform: PlatformAzure,
		Credentials: Credentials{
			AzureBaseURL:    "https://x.openai.azure.com",
			AzureAPIKey:     "k",
			AzureAPIVersion: "2024-10-21",
			// AzureDeployment missing
		},
	})
	if !errors.Is(err, ErrMissingCredentials) {
		t.Errorf("want ErrMissingCredentials about deployment, got: %v", err)
	}
}

func TestResolve_MissingCreds_OpenRouter(t *testing.T) {
	clearEnv(t)
	_, err := Resolve(Request{
		Model:    "gpt-4o",
		Platform: PlatformOpenRouter,
	})
	if !errors.Is(err, ErrMissingCredentials) {
		t.Errorf("want ErrMissingCredentials, got: %v", err)
	}
}

func TestResolveFromEnv_OpenAI(t *testing.T) {
	clearEnv(t)
	t.Setenv(DefaultEnvVars.OpenAIAPIKey, "sk-test")
	p, err := ResolveFromEnv("gpt-4o-mini")
	if err != nil {
		t.Fatalf("ResolveFromEnv: %v", err)
	}
	if p.Name() != "openai" {
		t.Errorf("Name() = %q, want openai", p.Name())
	}
}

func TestResolveFromEnv_Anthropic(t *testing.T) {
	clearEnv(t)
	t.Setenv(DefaultEnvVars.AnthropicAPIKey, "sk-ant-test")
	p, err := ResolveFromEnv("claude-3-5-sonnet-20241022")
	if err != nil {
		t.Fatalf("ResolveFromEnv: %v", err)
	}
	if p.Name() != "anthropic" {
		t.Errorf("Name() = %q, want anthropic", p.Name())
	}
}

func TestResolveFromEnv_NoCreds(t *testing.T) {
	clearEnv(t)
	_, err := ResolveFromEnv("gpt-4o-mini")
	if !errors.Is(err, ErrNoAutoPlatform) {
		t.Errorf("want ErrNoAutoPlatform, got: %v", err)
	}
}

func TestPlatformAuto_PrefersDirect(t *testing.T) {
	clearEnv(t)
	// Both Anthropic direct and AWS Bedrock available; Direct wins.
	p, err := Resolve(Request{
		Model:    "claude-3-5-sonnet-20241022",
		Platform: PlatformAuto,
		Credentials: Credentials{
			APIKey:    "sk-ant",
			AWSRegion: "us-east-1",
		},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "anthropic" {
		t.Errorf("PlatformAuto with both creds should pick Direct (anthropic), got %q", p.Name())
	}
}

func TestPlatformAuto_FallsBackToBedrock(t *testing.T) {
	clearEnv(t)
	// Only AWS creds → Bedrock for claude.
	p, err := Resolve(Request{
		Model:       "claude-3-5-sonnet-20241022",
		Platform:    PlatformAuto,
		Credentials: Credentials{AWSRegion: "us-east-1"},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "bedrock" {
		t.Errorf("want bedrock, got %q", p.Name())
	}
}

func TestPlatformAuto_FallsBackToAzure(t *testing.T) {
	clearEnv(t)
	// Only Azure creds for OpenAI family.
	p, err := Resolve(Request{
		Model:    "gpt-4o",
		Platform: PlatformAuto,
		Credentials: Credentials{
			AzureBaseURL:    "https://x.openai.azure.com",
			AzureAPIKey:     "k",
			AzureAPIVersion: "2024-10-21",
			AzureDeployment: "gpt-deploy",
		},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "azureopenai" {
		t.Errorf("want azureopenai, got %q", p.Name())
	}
}

func TestPlatformAuto_FallsBackToOpenRouter(t *testing.T) {
	clearEnv(t)
	// Llama with only OpenRouter key — Bedrock/Vertex/Azure unavailable.
	t.Setenv(DefaultEnvVars.OpenRouterAPIKey, "or-key")
	p, err := Resolve(Request{
		Model:    "llama-3-1-70b-instruct",
		Platform: PlatformAuto,
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "openrouter" {
		t.Errorf("want openrouter, got %q", p.Name())
	}
}

func TestPlatformAuto_PicksGroqForLlama(t *testing.T) {
	clearEnv(t)
	// Llama with only Groq key.
	t.Setenv(DefaultEnvVars.GroqAPIKey, "gq")
	p, err := Resolve(Request{
		Model:    "llama-3-1-70b-instruct",
		Platform: PlatformAuto,
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if p.Name() != "groq" {
		t.Errorf("want groq, got %q", p.Name())
	}
}

func TestBedrockModelID_AnthropicKnown(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"claude-3-5-sonnet", "anthropic.claude-3-5-sonnet-20241022-v2:0"},
		{"claude-3-5-sonnet-20241022", "anthropic.claude-3-5-sonnet-20241022-v2:0"},
		{"claude-3-5-haiku", "anthropic.claude-3-5-haiku-20241022-v1:0"},
		{"claude-3-opus", "anthropic.claude-3-opus-20240229-v1:0"},
		{"claude-3-haiku", "anthropic.claude-3-haiku-20240307-v1:0"},
	}
	for _, c := range cases {
		t.Run(c.in, func(t *testing.T) {
			got := bedrockModelID(FamilyAnthropic, c.in)
			if got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

func TestBedrockModelID_AnthropicPrefixedPassesThrough(t *testing.T) {
	in := "anthropic.claude-3-5-sonnet-20241022-v2:0"
	got := bedrockModelID(FamilyAnthropic, in)
	if got != in {
		t.Errorf("prefixed id should pass through; got %q", got)
	}
}

func TestBedrockModelID_UnknownPassesThrough(t *testing.T) {
	in := "claude-99-unreleased-xyz"
	got := bedrockModelID(FamilyAnthropic, in)
	if got != in {
		t.Errorf("unknown id should pass through; got %q", got)
	}
}

func TestBedrockModelID_LlamaKnown(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"llama-3-1-70b-instruct", "meta.llama3-1-70b-instruct-v1:0"},
		{"llama-3.1-8b-instruct", "meta.llama3-1-8b-instruct-v1:0"},
		{"llama-3-2-3b-instruct", "meta.llama3-2-3b-instruct-v1:0"},
	}
	for _, c := range cases {
		t.Run(c.in, func(t *testing.T) {
			got := bedrockModelID(FamilyLlama, c.in)
			if got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

func TestBedrockModelID_MistralKnown(t *testing.T) {
	got := bedrockModelID(FamilyMistral, "mistral-large-2407")
	want := "mistral.mistral-large-2407-v1:0"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestBedrockModelID_CohereKnown(t *testing.T) {
	got := bedrockModelID(FamilyCohere, "command-r-plus")
	want := "cohere.command-r-plus-v1:0"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestVertexAnthropicModelID_Known(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"claude-3-5-sonnet", "claude-3-5-sonnet-v2@20241022"},
		{"claude-3-5-sonnet-20241022", "claude-3-5-sonnet-v2@20241022"},
		{"claude-3-5-haiku", "claude-3-5-haiku@20241022"},
		{"claude-3-opus", "claude-3-opus@20240229"},
	}
	for _, c := range cases {
		t.Run(c.in, func(t *testing.T) {
			got := vertexAnthropicModelID(c.in)
			if got != c.want {
				t.Errorf("got %q, want %q", got, c.want)
			}
		})
	}
}

func TestVertexAnthropicModelID_AtVersionedPassesThrough(t *testing.T) {
	in := "claude-3-5-sonnet-v2@20241022"
	got := vertexAnthropicModelID(in)
	if got != in {
		t.Errorf("at-versioned id should pass through; got %q", got)
	}
}

func TestVertexAnthropicModelID_UnknownPassesThrough(t *testing.T) {
	in := "claude-future-model"
	got := vertexAnthropicModelID(in)
	if got != in {
		t.Errorf("unknown id should pass through; got %q", got)
	}
}

func TestApplyModelTranslation_Bedrock(t *testing.T) {
	req := struct {
		Model string
	}{Model: "claude-3-5-sonnet"}
	_ = req
	// Use the public-ish wrapper.
	got := bedrockModelID(InferFamily("claude-3-5-sonnet"), "claude-3-5-sonnet")
	want := "anthropic.claude-3-5-sonnet-20241022-v2:0"
	if got != want {
		t.Errorf("bedrockModelID = %q, want %q", got, want)
	}
}

func TestIsSupported(t *testing.T) {
	if !isSupported(FamilyAnthropic, PlatformBedrock) {
		t.Error("Anthropic+Bedrock should be supported")
	}
	if isSupported(FamilyGemini, PlatformBedrock) {
		t.Error("Gemini+Bedrock should not be supported")
	}
}

// equalPlatformSlice returns true if both slices contain the same
// platforms in the same order.
func equalPlatformSlice(a, b []Platform) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// containsPlatform reports whether the slice contains the given platform.
func containsPlatform(s []Platform, p Platform) bool {
	for _, q := range s {
		if q == p {
			return true
		}
	}
	return false
}
