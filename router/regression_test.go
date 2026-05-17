package router

import (
	"testing"

	"github.com/elloloop/llmrouter"
)

// --- ApplyModelTranslation -------------------------------------------------
//
// Regression coverage for the previously-uncovered ApplyModelTranslation
// entry point. Verifies the (platform, family) translation matrix:
//   - Bedrock translates Anthropic / Llama / Mistral / Cohere ids and
//     passes through unknowns / unsupported families unchanged.
//   - Vertex translates only Anthropic ids; other families pass through.
//   - All other platforms (Direct, Azure, OpenRouter, ...) never rewrite.

func TestApplyModelTranslation_Table(t *testing.T) {
	cases := []struct {
		name     string
		model    string
		platform Platform
		want     string
	}{
		// --- Bedrock + Anthropic -------------------------------------------
		{
			name:     "bedrock_claude_3_5_sonnet",
			model:    "claude-3-5-sonnet",
			platform: PlatformBedrock,
			want:     "anthropic.claude-3-5-sonnet-20241022-v2:0",
		},
		{
			name:     "bedrock_claude_3_5_haiku",
			model:    "claude-3-5-haiku",
			platform: PlatformBedrock,
			want:     "anthropic.claude-3-5-haiku-20241022-v1:0",
		},
		{
			name:     "bedrock_claude_3_opus",
			model:    "claude-3-opus",
			platform: PlatformBedrock,
			want:     "anthropic.claude-3-opus-20240229-v1:0",
		},

		// --- Bedrock + Llama -----------------------------------------------
		{
			name:     "bedrock_llama_70b",
			model:    "llama-3-1-70b-instruct",
			platform: PlatformBedrock,
			want:     "meta.llama3-1-70b-instruct-v1:0",
		},
		{
			name:     "bedrock_llama_8b_dotted",
			model:    "llama-3.1-8b-instruct",
			platform: PlatformBedrock,
			want:     "meta.llama3-1-8b-instruct-v1:0",
		},

		// --- Bedrock + Mistral ---------------------------------------------
		{
			name:     "bedrock_mistral_large",
			model:    "mistral-large-2407",
			platform: PlatformBedrock,
			want:     "mistral.mistral-large-2407-v1:0",
		},
		{
			name:     "bedrock_mistral_small",
			model:    "mistral-small",
			platform: PlatformBedrock,
			want:     "mistral.mistral-small-2402-v1:0",
		},

		// --- Bedrock + Cohere ----------------------------------------------
		{
			name:     "bedrock_cohere_command_r_plus",
			model:    "command-r-plus",
			platform: PlatformBedrock,
			want:     "cohere.command-r-plus-v1:0",
		},
		{
			name:     "bedrock_cohere_command_r",
			model:    "command-r",
			platform: PlatformBedrock,
			want:     "cohere.command-r-v1:0",
		},

		// --- Bedrock + Unknown ---------------------------------------------
		{
			name:     "bedrock_unknown_passthrough",
			model:    "claude-future-model",
			platform: PlatformBedrock,
			want:     "claude-future-model",
		},
		{
			name:     "bedrock_already_prefixed_passthrough",
			model:    "anthropic.claude-3-5-sonnet-20241022-v2:0",
			platform: PlatformBedrock,
			want:     "anthropic.claude-3-5-sonnet-20241022-v2:0",
		},

		// --- Vertex + Anthropic --------------------------------------------
		{
			name:     "vertex_claude_3_5_sonnet",
			model:    "claude-3-5-sonnet",
			platform: PlatformVertex,
			want:     "claude-3-5-sonnet-v2@20241022",
		},
		{
			name:     "vertex_claude_3_opus",
			model:    "claude-3-opus",
			platform: PlatformVertex,
			want:     "claude-3-opus@20240229",
		},

		// --- Vertex + non-Anthropic (no translation) -----------------------
		{
			name:     "vertex_gemini_unchanged",
			model:    "gemini-1.5-pro",
			platform: PlatformVertex,
			want:     "gemini-1.5-pro",
		},
		{
			name:     "vertex_llama_unchanged",
			model:    "llama-3-1-70b-instruct",
			platform: PlatformVertex,
			want:     "llama-3-1-70b-instruct",
		},

		// --- Other platforms: never translate ------------------------------
		{
			name:     "direct_claude_unchanged",
			model:    "claude-3-5-sonnet",
			platform: PlatformDirect,
			want:     "claude-3-5-sonnet",
		},
		{
			name:     "azure_gpt_unchanged",
			model:    "gpt-4o",
			platform: PlatformAzure,
			want:     "gpt-4o",
		},
		{
			name:     "openrouter_claude_unchanged",
			model:    "claude-3-5-sonnet",
			platform: PlatformOpenRouter,
			want:     "claude-3-5-sonnet",
		},
		{
			name:     "together_llama_unchanged",
			model:    "llama-3-1-70b-instruct",
			platform: PlatformTogether,
			want:     "llama-3-1-70b-instruct",
		},
		{
			name:     "groq_mixtral_unchanged",
			model:    "mixtral-8x7b-instruct",
			platform: PlatformGroq,
			want:     "mixtral-8x7b-instruct",
		},
		{
			name:     "auto_passthrough",
			model:    "claude-3-5-sonnet",
			platform: PlatformAuto,
			want:     "claude-3-5-sonnet",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := llmrouter.ChatRequest{Model: tc.model}
			out := ApplyModelTranslation(in, tc.platform)
			if out.Model != tc.want {
				t.Fatalf("Model: got %q, want %q", out.Model, tc.want)
			}
			// Original request must not be mutated.
			if in.Model != tc.model {
				t.Errorf("input mutated: in.Model=%q, want %q", in.Model, tc.model)
			}
		})
	}
}

// --- translateMistralBedrock branch coverage -------------------------------

func TestTranslateMistralBedrock_AllBranches(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"alias_mistral_large", "mistral-large", "mistral.mistral-large-2407-v1:0"},
		{"alias_mistral_large_latest", "mistral-large-latest", "mistral.mistral-large-2407-v1:0"},
		{"explicit_mistral_large_2407", "mistral-large-2407", "mistral.mistral-large-2407-v1:0"},
		{"mistral_large_2402", "mistral-large-2402", "mistral.mistral-large-2402-v1:0"},
		{"alias_mistral_small", "mistral-small", "mistral.mistral-small-2402-v1:0"},
		{"alias_mistral_small_latest", "mistral-small-latest", "mistral.mistral-small-2402-v1:0"},
		{"mixtral_8x7b_instruct", "mixtral-8x7b-instruct", "mistral.mixtral-8x7b-instruct-v0:1"},
		{"unknown_mistral_passthrough", "mistral-future", "mistral-future"},
		{"unknown_random_passthrough", "totally-unknown", "totally-unknown"},
		{"empty_passthrough", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := translateMistralBedrock(tc.in)
			if got != tc.want {
				t.Errorf("translateMistralBedrock(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

// Through the top-level bedrockModelID path, already-prefixed ids must
// pass through untouched (the prefix-guard runs before family dispatch).
func TestBedrockModelID_MistralAlreadyPrefixedPassesThrough(t *testing.T) {
	in := "mistral.mistral-large-2407-v1:0"
	got := bedrockModelID(FamilyMistral, in)
	if got != in {
		t.Errorf("got %q, want %q (prefixed id must pass through)", got, in)
	}
}

// --- directAPIKeyForFamily coverage ----------------------------------------

func TestDirectAPIKeyForFamily_AllBranches(t *testing.T) {
	env := DefaultEnvVars
	cases := []struct {
		name   string
		family ModelFamily
		want   string
	}{
		{"openai", FamilyOpenAI, env.OpenAIAPIKey},
		{"anthropic", FamilyAnthropic, env.AnthropicAPIKey},
		{"mistral", FamilyMistral, env.MistralAPIKey},
		{"cohere", FamilyCohere, env.CohereAPIKey},
		{"gemini", FamilyGemini, env.GoogleAPIKey},
		{"grok", FamilyGrok, env.GrokAPIKey},
		{"deepseek", FamilyDeepSeek, env.DeepSeekAPIKey},
		{"llama_no_direct", FamilyLlama, ""},
		{"other_no_direct", FamilyOther, ""},
		{"unknown_family_no_direct", ModelFamily("totally-made-up"), ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := directAPIKeyForFamily(tc.family, env)
			if got != tc.want {
				t.Errorf("directAPIKeyForFamily(%s) = %q, want %q", tc.family, got, tc.want)
			}
		})
	}
}

// Validates the env table is plumbed through (custom env names propagate).
func TestDirectAPIKeyForFamily_HonoursCustomEnvTable(t *testing.T) {
	custom := EnvVars{
		OpenAIAPIKey:    "MY_OPENAI",
		AnthropicAPIKey: "MY_ANTHROPIC",
		MistralAPIKey:   "MY_MISTRAL",
		CohereAPIKey:    "MY_COHERE",
		GoogleAPIKey:    "MY_GEMINI",
		GrokAPIKey:      "MY_GROK",
		DeepSeekAPIKey:  "MY_DEEPSEEK",
	}
	cases := []struct {
		family ModelFamily
		want   string
	}{
		{FamilyOpenAI, "MY_OPENAI"},
		{FamilyAnthropic, "MY_ANTHROPIC"},
		{FamilyMistral, "MY_MISTRAL"},
		{FamilyCohere, "MY_COHERE"},
		{FamilyGemini, "MY_GEMINI"},
		{FamilyGrok, "MY_GROK"},
		{FamilyDeepSeek, "MY_DEEPSEEK"},
	}
	for _, tc := range cases {
		t.Run(string(tc.family), func(t *testing.T) {
			if got := directAPIKeyForFamily(tc.family, custom); got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}
