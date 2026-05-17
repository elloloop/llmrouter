// Package router resolves model+platform requests into llmrouter.Providers
// without the caller needing to know which underlying provider package
// handles the combination.
//
// The library's headline value proposition is to decouple the model vendor
// (OpenAI, Anthropic, Meta/Llama, Mistral, Cohere, Google, xAI, DeepSeek)
// from the hosting platform (the vendor's own API, AWS Bedrock, Google
// Vertex AI, Azure AI Foundry, OpenRouter, Together, Groq, Fireworks,
// Cerebras, DeepSeek, Perplexity, xAI). A user wants "Claude on Bedrock"
// or "Llama on Groq" — they should not need to know that the first answer
// lives in providers/bedrock and the second in providers/groq.
//
// Resolve takes a Request describing the desired model + platform +
// credentials and returns a working llmrouter.Provider. The router infers
// the model family from the model id (claude-* → Anthropic, gpt-* →
// OpenAI, etc.) and picks the right underlying provider package for the
// (family, platform) combination.
//
// Example: Claude on Bedrock.
//
//	p, err := router.Resolve(router.Request{
//	    Model:    "claude-3-5-sonnet-20241022",
//	    Platform: router.PlatformBedrock,
//	    Credentials: router.Credentials{
//	        AWSRegion: "us-east-1",
//	    },
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	stream, err := p.CompletionStream(ctx, llmrouter.ChatRequest{...})
//
// Example: pick the first available platform from env vars.
//
//	p, err := router.ResolveFromEnv("gpt-4o-mini")
package router
