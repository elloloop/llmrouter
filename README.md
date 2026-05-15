# llmrouter

[![Go Reference](https://pkg.go.dev/badge/github.com/elloloop/llmrouter.svg)](https://pkg.go.dev/github.com/elloloop/llmrouter)
[![CI](https://github.com/elloloop/llmrouter/actions/workflows/ci.yml/badge.svg)](https://github.com/elloloop/llmrouter/actions/workflows/ci.yml)

A polyglot Go client for LLM providers. One OpenAI-shaped API surface across chat, embeddings, text-to-speech, and speech-to-text. Pluggable provider backends; configurable base URL + API key per provider; streaming-first.

> **v0.3 status:** 20 providers across chat, embeddings, TTS, and STT. Four root interfaces (`Provider`, `Embedder`, `Speaker`, `Transcriber`). See the [roadmap](https://elloloop.github.io/llmrouter/docs/project/roadmap) for v0.4 plans (WebSocket streaming for Deepgram + Cartesia).

## Why

There are great per-vendor SDKs in Go (`openai-go`, `anthropic-sdk-go`, `google.golang.org/genai`) and one unified library worth knowing about (`mozilla-ai/any-llm-go`). `llmrouter` exists for one specific shape of project:

- You want a **single API surface** across multiple vendors and multiple capabilities — chat, embeddings, TTS, STT — so application code doesn't branch on provider.
- You want **byte-level passthrough** where possible — for proxies, gateways, or any "intercept the OpenAI request, route it somewhere, stream it back" use case.
- You want to use **any URL with any API key** for OpenAI-compatible vendors (OpenRouter, Together, Groq, self-hosted) without a per-vendor SDK.
- You want **first-class streaming** with proper `context.Context` cancellation that propagates to the upstream HTTP request — across chat, audio output, and live transcription.

## Install

```bash
go get github.com/elloloop/llmrouter@v0.3.0
```

## Quick start

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/elloloop/llmrouter"
    "github.com/elloloop/llmrouter/providers/openai"
)

func main() {
    p, err := openai.New(llmrouter.WithAPIKey(os.Getenv("OPENAI_API_KEY")))
    if err != nil { log.Fatal(err) }

    stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
        Model:    "gpt-4o-mini",
        Messages: []llmrouter.Message{llmrouter.TextMessage("user", "Say hi in 5 words.")},
    })
    if err != nil { log.Fatal(err) }

    for chunk := range stream.Chunks() {
        for _, c := range chunk.Choices {
            fmt.Print(c.Delta.Content)
        }
    }
    if err := stream.Err(); err != nil { log.Fatal(err) }
}
```

## Providers (capability matrix)

| Provider | Chat | Embeddings | TTS | STT |
|---|:---:|:---:|:---:|:---:|
| OpenAI | ✓ | ✓ | ✓ | Whisper |
| Anthropic | ✓ | — *(use Voyage)* | — | — |
| Azure OpenAI | ✓ | ✓ | ✓ | Whisper |
| AWS Bedrock | ✓ | Titan + Cohere | — | — |
| Google Vertex AI | ✓ | ✓ | partial | — |
| Google Gemini (AI Studio) | ✓ | ✓ | ✓ | audio understanding |
| Cohere | ✓ | ✓ | — | — |
| Mistral | ✓ | ✓ | — | — |
| Together | ✓ | delegated | — | — |
| Groq | ✓ | — | — | Whisper |
| OpenRouter | ✓ | — | — | — |
| Fireworks | ✓ | — | — | — |
| DeepSeek | ✓ | — | — | — |
| xAI (Grok) | ✓ | — | — | — |
| Perplexity | ✓ | — | — | — |
| Cerebras | ✓ | — | — | — |
| **ElevenLabs** | — | — | ✓ | Scribe |
| **Deepgram** | — | — | — | Nova-3 |
| **Cartesia** | — | — | Sonic-2 | — |
| **Voyage AI** | — | ✓ | — | — |

Each row links to a per-provider docs page: [docs.tinykite.co/llmrouter/docs/providers/](https://elloloop.github.io/llmrouter/docs/providers/openai).

## Custom base URL — same provider, any endpoint

```go
// OpenRouter (OpenAI-compatible)
openrouter, _ := openai.New(
    llmrouter.WithAPIKey(os.Getenv("OPENROUTER_API_KEY")),
    llmrouter.WithBaseURL("https://openrouter.ai/api/v1"),
)

// Together
together, _ := openai.New(
    llmrouter.WithAPIKey(os.Getenv("TOGETHER_API_KEY")),
    llmrouter.WithBaseURL("https://api.together.xyz/v1"),
)

// Self-hosted vLLM / Ollama / anything OpenAI-compatible
local, _ := openai.New(
    llmrouter.WithAPIKey("not-needed-but-required-by-config"),
    llmrouter.WithBaseURL("http://localhost:11434/v1"),
)
```

## Anthropic with the OpenAI request shape

The Anthropic provider translates both the request body (OpenAI `messages` → Anthropic `/v1/messages`, system role lifted to top-level `system` field) and the SSE event stream. You write the same `ChatRequest`; the wire-level translation is hidden.

```go
import "github.com/elloloop/llmrouter/providers/anthropic"

p, _ := anthropic.New(llmrouter.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))

stream, _ := p.CompletionStream(ctx, llmrouter.ChatRequest{
    Model: "claude-3-7-sonnet-latest",
    Messages: []llmrouter.Message{
        llmrouter.TextMessage("system", "You are concise."),
        llmrouter.TextMessage("user", "Why is the sky blue?"),
    },
    MaxTokens: 256,
})
```

## Embeddings

```go
import "github.com/elloloop/llmrouter/providers/voyage"

p, _ := voyage.New(llmrouter.WithAPIKey(os.Getenv("VOYAGE_API_KEY")))

resp, _ := p.Embed(ctx, llmrouter.EmbedRequest{
    Model:    "voyage-3",
    Inputs:   []string{"What is RAG?", "Document chunk text..."},
    TaskType: "RETRIEVAL_QUERY", // mapped per-vendor (Cohere search_query, Voyage query, ...)
})

for i, vec := range resp.Embeddings {
    fmt.Printf("input %d: %d dims\n", i, len(vec))
}
```

See the [Embeddings concept page](https://elloloop.github.io/llmrouter/docs/concepts/embeddings) for the cross-vendor task-type mapping table.

## Audio: TTS

```go
import "github.com/elloloop/llmrouter/providers/cartesia"

p, _ := cartesia.New(llmrouter.WithAPIKey(os.Getenv("CARTESIA_API_KEY")))

stream, _ := p.Speak(ctx, llmrouter.SpeechRequest{
    Model:  "sonic-2",
    Input:  "Hello from Cartesia, streamed in under 100 milliseconds.",
    Voice:  "d46abd1d-2d02-43e8-819f-51fb652c1c61",
    Format: "mp3",
    Stream: true,
})

out, _ := os.Create("hello.mp3")
defer out.Close()
for chunk := range stream.Chunks() {
    out.Write(chunk.Data)
}
```

## Audio: STT

```go
import "github.com/elloloop/llmrouter/providers/deepgram"

p, _ := deepgram.New(llmrouter.WithAPIKey(os.Getenv("DEEPGRAM_API_KEY")))

f, _ := os.Open("meeting.wav")
defer f.Close()

stream, _ := p.Transcribe(ctx, llmrouter.TranscribeRequest{
    Model:          "nova-3",
    Audio:          f,
    AudioFormat:    "audio/wav",
    Language:       "en-US",
    ResponseFormat: "verbose_json",
})

for seg := range stream.Segments() {
    fmt.Printf("[%s] %s\n", seg.Start, seg.Text)
}
```

## API surface

| | |
|---|---|
| `llmrouter.Provider` | interface: `Name()`, `CompletionStream(ctx, req) (*Stream, error)` |
| `llmrouter.Embedder` | interface: `Embed(ctx, EmbedRequest) (*EmbedResponse, error)` |
| `llmrouter.Speaker` | interface: `Speak(ctx, SpeechRequest) (*AudioStream, error)` |
| `llmrouter.Transcriber` | interface: `Transcribe(ctx, TranscribeRequest) (*TranscriptStream, error)` |
| `llmrouter.ChatRequest` | OpenAI-shaped chat request; `Raw json.RawMessage` for byte passthrough |
| `llmrouter.Stream` / `AudioStream` / `TranscriptStream` | `Chunks() / Segments()`, `Err()`, `Cancel()` — same lifecycle |
| `llmrouter.WithAPIKey`, `WithBaseURL`, `WithHTTPClient`, `WithTimeout`, `WithExtra` | provider config options |
| `llmrouter.ErrUpstream` | non-2xx response wrapper with `Provider`, `StatusCode`, `Body` |

## Streaming semantics

- The producer goroutine pushes values into a buffered channel; the consumer reads until close.
- Cancelling the `ctx` passed to `CompletionStream` / `Speak` / `Transcribe` propagates to the upstream HTTP request.
- After the channel closes, call `Err()` for the terminal error (nil on success).
- Single-consumer: only one goroutine should read from the channel.

The same lifecycle applies across all three stream types — if you've used `Stream`, you already know how to use `AudioStream` and `TranscriptStream`.

## Roadmap

- **Shipped — v0.3**: Embeddings + TTS + STT root interfaces. Four new specialist providers (ElevenLabs, Deepgram, Cartesia, Voyage AI). OpenAI/Azure/Gemini gain TTS+STT+embeddings. Bedrock/Vertex/Cohere/Mistral gain embeddings.
- **Shipped — v0.2**: Azure OpenAI Service, AWS Bedrock, Google Vertex AI, Gemini (AI Studio), Cohere, Mistral. Typed tool-call passthrough. Extended thinking. Prompt caching. Multimodal content helpers. Ten OpenAI-compatible vendors verified.
- **Planned — v0.4**: WebSocket streaming for Deepgram + Cartesia (real-time). Anthropic embeddings if/when GA. First-class typed `ToolResultMessage`.
- **v1.0**: API freeze. Until then, expect minor breakages between minor versions.

## Comparisons

| | `llmrouter` (this) | `mozilla-ai/any-llm-go` | per-vendor SDKs |
|---|---|---|---|
| Chat providers (v0.3) | 16 (incl. all 3 cloud triple + 8 OpenAI-compat) | 10 | one per package |
| Embedding providers | 9 (incl. Voyage) | partial | one per package |
| TTS providers | 5 (OpenAI, Azure, Gemini, ElevenLabs, Cartesia) | no | yes (per vendor) |
| STT providers | 5 (Whisper variants + Gemini + Scribe + Nova-3) | no | yes (per vendor) |
| Byte passthrough | yes (`Chunk.Raw`, `AudioChunk.Raw`) | no (parses to typed shapes) | n/a |
| Cloud triple | yes (Azure / Bedrock / Vertex) | partial / no / no | yes (each vendor) |
| Streaming | channel + context cancel | channel + context cancel | varies |
| Mid-stream errors | yes (since v0.2) | no | varies |
| Pre-1.0 churn | expect API changes between minors | expect API changes | stable |

## License

[Apache 2.0](./LICENSE)

## Contributing

Issues and PRs welcome. For larger changes, please open an issue first to discuss the shape.

The library was extracted from [Kite AI Router](https://github.com/tinykite-co) — a self-hosted LLM gateway. The provider code originated there and got cleaner along the way; thanks to the Tollgate codebase for proving it out in production.
