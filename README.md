# llmrouter

[![Go Reference](https://pkg.go.dev/badge/github.com/elloloop/llmrouter.svg)](https://pkg.go.dev/github.com/elloloop/llmrouter)
[![CI](https://github.com/elloloop/llmrouter/actions/workflows/ci.yml/badge.svg)](https://github.com/elloloop/llmrouter/actions/workflows/ci.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/elloloop/llmrouter)](https://goreportcard.com/report/github.com/elloloop/llmrouter)
[![Latest release](https://img.shields.io/github/v/release/elloloop/llmrouter?sort=semver)](https://github.com/elloloop/llmrouter/releases)
[![Go version](https://img.shields.io/github/go-mod/go-version/elloloop/llmrouter)](go.mod)
[![License](https://img.shields.io/github/license/elloloop/llmrouter)](LICENSE)

**One Go API across 22 LLM providers — chat, embeddings, TTS, STT, realtime voice, and rerank — so your application code never branches on provider.**

Docs: **<https://elloloop.github.io/llmrouter/>**

## Why llmrouter

- **One API surface across 22 vendors.** OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, Google Vertex AI, Google Gemini, Cohere, Mistral, Groq, Together, OpenRouter, Fireworks, DeepSeek, xAI (Grok), Perplexity, Cerebras, ElevenLabs, Deepgram, Cartesia, Voyage AI, plus dedicated realtime packages for OpenAI Realtime and Gemini Live. Same `ChatRequest`, same streaming lifecycle, no provider-specific branching in your app code.
- **Six capabilities, not just chat.** Streaming chat, embeddings, text-to-speech (TTS), speech-to-text (STT) with word-level timing, full-duplex realtime sessions over WebSocket for voice agents, rerank for RAG retrieval, and structured outputs via JSON Schema (native on OpenAI, forced tool-use on Anthropic, `ResponseMIMEType` on Vertex/Gemini).
- **Byte-level passthrough for LLM gateways.** `ChatRequest.Raw` and `Chunk.Raw` let you proxy upstream OpenAI traffic byte-identically — ideal for self-hosted gateways, observability layers, and routing/caching proxies.
- **Boring dependencies.** Standard library plus well-known major-vendor SDKs only (AWS SDK v2, `google.golang.org/genai`, `coder/websocket`, `google/uuid`). Apache-2.0. No exotic transitive footprint.

## Install

```bash
go get github.com/elloloop/llmrouter@latest
```

Requires Go 1.24+.

## Quick start

### Streaming chat (OpenAI)

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
    if err != nil {
        log.Fatal(err)
    }

    stream, err := p.CompletionStream(context.Background(), llmrouter.ChatRequest{
        Model: "gpt-4o-mini",
        Messages: []llmrouter.Message{
            llmrouter.TextMessage("user", "Say hi in 5 words."),
        },
    })
    if err != nil {
        log.Fatal(err)
    }

    for chunk := range stream.Chunks() {
        for _, c := range chunk.Choices {
            fmt.Print(c.Delta.Content)
        }
    }
    if err := stream.Err(); err != nil {
        log.Fatal(err)
    }
}
```

### Same code, different provider (Anthropic Claude)

Only the import and the model name change. The Anthropic provider translates the request body and the SSE event stream to the same shape.

```go
import (
    "github.com/elloloop/llmrouter"
    "github.com/elloloop/llmrouter/providers/anthropic"
)

p, _ := anthropic.New(llmrouter.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))

stream, _ := p.CompletionStream(ctx, llmrouter.ChatRequest{
    Model:     "claude-3-7-sonnet-latest",
    MaxTokens: 256,
    Messages: []llmrouter.Message{
        llmrouter.TextMessage("system", "You are concise."),
        llmrouter.TextMessage("user", "Why is the sky blue?"),
    },
})
```

### Embeddings

```go
import (
    "github.com/elloloop/llmrouter"
    "github.com/elloloop/llmrouter/providers/voyage"
)

p, _ := voyage.New(llmrouter.WithAPIKey(os.Getenv("VOYAGE_API_KEY")))

resp, _ := p.Embed(ctx, llmrouter.EmbedRequest{
    Model:    "voyage-3",
    Inputs:   []string{"What is RAG?", "Document chunk text..."},
    TaskType: "RETRIEVAL_QUERY",
})

for i, vec := range resp.Embeddings {
    fmt.Printf("input %d: %d dims\n", i, len(vec))
}
```

### Text-to-speech (ElevenLabs)

```go
import (
    "github.com/elloloop/llmrouter"
    "github.com/elloloop/llmrouter/providers/elevenlabs"
)

p, _ := elevenlabs.New(llmrouter.WithAPIKey(os.Getenv("ELEVENLABS_API_KEY")))

stream, _ := p.Speak(ctx, llmrouter.SpeechRequest{
    Model:  "eleven_turbo_v2_5",
    Input:  "Hello from ElevenLabs.",
    Voice:  "21m00Tcm4TlvDq8ikWAM",
    Format: "mp3",
})

out, _ := os.Create("hello.mp3")
defer out.Close()
for chunk := range stream.Chunks() {
    out.Write(chunk.Data)
}
if err := stream.Err(); err != nil {
    log.Fatal(err)
}
```

### Custom base URL — any OpenAI-compatible endpoint

```go
// Together
together, _ := openai.New(
    llmrouter.WithAPIKey(os.Getenv("TOGETHER_API_KEY")),
    llmrouter.WithBaseURL("https://api.together.xyz/v1"),
)

// Self-hosted vLLM / Ollama / anything OpenAI-compatible
local, _ := openai.New(
    llmrouter.WithAPIKey("not-needed"),
    llmrouter.WithBaseURL("http://localhost:11434/v1"),
)
```

## Capability matrix

22 provider packages. ✓ = supported, — = not applicable for this vendor.

| Provider | Chat | Embed | TTS | STT | Realtime | Rerank | Structured outputs |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenAI | ✓ | ✓ | ✓ | ✓ Whisper | — | — | ✓ native |
| Anthropic | ✓ | — (Voyage shim) | — | — | — | — | ✓ forced tool-use |
| Azure OpenAI | ✓ | ✓ | ✓ | ✓ Whisper | — | — | ✓ |
| AWS Bedrock | ✓ | ✓ Titan + Cohere | — | — | — | — | — |
| Google Vertex AI | ✓ | ✓ | ✓ | ✓ | — | — | ✓ `ResponseMIMEType` |
| Google Gemini | ✓ | ✓ | ✓ | ✓ | — | — | ✓ `ResponseMIMEType` |
| Cohere | ✓ | ✓ | — | — | — | ✓ `rerank-v3.5` | — |
| Mistral | ✓ | ✓ | — | — | — | — | — |
| Groq | ✓ | — | — | ✓ Whisper | — | — | — |
| Together | ✓ | ✓ | — | — | — | ✓ `Llama-Rank-V1` | — |
| OpenRouter | ✓ | — | — | — | — | — | — |
| Fireworks | ✓ | ✓ | — | — | — | — | — |
| DeepSeek | ✓ | ✓ | — | — | — | — | — |
| xAI (Grok) | ✓ | — | — | — | — | — | — |
| Perplexity | ✓ | — | — | — | — | — | — |
| Cerebras | ✓ | — | — | — | — | — | — |
| ElevenLabs | — | — | ✓ + `SpeakRealtime` | ✓ Scribe | — | — | — |
| Deepgram | — | — | — | ✓ Nova-3 (WS) | — | — | — |
| Cartesia | — | — | ✓ Sonic-2 + `SpeakRealtime` | — | — | — | — |
| Voyage AI | — | ✓ | — | — | — | ✓ `rerank-2` | — |
| **OpenAI Realtime** | session | — | session | session | ✓ full duplex + tools | — | — |
| **Gemini Live** | session | — | session | session | ✓ full duplex + tools | — | — |

Per-provider docs: <https://elloloop.github.io/llmrouter/docs/providers/openai>

## Architecture

llmrouter exposes a small set of root interfaces (`Provider`, `Embedder`, `Speaker`, `Transcriber`, `Reranker`) backed by per-vendor packages under `providers/`. Each provider package implements only the interfaces it actually supports — Anthropic implements `Provider`, Voyage implements `Embedder` and `Reranker`, Deepgram implements `Transcriber` (over WebSocket), and so on. There is no central dispatcher and no plugin registry; you import the provider package(s) you need and pass them around as the relevant interface.

The library is OpenAI-shaped on the wire: requests use the OpenAI `messages` array and Chat Completion chunk format. Non-OpenAI providers translate request/response at the package boundary — Anthropic lifts the `system` role to a top-level field, Bedrock maps to Converse Stream, Vertex/Gemini hit the Generative Language API. For OpenAI-compatible vendors (Together, Groq, OpenRouter, Fireworks, DeepSeek, xAI, Perplexity, Cerebras, vLLM, Ollama) the request flows through byte-passthrough — `ChatRequest.Raw` is forwarded as-is, and `Chunk.Raw` carries the original wire bytes back to the caller. This makes llmrouter usable as the engine behind a self-hosted LLM gateway that needs to stay byte-identical to upstream OpenAI.

Streaming is uniform across capabilities: every long-lived call returns a typed stream (`Stream`, `AudioStream`, `TranscriptStream`) exposing a `Chunks()` / `Segments()` channel, an `Err()` terminal error, and a `Cancel()` method. Cancelling the `context.Context` passed in propagates to the upstream HTTP request, and mid-stream errors surface through `Err()` rather than being silently swallowed.

## API surface

| Symbol | Purpose |
|---|---|
| `Provider` | interface: `Name()`, `CompletionStream(ctx, req) (*Stream, error)` |
| `Embedder` | interface: `Embed(ctx, EmbedRequest) (*EmbedResponse, error)` |
| `Speaker` | interface: `Speak(ctx, SpeechRequest) (*AudioStream, error)` |
| `Transcriber` | interface: `Transcribe(ctx, TranscribeRequest) (*TranscriptStream, error)` |
| `Reranker` | interface: `Rerank(ctx, RerankRequest) (*RerankResponse, error)` |
| `ChatRequest` | OpenAI-shaped chat request; `Raw json.RawMessage` for byte passthrough |
| `EmbedRequest` / `SpeechRequest` / `TranscribeRequest` / `RerankRequest` | per-capability request types |
| `ResponseSchema` | JSON-schema constraint attached via `ChatRequest.ResponseSchema` |
| `Message` / `TextMessage(role, text)` | typed message; `Content` is `json.RawMessage` for multimodal arrays |
| `MultipartMessage(role, parts...)` | helper for vision / audio content blocks |
| `ToolResultMessage(toolCallID, content)` | helper for tool-result messages with cross-vendor translation |
| `Stream` / `AudioStream` / `TranscriptStream` | `Chunks() / Segments()`, `Err()`, `Cancel()` — same lifecycle |
| `openairealtime.Provider.Connect` | full-duplex `gpt-4o-realtime` session with typed tool use |
| `geminilive.Provider.Connect` | full-duplex Gemini `BidiGenerateContent` session |
| `cartesia.SpeakRealtime` / `elevenlabs.SpeakRealtime` | WebSocket TTS with multi-turn append |
| `anthropic.NewRecommendedEmbedder` | Voyage-backed `Embedder` shim (Anthropic's documented recommendation) |
| `WithAPIKey` / `WithBaseURL` / `WithHTTPClient` / `WithTimeout` / `WithExtra` | provider config options |
| `ErrUpstream` | non-2xx response wrapper with `Provider`, `StatusCode`, `Body` |
| `ErrInvalidConfig` | constructor-time validation error |

## Comparison vs alternatives

| | `llmrouter` | per-vendor SDKs | `mozilla-ai/any-llm-go` | LiteLLM |
|---|---|---|---|---|
| Language | Go | Go | Go | Python only |
| Chat providers | 22 | one per package | ~10 | many |
| Cloud triple (Azure / Bedrock / Vertex) | ✓ all three | per vendor | partial / no / no | ✓ |
| TTS / STT / Rerank | ✓ all three | per vendor | — | partial |
| Realtime WebSocket (voice agents) | ✓ OpenAI + Gemini Live | per vendor | — | — |
| Byte passthrough for gateways | ✓ `Chunk.Raw`, `ChatRequest.Raw` | n/a | — | partial |
| Streaming lifecycle | `chan` + ctx cancel + `Err()` | varies per SDK | similar | sync wrappers |
| Mid-stream errors surfaced | ✓ | varies | ✗ | varies |

**vs per-vendor SDKs:** great if you only use one vendor. Painful once you support two — you write provider-specific branching, parse two SSE formats, handle two error shapes.

**vs `mozilla-ai/any-llm-go`:** closest Go alternative. llmrouter additionally supports Azure / Bedrock / Vertex, TTS/STT/Rerank/Realtime, and byte-passthrough for gateways.

**vs LiteLLM:** great if you're on Python. llmrouter is the Go-native equivalent for backend services and gateways that don't want a Python sidecar.

**vs direct HTTP:** you'd re-implement SSE parsing, error normalization, context cancellation, and retry semantics per vendor.

## Roadmap

See the [full roadmap](https://elloloop.github.io/llmrouter/docs/project/roadmap/). Highlights:

- **v0.6 (next):** OpenAI Files API + Assistants v2, cross-vendor batch APIs, audio in chat (`gpt-4o-audio-preview`), semantic caching, prompt management / versioning.
- **v0.7:** Anthropic native embeddings when GA, additional realtime providers, gateway-side circuit-breaker primitives.
- **v1.0:** API freeze. Until then, expect minor breakages between minor versions — pinned modules are recommended.

See the [CHANGELOG](./CHANGELOG.md) for shipped releases.

## Contributing

Issues and PRs welcome. For larger changes, please open an issue first to discuss the shape. See [CONTRIBUTING.md](./CONTRIBUTING.md) if present, and the [docs site](https://elloloop.github.io/llmrouter/) for design notes and per-provider implementation guides.

## License

[Apache 2.0](./LICENSE). See [NOTICE](./NOTICE) for attributions.

## Acknowledgements

llmrouter was extracted from the **Kite AI Router** (codename: Tollgate), a self-hosted LLM gateway built at `tinykite-co`. The OpenAI and Anthropic provider implementations originated there and were cleaned up during extraction in May 2026. See [NOTICE](./NOTICE) for full attributions.
