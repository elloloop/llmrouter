# llmrouter

[![Go Reference](https://pkg.go.dev/badge/github.com/elloloop/llmrouter.svg)](https://pkg.go.dev/github.com/elloloop/llmrouter)
[![CI](https://github.com/elloloop/llmrouter/actions/workflows/ci.yml/badge.svg)](https://github.com/elloloop/llmrouter/actions/workflows/ci.yml)

A polyglot Go client for LLM providers. One OpenAI-shaped API surface; pluggable provider backends; configurable base URL + API key per provider; streaming-first.

> **v0.1 status:** OpenAI + Anthropic shipped. Azure OpenAI, AWS Bedrock, Google Vertex on the v0.2 roadmap.

## Why

There are great per-vendor SDKs in Go (`openai-go`, `anthropic-sdk-go`, `google.golang.org/genai`) and one unified library worth knowing about (`mozilla-ai/any-llm-go`). `llmrouter` exists for one specific shape of project:

- You want a **single API surface** across multiple vendors so application code doesn't branch on provider.
- You want **byte-level passthrough** where possible — for proxies, gateways, or any "intercept the OpenAI request, route it somewhere, stream it back" use case.
- You want to use **any URL with any API key** for OpenAI-compatible vendors (OpenRouter, Together, Groq, self-hosted) without a per-vendor SDK.
- You want **first-class streaming** with proper `context.Context` cancellation that propagates to the upstream HTTP request.

If you only call one vendor, use that vendor's official SDK. If you want a Python LiteLLM equivalent with type-safe wrappers around tool use and embeddings, see `mozilla-ai/any-llm-go`.

## Install

```bash
go get github.com/elloloop/llmrouter
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

The Anthropic provider translates both the request body (OpenAI `messages` → Anthropic `/v1/messages`, system role lifted to top-level `system` field) and the SSE event stream (Anthropic `message_start` / `content_block_delta` / `message_delta` / `message_stop` events → OpenAI delta chunks). You write the same `ChatRequest`; the wire-level translation is hidden.

```go
import "github.com/elloloop/llmrouter/providers/anthropic"

p, _ := anthropic.New(llmrouter.WithAPIKey(os.Getenv("ANTHROPIC_API_KEY")))

stream, _ := p.CompletionStream(ctx, llmrouter.ChatRequest{
    Model: "claude-3-5-sonnet-latest",
    Messages: []llmrouter.Message{
        llmrouter.TextMessage("system", "You are concise."),
        llmrouter.TextMessage("user", "Why is the sky blue?"),
    },
    MaxTokens: 256,
})
```

## API surface

| | |
|---|---|
| `llmrouter.Provider` | interface: `Name()`, `CompletionStream(ctx, req) (*Stream, error)` |
| `llmrouter.ChatRequest` | OpenAI-shaped request; `Raw json.RawMessage` for byte passthrough |
| `llmrouter.Message` | `Role`, `Content` (json.RawMessage for multimodal support) |
| `llmrouter.Stream` | `Chunks() <-chan Chunk`, `Err() error`, `Cancel()` |
| `llmrouter.Chunk` | normalized OpenAI delta + `Raw` for byte-level passthrough |
| `llmrouter.WithAPIKey`, `WithBaseURL`, `WithHTTPClient`, `WithTimeout`, `WithExtra` | provider config options |
| `llmrouter.ErrUpstream` | non-2xx response wrapper with `Provider`, `StatusCode`, `Body` |

## Streaming semantics

- The producer goroutine pushes `Chunk` values into a buffered channel; the consumer reads until close.
- Cancelling the `ctx` passed to `CompletionStream` propagates to the upstream HTTP request (the consumer can also call `stream.Cancel()`).
- After `Chunks()` closes, call `stream.Err()` for the terminal error (nil on success).
- Single-consumer: only one goroutine should read from `Chunks()`.

## Roadmap

- **v0.2**: Azure OpenAI Service (deployment URL + `api-key` header + `api-version` query param), AWS Bedrock (SigV4 auth + per-model-family body shapes), Google Vertex AI (ADC auth, project/region).
- **v0.3**: optional tool-call passthrough, multimodal content helpers.
- **v1.0**: API freeze. Until then, expect minor breakages between minor versions.

## Comparisons

| | `llmrouter` (this) | `mozilla-ai/any-llm-go` | per-vendor SDKs |
|---|---|---|---|
| Providers (v0.1) | OpenAI + Anthropic + any OpenAI-compat via base URL | 10 providers | one per package |
| Byte passthrough | yes (`Chunk.Raw`) | no (parses to typed shapes) | n/a |
| Azure / Bedrock / Vertex | planned v0.2 | partial / no / no | yes (each vendor) |
| Streaming | channel + context cancel | channel + context cancel | varies |
| Pre-1.0 churn | expect API changes | expect API changes | stable |

## License

[Apache 2.0](./LICENSE)

## Contributing

Issues and PRs welcome. For larger changes, please open an issue first to discuss the shape.

The library was extracted from [Kite AI Router](https://github.com/tinykite-co) — a self-hosted LLM gateway. The provider code originated there and got cleaner along the way; thanks to the Tollgate codebase for proving it out in production.
