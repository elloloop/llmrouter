---
name: New provider request
about: Request support for an LLM provider not yet covered
title: 'feat: <provider name> provider'
labels: enhancement, new-provider
assignees: ''
---

<!--
Use this template when the provider you want isn't in the matrix at
https://elloloop.github.io/llmrouter/docs/providers/. For features on
an existing provider (e.g. "OpenAI Files API", "Anthropic batch")
use the feature request template instead.
-->

## Provider

- **Name:**
- **Homepage:**
- **API documentation:**
- **Pricing page (for context on adoption):**

## Capability

Which capability would the provider serve? Tick all that apply.

- [ ] Chat (`llmrouter.Provider`)
- [ ] Embeddings (`llmrouter.Embedder`)
- [ ] Text-to-speech (`llmrouter.Speaker`)
- [ ] Speech-to-text (`llmrouter.Transcriber`)
- [ ] Realtime / bidirectional WebSocket session
- [ ] Rerank (`llmrouter.Reranker`)

## Wire-level shape

- **API surface:** OpenAI-compatible / native / SDK-only / other (describe)
- **Auth shape:** Bearer / api-key header / SigV4 / ADC / OAuth / other
- **Streaming protocol:** SSE / WebSocket / AWS event-stream / batch only / other
- **Body format:** OpenAI-shaped / vendor-native (link to spec)

## Why this should ship

<!-- Who would use it? What's the use case llmrouter doesn't already
support via an existing provider? Specifically: if this provider is
OpenAI-compatible, can callers already reach it via providers/openai
with WithBaseURL? If yes, what's missing from that path that a
dedicated provider would add (rewriting Provider name in errors,
custom headers, etc.)? -->

## Available SDKs

- [ ] Vendor publishes an official Go SDK (link):
- [ ] Vendor publishes an official non-Go SDK we can reference for the wire format:
- [ ] Only HTTP docs are available

## Volunteer to implement?

- [ ] I plan to send a PR
- [ ] I'd like to but need guidance on the pattern
- [ ] Just requesting; happy for someone else to pick this up
