# Compatibility Promise

llmrouter v1.0 marks the first stable release. From this point forward, the public API listed below is covered by semantic versioning: breaking changes require a major-version bump (v2.0.0, v3.0.0, …) and ship with at least one full minor-version notice period.

This document is the source of truth for what counts as "the public API."

## What's stable in v1.x

### Root package (`github.com/elloloop/llmrouter`)

**Interfaces** — implementations and consumers can rely on the method set staying frozen.

- `Provider` — chat completion via `CompletionStream`
- `Embedder` — vector embeddings via `Embed`
- `Speaker` — TTS via `Speak`
- `Transcriber` — STT via `Transcribe`
- `Reranker` — RAG rerank via `Rerank`

**Concrete types** — fields may be ADDED in minor versions (always as optional `omitempty`). Existing fields will not be removed, renamed, or have their types changed within v1.x.

- `ChatRequest`, `Message`, `Chunk`, `Choice`, `Delta`, `Usage`
- `Tool`, `ToolFunction`, `ToolChoice`, `ToolCallDelta`, `ToolCallFunctionDelta`
- `ResponseSchema`
- `EmbedRequest`, `EmbedResponse`
- `SpeechRequest`, `AudioStream`, `AudioChunk`, `AudioProducerHooks`
- `TranscribeRequest`, `TranscriptStream`, `TranscriptSegment`, `TranscriptWord`, `TranscriptProducerHooks`
- `RerankRequest`, `RerankResponse`, `RerankResult`
- `ContentPart`, `Config`, `Stream`, `ProducerHooks`
- `ErrUpstream`, `ErrInvalidConfig`

**Constructors** — signatures frozen.

- `TextMessage`, `MultipartMessage`, `Text`, `ImageURL`, `ImageBytes`, `ToolResultMessage`
- `NewConfig`, `NewStream`, `NewAudioStream`, `NewTranscriptStream`
- All `With*` option constructors: `WithAPIKey`, `WithBaseURL`, `WithHTTPClient`, `WithTimeout`, `WithExtra`

### Provider packages

For every package under `providers/<vendor>/`:

- The exported `Provider` struct and its `New(opts ...llmrouter.Option) (*Provider, error)` constructor are stable
- The `Name() string` return value is frozen (used as the `Provider` field on `ErrUpstream`; downstream code may switch on it)
- Vendor-specific options (e.g. `azureopenai.WithDeployment`, `vertexanthropic.WithTokenSource`) are stable
- The implementation may evolve (better error messages, more event types decoded, performance work) as long as the public surface stays compatible

### Router package (`github.com/elloloop/llmrouter/router`)

The `Resolve` entry point + `Request` / `Credentials` / `EnvVars` structs are stable.

The routing matrix (which `Platform` × `ModelFamily` combinations work) will GROW in minor versions. New `Platform` and `ModelFamily` constants may be added; existing constants will not be removed or renumbered.

Model-ID translation helpers (`bedrockModelID`, `vertexAnthropicModelID`) may gain new entries; existing entries will not be removed.

## What's NOT covered

These can change without a major-version bump:

1. **Unexported APIs.** Anything starting with a lowercase letter in any package is implementation detail.
2. **Test helpers.** `package <pkg>` tests and test files are not part of the public API even when they import exported symbols.
3. **Examples.** Programs under `examples/` are illustrative — their CLI flags, env-var names, and stdout formats may change.
4. **Documentation site** at https://elloloop.github.io/llmrouter/. Content, navigation, and URL shape may change. The Go package documentation (godoc comments) is part of the stable API surface, but the rendered docs site is presentation.
5. **CI workflows** under `.github/workflows/`. We may add, remove, or restructure these freely.
6. **Vendor SDK dependencies.** When AWS, Google, or other vendor SDKs release new versions with breaking changes, we will update our dependency and propagate the necessary changes — but only when those changes are user-observable and only with release notes. We pin to specific SDK versions in `go.sum` for reproducibility.

## What about upstream behaviour?

llmrouter is a client library. The behaviour of the LLM providers themselves (OpenAI, Anthropic, Bedrock, Vertex, …) is not under our control. When an upstream:

- Changes a model name → callers update the model id; no library change
- Changes the wire format of an existing endpoint → we patch the provider package, ship as `fix:` or `feat:`. If the change requires breaking our public API to surface (rare), it ships as a major bump.
- Adds a new feature (new field, new event type) → we add support in a minor release with a new field on the relevant request/response struct (always `omitempty`).
- Deprecates an endpoint → we maintain support until the upstream removes it, then drop in a major release with a notice period in the prior minor release.

## Deprecation policy

When we need to remove a public API:

1. Mark it `Deprecated:` in the doc comment in release **N**
2. Keep it functional through at least one full minor cycle
3. Add a recommended replacement to the doc comment + release notes
4. Remove in release **N+1.0.0** (next major)

Example timeline:

- v1.4.0 — `OldFunc` marked Deprecated, recommends `NewFunc`
- v1.5.0 — `OldFunc` still works; doc comment now warns "will be removed in v2.0.0"
- v1.6.0 — `OldFunc` still works
- v2.0.0 — `OldFunc` removed; migration guide in release notes

## Behaviour changes

Behaviour changes — same API surface, different runtime behaviour — are documented in:

- The CHANGELOG.md entry for the release
- A `BREAKING CHANGE:` footer on the conventional commit, if the change is meaningful enough to warrant a major bump even though signatures didn't change
- A migration note on the docs site

Examples of behaviour changes that warrant a major bump:

- A default value changing (e.g. `MaxTokens` default flipping from 4096 to 8192)
- An option silently being applied differently (e.g. `WithTimeout` now affecting streaming reads where it didn't before)
- A sentinel error wrapping changing in a way that breaks `errors.Is` / `errors.As` chains

Examples that DON'T warrant a major bump:

- Tightening or relaxing input validation (callers receiving better error messages, or accepting input they couldn't before)
- Performance changes
- Internal restructuring that doesn't surface
- Bug fixes that bring observable behaviour into line with documented behaviour

## Experimental subpackages

There are currently NO experimental subpackages in v1.0. Should one become necessary in the future, it will live under `experimental/<name>/` and ship with its own README clearly stating "API may change without major-version bump."

## Reporting compatibility issues

If you find code that broke between v1.x.y and v1.x.z where the SemVer rules say it shouldn't have, please open an issue with the conventional commit prefix `bug:` and tag it `compat-regression`. We treat compatibility regressions as high-severity bugs and patch them in the latest minor within 7 days where possible.

---

For the full evolution of the public API across versions, see [CHANGELOG.md](./CHANGELOG.md).
