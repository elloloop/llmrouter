# CLAUDE.md

## How I expect you to write code

**No shortcuts. "Simple" never means "sloppy."** A small diff that hardcodes,
duplicates, or skips a test isn't simpler — it's deferred cost.

1. **Fix causes, not symptoms.** Find the root cause before fixing. If you're
   applying a workaround, say so explicitly and explain why. Never swallow an
   exception or silence an error to make a problem disappear.

2. **Think about consequences.** Before changing shared or widely-used code,
   trace its callers and the invariants they rely on. A fix that's locally
   correct but breaks something elsewhere — now or later — is not a fix.

3. **SOLID, sensibly.** One responsibility per class/widget/function. Separate
   pure logic from I/O so it can be tested. Inject dependencies that cross a
   boundary so they're mockable. Don't add abstractions for things that don't
   cross a boundary.

4. **DRY about knowledge, not appearance.** Don't duplicate a rule or decision.
   Code that merely looks similar but changes for different reasons stays
   separate. When unsure, prefer duplication over a premature/wrong abstraction.

5. **No hardcoded values.** No magic numbers or strings inline — give them
   names. Environment/tenant/feature-specific values go in typed config in
   application code, never scattered literals, never the database.

6. **Readable & maintainable.** Clear names, short flat functions, early
   returns over deep nesting. Comments explain *why*, not *what*. Match the
   existing style of the file you're editing.

7. **Testable, and prove it.** Ship a test for behavior you add or change. If
   something is hard to test, that's a design smell — restructure until it
   isn't. "Works but can't be tested" means it isn't done.

A change is done only when: the cause (not a symptom) is fixed, no new hardcoded
values, a test covers it, and the analyzer/formatter are clean.

## Project facts

> Keep these current as the repo evolves; only write what you've confirmed.

- **Setup:** `go mod download` (deps); per CONTRIBUTING the whole bootstrap is just `go test -race -count=1 ./...`. Go 1.24+.
- **Analyze/lint:** `go vet ./...` and `staticcheck ./...` (install via `go install honnef.co/go/tools/cmd/staticcheck@latest`); `govulncheck ./...` runs informationally.
- **Test (all):** `go test -race -count=1 ./...`
- **Test (single):** `go test -race -run TestName ./...` (add the package path, e.g. `./router`, to scope it).
- **Format:** `gofmt -w .` (CI fails on any `gofmt -d .` diff).
- **Run an app:** examples are runnable Go programs, e.g. `go run ./examples/chat-streaming` (requires the relevant provider API key env var).
- **Repo layout:** root package `llmrouter` (core types/interfaces: `llmrouter.go`, `audio.go`, `embeddings.go`, `rerank.go`, `stream.go`, `messages.go`, `options.go`, `errors.go`); `providers/` (24 per-vendor packages); `router/` (model-id/family/platform routing); `examples/` (runnable demos); `docs-site/` (Astro docs source); `docs/` (generated docs output).
- **State management / data layer:** stateless library, no database. Provider config flows through functional options (`WithAPIKey` / `WithBaseURL` / `WithHTTPClient` / `WithTimeout` / `WithExtra`); secrets come from env vars at the call site. Wire format is OpenAI-shaped; non-OpenAI providers translate at the package boundary, and OpenAI-compatible vendors use byte passthrough via `ChatRequest.Raw` / `Chunk.Raw`.
- **Generated files (do not hand-edit):** `/docs/` (built by docs-site, deployed to GitHub Pages — gitignored), `docs-site/dist/`, `go.sum`, and release-please artifacts (`CHANGELOG.md`, `.release-please-manifest.json`).
- **Gotchas:** single Go module (no per-example submodules) — module path `github.com/elloloop/llmrouter`. Each provider package implements only the interfaces it supports (`Provider`, `Embedder`, `Speaker`, `Transcriber`, `Reranker`); no central dispatcher or registry. Pre-v1.0: expect minor breakages between minor versions. CI runs vet, race tests, build, staticcheck, gofmt, govulncheck (informational), and coverage; releases are automated via release-please.
