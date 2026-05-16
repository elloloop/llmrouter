# Contributing to llmrouter

Thanks for considering a contribution. `llmrouter` is a small, deliberately narrow Go library with a high bar on stability, test coverage, and dependency footprint. The rules below exist so contributions land cleanly and stay landed.

Before you start work on anything beyond a typo or a one-file fix, please open an issue first so we can agree on the shape. Most PR rework happens because the design conversation didn't happen up front.

## Code of conduct

This project follows the [Contributor Covenant v2.1](./CODE_OF_CONDUCT.md). By participating you agree to uphold it. Report concerns privately to `conduct@elloloop.com`.

## Reporting bugs

File a [bug report](https://github.com/elloloop/llmrouter/issues/new?template=bug_report.md). Include all of the following — if any are missing, the first reply is usually going to ask for them:

- Go version (`go version`)
- OS and architecture (`go env GOOS GOARCH`)
- `llmrouter` version (the tag in your `go.mod`)
- Which provider you were calling, and its base URL if non-default
- A **minimal reproduction**: the smallest Go program that triggers the bug. If the bug only reproduces against a live provider, include the upstream response or SSE transcript (redact API keys).
- Expected behaviour and actual behaviour
- The full error message and stack trace, if there is one

## Suggesting features

Open a [feature request](https://github.com/elloloop/llmrouter/issues/new?template=feature_request.md) with the `type:feature` label. Describe the use case first — what are you trying to do today that's hard? — then sketch a proposed API. Alternatives considered and explicit out-of-scope notes help the review go faster.

For a new provider, use the [new provider template](https://github.com/elloloop/llmrouter/issues/new?template=new_provider.md). It collects the wire-level details we need (capability, auth shape, upstream docs link) up front.

## Development setup

Clone the repository and run the test suite — that's the whole bootstrap:

```bash
git clone https://github.com/elloloop/llmrouter
cd llmrouter
go test -race -count=1 ./...
```

Requirements:

- **Go 1.23+** — pinned in `go.mod`. We use generics and a few newer stdlib APIs.
- **`gofmt`** — bundled with the Go toolchain. Run it before every commit (most editors do it on save).
- **Nothing else.** No `make`, no `just`, no docker, no protobuf compiler, no pre-commit hooks. `go test`, `go vet`, `gofmt`, and your editor are the whole tool chain.

Optional but recommended:

- `go vet ./...` for the standard set of lints (CI runs it on every PR).
- `go test -race -count=1 ./...` to bypass the test cache and catch streaming concurrency issues.

## Project layout

The repository is intentionally flat:

```
llmrouter/
├── llmrouter.go            // core types: Provider, ChatRequest, Message, Chunk
├── options.go              // option pattern (WithAPIKey, WithBaseURL, ...)
├── stream.go               // Stream type, channel + goroutine lifecycle
├── errors.go               // ErrUpstream + helpers
├── *_test.go               // table-driven tests, mirroring each source file
├── providers/
│   ├── openai/             // OpenAI provider (passthrough)
│   ├── anthropic/          // Anthropic provider (translating)
│   └── <name>/             // one directory per provider
├── docs-site/              // Astro source for the docs site
├── docs/                   // GENERATED output of docs-site — do not edit by hand
├── README.md
├── LICENSE
└── go.mod
```

Two layout rules that matter:

- Anything user-facing lives in the root package or under `providers/<name>/`. There is no `internal/` directory yet; if private helpers grow enough to need one, we'll add it.
- **The `docs/` directory is generated output.** It is the built static site that GitHub Pages serves. Edits to files under `docs/` will be wiped by the next build. Edit the Astro source under `docs-site/` instead.

## Dependency policy

`llmrouter` has a strict and unusual dependency policy. The bar is:

1. The Go standard library, or
2. An official vendor SDK for a provider we ship — i.e. published by Google, AWS, Anthropic, or OpenAI directly.

That's it. We do not depend on community helper libraries, structured-logging frameworks, error packages, or any third-party utility.

The current tree is the standard library plus two justified exceptions:

- [`google/uuid`](https://github.com/google/uuid) — reimplementing RFC 4122 here would be silly, and Google's package is the de facto standard.
- [`coder/websocket`](https://github.com/coder/websocket) — the realtime providers need a maintained, context-aware WebSocket client. The stdlib has none, `gorilla/websocket` is in maintenance-only mode, and `coder/websocket` (formerly `nhooyr.io/websocket`) is the cleanest minimal-dependency option.

PRs that add a new third-party dependency outside this bar will be asked to remove it. If the functionality is small, reimplement it inside the library. If it is large enough that reimplementation is unreasonable, file an issue first — most such PRs are better handled by *not* implementing the feature in the library at all and pushing the dependency up into the calling code.

## Adding a new provider

Adding a provider is the most common substantial contribution. The full mechanics live in the docs at [/docs/providers/adding-a-provider](https://elloloop.github.io/llmrouter/docs/providers/adding-a-provider). The short version:

1. File an issue first describing the provider, its auth model, its request body shape, and its streaming format.
2. Create `providers/<name>/<name>.go` and `providers/<name>/<name>_test.go`, mirroring the existing providers.
3. Implement the relevant interface (`Provider`, `Embedder`, `Speaker`, `Transcriber`, or `Reranker`). Use the OpenAI provider as a reference for passthrough, the Anthropic provider as a reference for translating.
4. Mock the upstream with `httptest.NewServer`. Cover success path, non-2xx upstream, malformed SSE, mid-stream context cancellation, and empty deltas.
5. Add a docs page under `docs-site/src/pages/docs/providers/<name>.astro` and wire it into `docs-site/src/data/nav.ts`.

## Code style

Standard Go. We follow [Google's Go Style Guide](https://google.github.io/styleguide/go/) where it overlaps with the toolchain's defaults, and the toolchain's defaults everywhere else.

- `gofmt` for formatting. No exceptions.
- Table-driven tests with named subtests. Subtest names describe the scenario in English (`"rejects empty api key"`, not `"tc1"`).
- Doc comments on every exported identifier, in standard Godoc style.
- Error strings start lowercase, do not end with punctuation. Wrap with `fmt.Errorf("context: %w", err)` to keep the chain intact.
- `context.Context` is the first parameter of any function that does I/O. Never store a context in a struct.
- Channels are typed and bounded. Document buffer size and the producer/consumer contract above the `make` call.
- Every exported function gets a test. Edge cases (nil, empty, boundary, cancellation, non-2xx, malformed) are required.

Coverage targets:

- Root package: **100% statement coverage**.
- Each provider: **≥90% statement coverage**.

New code that drops coverage below these thresholds will be asked for more tests before it lands.

## Conventional commits

Commit messages and PR titles follow [Conventional Commits](https://www.conventionalcommits.org/). Release-please reads these to determine the next version bump.

| Prefix | When | Bumps |
|---|---|---|
| `feat:` | New functionality | minor |
| `fix:` | Bug fix | patch |
| `docs:` | Docs only | none |
| `test:` | Tests only | none |
| `refactor:` | No behaviour change | none |
| `chore:` | Deps, tooling, CI | none |
| `perf:` | Performance improvement | patch |
| `style:` | Formatting / whitespace | none |
| `feat!:` / `fix!:` | Breaking change | major (post-1.0) |

The first line of the PR description is what ends up in the squash-merge commit, so treat it like the commit message.

## Pull request process

1. **Open an issue first** for anything beyond a typo or a one-file bug fix.
2. **One logical change per PR.** Small PRs land fast; big PRs sit.
3. **Reference the issue** in the PR body (`Closes #123`).
4. **Tests are required** for any source change. Doc-only and test-only PRs are exempt.
5. **CI must be green.** CI runs `gofmt` check, `go vet ./...`, `go test -race -count=1 ./...`, and a coverage gate.
6. **Squash-merge.** All PRs land as a single commit on `main`.
7. **Reviewer turnaround target: ~3 days.** Ping the PR if you haven't heard back after that.

## Release process

Releases are cut by maintainers; contributors don't need to bump versions or touch a changelog file.

1. [release-please](https://github.com/googleapis/release-please) watches `main`. Each `feat:` / `fix:` / `feat!:` commit feeds the next release PR.
2. A maintainer reviews and merges the release PR. That cuts the `vX.Y.Z` tag.
3. CI builds the docs site and publishes a GitHub Release with the generated notes.
4. `pkg.go.dev` picks up the new tag within a few minutes.

Version bumps follow SemVer. Pre-1.0, minor versions may break the API.

## Security disclosure

Please **do not** file security vulnerabilities as public GitHub issues. See [SECURITY.md](./SECURITY.md) for the private disclosure process and what counts as in-scope.
