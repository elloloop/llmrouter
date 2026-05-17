<!--
Thanks for contributing to llmrouter. A few things to know before
opening the PR:

- The PR title is the commit message that release-please reads to
  determine the next version bump. Use conventional commits:
    feat: / fix: / refactor: / docs: / test: / chore: / ci: / perf:
  Add a scope when applicable: feat(openai): ...
  Use `feat!:` or `fix!:` for breaking API changes (bumps minor in
  0.x, major in 1.x).
- One logical change per PR. If you discovered a bug while building
  a feature, fix it in a separate PR.
- All CI must pass: go vet, go test -race -count=1 ./..., staticcheck,
  gofmt.
- For new providers / capabilities, open an issue first to align on
  the surface before writing code.
-->

## Summary

<!-- One or two sentences on what this PR changes. -->

## Motivation

<!-- Which issue does this close? What user problem does this solve? -->

Closes #

## Type of change

<!-- Tick all that apply. -->

- [ ] `feat` — new provider, capability, or public API addition
- [ ] `fix` — bug fix (please link the issue)
- [ ] `refactor` — internal restructuring, no behaviour change
- [ ] `docs` — documentation only
- [ ] `test` — tests only
- [ ] `chore` — tooling, dependencies, maintenance
- [ ] `ci` — CI workflow changes
- [ ] `perf` — performance improvement
- [ ] **Breaking change** (`feat!` / `fix!`) — public API contract changes

## Test plan

<!-- How did you verify this works? How can a reviewer verify? -->

## Checklist

- [ ] PR title uses a conventional commit prefix
- [ ] `go test -race -count=1 ./...` is green locally
- [ ] `go vet ./...` is clean
- [ ] `staticcheck ./...` is clean
- [ ] `gofmt -l .` is empty
- [ ] New public functions have doc comments
- [ ] New behaviour has table-driven tests (with realistic fixtures, not just synthesized payloads)
- [ ] User-facing changes have docs site updates (under `docs-site/`)
- [ ] Breaking changes are flagged with `!` and documented in the PR body
- [ ] CHANGELOG.md is left untouched — release-please owns it
