---
name: Bug report
about: Something doesn't work as documented
title: 'bug: '
labels: bug
assignees: ''
---

<!--
Thanks for taking the time to file a bug. To help us reproduce and fix
quickly, please fill in every section below — vague reports often
take several rounds of clarification.
-->

## Description

<!-- What were you trying to do? What happened instead? -->

## Steps to reproduce

<!--
A minimal Go program that triggers the bug.
- No API keys — use httptest fakes where possible.
- If the bug is provider-specific, include the exact request/response shape.
-->

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/elloloop/llmrouter"
    "github.com/elloloop/llmrouter/providers/openai"
)

func main() {
    // ...
}
```

## Expected behaviour

<!-- What did you expect to happen? -->

## Actual behaviour

<!-- What actually happened? Include stack traces, error messages, log output. -->

## Environment

- `go version`:
- OS / arch:
- llmrouter version (e.g. `v0.8.0`):
- Provider in question (if applicable):
- Relevant upstream provider model/endpoint:

## Additional context

<!-- Anything else that might help — relevant docs, prior issues, screenshots, etc. -->
