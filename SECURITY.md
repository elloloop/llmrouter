# Security Policy

## Supported versions

llmrouter is pre-1.0. Only the latest minor version receives security patches. When v1.0 ships this policy will widen to cover the current major plus one previous minor.

| Version | Supported |
|---|---|
| 0.8.x (latest) | ✓ |
| < 0.8 | — please upgrade |

## Reporting a vulnerability

**Do NOT open a public GitHub issue for a suspected vulnerability.**

Use one of:

1. **GitHub Security Advisory** — https://github.com/elloloop/llmrouter/security/advisories/new (preferred — keeps the conversation in the repo and lets us coordinate the fix + disclosure)
2. **Email** — security@elloloop.com

Please include:

- The version (`llmrouter/v0.x.y`) where you observed the issue
- A minimal reproduction (Go program; no API keys)
- Your assessment of impact (information disclosure, RCE, credential leak, etc.)
- Any suggested fix or workaround

## Response targets

- **Acknowledge** within 72 hours of receiving the report
- **Fix in latest minor** within 14 days for high-severity (CVSS ≥ 7.0)
- **Fix in latest minor** within 30 days for medium-severity (CVSS 4.0–6.9)
- **Coordinated disclosure** — we'll work with you on a timeline that gives users a chance to upgrade before details are public

## Scope

In scope:

- Vulnerabilities in llmrouter source code (root package, provider packages, router subpackage)
- Documentation that, if followed, leads to insecure deployment
- Default option values that produce insecure behaviour

Out of scope:

- Vulnerabilities in upstream LLM providers (OpenAI, Anthropic, Bedrock, Vertex, etc.) — report to the provider directly
- API key leakage caused by caller misuse (logging requests, committing keys, etc.) — this is a deployment concern, not a library bug
- Denial-of-service via maliciously large request bodies — llmrouter is a client library, not a server; large-body protection belongs at your gateway layer
- Vulnerabilities in dependencies that don't affect llmrouter's reachable code paths — please report to the upstream maintainer

## Acknowledgements

Contributors who responsibly disclose vulnerabilities are credited in:

- The GitHub Security Advisory once published
- The release notes for the patched version
- An optional listing in this file (with your permission)

## Hall of fame

_(Reporters added here once we have any to thank — please be the first.)_

## Past advisories

_(None as of v0.8.0.)_
