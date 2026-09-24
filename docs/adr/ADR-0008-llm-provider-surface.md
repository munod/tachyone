# ADR-0008: LLM backend uses an OpenAI-compatible surface with an injectable transport

**Status:** Accepted
**Date:** 2026-09-24

## Context

M2 needed a way to answer primitives with existing LLMs before any encoder is trained.
Reviewing providers (OpenAI, Anthropic, local Ollama/llama.cpp, vLLM) showed that nearly all
expose an OpenAI-style `/chat/completions` endpoint, but SDKs and response envelopes differ.
OD-1 asked whether to build a provider-plugin registry or a single provider-agnostic call.
The project is also local-first: the base install must run with no hosted dependency and no
new mandatory package (ADR-0004).

## Decision

`LLMBackend` targets a single provider-agnostic surface: any server exposing
`POST {base_url}/chat/completions` with a JSON body, returning the assistant message content.
Provider choice is configuration (`JEBA_LLM_BASE_URL`, `JEBA_LLM_API_KEY`, `JEBA_LLM_MODEL`).
The HTTP call is injected as a `Transport` callable, and the default transport uses the
standard library (`urllib`) — no third-party dependency. Structured output is requested with
`response_format={"type": "json_object"}`; the reply is parsed tolerantly, validated, and
normalized, with bounded retries, then mapped to a wire `BackendError` on failure.

## Consequences

**Positive:** Works with OpenAI-compatible hosted and local providers unchanged; tests run
offline via an injected transport; the core install stays dependency-free and local-first.
**Negative:** Providers without an OpenAI-compatible endpoint are not first-class and need a
custom transport or a future registry.
**Neutral / follow-ups:** A provider registry (OD, "Future considerations") can be added later
without changing the `Backend` seam or the wire.

## Alternatives Considered

- **Provider plugin registry** — more flexible but materially more code and no M2 user; defer.
- **Per-provider SDKs (e.g. openai, anthropic)** — adds dependencies and breaks the offline core.
- **Local-only (Ollama/llama.cpp)** — narrower than the OpenAI-compatible surface, which already
  covers local servers.

## Related

- `docs/architecture.md` (`backends/llm.py`), `docs/protocol.md`, ADR-0002, ADR-0004
- Requirements: BACK-02, BACK-05, NFR-R01
