# ADR-0004: Local-first, offline-capable core

**Status:** Accepted
**Date:** 2026-09-24

## Context

The hosted decision-API model (Jev) is fast but remote, closed, and metered. Many target users
(edge, on-prem, air-gapped, privacy-sensitive) cannot send data to a hosted service. jeba's
differentiation is running the same contract on the user's own hardware. If the core required
a hosted service, that differentiation would disappear.

## Decision

The base install of jeba must:

- import and answer **fully offline**, with **no API key** and **no network egress**;
- keep heavy or remote dependencies behind optional pip extras (`serve`, `train`, `onnx`,
  `langchain`, `mcp`, `fast`);
- treat the HTTP server as **one mode**, not the only mode (CLI and in-process SDK are first-class);
- default to the local encoder backend once available (M3).

The LLM backend is an optional extra and may require a provider key, but it must never be
required by the core.

## Consequences

**Positive:**
- Privacy, latency, and cost benefits; works air-gapped.
- No vendor lock-in; no mandatory phone-home.
- Extras keep the base install small and dependency-light.

**Negative:**
- Cannot assume cloud capacity for defaults; local hardware is the performance envelope.
- Two-plus backends to maintain (see ADR-0002).

**Neutral / follow-ups:**
- Weights distribution must be offline-friendly (OD-3).
- Telemetry, if any, must be opt-out (ADR-0006).

## Alternatives Considered

- **Hosted-first with local fallback** — rejected: inverts the value proposition.
- **Local-only, no LLM backend** — rejected: delays end-to-end delivery (ADR-0002).
- **Cloud-required training** — rejected: contradicts offline/on-prem goals.

## Related

- `docs/requirements/non-functional.md` (NFR-S03, NFR-S04, NFR-R01)
- ADR-0002, ADR-0006