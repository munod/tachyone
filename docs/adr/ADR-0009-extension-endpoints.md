# ADR-0009: Extension endpoints mirror the canonical wire response

**Status:** Accepted
**Date:** 2026-09-24

## Context

The canonical `POST /v1/systemone` contract is frozen (ADR-0001). Product ergonomics still
need convenience surfaces: a batch call, a request without an explicit `model`, and a health
probe (NFR-O01, EXT-04, SERVE-02). OD-5 asked whether to expose an extension shape distinct
from `/v1/systemone` and, if so, what it should look like. A bespoke shape risks drifting from
the canonical contract and confusing existing Jev clients.

## Decision

Expose three additive endpoints that never alter the canonical response:

- `POST /predict` — same response shape as `/v1/systemone`; accepts `{state, questions, model?}`
  with a default model.
- `POST /predict/batch` — `{requests: [...]}` returning `{results: [...]}` in input order, each
  result identical in shape to a canonical response.
- `GET /health` — status, version, backend, device, and configured models.

`/v1/systemone` remains the only compatibility surface; the extensions are additive and
inherit the same auth rule.

## Consequences

**Positive:** Batch and health ergonomics without touching the frozen wire; SDK/CLI can reuse
the same response model; existing Jev clients are unaffected.
**Negative:** Extension request bodies are validated by FastAPI's default 422 body, not the
canonical error envelope (acceptable: extensions are out of the Jev contract).
**Neutral / follow-ups:** `return_details` is accepted but honored only when calibrated
distributions land (CAL-05, M3).

## Alternatives Considered

- **Only `/v1/systemone`** — safe but forces batching clients to loop; no health probe.
- **A distinct extension envelope** — inconsistent with the canonical response and easier to drift.

## Related

- `docs/protocol.md` ("Additive extensions"), `docs/architecture.md` (`serve.py`)
- Requirements: SERVE-02, EXT-04, NFR-O01, NFR-X02
