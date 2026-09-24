# ADR-0002: Pluggable backend with phased LLM → encoder strategy

**Status:** Accepted
**Date:** 2026-09-24

## Context

The strongest long-term engine is a local, non-autoregressive encoder. But training a
competitive encoder takes a full phase of data generation, fine-tuning, and calibration. If
the product waits for that, there is no end-to-end system to validate the contract, the
server, the SDK, or the CLI. Meanwhile, existing LLMs can already answer structured questions
well enough to prove the whole pipeline.

## Decision

Define a stable `Backend` interface (`backends/base.py`) behind a frozen `wire.py`. Deliver in
phases:

1. **M2:** an LLM backend using structured outputs of existing LLMs (optional extra).
2. **M3:** a local encoder backend (ModernBERT/mmBERT + 3 heads).
3. **M5:** an ONNX runtime backend.

The wire never depends on which backend is active. Every backend must pass the same contract
test suite.

## Consequences

**Positive:**
- An end-to-end, verifiable system exists in M2, de-risking the harder encoder work.
- Backends are swappable (LLM for quality, encoder for offline/low-latency, ONNX for portability).
- Third parties can add backends without touching the contract.

**Negative:**
- An extra abstraction layer and two-plus implementations to keep in parity.
- LLM backend introduces an optional external dependency (kept out of core).

**Neutral / follow-ups:**
- Backend selection is config-driven (`JEBA_BACKEND`).
- Provider abstraction surface is still open (OD-1).

## Alternatives Considered

- **Encoder-only from day one** — rejected: no end-to-end system until training succeeds.
- **LLM-only** — rejected: contradicts local-first/offline differentiation.
- **Hard-coding one engine** — rejected: blocks portability and evolution.

## Related

- `docs/architecture.md` (Backend strategy)
- `.specs/features/llm-backend/spec.md`, `.specs/features/encoder-backend/spec.md`
- ADR-0004 (local-first)
