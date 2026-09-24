# ADR-0007: Multilingual from Phase 3 via mmBERT-base

**Status:** Accepted
**Date:** 2026-09-24

## Context

jeba targets multilingual use from early on. Retrofitting multilingual support after an
English-only model is trained is expensive: it requires a second checkpoint, a routing layer,
and re-validation of calibration across languages. mmBERT-base provides broad language
coverage (100+ languages) with a manageable parameter count, and script/language detection is
cheap enough to run before inference.

## Decision

Ship multilingual support starting in Phase 3:

- Maintain an **English checkpoint** (ModernBERT-class) and a **multilingual checkpoint**
  (mmBERT-base, 100+ languages).
- Add a **`Router`** that detects script/language without a model forward pass (target
  overhead < 0.5 ms) and selects the appropriate checkpoint.
- Manage checkpoint lifecycle (`preload`, `max_loaded`, LRU `evict`, `unload`, `attach`).
- Fall back to the multilingual checkpoint when language detection is uncertain.
- Include multilingual probes in the benchmark suite.

## Consequences

**Positive:**
- Multilingual capability is a first-class differentiator from day one of the encoder phase.
- Routing is cheap and transparent; users do not choose checkpoints manually.
- Calibration is validated per language group.

**Negative:**
- Two checkpoints to load and manage; memory pressure on constrained devices.
- Router adds a small amount of complexity and a new failure surface.

**Neutral / follow-ups:**
- Additional checkpoints (typed-decisions) are future work.
- Language coverage list must be documented and tested.

## Alternatives Considered

- **English-only first, multilingual later** — rejected: expensive retrofit, weaker story.
- **Single multilingual checkpoint only** — rejected: English quality/latency trade-off.
- **Per-request manual checkpoint selection** — rejected: poor UX; router is better.

## Related

- `docs/architecture.md` (Routing & multilingual strategy)
- `.specs/features/encoder-backend/spec.md` (ROUTE-01..05)
- ADR-0005