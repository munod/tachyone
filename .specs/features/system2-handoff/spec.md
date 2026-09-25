# Confidence Thresholding & System-2 Handoff Specification

**Phase:** Post-M6 (B-3, `.specs/project/BACKLOG.md`)
**Status:** Implemented (B3-T1–T4); all goals met, contract suite unchanged.
**Related docs:** `docs/protocol.md` (confidence semantics), `docs/cookbook-handoff.md`,
`.specs/features/training-calibration/spec.md`, `STATE.md` L-004,
`docs/adr/ADR-0001-jev-drop-in-protocol.md`, `NFR-C06`.

## Problem Statement

Every `choice`/`score` answer already carries `confidence` (the selected mass, `calibration.py`) and
`noul` carries a probability, but the engine exposes no first-class uncertainty metric or
documented pattern for "abstain / hand off to a System-2 LLM when confidence < τ". Users
reimplement the threshold by hand, and there is no entropy or margin signal beyond the selected
mass.

## Goals

- [x] Add uncertainty helpers (normalized entropy and margin) beside `confidence` in `calibration.py`.
- [x] Add a `handoff.py` module with an `assess`/`assess_response` API returning a typed
      non-confident signal callers can route on.
- [x] Expose the helpers through the SDK and a CLI `--threshold` option that appends a sibling
      `handoff` object without touching the canonical response.
- [x] Document the pattern (when to hand off, suggested τ, composing with an LLM).

## Out of Scope

| Feature | Reason |
| --- | --- |
| A new wire field or primitive | Contract frozen (ADR-0001); helper is client-side/additive |
| Server-side abstention | Would mutate `/v1/systemone`; the caller decides (L-004) |
| Changing calibration/ECE | B-3 does not affect accuracy or ECE; NFR-C06 stays open |
| A universal default τ | Task-dependent; must stay a required, configurable knob |

---

## User Stories

### P1: Uncertainty metrics ⭐ MVP

**User Story:** As a decision consumer, I want more than the selected mass to judge uncertainty.

**Acceptance Criteria:**

1. WHEN `normalized_entropy` is given a distribution THEN it SHALL return `[0, 1]` (0 certain,
   1 uniform), normalizing by `log(n)`; empty/degenerate (n ≤ 1) SHALL return `0.0`.
2. WHEN `margin` is given a distribution THEN it SHALL return the gap between the top two
   probabilities in `[0, 1]`; empty or single-key SHALL return `0.0`.
3. WHEN input is unnormalized THEN both SHALL behave consistently with `confidence` (normalize
   first, treat negatives as zero).

### P1: Typed handoff signal ⭐ MVP

**User Story:** As an agent builder, I want a typed "not confident" signal I can route on.

**Acceptance Criteria:**

1. WHEN `assess(probabilities, threshold=τ)` is called THEN it SHALL return an `Uncertainty`
   carrying `confidence`, `entropy`, `margin`, `threshold`, and `abstain = confidence < τ`.
2. WHEN `assess_response(response, threshold=τ)` is called THEN it SHALL return a `HandoffReport`
   with a per-question `HandoffSignal`; the aggregate `abstain` SHALL be true when any question
   abstains.
3. WHEN the answer is `noul` THEN the signal SHALL use the `noul` probability directly (no
   separate `confidence`).
4. WHEN the answer is `choice`/`score` THEN the signal SHALL use the answer `confidence` and that
   answer's probability distribution.
5. WHEN τ is omitted by a caller THEN the API SHALL require it (no silent universal default).

### P1: CLI/SDK surface ⭐ MVP

**User Story:** As a CLI user, I want to see the handoff decision without breaking the canonical output.

**Acceptance Criteria:**

1. WHEN `jeba --predict --threshold τ` runs THEN stdout SHALL keep `model`, `answers`, `usage`
   unchanged and append a sibling `handoff` object.
2. WHEN `--threshold` is absent THEN output SHALL be byte-identical to today.
3. WHEN `assess`/`assess_response` are imported from `jeba` THEN they SHALL import without
   torch/transformers (base install, NFR-R01).

### P2: Documentation

**User Story:** As a new user, I want a documented, tested pattern for `if confidence < τ: handoff()`.

**Acceptance Criteria:**

1. WHEN a reader opens the cookbook THEN they SHALL find when to hand off, suggested τ per task
   shape, and how to compose with an LLM.

---

## Edge Cases

- WHEN a distribution is empty or all zeros THEN entropy/margin SHALL be `0.0` and confidence `0.0`
  (the answer abstains for any positive τ).
- WHEN `n = 1` THEN normalized entropy SHALL avoid division by `log(1) == 0` and return `0.0`.
- WHEN two options tie for the top THEN margin SHALL be `0.0`.
- WHEN a `score` answer is assessed THEN the helper SHALL use `confidence` and the level
  distribution, never the expected `score` position.
- WHEN `threshold` is outside `[0, 1]` THEN the API SHALL reject it with `ValueError`.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| CAL-06 | P1 metrics | Design | Done |
| CAL-03 | P1 handoff | Design | Done |
| CAL-05 | P1 CLI/SDK | Design | Done |
| EXT-02 | P1 additive | Design | Done |
| NFR-R01 | P1 base import | Design | Done |

**Coverage:** 5 requirements mapped to `.specs/features/system2-handoff/tasks.md`.

## Success Criteria

- [x] `assess`/`assess_response` covered by unit tests; no change to the `/v1/systemone` shape
      (contract suite unchanged).
- [x] Documented, tested example of `if confidence < τ: handoff()`.
- [x] CLI `--threshold` works with `--backend fake` and needs no API key.
