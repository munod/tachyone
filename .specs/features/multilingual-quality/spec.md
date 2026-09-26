# Multilingual `choice`/`score` Quality & Per-Language Calibration Specification

**Phase:** Post-M6 (B-1, `.specs/project/BACKLOG.md`)
**Status:** Code landed, retrained on the RTX 3060, and adapters published. Multilingual
`choice` 0.40 → 0.68 and English overall 0.72 → 0.76 (accuracy targets met). A follow-up raised the
multilingual LoRA rank to 64: overall 0.702 → 0.853 and `es` ECE 0.170 → 0.038, so 2/6 languages
meet ECE ≤ 0.05 (`es` 0.038, `pt` 0.024); `de`/`fr`/`it`/`nl` (0.104) remain (NFR-C06 open). See
`.specs/features/lora-rank-experiment/`.
**Related docs:** `docs/training.md` (§3–4, limitations), `.specs/features/training-calibration/spec.md`,
`STATE.md` L-002/L-003, `docs/requirements/non-functional.md` (NFR-C06), `docs/requirements/functional.md`
(CAL-02, CAL-04).

## Problem Statement

After the v3 dedicated `choice` head, English `choice` reached 0.78 but multilingual `choice` is
0.40 and multilingual `score` ECE is 0.178 (`benchmarks/report.md`). The multilingual checkpoint
carries six Latin training languages, but the synthetic data is shallow (2 templates per tone,
English-only instructions, English entities embedded in localized phrases) and calibration is
fitted **per primitive**, not per language. One global temperature cannot calibrate languages with
different confidence behaviour.

## Goals

- [x] Deepen and fully localize synthetic `choice`/`score`/`noul` data per language.
- [x] Fit temperature per `(primitive, language)`, keeping the per-primitive aggregate.
- [x] Apply per-language temperature at runtime with no wire/contract change.
- [x] Publish per-language accuracy/ECE and gate on the worst language.
- [x] Retrain and publish multilingual numbers/adapters (GPU).

## Out of Scope

| Feature | Reason |
| --- | --- |
| Non-Latin training languages | Runtime detection is Latin-only; would need a lang hint (deferred) |
| A new primitive or wire field | Contract frozen (ADR-0001) |
| Hosted data sources | Local-first, synthetic + public probes only (ADR-0004) |
| Architectural head changes | L-002: the head exists; this is data + calibration |

---

## User Stories

### P1: Language-stratified, localized data ⭐ MVP

**User Story:** As an ML engineer, I want realistic per-language supervision so multilingual
`choice`/`score` learn language-specific cues.

**Acceptance Criteria:**

1. WHEN `generate_data.py` runs THEN `instructions` and `noul`/`score` entities SHALL be localized
   to the record's language.
2. WHEN generating `choice` THEN each language SHALL have expanded team cue vocabulary and a
   localized hard-negative distractor.
3. WHEN re-run with the same seed/config THEN output SHALL remain byte-identical.
4. WHEN `per_type` is increased THEN support per language per primitive SHALL scale evenly.

### P1: Per-(primitive, language) temperature ⭐ MVP

**User Story:** As a decision consumer, I want confidence calibrated even for weakly served
languages.

**Acceptance Criteria:**

1. WHEN `fit_calibration.py` runs on predictions carrying `lang` THEN it SHALL emit a per-language
   temperature (and ECE before/after) in addition to the per-primitive aggregate.
2. WHEN a language has fewer than `min_samples` examples THEN the report SHALL warn and fall back
   to the per-primitive fit.
3. WHEN the grid boundary is optimal THEN the grid SHALL be wide enough not to truncate the fit.
4. WHEN test data is used THEN the fit SHALL NOT consume it.

### P1: Runtime application (additive) ⭐ MVP

**User Story:** As a runtime user, I want calibration applied automatically without changing the
contract.

**Acceptance Criteria:**

1. WHEN an `EncoderModel` loads temperatures THEN it SHALL apply `kind:lang` → `kind` → global
   precedence without altering the canonical response shape (EXT-02).
2. WHEN no language is supplied THEN the model SHALL detect it from the state text using the
   existing router heuristics, falling back to the per-primitive temperature.
3. WHEN an older `per_primitive`-only temperature file is loaded THEN it SHALL still apply.

### P2: Per-language reporting

**User Story:** As an evaluator, I want per-language accuracy/ECE to judge the worst case.

**Acceptance Criteria:**

1. WHEN `benchmarks/report.py` renders an evaluation artifact THEN it SHALL include `per_language`
   tables and a worst-language summary.

---

## Edge Cases

- WHEN detection returns `None`/`"und"` THEN use the per-primitive temperature (never crash).
- WHEN a language is absent from the fitted map THEN fall back rather than use a neighbour's value.
- WHEN only some primitives have per-language fits THEN apply per-language where present.
- WHEN prediction rows lack `lang` THEN default to `"und"` and warn.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| TRAIN-01 | P1 data | Design | Pending |
| TRAIN-05 | P2 reporting | Design | Pending |
| CAL-02 | P1 calibration | Design | Pending |
| CAL-04 | P1 calibration | Design | Pending |
| NFR-C06 | P1 calibration | Design | Pending |

**Coverage:** 5 requirements mapped to `.specs/features/multilingual-quality/tasks.md`.

## Success Criteria

- [x] Multilingual `choice` accuracy ≥ 0.60 on the held-out synthetic set (0.734).
- [ ] Per-language ECE ≤ 0.05 for `choice`/`score` — partial: `score` 0.09–0.16,
      `choice` 0.10 overall but `es` 0.30 / `de` 0.25 / `nl` 0.15 (power scaling insufficient).
- [x] No English regression (overall 0.781 ≥ 0.72).
- [x] Contract suite passes unchanged.
