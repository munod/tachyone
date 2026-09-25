# Input-Noise Robustness Specification

**Phase:** Post-M6 (B-4, `.specs/project/BACKLOG.md`)
**Status:** Implemented (code + tests landed; GPU retrain/publish pending).
**Related docs:** `docs/training.md` (§1), `STATE.md` L-003, `docs/requirements/functional.md`
(TRAIN-01, TRAIN-08), `.specs/features/multilingual-quality/spec.md`.

## Problem Statement

Synthetic states are clean templates. Real chat/user input carries typos, missing accents, dropped
punctuation, casing variation and regional slang, so accuracy outside controlled templates
degrades. L-003 showed that data-template artifacts are a real failure mode: the model can learn
surface tokens that do not transfer to messy input. We need a **seeded, deterministic, opt-in**
noise augmentation and a **clean vs noisy** evaluation split so robustness can be measured without
confusing it with clean-set regressions.

## Goals

- [x] Add deterministic, opt-in noise injection to `training/generate_data.py` (char swaps/deletes,
      accent stripping, casing/punctuation drops) applied to a configurable fraction of records.
- [x] Preserve byte-for-byte determinism: same seed + config → identical JSONL.
- [x] Keep clean and noisy evaluation separate and report both.
- [ ] Retrain and compare accuracy/ECE on both splits (GPU step; reported honestly).

## Out of Scope

| Feature | Reason |
| --- | --- |
| Changing the canonical `.jsonl` schema | Training data format is documented and stable (TRAIN-01) |
| Noise at inference time | Runtime is the caller's input; augmentation is a training concern |
| Semantic paraphrasing / back-translation | Larger scope; this is a cheap surface-noise stage |
| Changing the wire contract | Training-only; contract suite untouched |

---

## User Stories

### P1: Seeded opt-in noise augmentation ⭐ MVP

**User Story:** As an ML engineer, I want the training data to include realistic surface noise so
the model stops relying on clean templates.

**Acceptance Criteria:**

1. WHEN noise is enabled with a rate `r` THEN approximately `r` of generated records SHALL have
   their `state` perturbed (the label and question SHALL stay unchanged).
2. WHEN the same seed + config are used twice THEN the output SHALL be byte-identical.
3. WHEN noise is disabled (default) THEN generation SHALL be byte-identical to today.
4. WHEN applied THEN noise SHALL draw from the record's own RNG so it stays independent of the
   cyclic label (L-003), and SHALL never produce a crash on boundary/empty/long states.

### P1: Clean vs noisy evaluation ⭐ MVP

**User Story:** As an evaluator, I want to see robustness gains separately from clean accuracy.

**Acceptance Criteria:**

1. WHEN an evaluation run carries a noise setting THEN the report SHALL expose accuracy/ECE for the
   clean and noisy views separately (or two artifacts), so a gain on one is not confused with the
   other.
2. WHEN comparing runs THEN the public-probe evaluation SHALL stay on unmodified inputs for
   comparability.

---

## Edge Cases

- WHEN a state is empty THEN noise SHALL leave it empty (no index errors).
- WHEN a state is a single character or a single token THEN deletion SHALL not produce an
  out-of-range error.
- WHEN the noise rate is `0.0` or `1.0` THEN all/none of the records SHALL be perturbed.
- WHEN the accent-stripping language has no accents THEN the record SHALL pass through unchanged.
- WHEN two runs use different seeds THEN the noise SHALL differ.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| TRAIN-01 | P1 augmentation | Design | Done |
| TRAIN-08 | P1 augmentation | Design | Done |
| TRAIN-05 | P1 evaluation | Design | Done |

**Coverage:** 3 requirements mapped to `.specs/features/input-noise/tasks.md`.

## Success Criteria

- [x] Noise is seeded, reproducible, and opt-in via a config flag (existing determinism tests pass).
- [ ] Noisy-split accuracy improves vs baseline with no clean-split regression beyond tolerance
      (GPU run pending).
- [ ] Both splits reported in `benchmarks/report.md` (or honestly reported if the GPU run is
      pending).
