# Multilingual `choice`/`score` Quality — Tasks

**Spec:** `.specs/features/multilingual-quality/spec.md`
**Status:** A1–A5 done (retrained on the RTX 3060 and adapters published); per-language ECE
target remains open (NFR-C06), tracked in `BACKLOG.md` B-1.

---

## A1: Language-stratified, localized synthetic data

**What:** Localize `instructions` and `noul`/`score` entities; expand phrase templates and per-team
cue vocabulary; increase localized hard negatives; scale `per_type` via a committed multi config.
**Where:** `training/generate_data.py`, `training/data/phrases.json`, `training/configs/data_multi.json`.
**Depends on:** — · **Requirement:** TRAIN-01.
**Done when:** generated records are same-language (state + question), deterministic, and language
support scales evenly; existing generator tests still pass.
**Tests:** `tests/test_training_generate.py` (add localization + coverage) · **Gate:** full.
**Commit:** `feat(training): localize and deepen multilingual synthetic data`.

## A2: Per-(primitive, language) temperature fitting

**What:** Carry `lang` through calibration examples; emit per-language fits beside the per-primitive
aggregate; widen the grid; add a shared report parser.
**Where:** `training/fit_calibration.py`, `src/tachyone/calibration.py`,
`training/configs/calibration.json`.
**Depends on:** A1 · **Requirement:** CAL-02, CAL-04, NFR-C06.
**Done when:** report contains `per_primitive[kind]` (unchanged shape) plus per-language entries with
warnings below `min_samples`; the grid no longer truncates.
**Tests:** `tests/test_training_calibration.py`, `tests/test_calibration.py` · **Gate:** full.
**Commit:** `feat(training): fit per-language temperature`.

## A3: Runtime per-language temperature (additive)

**What:** Flatten temperature keys to `kind`/`kind:lang`; `EncoderModel` applies `kind:lang` →
`kind` → global, detecting language from the state text when not given; offline `predict` passes the
ground-truth `lang`.
**Where:** `src/tachyone/backends/encoder.py`, `training/predict.py`.
**Depends on:** A2 · **Requirement:** CAL-02, EXT-02.
**Done when:** old and new temperature files both apply; response shape unchanged.
**Tests:** `tests/test_encoder.py` · **Gate:** full · **Contract:** unchanged.
**Commit:** `feat(encoder): apply per-language temperature at runtime`.

## A4: Per-language benchmark reporting

**What:** Render `per_language` tables and a worst-language summary from evaluation artifacts.
**Where:** `benchmarks/report.py`.
**Depends on:** A2 · **Requirement:** TRAIN-05.
**Done when:** the report includes per-language accuracy/ECE and the worst language.
**Tests:** `tests/test_benchmark_report.py` · **Gate:** full.
**Commit:** `feat(benchmarks): report per-language metrics`.

## A5: GPU retrain and publish

**Status:** Done. Retrained on the RTX 3060 and adapters republished to the Hub
(`munod/tachyone-en`, `munod/tachyone-multi`); per-language ECE target not fully met (see spec).
**What:** Run data → LoRA → calibration → eval → report on the RTX 3060; update model card, changelog,
and STATE/BACKLOG; publish adapters.
**Where:** `benchmarks/report.md`, `docs/model-card.md`, `CHANGELOG.md`,
`training/package_hf.py`, `.specs/project/STATE.md`, `.specs/project/BACKLOG.md`.
**Depends on:** A1–A4 · **Requirement:** CAL-04, NFR-C06.
**Done when:** success criteria in the spec are measured and published (or honestly reported).
**Tests:** none (process) · **Gate:** build.
**Commit:** `chore(training): retrain multilingual adapters and publish numbers`.
