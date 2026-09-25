# Input-Noise Robustness — Tasks

**Spec:** `.specs/features/input-noise/spec.md`
**Status:** B4-T1–T3 done; B4-T4 (GPU retrain) pending the RTX 3060 run.

---

## B4-T1: Seeded opt-in noise in `generate_data.py`

**What:** Add a deterministic noise stage (char swap/delete, accent strip, casing/punctuation
drop) behind a `noise_rate` config field; apply it to the `state` only, drawing from the record's
own RNG; leave labels/questions untouched.
**Where:** `training/generate_data.py`, `tests/test_training_generate.py`.
**Depends on:** — · **Requirement:** TRAIN-01, TRAIN-08.
**Done when:** same seed → byte-identical; `noise_rate=0` is byte-identical to today; boundary
states (empty/long) never raise; existing determinism tests pass.
**Tests:** `tests/test_training_generate.py` · **Gate:** full · **Contract:** unchanged.
**Commit:** `feat(training): add seeded input-noise augmentation`.

## B4-T2: Clean vs noisy evaluation

**What:** Let the evaluation harness produce separate clean and noisy views so robustness is not
confused with clean accuracy; keep public-probe evaluation unmodified.
**Where:** `training/evaluate.py`, `training/configs/eval.json`, `tests/test_training_evaluate.py`.
**Depends on:** B4-T1 · **Requirement:** TRAIN-05.
**Done when:** a run can report both views and the report differentiates them.
**Tests:** `tests/test_training_evaluate.py` · **Gate:** full · **Contract:** unchanged.
**Commit:** `feat(training): report clean and noisy evaluation splits`.

## B4-T3: Document the augmentation and trace TRAIN-08

**What:** Document noise injection and the clean/noisy protocol in the training docs; register
`TRAIN-08` and update the traceability matrix.
**Where:** `docs/training.md`, `docs/requirements/functional.md`,
`docs/requirements/traceability.md`.
**Depends on:** B4-T2 · **Requirement:** TRAIN-08.
**Done when:** a reader can enable noise and interpret both splits; `TRAIN-08` is traced.
**Tests:** `tests/test_docs_site.py` · **Gate:** full.
**Commit:** `docs: document input-noise augmentation and splits`.

## B4-T4: GPU retrain and publish (blocked on GPU)

**What:** Retrain with noise, evaluate clean vs noisy, and record the numbers.
**Where:** `benchmarks/report.md`, `training/configs/*`, `CHANGELOG.md`.
**Depends on:** B4-T3 · **Requirement:** TRAIN-05, TRAIN-08.
**Done when:** noisy-split accuracy improves with no clean-split regression beyond tolerance, or
the result is honestly reported.
**Tests:** none (process) · **Gate:** build.
**Commit:** `chore(training): retrain with input noise and report splits`.

## B4-T5: Close B-4

**What:** Run all gates and update the backlog/state.
**Where:** `.specs/project/BACKLOG.md`, `.specs/project/STATE.md`.
**Depends on:** B4-T1–T4 · **Requirement:** —.
**Done when:** gates green and B-4 marked done with a pointer to the feature spec.
**Tests:** none (process) · **Gate:** build.
**Commit:** `chore(specs): close B-4 input-noise robustness`.
