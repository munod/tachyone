# Training & Calibration Specification

**Phase:** M4 (Phase 4)
**Status:** Implemented (scripts + tests; full RTX 3060 training run done and numbers published;
per-language ECE target still open, see `.specs/features/multilingual-quality/`)
**Related docs:** `docs/training.md`, `docs/adr/ADR-0005-encoder-rlcd.md`

## Problem Statement

The encoder backend is only as good as its training and calibration. We need a reproducible
pipeline that generates supervised data, fine-tunes encoders with LoRA/QLoRA, optimizes
probabilities with RLCD under strictly proper scoring rules, and fits temperature — all
within a single RTX 3060 12GB.

## Goals

- [x] Generate reproducible synthetic supervision as JSONL.
- [x] Fine-tune English and multilingual checkpoints with LoRA/QLoRA (no OOM on 12GB).
- [x] Train with RLCD against strictly proper scoring rules for calibration.
- [x] Fit temperature and report accuracy, ECE, and latency.
- [x] Make every stage reproducible from committed scripts + configs.

## Out of Scope

| Feature | Reason |
| --- | --- |
| Serving/training integration | `encoder-backend` (M3) |
| Foundation-model pretraining | We fine-tune existing encoders |
| Distributed multi-GPU training | Single 12GB target |

---

## User Stories

### P1: Synthetic data generation ⭐ MVP

**User Story:** As an ML engineer, I want deterministic synthetic examples so training is
reproducible.

**Acceptance Criteria:**

1. WHEN `generate_data.py` runs with a seed THEN it SHALL emit JSONL with identical output on re-run.
2. WHEN generating in `choice`/`score`/`noul` modes THEN each record SHALL contain the primitive
   type, inputs, and a target label/distribution.
3. WHEN the dataset is large THEN generation SHALL stream to disk (no full in-memory hold).

### P1: LoRA/QLoRA fine-tuning ⭐ MVP

**User Story:** As an ML engineer, I want to fine-tune on a 12GB GPU.

**Acceptance Criteria:**

1. WHEN `finetune_rlcd.py` runs with the documented config THEN training SHALL complete within
   12GB VRAM (gradient checkpointing + accumulation + quantized base).
2. WHEN training finishes THEN a checkpoint SHALL be saved with a config recording hyperparameters and seed.
3. WHEN re-run with the same seed/config THEN metrics SHALL be within documented tolerance.

### P1: RLCD proper-scoring calibration ⭐ MVP

**User Story:** As a decision consumer, I want calibrated probabilities.

**Acceptance Criteria:**

1. WHEN training uses RLCD THEN the loss SHALL be a strictly proper scoring rule (e.g., log/Brier)
   over the primitive distributions.
2. WHEN training completes THEN probabilities SHALL be normalized and `confidence` derivable.
3. WHEN temperature fitting runs THEN it SHALL minimize ECE on a calibration split without using test data.

### P2: Evaluation harness

**User Story:** As a maintainer, I want objective numbers to compare against references.

**Acceptance Criteria:**

1. WHEN evaluation runs THEN it SHALL report accuracy, ECE, and latency (p50/p95) per primitive and language.
2. WHEN public probes are run (MASSIVE / XNLI / typed-decisions) THEN results SHALL be saved as
   reproducible artifacts.

---

## Edge Cases

- WHEN a class/level is unseen in a split THEN it SHALL be excluded or mapped per documented rule.
- WHEN a batch would exceed VRAM THEN accumulation SHALL split it rather than crash.
- WHEN calibration data is too small THEN the fit SHALL warn rather than overfit silently.
- WHEN probabilities underflow THEN a numerically stable implementation SHALL be used.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| --- | --- | --- | --- |
| TRAIN-01 | P1 data | Design | Done |
| TRAIN-02 | P1 fine-tune | Design | Done (run done on GPU) |
| TRAIN-03 | P1 RLCD | Design | Done |
| TRAIN-04 | P1 RLCD | Design | Done |
| TRAIN-05 | P2 evaluation | Design | Done |
| TRAIN-06 | P1 fine-tune | Design | Done |
| TRAIN-07 | P1 data | Design | Done |
| CAL-01 | P1 RLCD | Design | Done |
| CAL-02 | P1 RLCD | Design | Done |
| CAL-04 | P1 RLCD | Design | Pending (target ECE after real run) |

**Coverage:** 10 requirements mapped to `docs/tasks.md` M4.

## Success Criteria

- [x] Full pipeline reproducible from scripts + configs on 12GB.
- [ ] ECE at or below the target stated in `docs/training.md` (after a real run).
- [ ] Benchmark numbers published with exact reproduction commands (M6).
