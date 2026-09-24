# ADR-0005: Encoder backend trained with RLCD proper-scoring calibration

**Status:** Accepted
**Date:** 2026-09-24

## Context

jeba's local backend must be fast (single forward pass), multilingual, and produce
**trustworthy confidence**. Autoregressive decoding is too slow and its token probabilities
are poorly calibrated for atomic decisions. Laya demonstrates that a non-autoregressive
encoder with task heads, trained with reinforcement learning against strictly proper scoring
rules (RLCD), yields calibrated probabilities. The available hardware is a single RTX 3060
12GB, so full fine-tuning of large encoders is infeasible.

## Decision

The local backend is:

- an **encoder** (ModernBERT-class for English, mmBERT-base for multilingual) with **three
  task heads** (`noul`, `choice`, `score`);
- trained with **LoRA/QLoRA** to fit 12GB VRAM;
- optimized with **RLCD against a strictly proper scoring rule** (e.g., log/Brier) so
  probabilities are calibrated;
- post-processed with **temperature fitting** to minimize ECE on a held-out calibration split;
- evaluated and reported on accuracy, ECE, and latency.

`confidence` for `choice`/`score` is derived from the calibrated distribution; `noul` returns
a single probability with no separate confidence.

## Consequences

**Positive:**
- Low-latency, offline inference with meaningful confidence.
- Calibration is measurable (ECE) and reproducible.
- Fits consumer hardware via parameter-efficient fine-tuning.

**Negative:**
- Requires data generation and calibration infrastructure (M4).
- Bounded by 12GB VRAM; larger models need quantization or are out of reach.
- Synthetic-data quality directly bounds model quality.

**Neutral / follow-ups:**
- ECE target is provisional (NFR-C06) until measured.
- Checkpoint variants (typed-decisions) are future work.

## Alternatives Considered

- **Autoregressive LLM only** — rejected: latency and calibration.
- **Full fine-tuning** — rejected: does not fit 12GB.
- **Softmax cross-entropy without proper scoring** — rejected: weaker calibration.
- **Training a foundation model from scratch** — rejected: out of scope and budget.

## Related

- `docs/training.md`
- `.specs/features/encoder-backend/spec.md`, `.specs/features/training-calibration/spec.md`
- ADR-0002, ADR-0007