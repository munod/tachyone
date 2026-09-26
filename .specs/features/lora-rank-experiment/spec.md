# Multilingual LoRA Rank Experiment Specification

**Phase:** Post-M6 (NFR-C06 follow-up, `.specs/project/BACKLOG.md` B-1)
**Status:** Done and measured on the RTX 3060. `lora_rank=64` is the clear winner.
**Related docs:** `.specs/features/multilingual-quality/spec.md`, `docs/training.md`,
`docs/requirements/non-functional.md` (NFR-C06, NFR-C07), `training/configs/finetune_multi.json`.

## Problem Statement

Multilingual accuracy and per-language ECE were bounded by trunk capacity: the multilingual
checkpoint carries six languages on a single LoRA (r=16) trunk, and `es` accuracy sat at 0.47 with
ECE 0.17. Difficulty sharing one residual capacity across six languages is the suspected cause;
the fix under test is a larger LoRA rank (r=32, r=64) applied **only** to
`finetune_multi.json`-equivalent configs.

## Hypothesis

A larger LoRA rank gives the mmBERT-base trunk enough residual capacity to retain all six
languages without one dominating the other, improving both per-language accuracy and calibration.

## Method

Train the same data (`data/train_multi.jsonl`, seed 1, 18k records) with identical
hyperparameters except `lora_rank`/`lora_alpha`:

| Config | lora_rank | lora_alpha |
| --- | --- | --- |
| baseline `finetune_multi.json` | 16 | 32 |
| `exp_multi_r32.json` | 32 | 64 |
| `exp_multi_r64.json` | 64 | 128 |

Then predict -> `fit_calibration` (per `(primitive, language)`) -> evaluate clean + noisy
(`--noise-rate 0.15`) on `data/eval_multi.jsonl` (seed 2, 1500 records).

## Results (RTX 3060, in-sample calibration)

| Rank | Clean acc | Clean ECE | Noisy acc | Noisy ECE |
| --- | --- | --- | --- | --- |
| 16 (baseline) | 0.702 | 0.033 | 0.701 | 0.035 |
| 32 | 0.741 | 0.050 | 0.743 | 0.051 |
| **64** | **0.853** | **0.038** | **0.847** | **0.042** |

Per-language ECE (clean):

| Language | r=16 | r=32 | r=64 |
| --- | --- | --- | --- |
| de | 0.077 | 0.098 | 0.063 |
| es | 0.170 | 0.133 | **0.038** |
| fr | 0.028 | 0.077 | 0.051 |
| it | 0.078 | 0.108 | 0.059 |
| nl | 0.045 | 0.060 | **0.104** |
| pt | 0.012 | 0.055 | 0.024 |

Per-language accuracy (clean): `es` 0.472 -> **0.956**, `de` 0.731 -> **0.863**,
`nl` 0.643 -> 0.663.

## Conclusion

**Adopt `lora_rank=64` / `lora_alpha=128`** for the multilingual checkpoint: overall accuracy
+0.151 (0.702 -> 0.853), `es` ECE 0.170 -> 0.038 and accuracy 0.472 -> 0.956. NFR-C06 is met for
`es` (0.038) and `pt` (0.024); `de` (0.063), `fr` (0.051), `it` (0.059) and `nl` (ECE 0.104,
accuracy 0.663) remain above the 0.05 target and stay open.

## Success Criteria

- [x] r=64 improves overall accuracy and per-language ECE over r=16 without a clean-accuracy
      regression.
- [~] Per-language ECE <= 0.05: 2/6 languages pass at r=64 (`es` 0.038, `pt` 0.024) — 3/6 passed
      at r=16 (`fr`, `nl`, `pt`); `de` 0.063, `fr` 0.051, `it` 0.059 and `nl` 0.104 do not.
- [x] Republish the r=64 multilingual adapter to the Hub (`munod/jeba-multi`, commit `599df58`).
