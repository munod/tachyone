# Multilingual LoRA Rank — Tasks

**Spec:** `.specs/features/lora-rank-experiment/spec.md`
**Status:** Done; `lora_rank=64` adopted, republish pending decision.

---

## R-T1: Experiment configs and training runs

**What:** Add `exp_multi_r32.json` / `exp_multi_r64.json` (same hyperparameters, larger rank) and
train both on `data/train_multi.jsonl`.
**Where:** `training/configs/exp_multi_r32.json`, `training/configs/exp_multi_r64.json`,
`checkpoints/multi_r32`, `checkpoints/multi_r64`.
**Depends on:** — · **Requirement:** NFR-C06, NFR-C07.
**Done when:** both runs complete on the RTX 3060.
**Tests:** none (GPU run) · **Gate:** build. · **Status:** Done.

## R-T2: Calibrate and evaluate clean + noisy

**What:** Predict, fit per-`(primitive, language)` temperature, and evaluate both views for each
rank against the r=16 baseline.
**Where:** `benchmarks/results/*_split.json`.
**Depends on:** R-T1 · **Requirement:** NFR-C06.
**Done when:** a comparison table (rank × accuracy × ECE, per language) is produced.
**Tests:** none (GPU run) · **Gate:** build. · **Status:** Done.

## R-T3: Adopt the winner and record the result

**What:** Set `training/configs/finetune_multi.json` to `lora_rank=64`/`lora_alpha=128`, update
`benchmarks/report.md`, the model card, STATE/BACKLOG, and decide on HF republish.
**Where:** `training/configs/finetune_multi.json`, `benchmarks/report.md`, `docs/model-card.md`,
`.specs/project/STATE.md`, `.specs/project/BACKLOG.md`.
**Depends on:** R-T2 · **Requirement:** NFR-C06.
**Done when:** the config reflects r=64 and the report/card carry the measured numbers.
**Tests:** `uv run pytest` · **Gate:** full. · **Status:** In progress.
