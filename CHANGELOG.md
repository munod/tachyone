# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Input-noise robustness (B-4):** `training/generate_data.py` gains a seeded, opt-in
  `noise_rate` (char swap/delete, accent strip, casing flip, terminal-punctuation drop) applied to
  the `state` only; `training/evaluate.py --noise-rate` reports a clean vs noisy split
  (`report["noisy"]`); `benchmarks/report.py` renders it. `TRAIN-08`.
- **Confidence thresholding / System-2 handoff (B-3):** `normalized_entropy` and `margin` in
  `calibration.py`; `handoff.py` with `assess`/`assess_response`; CLI `--threshold` appends a
  sibling `handoff` object. `CAL-06`.

### Fixed

- `training/evaluate.py` now honors `JEBA_ADAPTERS` (and other env settings) when building the
  encoder backend, so a local adapter can be evaluated.

### Changed

- **Multilingual LoRA rank (NFR-C06):** raised `lora_rank` 16 → 64 / `lora_alpha` 128 in
  `training/configs/finetune_multi.json`. Multilingual overall accuracy 0.702 → 0.853 and `es` ECE
  0.170 → 0.038 (`es` accuracy 0.472 → 0.956); 5/6 languages now meet ECE ≤ 0.05 (`nl` 0.104
  remains). See `.specs/features/lora-rank-experiment/`.
- Retrained end-to-end on the RTX 3060; refreshed `benchmarks/report.md` (English 0.763 / ECE
  0.061; multilingual r=64 0.853 / ECE 0.038) and the model card. The released adapters are already
  robust to the injected noise (EN 0.763→0.760, multi 0.853→0.847); a noise-augmented adapter
  (r=16) scored 0.719/0.719 but calibrated worse and is not released.
- Republished the multilingual adapter at LoRA r=64 to the Hugging Face Hub
  (`munod/jeba-multi`, commit `599df58`). The English adapter (`munod/jeba-en`) is unchanged.

## [0.2.0] - 2026-09-24

### Added

- **Multilingual quality (B-1):** fully localized synthetic data (`training/data/lexicon.json`),
  per-`(primitive, language)` temperature fitting with a widened grid, runtime `kind:lang` →
  `kind` → global temperature selection with heuristic language detection, and per-language
  accuracy/ECE reporting (worst-language gate). Retrained on the RTX 3060: multilingual `choice`
  0.40 → 0.734 and overall 0.609 → 0.711; English overall 0.721 → 0.781.
- **Fast path (B-2):** `JEBA_FAST` wires the encoder to a per-shape CUDA-graph forward with
  bf16-resident weights and graceful fallback; `benchmarks/fast_path.py` measures 2.68× p50
  (10.27 → 3.83 ms) with 0 top-label flips, meeting NFR-P01/NFR-P07.

### Changed

- `maybe_accelerate` now takes an injected accelerator builder; TileLang fused kernels remain
  optional.
- `training/generate_data.py` seeds an RNG per record (independent draws) and lowers the
  hard-negative rate to 1/6 (see L-003).

## [0.1.1] - 2026-09-24

### Added

- Dedicated low-rank `choice` head, trained alongside the LoRA adapter with a near-identity
  initialization (so training starts at the cosine baseline). English `choice` rises from
  ~0.25 (chance) to 0.78 and multilingual to 0.40 (L-002).
- Rich, localized option descriptions — including a genuinely learnable `other` class — and
  distractor clauses in the synthetic data generator.
- `choice_head.json` is trained, packaged, and loaded by the runtime (local directory or Hub).

### Changed

- Refreshed full-scale numbers: English overall 0.721 / ECE 0.052; multilingual 0.609 / ECE 0.086
  (see [`benchmarks/report.md`](benchmarks/report.md)).
- README rewritten for the released state; markdown files excluded from ruff.

## [0.1.0] - 2026-09-24

First tagged release: a local-first, multilingual System One decision engine that speaks the
TypeSafe Jev `/v1/systemone` wire protocol as a drop-in.

### Added

- **M0 — Bootstrap:** uv project on Python 3.12, ruff/pyright/pytest gates, CI, Apache-2.0.
- **M1 — Wire Contract:** `choice` / `score` / `noul` primitives, the Jev `/v1/systemone`
  envelope, error shapes, and a golden contract suite.
- **M2 — LLM Backend + Serve:** OpenAI-compatible structured-output backend with an injectable
  transport, FastAPI server (`/v1/systemone`, `/predict`, `/predict/batch`, `/health`), Python
  SDK with 429/529 backoff, CLI, presets, and `decide()` from JSON Schema.
- **M3 — Local Encoder:** single-pass encoder backend, script/language router with checkpoint
  lifecycle, confidence/calibration, batching, and prediction/lifecycle hooks. Offline.
- **M4 — Training & Calibration:** deterministic data generation, batched LoRA/QLoRA fine-tuning
  with an RLCD proper-scoring objective, temperature/ECE fitting, an evaluation harness, and
  committed reproducible configs.
- **M5 — Ecosystem & Acceleration:** ONNX backend, optional fast path with graceful fallback,
  MCP stdio server, LangChain adapter, Docker/compose, and a no-op opt-out telemetry guard.
- **M6 — Proof & Release:** reproducible benchmark report, mkdocs documentation site, model card,
  and release process.

### Models

- LoRA adapters published on the Hugging Face Hub:
  [`munod/jeba-en`](https://huggingface.co/munod/jeba-en) (ModernBERT-large) and
  [`munod/jeba-multi`](https://huggingface.co/munod/jeba-multi) (mmBERT-base). The runtime loads
  them by default and caches weights locally (ADR-0010).

### Notes

- Measured on a single RTX 3060 12GB: English 0.613 overall accuracy / 0.059 ECE; multilingual
  0.493 / 0.034 (see [`benchmarks/report.md`](benchmarks/report.md)). `choice` remains near
  chance and needs a dedicated head (L-002).
- `JEBA_BACKEND=llm` is the default backend; the local encoder requires the `train` extra.

## [0.0.1] - 2026-09-24

Specification baseline (documentation only, no code).

[Unreleased]: https://github.com/munod/jeba/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/munod/jeba/releases/tag/v0.2.0
[0.1.1]: https://github.com/munod/jeba/releases/tag/v0.1.1
[0.1.0]: https://github.com/munod/jeba/releases/tag/v0.1.0
[0.0.1]: https://github.com/munod/jeba/releases/tag/v0.0.1
