# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

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

[Unreleased]: https://github.com/munod/jeba/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/munod/jeba/releases/tag/v0.1.0
[0.0.1]: https://github.com/munod/jeba/releases/tag/v0.0.1
