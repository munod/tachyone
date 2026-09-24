# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **M0 — Bootstrap:** uv project on Python 3.12, ruff/pyright/pytest gates, CI, Apache-2.0.
- **M1 — Wire Contract:** `choice` / `score` / `noul` primitives, the Jev `/v1/systemone`
  envelope, error shapes, and a golden contract suite.
- **M2 — LLM Backend + Serve:** OpenAI-compatible structured-output backend with an injectable
  transport, FastAPI server (`/v1/systemone`, `/predict`, `/predict/batch`, `/health`), Python
  SDK with 429/529 backoff, CLI, presets, and `decide()` from JSON Schema.
- **M3 — Local Encoder:** single-pass encoder backend, script/language router with checkpoint
  lifecycle, confidence/calibration, batching, and prediction/lifecycle hooks. Offline.
- **M4 — Training & Calibration:** deterministic data generation, LoRA/QLoRA fine-tuning with an
  RLCD proper-scoring objective, temperature/ECE fitting, an evaluation harness, and committed
  reproducible configs.
- **M5 — Ecosystem & Acceleration:** ONNX backend, optional fast path with graceful fallback,
  MCP stdio server, LangChain adapter, Docker/compose, and a no-op opt-out telemetry guard.

### Notes

- Weights are fetched from the Hugging Face Hub on demand and cached locally (ADR-0010).
- The RTX 3060 training run and published benchmark numbers are still pending.

## [0.0.1] - 2026-09-24

Initial pre-release: specification baseline and the milestones above.
