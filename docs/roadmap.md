# Roadmap

> Canonical machine-facing roadmap lives in `.specs/project/ROADMAP.md`. This page is the
> human-facing view with exit criteria and risk notes. Task-level detail: `docs/tasks.md`.

**Current phase:** M6 Proof & Release ✅ complete and released (`v0.3.0`), including the post-M6
B-1 (multilingual quality + per-language calibration), B-2 (CUDA-graph fast path), B-3
(confidence thresholding / System-2 handoff) and B-4 (input-noise robustness) work.
**Delivery model:** one shippable increment per milestone; no fixed dates.

```mermaid
graph LR
    M0[M0 Bootstrap] --> M1[M1 Wire Contract]
    M1 --> M2[M2 LLM Backend + Serve]
    M2 --> M3[M3 Local Encoder]
    M3 --> M4[M4 Training + Calibration]
    M4 --> M5[M5 Ecosystem + Accel]
    M5 --> M6[M6 Proof + Release]
```

## M0 — Bootstrap

**Goal:** Reproducible Python 3.12 + uv project with quality gates and license.
**Depends on:** nothing.
**Exit criteria:**
- `uv sync` installs from `uv.lock` on Python 3.12.
- `uv run ruff check .`, `uv run pyright`, `uv run pytest` all pass.
- CI runs the same gates on push and PR.
- Apache-2.0 `LICENSE`, `README.md`, `CONTRIBUTING.md`, `AGENTS.md` present.

**Risks:** Python version drift (see B-001); uv lockfile churn.

## M1 — Wire Contract

**Goal:** Freeze the Jev-exact contract and primitives before backends.
**Depends on:** M0.
**Exit criteria:**
- Primitives validated with limits (choice ≤255, score 2–10 levels).
- `wire.py` matches Jev field names/types; golden fixtures pass.
- All documented error shapes (401/422/429/529) reproduce.
- A `FakeBackend` answers the full cycle offline.

**Risks:** misreading Jev edge behavior; mitigated by golden fixtures and `docs/protocol.md`.

## M2 — LLM Backend + Serving

**Goal:** End-to-end product using structured outputs of existing LLMs.
**Depends on:** M1.
**Exit criteria:**
- `POST /v1/systemone` served by FastAPI; SDK + CLI work.
- A real Jev client repointed at `tachyone-serve` returns correct answers.
- Backend is swappable behind `backends/base.py`; no wire change.
- Retry/backoff verified for 429/529.

**Risks:** structured-output reliability across providers (OD-1); provider drift.

## M3 — Local Encoder Backend

**Goal:** Offline single-pass encoder with routing and calibrated confidence.
**Depends on:** M2 (server reused), M4 (for a trained checkpoint — can start with base weights).
**Exit criteria:**
- All three primitives answered offline, no key, no network.
- Router selects English vs multilingual checkpoint with documented overhead.
- Confidence derived from distribution; `return_details` exposes probabilities.
- Batching (`predict_batch`, `sort_by_length`) and lifecycle (`preload`, `max_loaded`, eviction) work.

**Risks:** base-model quality before tuning; VRAM pressure with two checkpoints.

## M4 — Training & Calibration

**Goal:** Reproducible training and calibration within 12GB VRAM.
**Depends on:** M3 (architecture), M1 (data contract).
**Exit criteria:**
- `generate_data.py` → deterministic JSONL.
- LoRA/QLoRA fine-tuning completes on RTX 3060 12GB.
- RLCD proper-scoring training produces calibrated probabilities.
- Temperature fitting reduces ECE on a held-out split.
- Accuracy/ECE/latency report reproducible from committed scripts.

**Risks:** OOM (B-002); small calibration sets; synthetic-data bias.

## M5 — Ecosystem & Acceleration

**Goal:** Integration surface and speed paths, contract unchanged.
**Depends on:** M3, M4.
**Exit criteria:**
- ONNX backend + optional fast path (TileLang/CUDA graphs) behind extras.
- MCP stdio server, LangChain adapter, Docker/compose all work.
- Extras isolate optional dependencies; graceful fallback when unavailable.

**Risks:** ONNX op coverage; CUDA-graph constraints (OD-4).

## M6 — Proof & Release

**Goal:** Public credibility.
**Depends on:** M5.
**Exit criteria:**
- Benchmark report comparable to MASSIVE / XNLI / typed-decisions — **met (2026-09-27)**: the
  three named probes are measured on public data through the same metric code and published with
  licences, citations, chance levels and reproduction commands in
  [`benchmarks/probes.md`](https://github.com/munod/tachyone/blob/main/benchmarks/probes.md)
  (0.323 typed-decisions · 0.033 MASSIVE · 0.334 XNLI), alongside the head-to-head in
  [`compare.md`](compare.md) §3, which reproduces the peer model's own card to three decimals.
- Docs site published.
- Hugging Face weights + model card and a GitHub release.

**Risks:** benchmark comparability; weights distribution model (OD-3).

## Milestone → feature mapping

| Milestone | Feature spec |
| --- | --- |
| M0, M1 | `.specs/features/wire-contract/` |
| M2 | `.specs/features/llm-backend/` |
| M3 | `.specs/features/encoder-backend/` |
| M4 | `.specs/features/training-calibration/` |
| M5, M6 | `.specs/features/ecosystem/` |

## Post-M6 — Backlog (delivered)

The active backlog items shipped across `v0.2.0` and `v0.3.0` (figures are the ones measured at
the time — **B-11** later restated every dataset-level number, see below):

- **B-1 Multilingual `choice`/`score` quality** ✅ (`v0.2.0`) — localized per-language data,
  per-record RNG, per-`(primitive, language)` temperature fitting, runtime language detection, and
  per-language ECE reporting. Follow-up raised the multilingual LoRA rank to 64: overall accuracy
  0.702 → 0.853 and `es` ECE 0.170 → 0.038 (pre-B-11 labels; 2/6 languages ≤ 0.05 — `es`, `pt`).
- **B-2 Fast-path (TileLang/CUDA graphs)** ✅ (`v0.2.0`) — `TACHYONE_FAST=1` wires per-shape CUDA graphs
  with bf16 weights and graceful fallback; measured 2.68× p50 at the time, **2.72× re-measured on
  the B-11 adapters** (NFR-P01/P07 met).
- **B-3 Confidence thresholding / System-2 handoff** ✅ (`v0.3.0`) — `normalized_entropy`/`margin`
  and `assess`/`assess_response`; CLI `--threshold` appends a sibling `handoff` object (`CAL-06`).
- **B-4 Input-noise robustness** ✅ (`v0.3.0`) — seeded `noise_rate` augmentation and a clean/noisy
  evaluation split; the released adapters are already robust to the injected noise (`TRAIN-08`).

Shipped in `v0.4.0`:

- **B-7 Public probes** ✅ — `benchmarks/probes.py` loaders + `benchmarks/probes.md`:
  typed-decisions, MASSIVE and XNLI against their own chance levels, with licences, citations and
  reproduction commands; `OPS-06` implemented. **Re-run on the B-11 adapters (2026-09-28):**
  0.323 → 0.330, 0.033 → **0.013** (below MASSIVE's 0.017 chance), 0.334 → 0.333.
- **B-8 Calibration-asset hardening** ✅ — an adapter asset that cannot be loaded warns by name
  instead of degrading silently (`docs/huggingface.md` §4).
- **B-10 LLM prompt wrapper** ✅ — the `answers` wrapper documented for all three primitives;
  probe `JSON ok` 0.083 → 0.889 with parsing kept strict.
- **B-5a Multi-domain data** ✅ — the generator is split per domain with byte-identical output for
  existing configs, reports `per_domain`, and ships four new English domains plus presets. Six
  runs; run 5 is the artifact. On the pre-B-11 labels it met 1 of 2 gates (support 0.785 ✗,
  worst new domain 0.739 ✓) and released nothing; **re-measured on the B-11 labels both gates
  pass** (support 0.861 ✓, voice 0.807 ✓) — `.specs/project/BACKLOG.md` **B-5**.

Delivered 2026-09-28 (CHANGELOG `[Unreleased]`):

- **B-11 `noul` labels from the text** ✅ (ADR-0014) — the label rule, the acceptance tests, the
  label audit published beside the accuracy, seven datasets regenerated, both adapters retrained
  and republished (Hub `224c8a74` / `b7747756`): English 0.859 → **0.945**, multilingual restated
  at **0.718** (the old 0.853 was a language→label shortcut). Head-to-head, probes, fast path and
  the B-5a artifact were all re-measured in the same pass.
- **B-12 `score`/`choice` labels from the text** ✅ (ADR-0015, 2026-09-29) — the `index % 13`
  near-tie (7.8% of `score` rows, ceiling 0.888) and the index-derived labels of the empty
  boundary states are gone; `score` empty → middle level, `choice` empty → the catch-all
  (`ecommerce` gained `other` as a fifth option). Seven datasets regenerated, all three
  checkpoints retrained, and the **published set re-chosen by merit**: English **0.972** (B-11
  weights), multilingual **0.743** (B-12 retrain), five-domain run 5 **0.879** with **both gates
  passing** (0.886 / 0.815) — the retrain-on-corrected-labels option was tried and lost
  (`choice` 0.303, support 0.763), so B-5's structural decision is what remains. Fast path
  2.65×, probes 0.330 / 0.011 / 0.333, head-to-head 0.236 / 0.974 re-measured in the same pass.

Carried-over ideas (canonical list: `.specs/project/BACKLOG.md`):

- Provider registry for LLM backends (OpenAI-compatible, local llama.cpp).
- Additional checkpoints (typed-decisions variant, larger multilingual).
- Streaming multi-question responses.
- Web console for interactive triage.
