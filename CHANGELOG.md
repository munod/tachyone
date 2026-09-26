# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`docs/compare.md` — the three questions a prospective user asks first:** why not a small
  model on Ollama/llama.cpp, why not Jev, why not another open System One scorer. Measured
  figures where they exist, explicit notes where they do not, and the composition answer
  (Tachyone first, an LLM when `confidence < τ`) linking the System-2 cookbook. The tagline
  *"LLMs generate text. Tachyone produces calibrated decisions"* now sits in the landing hero,
  the README blockquote, the overview value proposition and the mkdocs site description.
- **`docs/use-cases.md` — five recipes with real JSON in and JSON out:** support-ticket
  classification, ticket routing, incident prioritization, risk assessment and document triage.
  The payloads live in `docs/assets/examples/*.json` and are embedded with `pymdownx.snippets`;
  `tests/test_docs_examples.py` validates every pair against the wire (question/answer key parity,
  probabilities over the declared options, `confidence` = selected mass, `noul` without
  `confidence`) and fails if an example stops being embedded.
- **Integration hub and recipes:** `docs/integrations.md` plus FastAPI, n8n, Power Automate,
  Azure Functions and LangGraph pages. Every HTTP payload shown was executed against a local
  `tachyone-serve` (200 with the documented shape; 422/401/429/529 rows come from
  `test_serve.py`), and each recipe carries a verification label distinguishing *shipped +
  tested* from *payload verified* — the third-party runtimes are not run in CI. The LangGraph
  flow is a real script (`docs/assets/examples/langgraph_flow.py`) executed by the test suite.
- **Head-to-head comparison harness (`benchmarks/compare.py`)** answering the same rows with the
  local encoder, the open peer scorer `pngwn/system-one-qwen3.5-4b-scorer` (loaded from a local
  download through its own `system_one.py` — nothing third-party is vendored) and any
  OpenAI-compatible server behind Tachyone's `llm` backend. One metric implementation, both
  evaluation distributions, temperature fitted per engine on the capped validation split, peak
  RSS/VRAM of the process that did the inference, and contract compliance counted per question.
  `tests/test_benchmark_compare.py` covers the metrics, the temperature fit, the row → wire
  mapping and the renderer. Results, method and limitations published in `docs/compare.md` §3;
  the run reproduces the peer's own model card (T 1.75, accuracy 0.705, ECE 0.046).

### Changed

- **Renamed `jeba` → Tachyone (ADR-0013).** Package `src/tachyone/`, console scripts `tachyone`,
  `tachyone-serve`, `tachyone-mcp-server`, env prefix `TACHYONE_*`, SDK classes `TachyoneClient` /
  `Tachyone*Error`, wire default model id `tachyone-latest`, Hub repos `munod/tachyone-en` /
  `munod/tachyone-multi`, site `munod.github.io/tachyone/`. No alias period: `JEBA_*`, `jeba-serve`
  and `JebaClient` are gone. ADR-0001..ADR-0012 and the released sections below keep the original
  name as historical record.

- **English adapter reseeded (B-9).** The published `training/configs/finetune_en.json` recipe run
  again from `seed: 2` produced a checkpoint that beats the previously published one on seven of
  eight metrics: overall accuracy **0.763 → 0.859**, overall ECE **0.061 → 0.023**, `choice`
  0.834 → 0.948, `score` 0.712 → 0.910 and their ECEs 0.074 → 0.020 / 0.042 → 0.039. Only `noul`
  accuracy moved against it (0.744 → 0.718), while `noul` ECE improved 0.101 → 0.020. The config
  now pins `seed: 2`, `munod/tachyone-en` was republished from it, and `benchmarks/report.md` was
  regenerated. Background: `.specs/project/BACKLOG.md` B-9.

### Fixed

- `benchmarks/report.py` split `--entry` on the **first** `=`, which broke on the report's own
  entry names (`english (... r=16 ...)`) with a misleading `FileNotFoundError`; it now splits on
  the last one.
- The report's hardcoded Reproduce block pointed at files that do not exist (`data/eval.jsonl`)
  and omitted the predict / fit-calibration / evaluate / report steps, so a regenerated report
  could not be reproduced from its own instructions.
- **Landing hero title was nearly invisible on the light theme.** The hero rules were bare
  classes (specificity 0-1-0), so Material's `.md-typeset h1` (0-1-1) won despite `extra.css`
  loading last and painted the title with `--md-default-fg-color--light` (`#0000008a`) over the
  dark hero gradient — it also silently replaced the title's size, weight and margins. Every
  `.tachyone-hero__*` rule is now scoped under `.tachyone-hero` (0-2-0, no `!important`), guarded by
  `tests/test_docs_site.py::test_hero_styles_are_scoped_against_theme_overrides`.

### Documentation

- **Continuity record for the open backlog**, so the work resumes from the repository rather than
  from a conversation: `BACKLOG.md` carries full execution detail for `B-7` (probe inventory with
  HF ids, licenses and sizes plus the `benchmarks/probes.py` loader design), `B-5` (generator
  refactor with a byte-identical guarantee for existing configs, the five-domain table,
  English-first scope, worst-domain gates and the 8–12 h training estimate) and `B-10` (decision:
  stay strict, no inner-shape fallback, so the compliance metric keeps meaning something).
  `STATE.md` opens **OD-6 / OD-7 / OD-8** for the three scope questions still unanswered.
  `benchmarks/README.md` warns that the peer scorer's default `--option-batch 16` OOMs a 12 GB
  card (use `4`) and records the exact eight commands behind the published tables. Roadmap,
  overview, testing and training pages now read *partially delivered* where a public probe exists.
- Corrected the per-language ECE claim (**2/6** languages meet ECE ≤ 0.05, not 5/6), the
  requirement counts (60 functional / 43 non-functional / 104 traced), stale `v0.2.0` status
  headers, references to files and `(planned)` markers that no longer exist, and the documented
  default backend (`llm`). Added CLI, MCP, LangChain and Docker guides.

## [0.3.0] - 2026-09-26

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
  0.170 → 0.038 (`es` accuracy 0.472 → 0.956); 2/6 languages now meet ECE ≤ 0.05 (`es` 0.038,
  `pt` 0.024 — `de` 0.063, `fr` 0.051, `it` 0.059 and `nl` 0.104 remain). See
  `.specs/features/lora-rank-experiment/`.
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

[Unreleased]: https://github.com/munod/tachyone/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/munod/tachyone/releases/tag/v0.3.0
[0.2.0]: https://github.com/munod/tachyone/releases/tag/v0.2.0
[0.1.1]: https://github.com/munod/tachyone/releases/tag/v0.1.1
[0.1.0]: https://github.com/munod/tachyone/releases/tag/v0.1.0
[0.0.1]: https://github.com/munod/tachyone/releases/tag/v0.0.1
