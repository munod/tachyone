# Backlog

Active next steps and carried-over ideas for jeba. This is the canonical "what's next" list;
`.specs/project/STATE.md` points here. Each item names its motivation, plan, acceptance criteria,
and known risks. Items are not scheduled until promoted into `docs/tasks.md`.

Legend: **Ready** (can start now) · **Blocked** (needs a prerequisite) · **Idea** (unscoped).

---

## B-1 — Multilingual `choice`/`score` quality  · Done (accuracy) / partial (calibration)

**Why.** After v3 (dedicated `choice` head, rich option descriptions), English `choice` reached
0.78 but multilingual `choice` is 0.40 and multilingual `score` ECE is 0.18 (see
`benchmarks/report.md`). The multilingual checkpoint carries six training languages with imbalanced
support, and calibration is fitted **per primitive**, not per language.

**Status (retrained).** Data is fully localized with per-record RNG and a low distractor rate;
calibration is per `(primitive, language)`; the runtime applies `kind:lang` → `kind` → global.
Measured: multilingual `choice` 0.40 → **0.734**, overall 0.609 → **0.711**; English overall
0.721 → **0.781**; the CUDA-graph fast path adds 2.7× p50. Per-language ECE still misses 0.05 for
`es`/`nl`/`de` and multilingual `score` (0.09–0.16), so calibration remains **partial**. See
`STATE.md` L-003 (shared-RNG shortcut). Spec/tasks: `.specs/features/multilingual-quality/`.

**Plan.**
1. Data: stratify generation per language (equal support per language for `choice`/`score`); add
   language-specific hard negatives and more cue vocabulary; consider adding 2–3 more languages.
2. Calibration: fit temperature per **(primitive, language)** — extend `training/fit_calibration.py`
   to key `per_primitive` by language and apply it in `EncoderModel` (`temperatures` already keys
   by primitive; widen to `"choice:pt"`-style keys or a nested map).
3. Report per-language accuracy/ECE in `benchmarks/report.md` (the harness already aggregates by
   language) and gate on the worst language, not just the average.

**Acceptance.**
- Multilingual `choice` accuracy ≥ 0.60 on the held-out synthetic set.
- Per-language ECE ≤ 0.05 for `choice`/`score` (NFR-C06), with per-language numbers published.
- No regression in English (overall ≥ 0.72).

**Risks / notes.** Small per-language calibration sets are noisy (warn below `min_samples`); more
data volume does not fix an architectural gap, but here the head already exists. Effort ~1 day +
retrain.

**Related.** `docs/training.md` (§3–4, limitations), `NFR-C06`, `CAL-02`/`CAL-04`, `STATE.md` L-002.

---

## B-2 — Fast path kernels (TileLang / CUDA graphs) · Done

**Why.** M5 shipped the fast-path **seam and graceful fallback** only
(`src/jeba/fast.py`: `maybe_accelerate` returns the stock target with reason "no accelerated kernels
registered"). There is no measured latency improvement yet, and `NFR-P01` is still `[open]`.

**Status (done).** `maybe_accelerate` now takes an injected accelerator builder and is wired by
`JEBA_FAST` in `load_encoder` to a per-shape CUDA-graph forward with bf16-resident weights (graceful
fallback on CPU/MPS/absent extras). `benchmarks/fast_path.py` measures stock vs fast: **p50 9.13 →
3.70 ms, p95 10.75 → 4.09 ms (2.47× p50)**, answer parity max abs **0.0037**, **0 top-label flips**.
NFR-P01 and NFR-P07 met; results in `benchmarks/report.md`. TileLang fused kernels remain optional
(no measured benefit yet). Spec/tasks: `.specs/features/fast-path/`.

**Plan.**
1. Implement fused kernels for the encoder forward (GEMM+activation epilogues, GEMM+GEGLU,
   residual+LayerNorm, in-place RoPE, sliding-window attention) behind the `fast` extra.
2. Capture per `(batch, length)` buckets as CUDA graphs; keep weights resident in bf16.
3. Wire `maybe_accelerate` to attach the accelerated forward when `tilelang_available()`; keep the
   stock forward on CPU/MPS or when the extra is missing.
4. Add `benchmarks/fast_path.py` micro-benchmark (stock vs fast, p50/p95, parity check) and record
   results in `benchmarks/report.md`.

**Acceptance.**
- On a supported CUDA device, p50 latency improves vs the stock forward with no response-shape
  change (parity within documented tolerance).
- Graceful fallback on CPU/MPS and when `tilelang` is absent (already covered by `tests/test_fast.py`).
- `NFR-P01` target measured and the micro-benchmark committed.

**Risks / notes.** CUDA-graph constraints (OD-4) and kernel/CUDA-version compatibility; guard
behind the extra and fall back cleanly. Effort ~2–3 days.

**Related.** `docs/adr/ADR-0012`, `docs/tasks.md` M5-T2, `NFR-P01`, `docs/overview.md` (success metrics).

---

## Carried over (from earlier planning)

- **Provider registry** for LLM backends (OpenAI-compatible, Anthropic, local llama.cpp) — `Idea`.
- **Additional checkpoints** (typed-decisions variant, larger multilingual) — `Idea`; overlaps B-1.
- **Streaming multi-question responses** — `Idea`.
- **Optional opt-out telemetry** — `Idea`; not to be added unless a future ADR decides (ADR-0011).
- **Web console** for interactive triage — `Idea` (would introduce accessibility requirements).
- **Process:** convert `docs/tasks.md` into `.specs/features/<phase>/tasks.md` at execution time;
  add `.specs/features/*/design.md` for the remaining phases; add the `mermaid-studio` skill for
  rendered diagrams.
