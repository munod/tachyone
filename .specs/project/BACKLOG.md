# Backlog

Active next steps and carried-over ideas for jeba. This is the canonical "what's next" list;
`.specs/project/STATE.md` points here. Each item names its motivation, plan, acceptance criteria,
and known risks. Items are not scheduled until promoted into `docs/tasks.md`.

Legend: **Ready** (can start now) · **Blocked** (needs a prerequisite) · **Idea** (unscoped).

---

## B-1 — Multilingual `choice`/`score` quality  · Done (published) / `de` `fr` `it` `nl` ECE open

**Why.** After v3 (dedicated `choice` head, rich option descriptions), English `choice` reached
0.78 but multilingual `choice` is 0.40 and multilingual `score` ECE is 0.18 (see
`benchmarks/report.md`). The multilingual checkpoint carries six training languages with imbalanced
support, and calibration is fitted **per primitive**, not per language.

**Status (retrained, published, then rank-bumped).** Data is fully localized with per-record RNG and
a low distractor rate; calibration is per `(primitive, language)`; the runtime applies `kind:lang` →
`kind` → global. The B-1 adapters are published (`munod/jeba-en`, `munod/jeba-multi`).
**Follow-up (lora-rank experiment):** the multilingual LoRA rank was raised 16 → **64**
(`lora_alpha` 128), removing the cross-language capacity bottleneck. Measured multilingual overall
accuracy **0.702 → 0.853** and `es` ECE **0.170 → 0.038** (`es` accuracy 0.472 → 0.956). Two of
six languages now meet ECE ≤ 0.05 (`es` 0.038, `pt` 0.024); `de` 0.063, `fr` 0.051, `it` 0.059 and
`nl` (ECE 0.104, accuracy 0.663) remain. English is unchanged
(overall 0.763). Spec/tasks: `.specs/features/multilingual-quality/`,
`.specs/features/lora-rank-experiment/`.

**Remaining plan.**
1. Four languages sit above the 0.05 ECE target (`nl` 0.104, `de` 0.063, `it` 0.059, `fr` 0.051),
   and `nl` also has the lowest accuracy (0.663). Investigate
   whether it needs more per-language support, a language-specific learning rate, or a richer
   calibration mapping; the scalar per-language temperature already exists.
2. [x] Republished the r=64 multilingual adapter to the Hub (`munod/jeba-multi`, commit `599df58`).
3. Keep per-language accuracy/ECE reporting and gate on the worst language, not the average.

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
fallback on CPU/MPS/absent extras). `benchmarks/fast_path.py` measures stock vs fast: **p50 10.27 →
3.83 ms, p95 11.05 → 4.12 ms (2.68× p50)**, answer parity max abs **0.0044**, **0 top-label flips**.
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

## B-3 — Confidence thresholding & System-2 handoff · Done

**Why.** Every `choice`/`score` answer already carries `confidence` (selected mass,
`calibration.py`) and `noul` returns a probability, but there is no first-class helper or
documented pattern for "abstain / hand off to a System-2 LLM when confidence < τ". Users
currently reimplement the threshold by hand, and the project has no entropy/uncertainty metric
beyond the selected mass.

**Status (done, 2026-09-25).** `calibration.py` gains `normalized_entropy` and `margin` beside
`confidence`; `handoff.py` exposes `assess`/`assess_response` returning typed `Uncertainty`/
`HandoffSignal`/`HandoffReport`; the SDK re-exports them and the CLI appends a sibling `handoff`
object via `--threshold` (canonical output unchanged when the flag is absent). The pattern is
documented in `docs/cookbook-handoff.md`; `CAL-06` is traced. Spec/tasks:
`.specs/features/system2-handoff/`.

**Plan.**
1. [x] Uncertainty helpers (`normalized_entropy`, `margin`) in `calibration.py` + tests.
2. [x] `handoff.py` with `assess`/`assess_response`; SDK export and CLI `--threshold`; client-side,
   wire-shape unchanged.
3. [x] Documented pattern (when to hand off, suggested τ, composing with an LLM).

**Acceptance.**
- [x] Helpers covered by unit tests; no change to `/v1/systemone` shape (contract suite unchanged).
- [x] A documented, tested example of `if confidence < τ: handoff()`.
- [x] Usable from the CLI without an API key (`--backend fake`).

**Risks / notes.** τ stays a required, configurable knob (no universal default).
`noul` has no separate `confidence`, so the helper uses its binary certainty `max(p, 1-p)`, which
keeps τ pointing the same way for every primitive.

**Related.** `docs/protocol.md` (confidence semantics), `docs/cookbook-handoff.md`,
`docs/adr/ADR-0005`, `NFR-C06`.

---

## B-4 — Input-noise robustness (typos / accents / slang) · Done (measured: neutral-negative)

**Why.** Synthetic states are clean templates. Real chat/user input has typos, missing accents,
absent punctuation and regional slang, so accuracy outside controlled templates degrades. L-003
shows data-template artifacts are a real failure mode.

**Status (measured, 2026-09-25).** Code plus an RTX 3060 run with `noise_rate=0.15`.
`training/generate_data.py` gains a seeded, opt-in `noise_rate` that applies one surface edit
(char swap/delete, accent strip, casing flip, terminal-punctuation drop) to the `state` only,
drawing from a per-record RNG so it stays independent of the cyclic label (L-003).
`training/evaluate.py` adds `--noise-rate` and reports the noisy view under `report["noisy"]`;
`benchmarks/report.py` renders it. **Result:** the clean-trained adapters are already robust to
this noise model — English 0.763→0.760, multilingual 0.702→0.701 at the r=16 baseline and
0.853→0.847 for the released r=64 adapter. A
noise-augmented multilingual adapter scored 0.719 on both views but calibrated worse (ECE 0.090
vs 0.033) and regressed on `de`/`nl`, so it is **not** released. Spec/tasks:
`.specs/features/input-noise/`.

**Plan.**
1. [x] Seeded, deterministic noise-injection in `training/generate_data.py` (config field
   `noise_rate`, CLI `--noise-rate`); determinism preserved (same seed → byte-identical).
2. [x] Clean vs noisy evaluation split: `evaluate.py --noise-rate` returns `report["noisy"]`;
   `benchmarks/report.py` renders it.
3. [x] Retrain with `training/configs/finetune_multi_noisy.json` and compare accuracy/ECE on both
   splits.

**Acceptance.**
- [x] Noise is seeded, reproducible, and opt-in via a config flag.
- [~] Noisy-split accuracy improves vs baseline: **not met** — released adapters were already
  robust and the augmented adapter did not beat them (reported honestly).
- [x] Both splits reported in `benchmarks/report.md`.

**Risks / notes.** Over-noising can teach noise invariance at the cost of clean accuracy; tune
the rate. Keep probe evaluation on unmodified public inputs for comparability. Effort ~1 day +
retrain.

**Related.** `STATE.md` L-003, `docs/training.md` (§1), `TRAIN-01`, `TRAIN-08`.

---

## B-5 — Multi-domain coverage (5 domains) · Idea

**Why.** Today the generator's `choice` criteria are hard-coded to four support teams
(`_TEAMS`, `team_descriptions.json`) with a support-triage lexicon. Broadening to distinct
domains (support, e-commerce/logistics, voice/smart-home commands, agent tool/function
selection, document/media classification) would make the model valuable across more agent flows.

**Plan.**
1. Generalize the data model so `choice` criteria per domain come from committed data, rather
   than a single hard-coded team set.
2. Author domain lexicons/phrases across the existing 7 languages (or a reduced set first).
3. Report per-domain accuracy/ECE and gate on the worst domain.

**Acceptance.**
- Per-domain accuracy/ECE published; no regression on the existing support domain.
- Domain data is deterministic and localized like the current lexicon.

**Risks / notes.** Cost multiplies by language count; tool/function selection largely overlaps
the existing `choice` case, so it may not add a new capability, only vocabulary. Effort ~2–3 days
per language cohort.

**Related.** `training/generate_data.py`, `training/data/`, `docs/training.md` (§1), `B-1`.

---

## B-6 — Contrastive pre-fine-tuning (InfoNCE / Triplet) · Idea

**Why.** The local engine decides by **cosine similarity** (`EncoderModel.answer_state`), so the
geometry of the embedding space directly determines accuracy. A short contrastive stage that pulls
semantically equivalent intents together (across languages) and pushes opposite-sense near-misses
apart should give the choice scorer and `noul`/`score` heads cleaner vectors — the highest-upside
quality idea on the table, and complementary to RLCD/proper-scoring.

**Plan.**
1. Build triplets/pairs from generated records: `(state, its criterion/question)` as positives,
   cross-intent near-misses as negatives.
2. Add a short contrastive pre-stage (InfoNCE or triplet) before the LoRA/proper-scoring run,
   behind a config in `training/`.
3. Ablate: baseline vs contrastive-pretrained, measuring top-1 accuracy **and** ECE (contrastive
   must not degrade calibration).

**Acceptance.**
- Contrastive stage is deterministic, config-driven, and runs within the 12GB budget.
- Top-1 improves on the multilingual held-out set with ECE not worse than baseline.
- Results (with/without) recorded in `benchmarks/report.md`.

**Risks / notes.** Extra GPU stage and data preparation; risk of representation collapse without
careful negative sampling. Treat as an experiment with a strict ablation gate. Effort ~2–4 days.

**Related.** `backends/encoder.py` (cosine decision), `training/finetune_rlcd.py`, `STATE.md` L-002.

---

## B-7 — Public-probe evaluation (MASSIVE / XNLI / typed-decisions) · Ready

**Why.** Every published number today comes from the deterministic **synthetic** held-out split
(`benchmarks/report.md` says so explicitly), so OPS-06 and the M6 exit criterion "benchmark report
comparable to MASSIVE / XNLI / typed-decisions" are only partially met. Public probes give an
externally comparable number.

**Status (not started).** Harness, model card, adapters and the synthetic report are published;
no public-probe run exists. Tracked as `OPS-06` partial and as the unmet M6 exit criterion.

**Plan.**
1. Evaluation-only loaders for MASSIVE (intent), XNLI (entailment) and typed-decisions mapped to
   jeba's `choice`/`score`/`noul` primitives; no training data changes.
2. Run the released adapters (`munod/jeba-en`, `munod/jeba-multi`) on the probe sets and report
   accuracy/ECE per language under `benchmarks/results/`.
3. Render the comparison in `benchmarks/report.md` with the exact reproduction commands
   (`benchmarks/public_probes.py` is still undelivered — see `benchmarks/README.md`).

**Acceptance.**
- Probe results published in `benchmarks/report.md` with seed, hardware and exact commands.
- No regression claim is made on the synthetic numbers; both sets are labeled.
- `OPS-06` moves to `Implemented` and the M6 exit criterion is met.

**Risks / notes.** Probe licenses and download size; label/option spaces do not always map to a
jeba `choice` criterion, so some probes may need a `noul`/`score` formulation. Evaluation-only —
must not leak into training data (keeps comparability). Effort ~2–3 days (no GPU retrain).

**Related.** `docs/benchmarks.md` (Known limitations), `benchmarks/README.md`, `OPS-06`,
`docs/training.md` (§5), `NFR-D04`.

---

## B-8 · Hardening — silent degradation when calibration assets fail to load · Ready

**Why.** `load_temperatures()` and `load_choice_head()` in `src/jeba/backends/encoder.py` swallow
every exception (`except Exception: return {} / None`). On a partially populated cache — the exact
scenario `JEBA_OFFLINE=1` promises to make safe — the engine then runs **without per-language
temperature scaling and without the `choice` head**, silently, and reports confident numbers that
do not reflect the calibrated model. Reported by the 2026-09-26 documentation review: an
`JEBA_OFFLINE=1` run failed with `LocalEntryNotFoundError` because the Hub `refs/main` pointed at
an empty snapshot.

**Status (not started).** Documentation now describes the real prefetch step
(`docs/huggingface.md`, `docs/adr/README.md`); the code path is unchanged.

**Plan.**
1. Distinguish "asset genuinely absent" (fine, documented) from "asset present but unreadable /
   cache corrupt" (a bug) and `log.warning` on the latter, naming the file and the repo.
2. In `offline` mode, fail loudly (or at least warn on stderr) instead of degrading silently.
3. Add a regression test with a seeded fake encoder that asserts the warning is emitted when
   `temperature_calibration.json` / `choice_head.json` cannot be read.

**Acceptance.**
- A corrupt/partial cache produces a warning naming the missing asset; a clean first run does not.
- `tests/test_encoder.py` covers both branches; no wire-shape change (contract suite untouched).

**Risks / notes.** Behaviour change is log-only; do not turn it into a hard failure for users who
intentionally run without calibration. Effort ~0.5 day.

**Related.** `docs/huggingface.md` (§4), `docs/adr/README.md` (ADR-0010 note), `NFR-C05`.

---

## B-9 · Which English checkpoint is canonical? · Ready (measured; needs a decision)

**Why.** `munod/jeba-en` holds two different English checkpoints and the trade-off is real, not
theoretical. On 2026-09-26 both were evaluated on the same split (`data/eval_en.jsonl`,
1500 records, RTX 3060, `--backend encoder`):

| Metric | `bac4de41` (previous Hub) | `checkpoints/en` (published now) | Better |
| --- | ---: | ---: | --- |
| overall accuracy | **0.7813** | 0.7633 | `bac4de41` |
| overall ECE | 0.0767 | **0.0609** | current |
| `choice` accuracy | 0.7080 | **0.8340** | current |
| `choice` ECE | 0.1406 | **0.0739** | current |
| `noul` accuracy | 0.7440 | 0.7440 | tie |
| `noul` ECE | **0.0119** | 0.1011 | `bac4de41` |
| `score` accuracy | **0.8920** | 0.7120 | `bac4de41` |
| `score` ECE | 0.0897 | **0.0416** | current |

**Status.** Published as `checkpoints/en` per AD-008 (consistency with `benchmarks/report.md`,
README and the model card; `choice` is the primitive behind `triage`/`router`/`guard`). The
previous checkpoint is **not lost**: `hf_hub_download("munod/jeba-en", ..., revision="bac4de4")`.
Raw run: `benchmarks/results/en_legacy_bac4de4.json` (gitignored).

**Plan.**
1. Decide whether headline accuracy (`bac4de41`) or `choice` quality + calibration (current)
   matters more for the first release; the two should not both be called "the" English adapter.
2. Reproduce both runs before deciding:
   `JEBA_ADAPTERS="jeba-en=<path>" uv run python -m training.evaluate --data data/eval_en.jsonl --out <out>.json --backend encoder`.
3. If `bac4de41` wins, restore it and re-run `benchmarks/report.md` + model card together —
   never update the Hub and the report in separate steps (L-005).

**Acceptance.**
- One checkpoint is declared canonical in `benchmarks/report.md` and `docs/model-card.md`.
- The losing variant is either dropped or published under a clearly named revision/repo.
- The `noul` calibration gap (0.012 vs 0.101) is understood before it is accepted.

**Risks / notes.** Accuracy and ECE pull in opposite directions here, so "better" depends on the
product call: temperature scaling fixes calibration without retraining, but the `choice`/`score`
accuracy split comes from the adapter + choice head themselves. Effort ~0.5 day (no retrain) if
only relabelling; a retrain makes it a full training cycle.

**Related.** `.specs/project/STATE.md` AD-008, L-005; `docs/model-card.md`,
`benchmarks/report.md`, `NFR-C06`, `NFR-C07`.

---

## Evaluated and not pursued (for now)

These proposals were assessed against the frozen wire (ADR-0001) and the actual similarity-based
engine; they are deliberately **not** scheduled. Revisit only with a new ADR and measured
evidence.

- **`route` (multi-label) as a canonical primitive · rejected.** `wire._check_normalized` requires
  `sum(probabilities) == 1.0`, so independent sigmoids across categories are incompatible with the
  contract. The use case is already expressible as N independent `noul` questions composed by the
  client, which is the protocol's intended model. At most an additive extension endpoint.
- **`extract_span` (extractive QA / entity spans) · rejected for the current engine.** Requires
  token-level outputs and `start/end` fields; `load_encoder` mean-pools and exposes only pooled
  vectors, and the response has no span shape. Would be a new backend/endpoint, not an evolution
  of the similarity engine, and sits next to the "atomic structured decisions only" non-goal.
- **MoE of LoRA adapters / per-task adapter switching · deferred as premature.** One adapter is
  trained per checkpoint; there are no real per-task neural heads. The measured bottleneck is
  calibration (per-language ECE), not trunk capacity. Adapter routing complicates ONNX export and
  the CUDA-graph fast path with no demonstrated gain and overfit risk on the synthetic set. A real
  per-primitive head would be the more impactful change, and only after evidence of underfit.

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
