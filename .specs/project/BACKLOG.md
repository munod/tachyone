# Backlog

Active next steps and carried-over ideas for Tachyone. This is the canonical "what's next" list;
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
`kind` → global. The B-1 adapters are published (`munod/tachyone-en`, `munod/tachyone-multi`).
**Follow-up (lora-rank experiment):** the multilingual LoRA rank was raised 16 → **64**
(`lora_alpha` 128), removing the cross-language capacity bottleneck. Measured multilingual overall
accuracy **0.702 → 0.853** and `es` ECE **0.170 → 0.038** (`es` accuracy 0.472 → 0.956). Two of
six languages now meet ECE ≤ 0.05 (`es` 0.038, `pt` 0.024); `de` 0.063, `fr` 0.051, `it` 0.059 and
`nl` (ECE 0.104, accuracy 0.663) remain. The rank bump left English alone; English was
reseeded separately in B-9 (overall 0.859). Spec/tasks: `.specs/features/multilingual-quality/`,
`.specs/features/lora-rank-experiment/`.

**Remaining plan.**
1. Four languages sit above the 0.05 ECE target (`nl` 0.104, `de` 0.063, `it` 0.059, `fr` 0.051),
   and `nl` also has the lowest accuracy (0.663). Investigate
   whether it needs more per-language support, a language-specific learning rate, or a richer
   calibration mapping; the scalar per-language temperature already exists.
2. [x] Republished the r=64 multilingual adapter to the Hub (`munod/tachyone-multi`, commit `599df58`).
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
(`src/tachyone/fast.py`: `maybe_accelerate` returns the stock target with reason "no accelerated kernels
registered"). There is no measured latency improvement yet, and `NFR-P01` is still `[open]`.

**Status (done).** `maybe_accelerate` now takes an injected accelerator builder and is wired by
`TACHYONE_FAST` in `load_encoder` to a per-shape CUDA-graph forward with bf16-resident weights (graceful
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
this noise model — English 0.859→0.854, multilingual 0.702→0.701 at the r=16 baseline and
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

## B-5 — Multi-domain coverage (5 domains) · Ready (evidence measured 2026-09-26)

**Why.** Today the generator's `choice` criteria are hard-coded to four support teams
(`_TEAMS`, `team_descriptions.json`) with a support-triage lexicon. Broadening to distinct
domains (support, e-commerce/logistics, voice/smart-home commands, agent tool/function
selection, document/media classification) would make the model valuable across more agent flows.

**Evidence (2026-09-26, head-to-head).** The released English adapter scores **0.854** on its own
support records and **0.229** on the public nine-family probe (`pngwn/system-one-decisions`), even
on the four `tickets_*` families that nominally share its vocabulary (0.031–0.500) — because their
option spaces are 52 queues and 77 intents, not our four teams. A peer scorer trained on that data
scores 0.705 there. Domain coverage, not architecture, is the binding constraint; full tables in
`docs/compare.md` §3.

**Plan.**

1. **Refactor the generator without changing what already exists.** `DataConfig` gains
   `domains: tuple[str, ...] = ("support",)` — the default keeps today's behaviour. Per-domain
   content moves to committed data (`training/data/domains/<domain>.json`: option labels +
   descriptions, entities, phrase banks, score levels, `noul` criteria), reusing the current
   lexicon structure. Records gain a `domain` field.
   **Hard guarantee:** existing configs (`data.json`, `data_multi.json`, `data_noisy.json`, the
   B-9/B-4 runs) must produce **byte-identical** output — a hash regression test, because L-003 and
   B-4 already cost days to a determinism assumption that did not hold.
2. **Five domains**, chosen to line up with the five recipes in `docs/use-cases.md`:

   | Domain | `choice` | `score` | `noul` |
   | --- | --- | --- | --- |
   | `support` (exists) | team | urgency, frustration | churn risk |
   | `ecommerce` | order status / queue | priority | refund due? |
   | `agent_tools` | tool or model tier | complexity | needs tools? |
   | `documents` | document type | confidence | needs human review? |
   | `voice` | device / action | confidence | is it a command? |
3. **Scope: English first (`B-5a`), multilingual second (`B-5b`).** Localizing four new domain
   lexicons is the human cost; 7 languages at once multiplies it. `B-5b` follows only once the
   refactor and the gates are proven in `en`.
4. **Measure per domain.** `training/evaluate.py` aggregates per-primitive and per-language only —
   add `per_domain`, and render it in `benchmarks/report.py`.
5. **Training cost must be estimated before running.** 5× records ≈ 5× steps: at today's settings
   (9k records, 4 epochs) that is roughly 8–12 h on the RTX 3060. Mitigations: `per_type` ≈ 1,000
   per domain, or fewer epochs. New config keeps the pinned `seed: 2` (AD-009).
6. **Gates on the worst domain, never the average** (project convention): no regression on
   `support` (English overall stays ≥ 0.85); worst new domain ≥ 0.70; per-domain ECE published
   against the 0.05 target with the exceptions declared.
7. **Publish and re-measure.** Republish `munod/tachyone-en` as a new revision (the AD-009
   pattern), regenerate `benchmarks/report.md`, `docs/model-card.md`, README and CHANGELOG, then
   **re-run the `B-7` probes** — the harness takes minutes and the before/after table on public
   data is the real payoff of this item.

**Acceptance.**
- Existing configs regenerate byte-identically (hash test in `tests/`).
- Five domains with deterministic, localized, committed data; `B-5a` English first.
- Per-domain accuracy/ECE published and gated on the worst domain; no regression on `support`.
- Adapter republished; `benchmarks/report.md`, model card, README and CHANGELOG updated.
- Probe before/after table published (`B-7` harness).

**Risks / notes.** Five domains in one LoRA of fixed capacity may dilute per-domain accuracy —
that is what the worst-domain gate is for. Training time vs the 12 GB budget (ADR-0005). Tool/
function selection largely overlaps the existing `choice` case, so it may add vocabulary rather
than capability. `presets.py`, `docs/use-cases.md` and `docs/training.md` must follow the data.
Effort: `B-5a` ~2–3 days (refactor + data + train + evaluate), `B-5b` ~2 days more.

**Related.** `training/generate_data.py`, `training/data/`, `docs/training.md` (§1), `B-1`, `B-7`.

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

## B-7 — Public-probe evaluation (MASSIVE / XNLI / typed-decisions) · Done (2026-09-27)

**Why.** Every published number today comes from the deterministic **synthetic** held-out split
(`benchmarks/report.md` says so explicitly), so OPS-06 and the M6 exit criterion "benchmark report
comparable to MASSIVE / XNLI / typed-decisions" are only partially met. Public probes give an
externally comparable number.

**Status (2026-09-26 — external probe delivered).** `benchmarks/compare.py` runs an
**evaluation-only** head-to-head on the public `pngwn/system-one-decisions` test split (9 task
families: ag_news, banking77, go_emotions, mmlu, yelp_score, 4× tickets), with no training data
change and no leak. Results, method and caveats are published in `docs/compare.md` §3. Findings
worth carrying forward:

- Tachyone (released English adapter) scores **0.229** there against **0.854** on its own records —
  domain coverage, not an architecture verdict; recorded in `docs/benchmarks.md` §Known
  limitations and used as the evidence for **B-5**.
- The harness **reproduces the peer's own published metrics** (T 1.75 vs 1.75, acc 0.705 vs 0.707,
  ECE 0.046 vs 0.044, 537/576 rows, 8 of 9 per-task values identical), which is what makes the
  comparison usable as external evidence.

**Status (done, 2026-09-27 — all three named probes published).** `benchmarks/probes.py` loads
typed-decisions, MASSIVE and XNLI onto the harness row shape; `benchmarks.compare run --format
probe --probe <name>` evaluates them with the same engines, metrics and temperature fit as
everything else; `python -m benchmarks.probes render` composes the committed report
**`benchmarks/probes.md`** (tables, licences, citations, chance levels, reproduction commands).
Phase 1 ran with the released adapters through the language router (`--backend encoder`):

| Probe | n | Accuracy | Chance | ECE raw | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| typed-decisions (`all/test`, 4 configs) | 2000 | **0.323** | 0.20–0.50 | 0.408 | best config 0.402 (`customer_service`, `security_incidents`), worst 0.218 (`invoice_processing`) |
| MASSIVE (7 locales) | 3584 | **0.033** | 0.017 | 0.214 | per language 0.016 (`de`) … 0.047 (`en`) — at chance everywhere |
| XNLI (`en`) | 5010 | **0.334** | 0.333 | 0.452 | exactly chance |

- **XNLI licence verified as required**: **CC BY-NC 4.0**, read from
  `facebookresearch/XNLI`'s `LICENSE` — *not* CC BY-SA as this entry guessed, and the HF card has
  no tag at all. Publishing derived metrics (our scores) is fine; the corpus is never committed.
- **Per-language accuracy/ECE for `B-1`** comes out of the MASSIVE run (`task` = language): all
  seven languages are at chance on 60 intents, so these are public numbers for the *task*, not a
  per-language quality verdict — the synthetic per-language table in `docs/benchmarks.md` remains
  the in-domain one.
- All three temperature fits landed on the grid ceiling (**T=20.0**), so `ECE cal` is bought by
  flattening: `probes.md` publishes `Conf` and `Brier` next to it and says so (lesson L-007).

**Remaining plan.**

1. [x] **New loader module `benchmarks/probes.py`** (keep `compare.py` focused on metrics): one
   function per dataset converting it to the harness's standard row shape
   (`state / task / question / options / answer_index`), so every engine, metric, temperature fit
   and the renderer are reused untouched. `compare.py` gains `--format probe --probe <name>
   --data <dir>` next to the existing `probe`/`jsonl` formats.

   | Probe | HF repo | License | Size | Row mapping |
   | --- | --- | --- | --- | --- |
   | typed-decisions | `LocalLLaMA/typed-decisions` | **Apache-2.0** | 4 configs (`agent_trace_observability`, `customer_service`, `invoice_processing`, `security_incidents`) + `all`, n<1K each, **3.15 MB** total | already `noul`/`choice`/`score`; flatten multi-question rows to **one row per question** (deliberately conservative — it ignores the `choice` head's batching; say so) |
   | MASSIVE | `AmazonScience/massive` | **CC-BY-4.0** | 60 intents, 51 languages, `en` config tens of MB | `choice` over the 60 intent labels (max is 255 ✓), fixed `instructions` |
   | XNLI | `facebook/xnli` | **CC BY-NC 4.0** — verified 2026-09-27 in `facebookresearch/XNLI`'s `LICENSE` (the HF card has no tag and its Licensing Information section is a placeholder) | `en`: 5,010 test / 2,490 val, ~50 MB | `choice` over {entailment, neutral, contradiction}, `state` = premise + hypothesis; documented as a design decision against the alternative `noul` formulation |

   Bigger volume option after the three named probes: `tasksource/procedural-typed-decisions`
   (Apache-2.0, 100K–1M rows, answers **computed by rule** from the state rather than labeled by a
   model). Several other `typed-decisions` repos exist (tasksource 2.5M, pngwn CC-BY-SA) — the
   LocalLLaMA one is the default because it is independent, Apache-2.0 and already in the wire's
   shape.

2. [x] **What to run.** Phase 1 (required by `OPS-06`) **done**: `tachyone` with both released
   adapters (`munod/tachyone-en`, `munod/tachyone-multi`) on all three probes, routed per row by
   language. Phase 2 (optional, not run): the peer
   scorer and the LLMs on **typed-decisions only** — MASSIVE's 60-option prompts are the case we
   already measured as impractical for autoregressive models (46 s per 52-option attempt).

   MASSIVE across our six trained languages is deliberately in scope: it is the first **public**
   per-language accuracy/ECE, which is exactly what `B-1` needs to stop quoting synthetic
   per-language numbers.

3. [x] **Where results land.** Artifacts under `benchmarks/results/probe_*.json` (gitignored, each
   recording the exact command), rendered table committed to **`benchmarks/probes.md`** (separate
   from `report.md`, which `benchmarks/report.py` regenerates wholesale), plus a "Public probes"
   section in `docs/benchmarks.md` with hardware, seed, and the citation/license of each dataset.

**Acceptance.**
- [x] All three named probes published with the exact reproduction command, hardware and dataset
  citations; both synthetic and probe numbers labeled side by side (no regression claim made on
  either) — `benchmarks/probes.md` (renderer: `python -m benchmarks.probes render`).
- [x] `OPS-06` moves to `Implemented`; the M6 exit criterion "benchmark report comparable to
  MASSIVE/XNLI/typed-decisions" is met (`docs/requirements/traceability.md`, `docs/roadmap.md`).
- [x] MASSIVE covers all six trained languages **plus English**, with per-language accuracy/ECE
  published (`task` = language, 512 rows each).

**Risks / notes.** XNLI licence **confirmed: CC BY-NC 4.0** (see the table above); a 60-way
`choice` did land near chance (0.033 vs 0.017) — published anyway, that is the number; label
mapping has a test per probe (`tests/test_benchmark_probes.py`, 14 tests including the two
`noul` questions that ship without `criteria`). Evaluation-only: probe rows never reach
`training/` (that is what keeps the comparison comparable). Actual effort: one session, no GPU
retrain, ~120 MB of downloads (MASSIVE is a 40 MB official tarball, not a `snapshot_download`).

**Related.** `docs/benchmarks.md` (Known limitations), `docs/compare.md` §3,
`benchmarks/README.md`, `OPS-06`, `docs/training.md` (§5), `NFR-D04`.

---

## B-8 · Hardening — silent degradation when calibration assets fail to load · Done (2026-09-27)

**Why.** `load_temperatures()` and `load_choice_head()` in `src/tachyone/backends/encoder.py` swallow
every exception (`except Exception: return {} / None`). On a partially populated cache — the exact
scenario `TACHYONE_OFFLINE=1` promises to make safe — the engine then runs **without per-language
temperature scaling and without the `choice` head**, silently, and reports confident numbers that
do not reflect the calibrated model. Reported by the 2026-09-26 documentation review: an
`TACHYONE_OFFLINE=1` run failed with `LocalEntryNotFoundError` because the Hub `refs/main` pointed at
an empty snapshot.

**Status (done, 2026-09-27).** Both loaders now resolve the asset through one `_adapter_asset()`
helper and read it through `_read_asset_json()`, and every non-routine outcome logs a WARNING
naming **the asset and the source** before the engine degrades:

| Situation | Behaviour |
| --- | --- |
| adapter does not ship the file (online, hub says `EntryNotFoundError`) | silent — documented state |
| local adapter dir without the file, online | silent — documented state |
| `LocalEntryNotFoundError` (not cached; a partial prefetch looks exactly like this) | **warn** |
| `RepositoryNotFoundError` (mis-typed adapter id) / `HfHubHTTPError` (network, 500) | **warn** |
| file present but unreadable, not a JSON object, or a non-numeric temperature / broken scorer | **warn** |
| any of the above under `TACHYONE_OFFLINE=1`, including a missing file in a local dir | **warn** |

Warnings go through `logging.getLogger("tachyone.backends.encoder")`, which the package does not
configure, so Python's last-resort handler prints them to **stderr**. Nothing raises: the baseline
still answers (log-only, per the risk note below). The previously-raising path — a corrupt
temperature value escaping `parse_temperature_report()` as an unhandled `ValueError` — now warns
and degrades instead of crashing the backend.

**Plan.**
1. [x] Distinguish "asset genuinely absent" (fine, documented) from "asset present but unreadable /
   cache corrupt" (a bug) and `log.warning` on the latter, naming the file and the repo.
2. [x] In `offline` mode, fail loudly (or at least warn on stderr) instead of degrading silently.
3. [x] Add a regression test with a seeded fake encoder that asserts the warning is emitted when
   `temperature_calibration.json` / `choice_head.json` cannot be read.

**Acceptance.**
- [x] A corrupt/partial cache produces a warning naming the missing asset; a clean first run does not.
- [x] `tests/test_encoder.py` covers both branches; no wire-shape change (contract suite untouched).

**Risks / notes.** Behaviour change is log-only; do not turn it into a hard failure for users who
intentionally run without calibration. Effort ~0.5 day (actual: one session, 11 tests).

**Related.** `docs/huggingface.md` (§4), `docs/adr/README.md` (ADR-0010 note), `NFR-C05`.

---

## B-9 · Which English checkpoint is canonical? · Done (seed sweep → AD-009)

**Why.** `munod/tachyone-en` held two different English checkpoints and the trade-off looked like a
property of the model. On 2026-09-26 every candidate was evaluated on the same split
(`data/eval_en.jsonl`, 1500 records, RTX 3060, `--backend encoder`, calibration refit per
checkpoint with the identical procedure):

| Run | overall acc | overall ECE | `choice` acc | `score` acc | `noul` acc | `val_loss` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bac4de41` (previous Hub) | 0.7813 | 0.0767 | 0.7080 | 0.8920 | 0.7440 | — |
| seed 42 (first publication) | 0.7633 | 0.0609 | 0.8340 | 0.7120 | 0.7440 | 0.4177 |
| seed 42 (control, re-run) | 0.7853 | 0.0739 | 0.9000 | 0.7120 | 0.7440 | 0.4195 |
| seed 1 | 0.7720 | 0.0789 | 0.6700 | 0.9020 | 0.7440 | 0.3791 |
| **seed 2 — published** | **0.8587** | **0.0225** | **0.9480** | **0.9100** | 0.7180 | **0.3264** |

**Finding.** The loop is not deterministic and lands in distinct optima: two runs of the *same*
seed-42 config disagree (`choice` 0.834 vs 0.900, `val_loss` 0.4177 vs 0.4195), and seed 1
reproduces the old checkpoint's profile (low `choice`, high `score`). The accuracy-vs-`choice`
trade-off was initialization, not a property of the architecture. See `STATE.md` L-005 and L-006
for how each structural explanation (temperature, code, config, base model, package versions,
dataset) was ruled out before landing on variance.

**Status (done, 2026-09-26).** `seed: 2` won 7 of 8 metrics against every earlier checkpoint,
cleared both acceptance targets with margin, and posted the lowest English ECE measured. Promoted
per AD-009: `training/configs/finetune_en.json` pins `seed: 2`, `checkpoints/en` holds the run,
`benchmarks/report.md` + README + model card + requirements were regenerated together, and
`munod/tachyone-en` was republished. Nothing was lost: `bac4de41` stays at
`hf_hub_download("munod/tachyone-en", ..., revision="bac4de4")` and the pre-sweep `checkpoints/en`
was kept as `checkpoints/en_prev_pub0.7633`.

**Acceptance.**
- [x] One checkpoint declared canonical in `benchmarks/report.md` and `docs/model-card.md`.
- [x] Both targets met: overall ≥ 0.7813 → **0.8587**; `choice` ≥ 0.8340 → **0.9480**.
- [x] ECE no longer the loser: 0.0609 → **0.0225**.
- [x] The single regression — `noul` accuracy 0.7440 → 0.7180 — recorded and accepted
      (`noul` ECE improved 0.1011 → 0.0202).

**Risks / notes.** The loop is non-deterministic, so the recipe reproduces a *distribution*, not
this artifact; `seed: 2` is pinned to record what produced it, and a re-run of the recipe may
land elsewhere. The eval split shares 266 of 281 unique `score` states with the training data,
so these are in-sample synthetic numbers (already labelled as such in `docs/benchmarks.md`) —
not a measurement on unseen text. Public probes remain open in **B-7**.

**Related.** `.specs/project/STATE.md` AD-008, AD-009, L-005, L-006; `docs/model-card.md`,
`benchmarks/report.md`, `NFR-C06`, `NFR-C07`.

---

## B-10 · LLM backend: the prompt under-specifies the response wrapper · Done (2026-09-26)

**Why.** Measured on 2026-09-26 while benchmarking small local models through the shipped `llm`
backend: `ling-tiny` produced a contract-valid answer for **3 of 36** probe questions and **20 of
48** support questions; the recurring failure is `model output is missing an 'answers' object`.

**Root cause (in our prompt, not in the model).** `_SYSTEM_PROMPT` in `src/tachyone/backends/llm.py`
documents the wrapper only in the `noul` example:

```
The shape is {"answers": {"<question id>": {"noul": ...}}} for noul,
{"choice": "<option>", "probabilities": {...}} for choice, ...
```

A model that takes the `choice` example literally answers with the *inner* object. Strong
frontier models infer the wrapper; small ones do not, and every failure costs
`TACHYONE_LLM_RETRIES + 1` autoregressive attempts (a 52-option question took 46 s per attempt).

**Status (done, 2026-09-26).** `_SYSTEM_PROMPT` now states the wrapper once for all three
primitives, gives one complete example with question ids, and bans the value errors actually
observed (boolean `noul`, criteria text echoed back, bare string instead of an answer object,
option label used as the question id); `build_answers` names the missing `answers` key and echoes
the top-level keys it saw, and stays strict — no inner-shape fallback. Re-run of the four LLM rows
(same servers, flags and rows, greedy): probe `JSON ok` **0.083 → 0.889** (`ling-tiny`) and
**0.889 → 1.000** (`ornith-9b`), home **0.417 → 0.917** and **0.833 → 1.000**. Prompt selection was
gated on the **worst set** (project convention): four candidate prompts were measured on both sets
and the winner posts 0.889 / 0.917 against the runner-up's 0.917 / 0.729 — an earlier draft that
scored higher on the probe alone regressed home `noul` to 3/16 (below the original prompt's
10/16), which is why one set was not allowed to decide. Accuracy/ECE moved with the answer
distribution and are recorded, not compared, per the plan.

**Plan.**
1. [x] Spell out the wrapper **once** for all three primitives, with one complete example per type
   and an explicit rule: the outer object MUST contain an `answers` key mapping question id →
   answer.
2. **Decision (2026-09-26): stay strict — do not accept the inner shape as a fallback.**
   `build_answers` / `_answer_from_raw` do not change. Reason: tolerating the inner object would
   push measured compliance to ~100% by construction and destroy the `JSON ok` column in
   `benchmarks/compare.py`, which exists precisely to catch this. Improve the error instead — name
   the missing `answers` key and echo the top-level keys actually observed. If compliance is still
   poor after the prompt fix, revisit tolerance as its own decision, with the harness counting
   "model emitted the contract shape" separately from "backend answered". **Still strict after the
   fix** — residual failures are reported below.
3. [x] Re-run the LLM rows of `benchmarks/compare.py` (probe n=36, home n=48) and publish the
   before/after compliance.

**Acceptance.**
- [x] A small local model's contract compliance on the probe rises well above the measured 8.3%
  without changing `POST /v1/systemone` (contract suite untouched) → **0.889** (`ling-tiny`),
  **1.000** (`ornith-9b`).
- [x] `tests/test_backends_llm.py` covers the wrapper being documented for `choice`/`score`, and the
  inner-object failure path producing the actionable message.
- [x] Before/after numbers published in `docs/compare.md` §3 ("What the prompt fix changed").

**Risks / notes.** Prompt changes move answer distributions, so accuracy/ECE for LLM backends
are not comparable before vs after; record both. Effort ~0.5 day.

**Related.** `docs/compare.md` §3 (measured compliance), `tests/test_backends_llm.py`,
`benchmarks/compare.py`, ADR-0008.

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
