# State

**Last Updated:** 2026-09-26
**Current Work:** Post-M6 backlog. **B-2 (fast path) done**: `TACHYONE_FAST` wires
`maybe_accelerate` to a per-shape CUDA-graph forward (bf16 weights) with graceful fallback;
`benchmarks/fast_path.py` measures 2.68× p50 (10.27 → 3.83 ms) and 0 top-label flips, meeting
NFR-P01/NFR-P07. **B-1 (multilingual quality)** landed, retrained on the RTX 3060, and the
adapters were republished to the Hub (`munod/tachyone-en`, `munod/tachyone-multi`); the `v0.2.0` and
`v0.3.0` GitHub Releases exist. Multilingual `choice` 0.40 → **0.684**, overall 0.609 → **0.853**;
English overall 0.721 → **0.859**; with the multilingual LoRA at r=64, per-language ECE is now
≤ 0.05 for only 2/6 languages (`es` 0.038, `pt` 0.024; `de` 0.063, `fr` 0.051, `it` 0.059 and `nl`
0.104 remain open — NFR-C06 partially open).
**B-3 (confidence thresholding / System-2 handoff) done**:
`calibration.py` gains normalized `entropy`/`margin`, `handoff.py` exposes `assess`/
`assess_response`, and the CLI appends a sibling `handoff` object via `--threshold` without
changing the canonical response. **B-4 (input-noise robustness) done and measured on the RTX
3060**: seeded opt-in `noise_rate` in `generate_data.py` plus a clean/noisy split in `evaluate.py`
(`report["noisy"]`); the clean-trained adapters are already robust (EN 0.859→0.854; multilingual
0.702→0.701 at the r=16 baseline and 0.853→0.847 for the released r=64 adapter)
and the noise-augmented adapter did not beat them (ECE 0.090 vs 0.033), so it is not released.
**Multilingual LoRA rank (NFR-C06 follow-up) done and adopted**: `lora_rank` 16 → **64**
(`lora_alpha` 128) in `finetune_multi.json`; measured multilingual overall accuracy 0.702 → **0.853**,
`es` ECE 0.170 → **0.038** (accuracy 0.472 → 0.956). Two of six languages now meet ECE ≤ 0.05
(`es` 0.038, `pt` 0.024); `de` 0.063, `fr` 0.051, `it` 0.059 and `nl` (ECE 0.104, accuracy 0.663)
remain open. The r=64 multilingual adapter is republished to the
Hub (`munod/tachyone-multi`, commit `599df58`); the English adapter was reseeded separately in B-9
(AD-009: overall 0.859 / ECE 0.023).
**B-9 (English checkpoint)** was resolved by a seed sweep — see AD-009 and L-006.
**B-10 (LLM prompt wrapper) done**: `_SYSTEM_PROMPT` spells out the `answers` wrapper for all three
primitives with a complete example and bans the value errors observed; `build_answers` names the
missing key and echoes the top-level keys it saw, staying strict (no inner-shape fallback). The
four LLM rows were re-run on the same servers/flags/rows: probe `JSON ok` 0.083 → **0.889**
(`ling-tiny`) and 0.889 → **1.000** (`ornith-9b`), home 0.417 → **0.917** and 0.833 → **1.000**;
four candidate prompts were gated on the **worst set** because a probe-only winner regressed home
`noul` to 3/16. Before/after tables are in `docs/compare.md` §3.
**B-8 (calibration-asset hardening) done**: `load_temperatures()` / `load_choice_head()` no longer
swallow every exception. An asset that is absent because the adapter does not ship it stays silent;
anything else — not cached under `TACHYONE_OFFLINE=1`, a mis-typed repo id, a network error, an
unreadable/corrupt file, a non-numeric temperature — logs a WARNING naming the asset and the source
(stderr, since the package configures no logging) and then degrades to the uncalibrated baseline.
A corrupt temperature value, which used to raise out of the loader, now warns instead of crashing.
11 tests in `tests/test_encoder.py`; `docs/huggingface.md` §4 documents the message.
**B-7 (public probes) done**: `benchmarks/probes.py` loads typed-decisions / MASSIVE / XNLI onto the
harness row shape and `python -m benchmarks.probes render` composes the committed
`benchmarks/probes.md`. Released adapters, evaluation only: typed-decisions **0.323** (chance
0.20–0.50), MASSIVE **0.033** across seven locales (chance 0.017; per-language 0.016–0.047, so the
first *public* per-language accuracy/ECE exists), XNLI **0.334** (chance 0.333). All three fits hit
the grid ceiling (T=20.0), so `Conf`/`Brier` are published beside `ECE cal` (L-007). XNLI's licence
was **verified as CC BY-NC 4.0** upstream — not the CC BY-SA the plan guessed. `OPS-06` →
Implemented, M6 exit criterion met.
**B-5a (multi-domain, English) in progress**: OD-6/OD-7 resolved (English first; `per_type` 1000/domain
→ 15,000 records, the ~2–3 h option). The generator now reads committed per-domain content from
`training/data/domains/<domain>.json` (five domains: `support`, `ecommerce`, `agent_tools`,
`documents`, `voice`), the three legacy content files were merged into `domains/support.json`, and
`training/evaluate.py` reports `per_domain` (rendered by `benchmarks/report.py`, which gates on the
worst domain). **Byte-identity holds**: golden hashes and all five shipped datasets reproduce
exactly, and `support` records are identical inside and outside a five-domain run. `data/` gains
`train_en_domains.jsonl` and `eval_en_domains.jsonl` (7,500 — its `support` half **is**
`eval_en.jsonl`, so the published 0.859 stays the support gate). Two data records were corrected
on the way: `data_multi.json` and `data_noisy.json` claimed seed 42 while the shipped data used
seed 1 (hash-verified), so the configs now match the data, and `data_en.json` /
`data_eval_en.json` / `data_eval_multi.json` document the datasets that had no config at all.
**Runs 1 and 3 failed both gates** (measured 2026-09-27):

| run | data | LoRA / head | overall | support | worst new domain |
| --- | --- | --- | ---: | ---: | ---: |
| baseline (published) | 9,000 support | r=16 / 32 | 0.859 | **0.859** | 0.478 (zero-shot) |
| run 1 | 15,000 (support 3,000) | r=16 / 32 | 0.630 | **0.629** ✗ | 0.578 (`voice`) ✗ |
| run 3 | 21,000 (support 9,000) | r=64 / 32 | 0.540 | **0.597** ✗ | 0.379 (`voice`) ✗ |
| **run 4** (fixed loop) | 21,000 (support 9,000) | r=16 / 32 | **0.732** | **0.655** ✗ | **0.690** (`ecommerce`) ✗ |
| **run 5** (bigger choice head) | 21,000 (support 9,000) | r=16 / 128, 6 ep | **0.804** | **0.785** ✗ | **0.739** (`voice`) ✅ |
| **run 6** (rank 256, 8 ep) | 21,000 (support 9,000) | r=16 / 256, 8 ep | 0.755 | **0.795** ✗ | 0.659 (`ecommerce`) ✗ |

Runs 2 and 2b never finished: both were killed by a **silent process-group hangup** at ~60 min
(RAM 16.2/31 GB and VRAM 4.8/12 GB flat to the last sample, no traceback, no `exit=` line) —
L-009 has the fix, and run 3 was the same config launched `setsid nohup` in its own session.
The per-(domain, primitive) cross table — a script, because the reports have no such cell — shows
each run collapsing a *different* primitive: run 1 lost support `score` (0.910 → 0.492, level 3
never predicted: 127/127 `high` → `medium`), run 3 lost `noul` everywhere (2499/2500 predicted 1,
accuracy 0.244) while `score` reached 0.886–0.908. Restoring support's volume and doubling the
rank made things *worse*, so the diagnosis moved to the loop, where two real bugs were found:
(1) `val_split` took the **tail of the file**, so on domain-ordered data the validation set is the
last domain's tail — the published `val_loss` 0.326 was `support`/`score` alone and the column
never measured overall fit; (2) with `batch 8 × grad_accum 4` **every optimizer step was 32
consecutive records of one primitive and one domain** (file order) and every epoch ended on
`score` — textbook multi-domain interference, invisible to the single-domain baseline. Both were
fixed for run 4 (`split_records()` + `shuffled()` in `training/finetune_rlcd.py`, 3 tests), the
LoRA went back to r=16 (r=64 measured worse: 0.630 → 0.540) and support kept its full volume
(3,000/type through `DataConfig.per_domain`, 21,000 records).
**Run 4 cleared most of the gap (0.732 overall, `val_loss` 0.369 against 0.631/0.654)**: three new
domains now pass ≥ 0.70 (`agent_tools` 0.783, `voice` 0.782, `documents` 0.751), `noul` reached its
label-noise ceiling (0.744) and `score` 0.894. Two cells still fail — support **0.655** and
`ecommerce` **0.690** (0.010 short) — and both trace to **support `choice` 0.316** (published
0.948, chance 0.25) beside `ecommerce` 0.432. The predictions are biased to the fourth option
(`other` 423/500, `catalog` 357/500): `cos(criterion, question)` is a per-domain constant, so when
the `cos(criterion, state)` term is weak it wins outright. Zeroing the choice head (same
checkpoint, head written as zeros) drops `choice` 0.557 → 0.371 globally but leaves support
**unchanged** — the shared low-rank head learned to help the four new domains and not support.
**Run 5 raised the head's capacity and training time (`choice_rank` 32 → 128, epochs 4 → 6,
per-epoch per-primitive loss in the log) and cleared the second gate**: overall **0.804**,
`choice` 0.557 → 0.789, worst new domain **`voice` 0.739 ≥ 0.70** (`agent_tools` 0.849,
`ecommerce` 0.838, `documents` 0.810), `noul` 0.720 and `score` 0.905 at their ceilings,
`val_loss` 0.369 → 0.253. **Only support remains: 0.785 vs 0.85**, and it is entirely
`support/choice` **0.726** (needs ≈0.92; the support-only model posts 0.948). Three checks say it
is under-fit rather than over-fit: `choice` train loss flat from epoch 2, the checkpoint scores
**0.715 on the training records themselves** (eval 0.726), and the leak always goes into `other`
— whose description names the other three options and overlaps the question, so the per-domain
constant `cos(criterion, question)` keeps beating a state term the shared head never got enough
capacity or gradient share to strengthen.
**Run 6 tested the remaining capacity/time hypothesis (`choice_rank` 256, 8 epochs) and failed**:
overall 0.804 → 0.755, support only 0.785 → 0.795, and it took the second gate back (`ecommerce`
0.659). The confusion matrices say why: the fourth-option leak is not fixed, it **moves** — run 5
leaks `billing → other` on support, run 6 fixes that and leaks `billing → sales` there while
`ecommerce` grows `returns → catalog` 117. Support is 0.726 / 0.738 across the two, which is
run-to-run variance in *which* domain the shared `choice` head under-fits (L-006 / AD-009
territory), not a capacity curve — and seed 2 was already the winner of B-9's sweep.

**B-5a closed at 1 of 2 gates (2026-09-27)**: worst new domain **0.739 ✓**, support **0.785 ✗**
(needs 0.85). Run 5 is the designated artifact (`checkpoints/en_domains`,
`benchmarks/results/en_domains_r5.json`) and its tables are published in `docs/benchmarks.md`;
**nothing was republished** — the Hub adapters, the model card numbers and `benchmarks/report.md`
still describe the support-only checkpoint, and `CHANGELOG` records the outcome under
`[Unreleased]`. Closing the last 0.065 needs a structural decision (BACKLOG B-5 lists the three:
per-domain adapters, two-stage training, or a different `choice` scoring rule — each an ADR), and
**B-5b stays blocked** behind it. **B-6** remains an Idea.

## Milestone Status

| Milestone | Status | Notes |
| --- | --- | --- |
| M0 Bootstrap | ✅ Complete | uv + py3.12, ruff/pyright/pytest, CI |
| M1 Wire Contract | ✅ Complete | primitives + wire.py, contract suite |
| M2 LLM Backend + Serve | ✅ Complete | llm/fake backends, FastAPI, SDK, CLI, presets, decide, e2e |
| M3 Local Encoder | ✅ Complete | encoder backend, router+lifecycle, calibration, agent, hooks |
| M4 Training + Calibration | ✅ Complete | pipeline; full RTX 3060 run done, numbers published |
| M5 Ecosystem + Accel | ✅ Complete | ONNX, fast fallback, MCP, LangChain, Docker, telemetry no-op |
| M6 Proof + Release | ✅ Complete | benchmark report + script, docs site, model card, changelog, release process |

> This is the persistent memory for the Tachyone project across sessions. Decisions here are
> authoritative. Anything marked **DECIDED** must not be reopened without a new ADR.

---

## Recent Decisions (Last 60 days)

### AD-001: Drop-in Jev `/v1/systemone` with additive extensions (2026-09-24)

**Decision:** Tachyone speaks the exact TypeSafe Jev wire contract and accepts existing Jev
clients unchanged. Extensions (router control, hooks, `predict_batch`, integrations) are
additive and never mutate the canonical request/response shape.
**Reason:** Existing clients are the fastest path to adoption and to a verifiable contract;
a stable wire is a stronger asset than a bespoke API.
**Trade-off:** We inherit Jev's primitive semantics and error model; some ideas must be
expressed as extensions rather than first-class fields.
**Impact:** `docs/protocol.md` is the source of truth; contract tests gate every backend.
See `docs/adr/ADR-0001-jev-drop-in-protocol.md`.

### AD-002: Plugable backend with phased LLM → encoder (2026-09-24)

**Decision:** A stable `Backend` interface sits behind `wire.py`. Phase 2 ships an LLM
(structured outputs) backend; Phase 3 adds a local encoder backend; ONNX is a later
backend. The wire never depends on which backend is active.
**Reason:** Delivers an end-to-end system immediately while de-risking the harder encoder
training work.
**Trade-off:** An extra abstraction layer and two code paths to maintain.
**Impact:** `backends/base.py` is the seam; contract tests run against every backend.
See `docs/adr/ADR-0002-pluggable-backend-phasing.md`.

### AD-003: Python 3.12 + uv (2026-09-24)

**Decision:** Pin `requires-python = ">=3.12,<3.13"` and use `uv` for env + lockfile.
**Reason:** System Python is 3.14 but torch/transformers do not support it yet; uv is fast
and gives a committed `uv.lock` for reproducibility.
**Trade-off:** Contributors must use 3.12; slightly narrower dependency ceiling.
**Impact:** `.python-version`, `pyproject.toml`, CI all pin 3.12.
See `docs/adr/ADR-0003-python312-uv.md`.

### AD-004: Local-first, offline-capable core (2026-09-24)

**Decision:** The base install must run fully offline with no API key or hosted service.
LLM/remote backends are optional extras. Server HTTP mode is one mode, not the only one.
**Reason:** Privacy, latency, cost, and edge/on-prem requirements are the core differentiator.
**Trade-off:** We cannot assume a network or cloud for defaults.
**Impact:** No hosted dependency in core; extras isolate heavy/optional deps.
See `docs/adr/ADR-0004-local-first.md`.

### AD-005: Encoder backend + RLCD proper-scoring calibration (2026-09-24)

**Decision:** The local backend is a single-forward-pass encoder (ModernBERT / mmBERT) with
three task heads, trained with RLCD against strictly proper scoring rules, plus temperature
fitting to minimize ECE.
**Reason:** Non-autoregressive single-pass gives low latency; proper scoring yields
calibrated probabilities that make `confidence` meaningful.
**Trade-off:** Requires data generation and calibration infrastructure; bounded by 12GB VRAM.
**Impact:** `docs/training.md` defines the pipeline; `calibration.py` owns confidence.
See `docs/adr/ADR-0005-encoder-rlcd.md`.

### AD-006: Apache-2.0 and opt-out telemetry (2026-09-24)

**Decision:** License the project Apache-2.0. If telemetry is ever added it must be
opt-out (`DO_NOT_TRACK=1` / equivalent) and never required for function.
**Reason:** Matches the open-source ecosystem we build on (Laya) and keeps trust local-first.
**Trade-off:** No permissive-license leverage over downstream forks; telemetry is weak by design.
**Impact:** No secrets in repo; no mandatory phone-home.
See `docs/adr/ADR-0006-license-telemetry.md`.

### AD-007: Multilingual from Phase 3 via mmBERT-base (2026-09-24)

**Decision:** Ship a multilingual checkpoint (mmBERT-base, 100+ languages) alongside the
English checkpoint, selected automatically by a script/language router.
**Reason:** Multilingual capability is a primary differentiator and is not retrofittable cheaply.
**Trade-off:** Two checkpoints to load/manage; router adds a small amount of complexity.
**Impact:** `router.py`; benchmark suite must include multilingual probes.

### AD-008: Keep `checkpoints/en` as the published English adapter (2026-09-26)

> **Superseded by AD-009** the same day: a seed sweep found a checkpoint that dominates both
> contenders, so the choice below was overtaken rather than reversed.

**Decision:** `munod/tachyone-en` carries the local `checkpoints/en` — accuracy 0.763 / ECE 0.061 /
`choice` 0.834, the source of every number in `benchmarks/report.md` — instead of the older
Hub checkpoint `bac4de41` (accuracy 0.781 / ECE 0.077 / `choice` 0.708 / `score` 0.892), which
remains downloadable at `revision=bac4de4`.
**Reason:** The whole published surface (report, README, model card, docs) describes the local
checkpoint, and `choice` — the primitive behind the `triage`/`router`/`guard` presets — gains
12.6 points with it.
**Trade-off:** The Hub's English adapter loses 1.8 points of overall accuracy and 18 points of
`score` accuracy; the old checkpoint is also far better calibrated on `noul` (ECE 0.012 vs 0.101).
**Impact:** both measured on `data/eval_en.jsonl` (1500 records, RTX 3060); the old run is saved
as `benchmarks/results/en_legacy_bac4de4.json` (gitignored) and the open choice is tracked in
`BACKLOG.md` B-9.

### AD-009: Republish the English adapter from the seed sweep (2026-09-26)

**Decision:** `munod/tachyone-en` and the repository's `checkpoints/en` now carry the `seed: 2` run of
the **same** `training/configs/finetune_en.json` recipe — overall **0.859** / ECE **0.023** /
`choice` **0.948** / `score` **0.910** — superseding AD-008.
**Reason:** B-9 asked for headline accuracy *and* choice quality at once. Four runs of the
identical experiment (seed 42 twice, seed 1, seed 2) landed in three different optima; the seed-2
run beats every previous checkpoint on 7 of 8 metrics, clears both B-9 targets with margin
(`overall ≥ 0.7813`, `choice ≥ 0.8340`) and posts the lowest ECE ever measured for English.
**Trade-off:** `noul` accuracy drops 0.744 → 0.718 (−2.6 points), the only regression. The loop is
also **not deterministic** — four runs of one config gave `val_loss` 0.4177 / 0.4195 / 0.3791 /
0.3264 — so this exact checkpoint cannot be recreated by rerunning the recipe; only its quality
distribution can. `seed: 2` is pinned in the config to record what produced it.
**Impact:** `benchmarks/report.md`, README, model card, the requirements tables and the Hub were
regenerated from it; `BACKLOG.md` B-9 closed; lesson in L-006.

### AD-010: Rename the project to Tachyone (2026-09-26)

**Decision:** `jeba` → **Tachyone** everywhere: package `src/tachyone/`, console scripts
`tachyone`/`tachyone-serve`/`tachyone-mcp-server`, env prefix `TACHYONE_*`, SDK classes
`TachyoneClient`/`Tachyone*Error`, wire default model id `tachyone-latest`, Hub repos
`munod/tachyone-en`/`-multi`, site `munod.github.io/tachyone/`, cache `~/.cache/tachyone/models`.
Full record in `docs/adr/ADR-0013-rename-to-tachyone.md`.
**Reason:** the original name was a placeholder that outlived its usefulness; renaming is cheapest
before there are users, and the measurements confirmed there are none (0 stars/forks, 0 Hub
downloads, PyPI never published).
**Trade-off:** no alias period — `JEBA_*`, `jeba-serve` and `JebaClient` break immediately for
anyone who had automated against them; links to the old Hub repo ids depend on platform redirects.
**Impact:** 136 files / 1144 occurrences renamed in one commit; `uv.lock` re-locked; ADR-0001..0012
and the released `CHANGELOG` sections keep the original name as historical record.

### AD-011: Head-to-head comparison protocol (2026-09-26)

**Decision:** published comparisons must (a) make every engine answer **the same rows** through
one implementation of accuracy / 10-bin ECE / Brier, (b) show **both** evaluation sets — the
peer's public distribution and ours — with each engine's in-domain status labeled, (c) publish
`answered` beside `n` and always show `ECE raw`, `ECE cal`, `Brier` and `Conf` together, (d) run
serialized, (e) import third-party scoring code from a local download and **never vendor it**,
and (f) count a failed answer as *wrong*. Jev itself stays **qualitative**: no API key was
authorized, so no hosted numbers are claimed.
**Reason:** a benchmark run on another model's training data measures **domain coverage** as much
as the engine (our first run read 0.229 vs 0.705 and would have been quoted either way), and ECE
can be *lowered* by flattening distributions (Tachyone posted 0.040 vs their 0.044 only because
the fit pushed T into the grid ceiling while mean confidence collapsed to 0.263).
**Trade-off:** two tables are harder to read than one headline number; the LLM rows use fewer rows
(a failing question costs 2–3 autoregressive attempts, 46 s at 52 options), so `n` differs per
engine; and Tachyone publishes its own worst case (0.229 out of domain).
**Impact:** `benchmarks/compare.py` + `tests/test_benchmark_compare.py`; results in
`docs/compare.md` §3 with method and limitations; the harness reproduces the peer's published
card (T 1.75, acc 0.705, ECE 0.046, 537/576 rows) which is what licenses the comparison.
`B-7` partially delivered, `B-5` evidence recorded, `B-10` opened from a failure the run exposed.

---

## Active Blockers

### B-001: Python 3.14 vs torch/transformers

**Discovered:** 2026-09-24
**Impact:** System interpreter cannot run the model layer; any accidental use of 3.14 breaks deps.
**Workaround:** Pin 3.12 via `.python-version` and `requires-python`.
**Resolution:** Revisit when torch/transformers publish 3.14 wheels; tracked as AD-003 consequence.

### B-002: RTX 3060 12GB training budget

**Discovered:** 2026-09-24
**Impact:** Full fine-tuning of large encoders is infeasible; naive LoRA/QLoRA may OOM.
**Workaround:** LoRA/QLoRA + gradient checkpointing + small effective batch + grad accumulation.
**Resolution:** Pipeline implemented (M4-T2) and executed full-scale on the RTX 3060 12GB
(English + multilingual LoRA, 6k train / 1.5k eval, calibration). Numbers are in
`benchmarks/report.md` and the adapters are published (`munod/tachyone-en`, `munod/tachyone-multi`).

---

## Lessons Learned

### L-001: Contract-first prevents backend churn

**Context:** Borrowed from Laya's convention that public API changes require updating the
contract test in the same PR.
**Problem:** Without a frozen contract, swapping LLM → encoder silently changes responses.
**Solution:** Freeze `wire.py` and `docs/protocol.md` in Phase 1; gate every backend on the
same contract tests.
**Prevents:** Divergence between backends and broken existing clients.

### L-002: A dedicated choice head fixes what the similarity baseline cannot

**Context:** v2 templates left `choice` at chance (~0.25 over four teams); LoRA on the trunk and
more data did not help.
**Problem:** The parameter-free cosine-similarity head cannot separate a four-way team mapping.
**Solution:** Added a low-rank `choice` scorer (near-identity init so training starts at the
baseline): English `choice` 0.25 → 0.78, multilingual 0.26 → 0.40; `noul`/`score` keep the
similarity head.
**Prevents:** Spending more effort on data volume for a capability that needs a different head.

---

### L-003: A shared RNG in data generation teaches shortcuts

**Context:** B-1 multilingual quality. The generator used one `random.Random(seed)` for every
record, so phrase/term draws correlated with the cyclic team label across records.
**Problem:** `choice` learned phrase→team shortcuts that did not transfer to a fresh-seed eval
(clean-example accuracy ~0.40, near chance, while distractor cases looked high). Raising the
hard-negative rate to 1/3 added label ambiguity (the target is the *first* team) and compounded
the collapse.
**Solution:** Seed an RNG per record from `(seed, kind, index, language)` (independent draws,
still byte-deterministic) and keep the distractor rate low (1/6). Measure clean vs hard-negative
accuracy separately when diagnosing `choice`.
**Prevents:** Mistaking a generator artifact for model capability or for a hard task.

---

### L-004: The frozen wire and the similarity engine bound what a "new primitive" can be

**Context:** A v0.3 proposal suggested additive primitives (`route` multi-label, `extract_span`
for entities, `threshold`/OOD handoff) plus MoE-of-LoRA adapters.
**Problem:** The canonical `/v1/systemone` contract is frozen to three primitives (ADR-0001), and
the runtime engine is a cosine-similarity baseline (`EncoderModel.answer_state`), not trained
per-task heads. `wire._check_normalized` requires each distribution to sum to 1.0 — incompatible
with independent sigmoids — and `load_encoder` mean-pools to a single vector per text, so there
are no token-level outputs to point spans at.
**Solution:** Treat new capabilities as **additive extensions** (extension endpoints or composed
atomic questions), never as canonical `questions` types. `threshold`/handoff is largely a client-
side helper over the existing `confidence`; `route` is N independent `noul`; `extract_span` would
be a new backend/endpoint. MoE adapters were deferred as premature: the measured bottleneck is
calibration, not capacity.
**Prevents:** Redesigning the wire or adding adapter machinery to chase use cases the atomic,
composable contract already covers.

### L-005: A model card's disclaimer is not evidence about its weights

**Context:** The 2026-09-26 documentation review found `munod/tachyone-en` carrying a "pre-release /
weights and metrics pending" banner beside a `temperature_calibration.json` whose
`ece_after 0.1406` matched nothing in the repository.
**Problem:** From that stale banner I inferred the published artifacts were out of date and
republished the local checkpoint **without measuring the old one first**. The banner was the only
stale part: the card's numbers (0.781 / 0.077, `choice` 0.708, `score` 0.892) were real
measurements of the uploaded weights, and those weights beat the new ones on overall accuracy
(0.781 vs 0.763), on `score` (0.892 vs 0.712) and on `noul` calibration (0.012 vs 0.101).
**Solution:** Before replacing anything published, fetch both revisions and evaluate each:
`tachyone_ADAPTERS="tachyone-en=<path>" uv run python -m training.evaluate --data data/eval_en.jsonl
--out <out>.json --backend encoder`. The measured comparison lives in `BACKLOG.md` B-9.
**Prevents:** Swapping a published artifact for a different one that is better on some metrics
and worse on others, on the strength of a document rather than a number.

### L-006: Identical experiment, four runs, three different models

**Context:** B-9 began with two checkpoints of the *same* config disagreeing by 12 points on
`choice` (0.708 vs 0.834) and 18 on `score` (0.892 vs 0.712), and no obvious cause.
**Problem:** every structural explanation was checked and ruled out in turn — calibration
temperature (the evaluator uses argmax, invariant to any T > 0), training code, config and seed,
the base-model snapshot, the package versions, and finally the dataset (four generator versions
exist from 2026-09-24). The dataset looked like the answer until label agreement between each
candidate and the eval set showed the older checkpoint could only have been trained on the
*current* data (35.9% agreement for the old versions vs 89.6% for the current one). Only run-to-run
variance was left.
**Solution:** run a **control** — the identical config re-executed — before blaming anything. The
control alone moved `choice` 0.834 → 0.900 and `val_loss` 0.4177 → 0.4195; sweeping seeds then
found `seed: 2` at `choice` 0.948 / `score` 0.910 / ECE 0.023 (AD-009).
**Prevents:** Tuning hyper-parameters to "fix" a symptom whose cause was never established, and
underestimating how much a non-deterministic training loop can vary between identical runs.

### L-007: A benchmark on someone else's training data measures domain coverage

**Context:** the first head-to-head run put Tachyone at **0.229** against a peer scorer's
**0.705** on the peer's own public test split — a 3× gap that could have been published as a
verdict, in either direction.
**Problem:** the peer trained on that exact data (12,913 questions from the same repo) while the
released Tachyone adapter was trained on synthetic support tickets with four team labels; even the
four `tickets_*` families score 0.031–0.500 because their option spaces are 52 queues and 77
intents. A second failure compounded it: Tachyone's *calibrated* ECE (0.040) looked better than
the peer's (0.044) only because the fit pushed T to the 20.0 grid ceiling and flattened mean
confidence to 0.263 — "better calibration" bought by discarding all information.
**Solution:** run the comparison **on both turfs** and label who trained on what; publish `ECE
raw`, `ECE cal`, `Brier`, `Conf` and `answered` together so neither the domain shift nor the
flattening can hide; and validate the instrument first — the harness independently reproduced the
peer's card (T 1.75 vs 1.75, acc 0.705 vs 0.707, ECE 0.046 vs 0.044, 8 of 9 per-task values).
**Prevents:** quoting a single-distribution benchmark as a quality verdict, and quoting a
calibration metric without the confidence it was bought with.

### L-008: A label drawn from the index is a label the text cannot support

**Context:** B-5a, while authoring the new domains: English `noul` accuracy sits at 0.718 while
`choice` is 0.948 (AD-009), and the multilingual per-language spread is wide (`nl` 0.663 vs
`es` 0.956).
**Problem:** `_noul_record` sets `positive = index % 2 == 0` and only then forces `False` when the
tone is neutral, so `target = 1` iff *even index ∧ request tone*. Half of the request-toned states
— which read "I need a refund today." — are labeled `false`. Measured on the shipped eval sets:
English has 121 of 241 request-toned `noul` records labeled 0 (50% contradictory), and by language
**0%** for `fr`/`it`/`pt` against 40% `de`, 51% `es`, 55% `nl`, because the per-language RNG
happens to correlate tone with parity in some languages. The Bayes-optimal accuracy given the tone
is 0.746 on `eval_en.jsonl`; the model scores 0.718. The per-language spread is the same artifact,
not model quality.
**Solution:** leave the rule alone for B-5 (its hard guarantee is byte-identity, and changing it
would move every published number) and track the fix + the re-measurement it forces in
`BACKLOG.md` **B-11**. When writing a generator, derive the target from the text just emitted,
never from the loop index.
**Prevents:** reading a data artifact as a capability gap, and tuning a model against a ceiling
the labels impose.

### L-009: A background training run must own its session, or it dies with the harness

**Context:** B-5a run 2 and run 2b, both launched as background shells from the agent harness.
**Problem:** both died **silently** at ~60–65 min: no traceback, no `exit=` line (the wrapper
shell died with the child), no checkpoint, and no OOM — the sampler showed RAM 16.2 GB of 31 and
VRAM 4.8 GB flat until the last sample, and `journalctl` shows no reboot or suspend. The whole
process group took a hangup/kill from outside the training (a harness reap or restart looks
exactly like this), while run 1, which happened to finish first, was unaffected.
**Solution:** launch training with `setsid nohup … < /dev/null &` so it starts a **new session**
(SID = its own PID) and cannot receive the parent's hangup; the detached script appends `exit=$?`
to the log and touches a marker, and a *separate*, expendable watcher shell only waits for that
marker. Sample RAM/VRAM every 15 s from inside the same detached script, so even a kill leaves
evidence.
**Prevents:** an hour of GPU time lost to a death with no traceback to diagnose.

---

## Quick Tasks Completed

| #   | Description | Date | Commit | Status |
| --- | ----------- | ---- | ------ | ------ |
| M0-T1 | uv project skeleton (py3.12, extras, entry points, uv.lock) | 2026-09-24 | `build: bootstrap uv project (python 3.12)` | ✅ |
| M0-T2 | ruff + pyright + pytest config and dev deps | 2026-09-24 | `chore: configure ruff, pyright, pytest` | ✅ |
| M0-T3 | repo hygiene (.gitignore, .editorconfig, LICENSE, git init) | 2026-09-24 | `chore: initialize repository with project specifications` | ✅ |
| M0-T4 | CI workflow (ruff + pyright + pytest on py3.12) | 2026-09-24 | `ci: add lint, type, and test gates` | ✅ |
| M1-T1 | `noul` primitive | 2026-09-24 | `feat(primitives): add noul question and answer` | ✅ |
| M1-T2 | `choice` primitive (1..255 options) | 2026-09-24 | `feat(primitives): add choice question and answer` | ✅ |
| M1-T3 | `score` primitive (2..10 levels) | 2026-09-24 | `feat(primitives): add score question and answer` | ✅ |
| M1-T4 | wire envelope + errors + `answer()` | 2026-09-24 | `feat(wire): add systemone request and response` | ✅ |
| M1-T5 | backend seam + fake backend | 2026-09-24 | `feat(backends): add backend protocol and fake backend` | ✅ |
| M1-T6 | golden contract tests | 2026-09-24 | `test(contract): freeze systemone wire parity` | ✅ |
| M1-T7 | error-shape tests | 2026-09-24 | `test(contract): cover systemone error shapes` | ✅ |
| M2-T1 | env configuration | 2026-09-24 | `feat(config): add environment configuration` | ✅ |
| M2-T2 | structured-output LLM backend | 2026-09-24 | `feat(backends): add structured-output LLM backend` | ✅ |
| M2-T3 | `/v1/systemone` HTTP endpoint | 2026-09-24 | `feat(serve): add systemone HTTP endpoint` | ✅ |
| M2-T4 | predict/batch/health endpoints | 2026-09-24 | `feat(serve): add predict, batch, and health endpoints` | ✅ |
| M2-T5 | Python SDK client + backoff | 2026-09-24 | `feat(sdk): add python client with backoff` | ✅ |
| M2-T6 | CLI + presets | 2026-09-24 | `feat(cli): add predict CLI and presets` | ✅ |
| M2-T7 | `decide()` from JSON Schema | 2026-09-24 | `feat(schemas): add decide() from json schema` | ✅ |
| M2-T8 | repointed-Jev-client e2e | 2026-09-24 | `test(e2e): verify repointed jev client` | ✅ |
| M3-T1 | script/language routing | 2026-09-24 | `feat(router): add script and language routing` | ✅ |
| M3-T2 | checkpoint lifecycle | 2026-09-24 | `feat(router): add checkpoint lifecycle management` | ✅ |
| M3-T3 | confidence derivation | 2026-09-24 | `feat(calibration): derive confidence from distributions` | ✅ |
| M3-T4 | agent single-pass batching | 2026-09-24 | `feat(agent): add single-pass batching` | ✅ |
| M3-T5 | encoder backend (3 heads) | 2026-09-24 | `feat(backends): add encoder backend with three heads` | ✅ |
| M3-T6 | hooks registry | 2026-09-24 | `feat(hooks): add prediction and lifecycle hooks` | ✅ |
| M3-T7 | batch prediction endpoint | 2026-09-24 | `feat(agent): expose batch prediction` | ✅ |
| M3-T8 | offline multilingual smoke | 2026-09-24 | `test(integration): offline multilingual smoke` | ✅ |
| M4-T1 | deterministic data generation | 2026-09-24 | `feat(training): add deterministic data generation` | ✅ |
| M4-T3 | RLCD proper-scoring objective | 2026-09-24 | `feat(training): add RLCD proper-scoring objective` | ✅ |
| M4-T2 | LoRA/QLoRA fine-tuning | 2026-09-24 | `feat(training): add LoRA fine-tuning` | ✅ |
| M4-T4 | calibration temperature fitting | 2026-09-24 | `feat(training): fit calibration temperature` | ✅ |
| M4-T5 | accuracy/ECE/latency harness | 2026-09-24 | `feat(benchmarks): add accuracy, ECE, latency harness` | ✅ |
| M4-T6 | reproducible configs | 2026-09-24 | `chore(training): add reproducible configs` | ✅ |
| M5-T1 | ONNX runtime backend | 2026-09-24 | `feat(backends): add onnx runtime backend` | ✅ |
| M5-T2 | optional fast path + router override | 2026-09-24 | `perf: add optional fast path` | ✅ |
| M5-T3 | MCP stdio server | 2026-09-24 | `feat(mcp): add stdio server` | ✅ |
| M5-T4 | LangChain adapter | 2026-09-24 | `feat(langchain): add runnable adapter` | ✅ |
| M5-T5 | Docker + compose | 2026-09-24 | `build(docker): add image and compose` | ✅ |
| M5-T6 | telemetry guard | 2026-09-24 | `feat(telemetry): add opt-out telemetry guard` | ✅ |
| M6-T1 | reproducible benchmark report | 2026-09-24 | `docs(benchmarks): publish reproducible report` | ✅ |
| M6-T2 | documentation site | 2026-09-24 | `docs: add documentation site` | ✅ |
| M6-T3 | model card + changelog + release process | 2026-09-24 | `docs(release): add model card, changelog, and release process` | ✅ |
| B-1 | multilingual quality: localized data + per-language temperature + reporting + retrain | 2026-09-24 | `feat(training): localize and deepen multilingual data` | ✅ |
| B-2 | fast path: CUDA-graph encode + micro-benchmark (NFR-P01/P07) | 2026-09-24 | `perf(fast): wire acceleration seam and CUDA-graph encode` | ✅ |
| B-3 | confidence thresholding / System-2 handoff (entropy/margin, `handoff.py`, CLI `--threshold`) | 2026-09-25 | `feat(handoff): add confidence threshold and handoff signal` | ✅ |
| B-4 | input-noise robustness: seeded `noise_rate` + clean/noisy eval split (retrained + measured on GPU) | 2026-09-25 | `feat(training): add seeded input-noise augmentation` | ✅ |
| B-10 | LLM prompt spells out the `answers` wrapper + actionable error (probe `JSON ok` 0.083 → 0.889) | 2026-09-26 | `fix(llm): spell out the answers wrapper in the system prompt` | ✅ |
| B-8 | calibration assets that cannot be loaded now warn by name instead of degrading silently | 2026-09-27 | `fix(encoder): warn when a calibration asset cannot be loaded` | ✅ |
| B-7 | public probes: `benchmarks/probes.py` loaders for typed-decisions/MASSIVE/XNLI + `benchmarks/probes.md` | 2026-09-27 | `feat(benchmarks): add the public probe loaders and report` | ✅ |
| R-T1..T3 | multilingual LoRA rank 16 → 64 (accuracy 0.702 → 0.853, `es` ECE 0.170 → 0.038) | 2026-09-25 | `chore(training): raise multilingual LoRA rank to 64` | ✅ |

---

## Deferred Ideas

Canonical backlog: [`BACKLOG.md`](BACKLOG.md). **B-2** fast-path kernels are **done** (measured,
NFR-P01 met). **B-1** multilingual `choice`/`score` quality is **done**: localized data,
per-language temperature, per-language reporting, the GPU retrain/publish step and the r=64 rank
bump are all shipped and published. **B-3** and **B-4** are done and measured. **B-7** (public
probes) is **Done** (2026-09-27): the head-to-head harness answers a public nine-family probe
(`docs/compare.md` §3), and `benchmarks/probes.md` now publishes typed-decisions / MASSIVE / XNLI
with licences, citations and reproduction commands (OD-8 resolved — all three). **B-5** is
**Ready** with measured evidence, and **B-10** (LLM prompt wrapper) is **Done** (2026-09-26): the
wrapper is documented for
all three primitives, the parser stayed strict, and the four LLM rows were re-run — probe
`JSON ok` 0.083 → 0.889. **B-8** (calibration-asset hardening) is **Done** (2026-09-27). Carried-over
ideas (provider registry, extra checkpoints, streaming, web
console, process items) are listed in the backlog too. Promote an item into `docs/tasks.md` when
it is scheduled.

**New backlog entries (2026-09-25):** **B-3** confidence thresholding / System-2 handoff (Ready),
**B-4** input-noise robustness (Ready), **B-5** multi-domain coverage (Idea → Ready 2026-09-26),
**B-6** contrastive pre-fine-tuning (Idea). **B-10** was added 2026-09-26. `BACKLOG.md` also
records an **"Evaluated and not pursued (for now)"** section for `route`/`extract_span`/MoE
adapters with the rationale; revisit only with a new ADR and measured evidence (see L-004).

---

## Open Decisions (need resolution before the blocking phase)

- [x] **OD-1:** RESOLVED (2026-09-24, M2) — Provider-agnostic OpenAI-compatible surface with
      an injectable transport; no new core dependency. See `docs/adr/ADR-0008`.
- [x] **OD-2:** RESOLVED (2026-09-24) — No telemetry; `tachyone.telemetry` is a no-op guard and
      `TACHYONE_TELEMETRY`/`DO_NOT_TRACK` are reserved for a future opt-out. See ADR-0011.
- [x] **OD-3:** RESOLVED (2026-09-24) — Weights fetched from the Hugging Face Hub on demand and
      cached locally (`TACHYONE_MODELS_DIR` or `~/.cache/tachyone/models`), with an explicit prefetch step
      (`hf download <repo> --local-dir …`, or one warm-up prediction) and cache-only offline mode
      (`TACHYONE_OFFLINE=1`). Note: the `tachyone download` subcommand named in the original decision was
      **never implemented** — see `docs/adr/README.md` → Implementation notes. M3 uses public base
      encoders + untrained heads. See ADR-0010.
- [x] **OD-4:** RESOLVED (2026-09-24) — ONNX backend first, TileLang/CUDA-graph fast path
      afterwards behind the `fast` extra with graceful fallback. See ADR-0012.
- [x] **OD-5:** RESOLVED (2026-09-24, M2) — ``/predict`` and ``/predict/batch`` mirror the
      canonical response shape and are additive; ``/v1/systemone`` is untouched. See `docs/adr/ADR-0009`.
- [x] **OD-6:** RESOLVED (2026-09-27) — **`B-5` scope: English first (`B-5a`), multilingual
      afterwards.** Confirmed by the go-ahead for B-5a: four new domain lexicons localized once,
      `B-5b` (the other languages) follows only once the refactor and the gates are proven in `en`.
      The five domains tabled in `BACKLOG.md` B-5 are the ones implemented (`support`,
      `ecommerce`, `agent_tools`, `documents`, `voice`).
- [x] **OD-7:** RESOLVED (2026-09-27) — **`B-5` data volume: `per_type` 1000/domain.** 5 domains ×
      3 primitives × 1000 = **15,000 records** (1.67× the 9,000 the published English adapter was
      trained on), which is the ~2–3 h option rather than the 8–12 h full 5×. Config:
      `training/configs/data_en_domains.json` (seed 1) for training and
      `data_eval_en_domains.json` (seed 2, 7,500 rows) for evaluation; the fine-tune config keeps
      the pinned `seed: 2` (AD-009).
- [x] **OD-8:** RESOLVED (2026-09-27) — **`B-7` probe scope: all three probes.** MASSIVE, XNLI and
      typed-decisions were all implemented and published in `benchmarks/probes.md` (one session,
      no retrain); MASSIVE also covers the six trained languages **plus English**. XNLI's licence
      was verified before publishing: **CC BY-NC 4.0** (upstream `LICENSE`), and only derived
      metrics are published. `B-10`'s strict-vs-tolerant question was already
      **resolved** (stay strict) — recorded in `BACKLOG.md` B-10.

---

## Todos

- [x] Create `.specs/features/*` specs for each phase (done in this baseline).
- [ ] Convert `docs/tasks.md` into `.specs/features/<phase>/tasks.md` at execution time.
- [ ] Add `mermaid-studio` skill (recommended) for rendered architecture diagrams.
- [ ] Add `.specs/features/*/design.md` and `tasks.md` for the remaining phases at execution time
      (only `wire-contract` has a design.md so far; the rest are covered by `docs/architecture.md`).

## Documentation Baseline (originated 2026-09-24; kept current)

The full spec/design/planning documentation set was produced with **zero production code** and has
grown with the project:

- Root: `README.md`, `CONTRIBUTING.md`, `AGENTS.md`, `CHANGELOG.md`, `LICENSE` (Apache-2.0).
- `docs/`: `index.md`, `overview.md`, `roadmap.md`, `protocol.md`, `architecture.md`, `tasks.md`,
  `testing.md`, `training.md`, `benchmarks.md`, `cookbook-handoff.md`, `release.md`,
  `huggingface.md`, `model-card.md`.
- `docs/requirements/`: `functional.md` (60 reqs), `non-functional.md` (43 reqs), `traceability.md`.
- `docs/adr/`: `README.md` + ADR-0001..ADR-0012.
- `.specs/project/`: `PROJECT.md`, `ROADMAP.md`, `STATE.md`, `BACKLOG.md`.
- `.specs/features/`: `wire-contract/` (spec + design), `llm-backend/`, `encoder-backend/`,
  `training-calibration/`, `ecosystem/`, `fast-path/`, `multilingual-quality/`,
  `lora-rank-experiment/`, `input-noise/`, `system2-handoff/`.

**Verified:** 104/104 requirements traced (60 functional + 44 non-functional; 0 actionable
unmapped); 13/13 ADRs indexed and present;
all relative doc links resolve; 0 `.py` files created.

---

## Preferences

**Model Guidance Shown:** never

**Hugging Face token:** it is exported in `~/.zshrc`, and a non-interactive zsh reads only
`~/.zshenv` — so shells started by a tool do not see it. Load it without printing it:

```bash
eval "$(grep -hE '^(export )?(HF_TOKEN|HF_HOME|HUGGING_FACE_HUB_TOKEN)=' ~/.zshrc)"
```

Verified authenticated as `munod` (2026-09-27); used it to refresh the model cards for `v0.4.0`.
Never echo it, never commit it (rule 4).
