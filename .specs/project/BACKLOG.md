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
4. **Per-language `noul` inverted — CLOSED 2026-10-07** (interleaved continuation from the
   inverted trunk; all five pre-registered gates green — full trail in **B-14**). The
   per-language `noul` gate this bullet demanded now exists in practice: B-15's holdout
   tripwire (`noul` ≥ 0.60 in every language) passed everywhere (worst `it` 0.7807). Original
   report: the B-14 multi arm scored `noul`/`support`/`it` at **0.0723** on `eval_multi`
   (other languages 1.000) and anti-fit its own training rows (10%). Per-language `noul` was
   never gated in the acceptance below (it covers `choice`/`score`); it is gated now.

**Acceptance.**
- Multilingual `choice` accuracy ≥ 0.60 on the held-out synthetic set.
- Per-language ECE ≤ 0.05 for `choice`/`score` (NFR-C06), with per-language numbers published.
- No regression in English (overall ≥ 0.72).

**Evaluated 2026-10-08 against the B-15 holdout — the yardstick this was written for** (full
tables in B-15): `choice` **0.9248** ≥ 0.60 ✓ · English overall **0.9821** ≥ 0.72 ✓ ·
per-language ECE **NOT met** on multi — choice `es` 0.0753 / `it` 0.0968 / `pt` 0.1014, score
`de` 0.0540 / `es` 0.0951 / `it` 0.1495 / `nl` 0.0577 (en passes: choice 0.0155, score
0.0067). Recorded with the numbers, never re-fixed → **B-1 stays open on criterion 3.**

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

## B-5 — Multi-domain coverage (5 domains) · **Done** — B-5a both gates pass, isolate closes 91% of the support gap, **published as `munod/tachyone-en` `c00c174d`**, probes + head-to-head re-run, **`v0.5.0` released** (2026-10-01)

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

**Status (B-5a in progress).** Scope and budget are settled (OD-6: English first; OD-7:
`per_type` 1000/domain → 15,000 records, the ~2–3 h option). The refactor, the five domain files,
`per_domain` reporting and the byte-identity guarantee are **done** — see `STATE.md` and commit
`d9a974d`. The measurements so far:

| Run | Data | LoRA / head | overall | support | worst new domain | `val_loss` | Gates |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline (published) | 9,000 support | r=16 / 32 | 0.859 (support-only eval) | **0.859** | 0.478 (zero-shot) | 0.326* | support ✓ |
| B-5a run 1 | 15,000 (support 3,000) | r=16 / 32 | 0.630 | **0.629** ✗ | 0.578 (`voice`) ✗ | 0.631* | both ✗ |
| B-5a run 3 | 21,000 (support 9,000) | r=64 / 32 | 0.540 | **0.597** ✗ | 0.379 (`voice`) ✗ | 0.654* | both ✗ |
| **B-5a run 4** (shuffled loop) | 21,000 (support 9,000) | r=16 / 32 | **0.732** | **0.655** ✗ | **0.690** (`ecommerce`, 0.010 short) ✗ | 0.369 | both ✗ |
| B-5a run 5 | 21,000 (support 9,000) | r=16 / **128**, 6 epochs | **0.804** | **0.785** ✗ | **0.739** (`voice`) ✅ | 0.253 | support ✗, new domains ✓ |
| B-5a run 6 | 21,000 (support 9,000) | r=16 / **256**, 8 epochs | 0.755 | **0.795** ✗ | 0.659 (`ecommerce`) ✗ | 0.281 | both ✗ |

\* the `val_loss` column is **not** what it looked like — see the two loop bugs below. Runs 2 and
2b never finished: both were killed as a **silent process-group hangup** at ~60 min with RAM
16.2/31 GB and VRAM 4.8/12 GB flat, no traceback and no `exit=` line (`STATE.md` L-009).

**Run 4 fixed the loop and almost cleared the gates**: three of the four new domains pass
(`agent_tools` 0.783, `voice` 0.782, `documents` 0.751), `noul` reached its label-noise ceiling
(0.744) and `score` reached 0.894, with `val_loss` 0.369 against 0.631/0.654 before. The whole
remaining gap is one cell of the cross table: **support `choice` 0.316** (published 0.948, chance
0.25).

Two measurements localise it:

- **The predictions are biased to the fourth option**: `other` 423/500 on `support`, `catalog`
  357/500 on `ecommerce`, `other` 264/500 on `voice`. The scoring rule adds `cos(criterion,
  question)`, a per-domain constant, so when the `cos(criterion, state)` term is too weak the
  constant wins and every record falls to the same option.
- **Zeroing the choice head** (same checkpoint, head written as zeros) drops `choice` 0.557 →
  0.371 globally but leaves support *unchanged* (0.655 → 0.655): the shared low-rank head has
  learned to help the four new domains and **not** support. Capacity/time of that head is what
  run 5 changes (`choice_rank` 32 → 128, epochs 4 → 6).

**Run 5 moved every cell** (overall 0.732 → **0.804**, `choice` 0.557 → **0.789**, support 0.655 →
**0.785**) and the **second gate now passes**: `voice` 0.739 ≥ 0.70 with every new domain above it
(`agent_tools` 0.849, `ecommerce` 0.838, `documents` 0.810). `noul` (0.720) and `score` (0.905) sit
at their ceilings, so the entire remaining gap is one cell again — **support `choice` 0.726**, which
would need ≈ 0.92 for support to reach 0.85 (the support-only model posts 0.948).

Three measurements say it is **under-fit, not over-fit**:

- the per-epoch curve (`epoch N train_loss …`) shows `choice` bottoming at epoch 2 (0.049 → 0.046
  flat) while `score` keeps falling (0.717 → 0.272), so `choice` is the term that stops improving;
- the checkpoint scores **0.715 on the training records themselves** (eval 0.726 — the same), with
  `billing → other` 101/250 and `sales → other` 58/250 in both;
- the leak is always **into `other`**, whose description names the other three options *and*
  overlaps the question text, so `cos(criterion, question)` — a per-domain constant — keeps
  winning over a `cos(criterion, state)` that the shared head never had enough capacity or
  gradient share to strengthen.

`val_loss` fell 0.369 → **0.253** across the same change.

**Run 6 (choice_rank 256, 8 epochs) made it worse** — overall 0.804 → 0.755 and it took the
second gate back with it. Support only moved 0.785 → 0.795 (+0.010 for the whole change), while
`ecommerce` fell 0.838 → 0.659 and `agent_tools` 0.849 → 0.705. The per-epoch curve explains the
shape: at rank 256 the head sits at loss ≈0.75 for the first four epochs and only collapses to
0.05 at epoch 5, so eight epochs spent half the run learning to use a head the smaller one already
had.

The confusion matrices show the leak **moving between domains rather than being fixed**: run 5
leaks `billing → other` (82/125) and `sales → other` (48/125) on support, while run 6's support
leaks `billing → sales` (88/125) instead — support lands at 0.726 and 0.738 — and it is
`ecommerce` that develops a fourth-option leak (`returns → catalog` 117, `shipping → catalog` 91).
That is run-to-run variance in *which* domain the shared head under-fits (L-006/AD-009 territory),
not a monotone capacity effect: **run 5 remains the best measured run**.

Runs 1 and 3 each collapsed a *different* primitive (run 1 lost `support/score`: 0.910 → 0.492,
level 3 never predicted; run 3 lost `noul` everywhere: 2499/2500 predicted 1, accuracy 0.244,
because the head can only emit `sigmoid(cos/T)` ∈ [0.31, 0.69] at the fitted T = 0.83), which
pointed at the training loop rather than the data. Two real bugs were found there:

1. **`val_split` took the tail of the file.** Generated data is ordered domain → primitive, so a
   multi-domain run's "validation" set is the tail of the *last* domain (`voice`) — and the
   published `val_loss` 0.326 was `support`/`score` alone. That column never measured overall fit.
2. **Every optimizer step was a single (domain, primitive) gradient.** `batch_size 8 ×
   grad_accum 4` = 32 consecutive records of one primitive, and the file is domain-ordered, so
   each step trained one domain and every epoch ended on `score` — the textbook recipe for
   multi-domain interference. The published baseline never saw it because it has one domain.

Both are fixed for run 4 (`split_records()` + `shuffled()` in `training/finetune_rlcd.py`, 3 tests)
and the LoRA goes back to r=16: r=64 measured worse on every headline (0.630 → 0.540).

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

**Acceptance (B-5a evaluated 2026-09-27).**
- [x] Existing configs regenerate byte-identically (hash test in `tests/`, 3 golden hashes + a
  config-vs-dataset check over the five shipped datasets).
- [x] Five domains with deterministic, localized, committed data; `B-5a` English first.
- [x] Per-domain accuracy/ECE published and gated on the worst domain — worst new domain **0.739**
  (`voice`) ✓, per-domain ECE 0.082–0.116 published with the exceptions declared above the 0.05
  target.
- [x] **No regression on `support` (English overall ≥ 0.85)** → **0.785** ✗ on the pre-B-11
  labels (best of six runs); **0.861 ✓ re-measured on the B-11 labels** — see below.
- [x] Adapter republished; `benchmarks/report.md`, model card, README and CHANGELOG updated →
  **done 2026-10-01**: `munod/tachyone-en` commit [`c00c174d`](https://huggingface.co/munod/tachyone-en/commit/c00c174dde0208b186cb503abfa0e15bdde0400d)
  (five-domain bank, all six files sha256-verified against the local build, stale
  `finetune_config.json` deleted) and `munod/tachyone-multi` commit `d3f64cc0` (model card only);
  `benchmarks/report.md` (three entries incl. the gate row), `docs/benchmarks.md`,
  `docs/model-card.md`, `README.md`, `docs/roadmap.md`, `docs/huggingface.md` §3 and
  `CHANGELOG.md` all carry the new set. Latency was re-measured on the L4 box and labelled
  box-bound everywhere it appears.
- [x] Probe before/after table published (`B-7` harness) → **done 2026-10-01 for the probes**:
  all three re-run against the new revision (artifacts `benchmarks/results/probe_*_preb5.json`
  keep the previous ones) and `benchmarks/probes.md` re-rendered — typed-decisions **0.330 → 0.269**,
  XNLI **0.333 → 0.341**, MASSIVE flat at 0.011, with `Conf` 0.399 → 0.811 / 0.354 → 0.991 /
  0.021 → 0.365 and `ECE raw` 0.268 → 0.650 on XNLI (the head is sharp in-domain and confident
  off-domain; published, not absorbed). Fast-path parity also re-run: **0 flips**.
- [x] Head-to-head `choice` cells re-run against the new revision → **done 2026-10-01**: only
  Tachyone's engine was re-scored (peers and gold labels untouched, same rows / same metric code),
  artifacts `compare_tachyone_b5.json` + `compare_home_tachyone_b5.json` rendered as
  `compare_table_{a,b}_b5.md`. Table A (their turf) **0.236 → 0.233** with `ECE raw` 0.337 → 0.561
  and `Conf` 0.268 → 0.355 (the B-5 head is confident off-domain); table B (our turf) **0.974 →
  0.958** with `choice` 0.938 → **1.000** and `noul`/`score` 1.000/0.984 → 0.938/0.938 — the same
  support trade `docs/benchmarks.md` publishes. `docs/compare.md` §3 carries the tables, the
  per-task cells and the method rows.

**Verdict (2026-09-27): 1 of 2 gates.** Run 5 is the designated artifact —
`checkpoints/en_domains`, `data/preds_en_domains_r5.jsonl`, `benchmarks/results/en_domains_r5.json`
— and its numbers are the ones published in `docs/benchmarks.md`. Runs 1, 3, 4, 5 and 6 are kept
as `_rN` so the curve can be re-read.

**Re-measured on the B-11 labels (2026-09-28): both gates pass.** B-11 (ADR-0014) corrected the
`noul` labels, moving 602 of the 2,500 `noul` rows in `eval_en_domains.jsonl` and nothing else —
`choice` and `score` labels are untouched — so the *same* run-5 weights re-evaluate to overall
**0.880** (`benchmarks/results/en_domains_r5_postb11.json`):

| Gate | pre-B-11 labels | B-11 labels | verdict |
| --- | ---: | ---: | --- |
| `support` ≥ 0.85 | 0.785 ✗ | **0.861** ✓ | flipped |
| worst new domain ≥ 0.70 | 0.739 (`voice`) ✓ | **0.807** (`voice`) ✓ | held |

The entire delta is `noul`: run 5's `noul` 0.720 → **0.946**, while `choice` 0.789 and `score`
0.905 are the *same measurement as before*, because those labels never moved. Per domain now:
`agent_tools` 0.929, `ecommerce` 0.915, `documents` 0.887, `support` 0.861, `voice` 0.807;
per-domain ECE 0.045–0.067 (was 0.082–0.116) — `agent_tools` 0.045 and `documents` 0.046 meet
the 0.05 target, the other three stay declared exceptions.

So the three structural options below are no longer what the *written* gate needs, but two facts
change how they should be read: (a) the baseline moved — the released support-only adapter now
scores **0.972** on `eval_en`, so a five-domain adapter at **0.861** on support clears the
absolute gate while sitting 0.111 below the released one; and (b) run 5 itself was **trained on
pre-B-11 labels**, and retraining it on corrected data was the cheap untried option that should
come before any of the three structural decisions.

**Re-measured on the B-12 labels, and that cheap option was executed (2026-09-29).** The same
run-5 weights now post **0.879 overall, support 0.886 ✓, worst new domain 0.815 ✓** (per-domain ECE
0.036–0.080) — both gates pass on a *third* label generation, with `score` 0.905 → 0.946 and
`choice` 0.745. The retrain-on-corrected-labels option **was run and lost**: `noul` and `score`
both reached 1.000 while `choice` collapsed to **0.303** (train loss 0.027, eval at chance) and
`support` fell to **0.763 ✗** — the corrected labels make the two tone tasks trivially learnable
and the shared trunk starves the `choice` head (the same mechanism as the English control at
0.841). The first attempt also ran the **wrong recipe**: the config file still carried run 6's
settings (`choice_rank` 256, 8 epochs) instead of the designated run 5 (`128`, 6) and collapsed
outright (0.338) — restored in `aedb064`, lesson L-011. So the artifact keeps its B-11 weights, the
baseline is the released English adapter at **0.972**, and the structural decision below is the
only option left: *both* cheap ones (relabel, retrain) are measured and closed.

**What would actually close the last 0.065 (each needs a decision before code):**
**→ Decided in [ADR-0016](../../docs/adr/ADR-0016-per-domain-choice-heads.md) (Accepted,
2026-09-30): per-domain `choice` heads + a deterministic gate, with the scoring rule kept as
the sequenced fallback.** The three options, for the record:

1. **Per-domain adapters** (MoE of LoRA / adapter routing): gives support its own capacity so the
   shared `choice` head stops trading one domain for another. This reopens the item listed under
   *Evaluated and not pursued* — revisit only with a new ADR (the L-004 rule).
2. **Two-stage training**: five domains first, then a short support-only polish phase, with the new
   domains re-measured for forgetting. Also an ADR — it changes the recipe behind every published
   number.
3. **A different `choice` scoring rule**: the leak rides on the per-domain constant
   `cos(criterion, question)`; removing or rebalancing it invalidates published numbers and forces
   a full re-benchmark (the AD-009 pattern).

**The isolate ran and closed it (2026-10-01, ADR-0016 §5 first experiment).** Only the head
moved: the run-5 trunk stayed frozen, the 7,000 `choice` records were encoded **once**, and two
arms were fitted from that cache (`training/fit_choice_bank.py`, 8 epochs, lr **1e-4** — the
first config's 1e-3 destroyed the domain heads, 0.37 → 1.48 loss, and is recorded in the B-5
task). Four arms plus the released reference, one harness (`training.predict`, all 7,500 rows,
`--train-data` for the seen-text split, calibration refit per arm):

| arm | overall | `choice` | support | worst new domain | per-domain ECE |
| --- | ---: | ---: | ---: | ---: | --- |
| run-5 as shipped (baseline) | 0.879 | 0.745 | 0.886 | 0.815 (`voice`) | 0.036–0.080 |
| **shared head refit** (control) | 0.963 | 0.998 | **0.964 ✓** | **0.962 ✓** (`ecommerce`) | 0.022–0.027 |
| **bank + gate** | **0.964** | **0.9996** | **0.964 ✓** | **0.963 ✓** (`documents`) | 0.021–0.027 |
| bank + oracle (`--head-hint domain`) | 0.964 | 0.9996 | 0.964 | 0.963 (`documents`) | 0.021–0.027 |
| released `tachyone-en` (reference) | 0.511 | 0.452 | **0.972** | 0.328 (`agent_tools`) | 0.110–0.368 |

**Verdict — every ADR-0016 / B-5 gate passes:** `support` **0.964 ≥ 0.85**, worst new domain
**0.963 ≥ 0.70**, and the gap to the released adapter **shrank 0.086 → 0.008** (support cell
0.964 against its 0.972, i.e. **91% of the gap closed by moving the head alone**, with the trunk
untouched). The gate's own number — the ADR's required publication — is **strict 1.000, 0 fell
to shared, 0 wrong domain** over 2,500 `choice` rows, and the oracle arm is numerically
identical, so routing is not the bottleneck. Per-domain ECE **0.021–0.027 clears the 0.05
target in all five domains for the first time** (was 0.036–0.080 with three declared
exceptions). `noul`/`score` are unchanged at 0.946 in every arm, exactly as the ADR promised:
only `choice` cells moved.

**What the isolate actually proved — read before quoting the table.** The shared-head control
(0.9633) is **3 of 2,500 `choice` rows** behind the full bank (0.9637): per-domain *capacity*
is not what closed the 0.086. What closed it is refitting **one** head on the frozen trunk with
the B-12-corrected labels — a change both arms share, and therefore one the control-vs-bank
comparison cannot attribute to ADR-0016's mechanism (the arms were built so that difference is
*only* the structure). On the 473 `choice` rows whose text never occurs in training the ranking
holds — baseline 0.892 → control 0.992 → **bank 0.998** — so the gain is not the eval/train row
collision (81% of `choice` rows are byte-identical to a training row) buying a memorized
answer. **Open → resolved (2026-10-01):** which artifact to publish — the **bank** was chosen and
published (ADR-0016's decided structure; it is never worse than the control, ships the gate that
measures 1.000, and its format is what B-5b's language keys reuse), with the control kept
comparable at `checkpoints/en_domains_bank_ctrl`. The structural hypothesis itself still wants a
fresh reading: the measured axis was labels, not capacity (L-012).

**Artifacts (all gitignored):** `benchmarks/results/en_domains_{r5_baseline,bank_ctrl,bank_gate,bank_oracle}.json`,
`benchmarks/results/en_released_sameharness.json`, their `calibration_*.json`, and
`data/preds_en_domains_{r5_baseline,bank_ctrl,bank_gate,bank_oracle}.jsonl`.
Reproduce: `bash /tmp/opencode/b7_run.sh` (the four-arm loop) — see the B-7 task in
`.specs/features/choice-head-bank/tasks.md`.

**B-5b (the multilingual half) is now unblocked** — the bank was chosen and published
(2026-10-01), and its `{shared, domains}` format is exactly what B-5b's language keys reuse.

**B-5b cycle — scheduled, baseline fixed (2026-10-01).** Spec/design/tasks at
`.specs/features/multilingual-five-domains/` (four locked decisions in its `context.md`:
bank keyed by **domain**, **baseline-first gates**, **30k volume** keeping support's 18k,
**full B-5a publication pattern**). Landed so far: the four lexicons localized to the six
training languages (`19de6b9`), the 30k/7.5k datasets with support halves byte-identical to
`train_multi`/`eval_multi` (`367aa88`), and the `per_domain_language` report cell (`ea24e73`).
**Gates were fixed from the released-multilingual baseline before any training** (B5B-4,
`benchmarks/results/multi_domains_baseline.json` + `data/preds_multi_domains_baseline.jsonl`;
released `checkpoints/multi` on `data/eval_multi_domains.jsonl` through `training.predict` —
the explicit-adapter harness the arms will also use):

| cell | baseline (released multi, explicit adapter) |
| --- | ---: |
| overall | **0.5609** (ECE 0.172) |
| `support` | **0.8413** (ECE 0.027) ← the no-regression anchor |
| `ecommerce` | 0.5647 |
| `voice` | 0.5000 |
| `documents` | 0.4607 |
| `agent_tools` | **0.4380** (worst new domain, zero-shot) |
| worst `domain/lang` cell | `agent_tools/nl` **0.3012** |
| seen / unseen text | 0.5950 / 0.5097 |

**Fixed gates:** `support` ≥ **0.8413** (same harness, same rows) · worst new domain ≥
**0.70** (absolute, set in advance; baseline is 0.438 zero-shot) · per-domain ECE ≤ **0.05**
with exceptions declared · gate strict published beside `per_domain` (oracle arm separates
routing from head quality).

**Anchor correction — the published 0.743 is a routed number and does not reproduce.**
`training.evaluate --backend encoder` routes every record by detected language: on the
multilingual rows **13.4% (`eval_multi`) / 14.8% (`eval_multi_domains`) fall through to
`tachyone-en`** (Latin script detected as `en` or `None` — empty boundary states are 5.4%,
short/loanword-heavy states the rest). Measured today on the same rows and adapter: routed
**0.736** vs explicit **0.8413**. The published **0.743 ≠ today's 0.736** because
`munod/tachyone-en` was republished as the B-5 bank adapter on 2026-10-01 — the routed
harness silently answers with *whatever the English checkpoint is today*. Controls:
`/tmp/opencode/control_eval_multi.json` (predict) and `control_eval_backend.json` (routed).
**Rule (L-013): never gate one checkpoint's quality on a routed harness.**

**B5B-6 — three arms, one harness, gates PASS (2026-10-02).** The joint run finished
(`exit=0`, 8/8 epochs, `choice` loss oscillating but descending 0.51 → 0.035, no L-011
collapse signature; `finetune_config.json` matches the launch config). Arms via
`/tmp/opencode/b5b_eval.sh` (explicit adapter, calibration refit per arm, `--train-data`
split; the shared-only arm is the trained bank's `shared` payload written in the legacy
shape — it differs from the bank arm by exactly one factor: which head answers `choice`):

| arm | overall | ECE | `choice` | `support` | worst new domain | gate strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (released multi) | 0.5609 | 0.172 | 0.258 | 0.8413 | 0.438 | — (legacy asset) |
| bank + gate | 0.8732 | 0.023 | 0.6196 | **0.9453 ✓** | **0.847 ✓** | **1.000** (0 shared / 0 wrong, n=2500) |
| bank + oracle | 0.8732 (identical to gate — routing is not the bottleneck) | | | | | |
| **shared only** | **0.9295** | **0.012** | **0.7884** | **0.980** | **0.890** | — |

**Gates (bank+gate vs the numbers fixed before training): `support` 0.9453 ≥ 0.8413 PASS ·
worst new domain 0.8467 ≥ 0.70 PASS · per-domain ECE ≤ 0.05 → two declared exceptions
(`support` 0.071, `ecommerce` 0.052; `agent_tools` 0.044, `documents` 0.038, `voice` 0.029) ·
gate strict 1.0000 published.** `noul` and `score` are **1.000 in every arm and every domain**
(the B-12 corrected-label ceiling: both are the same tone detector — L-010; every difference
between arms rides on `choice`).

**The measured surprise — the shared head beats the per-domain bank in EVERY domain**
(`choice`: `ecommerce` +0.290, `voice` +0.224, `support` +0.104, `agent_tools` +0.130,
`documents` +0.096), taking overall **0.9295 vs 0.8732**. Per-language: shared 0.893–0.948 vs
bank 0.826–0.918; worst cell `ecommerce/de` 0.799 (shared) vs `agent_tools/fr` 0.763 (bank);
held-out-text 0.936 (shared) vs 0.897 (bank). This is the **first joint bank run ever
measured** (B-5a published the frozen-trunk *isolate*, where bank and shared-refit tied at
3 rows — L-012); the trainer code was re-read and each head is trained standalone against the
same target with the same lr — no train/inference mismatch, so the loss is real. Candidate
mechanism (unresolved, L-006: do not call it structure yet): each domain head gets ~1/5 of
the optimizer exposures while the trunk is still moving early in training, while the shared
head's criteria-conditioned residual pools evidence across all five option spaces.

**Open → the isolate decides the publish arm:** ADR-0016 §5's designated experiment —
`training/fit_choice_bank.py` on the **frozen trained trunk** (`checkpoints/multi_b5b`),
which equalizes optimization (both arms refit on one cached encode pass) and therefore
separates *structure* from *joint-training exposure*. Publish the best measured arm:
bank wins → multi ships ADR-0016's keyed format; shared wins → multi ships the legacy head
and B-5b publishes the structural finding.

**B5B-6b + B5B-7 — the isolate ran, gates PASS with zero ECE exceptions, publish arm decided
(2026-10-02).** `training/configs/fit_bank_multi_domains.json` (trunk `checkpoints/multi_b5b`,
10,000 `choice` records encoded once, lr 1e-4, 8 epochs, `choice_rank` 128) produced both fitted
arms; evaluated in the same harness with calibration refit per arm — the full candidate table:

| arm | overall | ECE | `choice` | `support` | worst new domain | gate strict | unseen text |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (released multi) | 0.5609 | 0.172 | 0.2580 | 0.8413 | 0.438 | — | 0.510 |
| joint bank | 0.8732 | 0.023 | 0.6196 | 0.9453 | 0.801 | 1.000 | 0.897 |
| joint shared | 0.9295 | 0.012 | 0.7884 | 0.9800 | 0.890 | — | 0.936 |
| **fitted bank (publish)** | **0.9975** | **0.001** | **0.9924** | **1.0000** | **0.993** | **1.000** | 0.996 |
| fitted shared | 0.9976 | 0.001 | 0.9928 | 1.0000 | 0.992 | — | 0.996 |

**Verdict — every B-5b gate passes on the publish arm:** `support` **1.0000 ≥ 0.8413** ✓,
worst new domain **0.9927 ≥ 0.70** ✓, per-domain ECE **0.0005–0.0038 (all five under 0.05 —
zero exceptions, first time either checkpoint family posts that)** ✓, gate **strict 1.000,
0 fell to shared, 0 wrong domain** over 2,500 `choice` rows ✓. Per-language 0.995–1.000,
worst cell `ecommerce/pt` 0.980, held-out-text 0.9957 (so the headline is not the known
train/eval row collision buying it — B7's discipline).

**What the isolate proves (and what it does not):** fitted bank vs fitted shared differ by
**1 row of 7,500** — under equal optimization the structure is *neutral*, replicating the
English isolate's 3-row tie (L-012, now measured twice); the joint bank's 5.6-point deficit
was optimization exposure, not capacity or routing (gate 1.000, oracle identical). The
fitted heads riding ~0.99 is the same mechanism as the published English `v0.5.0` isolate:
the generator's option-specific phrase pools make in-domain `choice` nearly deterministic
for a head fit on frozen, well-trained encodings — the published methodology, not a new
trick. Everything above rides on `noul`/`score` = 1.000 (the B-12 label ceiling).

**Publish decision: the fitted bank** (`checkpoints/multi_b5b_fit_bank` — trunk + keyed
`choice_head.json` + fit recipe, L-011). It is best-or-tied on every column, ships ADR-0016's
decided format with the gate that measures 1.000, and matches the English adapter's proven
trunk+fitter construction. B5B-8 promotes it to `checkpoints/multi` and republishes.

**B5B-8 — published and verified (2026-10-02).** `checkpoints/multi_b5b_fit_bank` promoted to
`checkpoints/multi` (the replaced build kept at `checkpoints/multi_preb5b`, the fit's own
calibration promoted beside it) and uploaded as **[`9a3ef5a5`](https://huggingface.co/munod/tachyone-multi/commit/9a3ef5a5d38972ffe119cdfe6b4b6c7bb14da9d1)**
— six files **sha256-verified one by one** against the local build, file set exact, stale
`finetune_config.json` deleted (the upload hit a fine-grained-token 403 on both the xet and LFS
paths — and a token with only PR-create scope, which the `pass create_pr=1` hint exposed — until
Write was granted). The whole published surface is one set: `benchmarks/report.md` (fourth
entry, reproduction commands fixed — the render invocation was stale), `docs/benchmarks.md`
(multilingual section replaced by the B-5b artifact, both splits labelled by harness), model
card, README, `docs/huggingface.md` (revision entry), `docs/roadmap.md`, `CHANGELOG.md`
(`[Unreleased]`), `docs/training.md` (localization completeness, the multi fitter config, the
`per_domain_language` row, the L-013 harness rule). Sweep: **MASSIVE re-run** 0.011 →
**0.039** (chance 0.017, every language above chance; `ECE raw` 0.354 → 0.518, `Conf` 0.365 →
0.557 — sharp off-domain, English's B-5 trade; the only probe off the grid ceiling at
**T=14.55**, so `probes.py` renders its temperature line from the artifacts and skips `*_pre*`
backups so the documented glob reproduces the page) and **fast-path parity** 3.66× / **0
flips**. The head-to-head was **not** re-run, by check not by habit: both quoted tables are
English-only rows (`system-one-decisions`, `eval_en`) that route to `tachyone-en`, untouched
since B-5.

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
| typed-decisions (`all/test`, 4 configs) | 2000 | **0.323** → **0.330** (B-11) | 0.20–0.50 | 0.408 → 0.498 | best config 0.412 (`security_incidents`), worst 0.250 (`agent_trace_observability`) |
| MASSIVE (7 locales) | 3584 | **0.033** → **0.013** (B-11) | 0.017 | 0.214 → 0.136 | now *below* chance; per language 0.006 (`es`) … 0.023 (`nl`) — the multilingual `choice` regression seen from outside |
| XNLI (`en`) | 5010 | **0.334** → **0.333** (B-11) | 0.333 | 0.452 → 0.269 | exactly chance, as before |

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

## B-11 · The `noul` target depends on the loop index, not on the text · **Done (2026-09-28)**

**Why.** `_noul_record` computes `positive = (index % 2 == 0)` and only then forces `False` when
the tone is neutral, so `target = 1` iff *even index ∧ request tone*. Half of the request-toned
states therefore carry a `false` label while reading like a request ("I need a refund today.").
Measured on the shipped eval sets (2026-09-27, `STATE.md` L-008):

| Set | request-toned `noul` records | of which labeled 0 |
| --- | ---: | ---: |
| `eval_en.jsonl` | 241 | 121 (**50%**) |
| `eval_multi.jsonl` → `fr`, `it`, `pt` | ~40 each | **0** (0%) |
| `eval_multi.jsonl` → `de`, `es`, `nl` | ~33–46 each | **40–55%** |

Consequences: the Bayes-optimal accuracy given the tone is **0.746** on `eval_en.jsonl` and the
published model scores **0.718** (so `noul` is label-bound, not model-bound), and the per-language
spread (`nl` 0.663 vs `es` 0.956) tracks how much the per-language RNG happens to correlate tone
with parity — a data artifact that reads like a capability gap.

**Plan.**
1. Decide the label rule: `target = 1 iff tone == "request"` — text-consistent and balanced
   (~50/50, versus the current ~25% positive).
2. **Decide the scope**: fix everything (all datasets regenerate, all published numbers move) or
   only newly generated domains (then `support` keeps the legacy noise and the domains are not
   labelled alike — the trade-off B-5a deliberately refused to hide inside a config flag).
3. If everything is fixed: regenerate the train/eval sets, retrain English **and** multilingual,
   refit calibration, and re-run `benchmarks/report.md`, `docs/compare.md` §3 and the three public
   probes (`benchmarks/probes.md`) — the numbers cannot be compared before/after, so they are
   republished as a set, like AD-009 did for the seed sweep.

**Executed (2026-09-27 → 28).** Code (`c2cfd4c`, `619cb01`), ADR-0014, datasets regenerated — only
`noul.target` moved (~24% of `noul` rows; zero bytes of `state`, questions, `choice`/`score`) —
both adapters retrained, and per L-005 the pre-B-11 checkpoints were measured against the
corrected eval sets **before** being replaced:

| checkpoint | on pre-B-11 labels | on B-11 labels (old weights) | on B-11 labels (retrained) |
| --- | ---: | ---: | ---: |
| English | 0.859 (`noul` 0.718) | 0.935 (`noul` 0.946) | **0.945** (`noul` **0.992**) |
| multilingual | 0.853 (`noul` 0.960) | 0.714 (`noul` 0.614) | **0.718** (`noul` **0.832**, 8 epochs) |

The English row is the label fix proving itself: the old weights already read the text (0.946) and
the retrain adds the rest (overall 0.945, `choice` 0.960). The multilingual retrain needed a sweep
— `noul` improved in **every** language (the language→label shortcut is gone: `de` 0.373 → 0.747,
`es` 0.452 → 0.940, `nl` 0.217 → 0.663) while `choice` lost ~0.20 and `score` ~0.20 at the
4-epoch setting:

| run (multilingual) | overall | ECE | `noul` | `choice` | `score` | `val_loss` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-B-11 weights | 0.714 | 0.076 | 0.614 | **0.674** | 0.854 | — |
| retrain, 4 epochs (seed 42) | 0.657 | 0.051 | 0.832 | 0.490 | 0.648 | 0.342 |
| **control** (identical config) | 0.663 | 0.059 | 0.832 | 0.496 | 0.660 | 0.340 |
| **8 epochs — published** | **0.718** | **0.042** | **0.832** | 0.468 | **0.854** | 0.352 |
| `choice_rank` 128, 8 epochs | 0.671 | 0.044 | 0.832 | 0.490 | 0.690 | 0.458 |

Three things the sweep settled (L-006 first: run the control before blaming anything):

1. **It is not variance.** The control lands 0.663 against run 1's 0.657 — 0.006 apart. The
   4-epoch `choice`/`score` drop is systematic.
2. **`score` was under-fit, not broken.** 8 epochs take it 0.648 → **0.854** (level with the
   pre-B-11 weights) and overall to **0.718**, above 0.714 — with the best ECE of the table
   (0.042 vs 0.076).
3. **`choice` capacity buys nothing.** `choice_rank` 32 → 128 collapsed the *training* loss
   (0.408 → **0.085**) while eval stayed at 0.490 and overall fell to 0.671 — pure overfit, the
   same pattern as B-5a run 6. The residual `choice` cost (~0.20 against the old weights) is the
   harder, balanced `noul` task sharing the trunk, not a head that is too small: it is the same
   interference B-5 lists, and it is recorded here rather than papered over.

**Published multilingual artifact: `checkpoints/multi_e8`** (promoted to `checkpoints/multi`;
the 4-epoch first retrain is kept as `checkpoints/multi_r1b11` so the sweep can be re-read).

**Published (2026-09-28).** Both adapters were republished to the Hub — `munod/tachyone-en`
commit `224c8a74`, `munod/tachyone-multi` commit `b7747756`, every uploaded file verified against
the local build by sha256 — and the measured surface moved with them: `benchmarks/report.md`,
`benchmarks/probes.md`, `docs/benchmarks.md`, `docs/compare.md` §3, `docs/model-card.md`, README,
the landing page and `docs/huggingface.md`. The fast path was re-measured on the new weights
(2.72×, parity 0.000771, 0 flips), the B-5a artifact was re-judged (both gates pass), and the
CHANGELOG entry sits under `[Unreleased]`.

**Acceptance.**
- [x] No request-toned record carries label 0, and no neutral-toned record label 1 (a test asserts
  it — over every domain × language, plus the three shipped eval sets).
- [x] The scope decision is recorded as an ADR (`docs/adr/ADR-0014-noul-label-from-text.md`) and the
  affected datasets/checkpoints re-measured (datasets regenerated, both adapters retrained, probes,
  fast path, head-to-head and the B-5a artifact all re-run).
- [x] Per-language `noul` accuracy is reported next to the contradictory-label rate, so the two can
  no longer be confused (`report["noul_per_language"]` + `report["noul_labels"]`, rendered in one
  table by `benchmarks/report.py` and in `docs/benchmarks.md`).

**Risks / notes.** Cost: up to two retrains plus a full benchmark re-run (blocker B-002); the
published numbers stay valid for the data that produced them, which is why this is a decision and
not a bugfix. Effort ~2 days + GPU.

**Related.** `STATE.md` L-008, AD-009 (the last time a re-measurement was published as a set),
`training/generate_data.py::_noul_record`, `NFR-C06`.

---

## B-12 · The `score` near-tie label also comes from the index · **Done (2026-09-29)**

**Why.** `_score_record` takes the level from the tone (`_LEVEL_BY_TONE`) and then, on
`index % 13 == 0`, lowers it one level as an "ambiguous near-tie label". The downgrade is a
property of the loop position, not of the text: **7.8%** of `score` records (39/500 in both
`eval_en.jsonl` and `eval_multi.jsonl`) carry a level their sentence does not read as, so a model
that reads the tone perfectly tops out at **12/13 = 0.922** — and the released checkpoints are
sitting on that ceiling, not below it:

| Set | `score` accuracy | text-consistent ceiling |
| --- | ---: | ---: |
| English (AD-009) | 0.910 | 0.922 |
| multilingual r=64 | 0.916 | 0.922 |

So `score` is label-bound exactly as `noul` was (L-008), only less visibly: the artifact caps the
metric instead of splitting it per language. Measured 2026-09-27 against the regenerated datasets,
whose `score` labels B-11 did **not** touch. (That 0.922 counts the empty boundary states as if
they were readable; counting them too — they inherit a tone the text never shows — the ceiling a
perfect tone-reader actually reaches is **0.888**, which is where the released checkpoints sit:
0.882 English, 0.854 multilingual.)

**Plan (the B-11 shape).**
1. Decide the rule: keep the near-tie but *author* it — mark the ambiguous phrases in the
   committed domain data (or derive the downgrade from the phrase just emitted), so the label is
   recoverable from the text; or drop the downgrade and keep the task unambiguous.
2. Scope decision + ADR, the same question ADR-0014 answered for `noul`: every `score` label moves,
   so datasets, checkpoints and published numbers move with it. **Bundle it with the next retrain**
   rather than paying for a third one of its own.
3. Publish the ceiling beside the accuracy, the way `noul_labels` now does, so it stays visible.

**Scope grew before execution (2026-09-28) — recorded in ADR-0015.** Measuring the near-tie
surfaced the *same* artifact one primitive over: the **empty boundary state** carries an
index-derived label in `score` (the tone drawn before the state was emptied) and in `choice`
(`options[index % len(options)]`) — 5.4% of every primitive's rows, worth ~4 points of `score`
ceiling and ~1.4 of `choice`. `noul` already had its default from B-11 ("" → 0). And `ecommerce`,
the one domain without a catch-all, blocked the `choice` default — it gained `other` as a fifth
option. Decision: **drop** the near-tie (ambiguity the text cannot express is label noise) and give
both empty states their defaults, in the same cycle as the B-5 retrain — `score` empty → middle
level, `choice` empty → `other`.

**Acceptance.**
- [x] No record whose text reads as level *k* carries *k−1* — the near-tie is gone and `score`
  levels come from the tone's banks, empty states taking the middle level; asserted by a test
  across all five domains × seven languages (`test_score_level_comes_from_the_text_not_the_index`).
- [x] The scope decision is recorded as an ADR (**ADR-0015**) and the affected datasets and
  checkpoints re-measured: seven datasets regenerated, all three checkpoints retrained, the whole
  surface re-run (fast path, report, three probes, both head-to-head tables) and republished.
- [x] Ceiling and accuracy reported together: the ceiling went 0.888 → **0.978** (same English
  weights on corrected labels) and the B-12 retrain reaches **0.998**; multilingual `score`
  0.854 → 0.870.

**Executed (2026-09-28/29) — and what it taught.** Code (`38580da`), ADR-0015, seven datasets
regenerated (golden hashes recaptured; `noul` untouched; `ecommerce`'s `choice` rows re-cycled
4 → 5 options), then all three checkpoints retrained in one cycle. The cycle is why the **published
set was re-chosen by merit**:

| checkpoint | retrained on B-12 labels | published instead | why |
| --- | ---: | ---: | --- |
| English | 0.962 (`choice` 0.888) | **0.972** (B-11 weights, `choice` 0.946) | an identical-recipe **control landed 0.841** — L-006 variance |
| multilingual | **0.743** (`choice` 0.468) | **0.743** | it won on merit *and* has clean provenance |
| five-domain run 5 | 0.768 (`choice` **0.303**) | **0.879** (B-11 weights) | the retrain **lost the support gate** (0.763) |

Mechanism, measured three times: once the near-tie noise is gone, `noul` and `score` become the
*same* tone detector and both reach 1.000, and the shared trunk starves `choice` (train loss 0.027
with eval at chance for five domains). One config bug was found on the way —
`finetune_en_domains.json` still carried run 6's settings (`choice_rank` 256, 8 epochs) instead of
the designated run 5 (`128`, 6), so the first domains retrain ran the wrong experiment and
collapsed outright; restored in `aedb064` (lesson **L-011**).

**Risks / notes.** Cost realized: one label fix, three retrains, one control, one wrong-recipe run
and a full re-benchmark (B-002) — the bundling advice above was right, and B-5's "retrain on
corrected labels" option is now *measured and closed* rather than open.

**Related.** `STATE.md` L-008, `docs/adr/ADR-0014-noul-label-from-text.md`,
`training/generate_data.py::_score_record`.

---

## B-13 — JevBench preparation (score well on the public board) · **P0…JB-13 done (2026-10-02) — artifact published as `munod/tachyone-en` [`f28103bf`](https://huggingface.co/munod/tachyone-en/commit/f28103bf4a85bacd10df90c21125c84522c82993); P3 executed and adopted with its gate recorded NOT met (52.6 < 60); **P4 filed as #182 — B-13 complete**

**Why.** [JevBench](https://github.com/fstandhartinger/jevbench) ranks Jev-class decision
models on four equally-weighted axes (Intelligence chance-corrected per tier, Calibration =
ECE + gold-distribution fidelity, Speed, Cost) with a `(I/50)²` multiplier below 50
Intelligence. It speaks exactly the frozen `/v1/systemone` wire (their adapter `typesafe` —
verified against our models, no change needed), so it is the external scoreboard the
project's quality work has never had. 534 decisions: 231 public (measurable today) + 308
sealed (measured by the evaluator on our offline artifact).

**Status — P0 baseline measured (2026-10-02), full record in
`.specs/features/jevbench/spec.md`:** 231/231 strict-valid, **0 failures** (wire needs
nothing), but on the public items: Intelligence **8.4** (Laya, same trunk class: 45.8),
Calibration **0.0** (ECE 0.543 ≥ 0.5 zeroes the axis), Speed **84.8**, Cost **78.8** →
composite **0.43** with the near-chance multiplier 0.028. Per tier: easy 0.500, standard
0.361, hard 0.288 (below its 0.336 chance). Per primitive: `noul` 0.473 (= binary chance),
`choice` 0.309, `score` 0.222. Two structural causes recorded: the runtime `noul` score
ignores the rubric (`cos(question, state)` only → policy/adequacy at chance), and English
inference truncates at **512 tokens** while hard states average 1,079 (max 3,746).

**Plan (locked decisions in the spec):**
1. **P1 hardening** — wire audit (done in P0); English `context` 512 → long enough for the
   hard tier, **A/B measured on the public run** (mean-pool shift risk, do not assume);
   fast path + p50/p95; cost basis documented ($0.01/M encoder class, one forward pass).
2. **P2 Intelligence** — family-shaped training data, two layers: (a) real public sources
   (MultiNLI/BoolQ/Banking77; SST-5/AG News only after licence review — the tev1 recipe's
   `DATA_SOURCES.md` is the reference) converted to our record shape with pinned
   provenance; (b) synthetic **executable rule trees** for `long_policy`/`multi_hop`/
   `temporal_numeric`/`trap` + the six original families. Targets: easy ≥ 0.95,
   standard ≥ 0.73 → **I ≥ 50** (kills the multiplier).
3. **P3 Calibration** — ECE 0.543 → ≤ 0.15 by flattening confidence when the best
   similarity is weak (argmax untouched ⇒ Intelligence independent of this axis); B-6
   contrastive is optional here, measured on these items.
4. **P4 submission** — offline-artifact issue (the #158 pattern): pinned Hub weights,
   licences, inference command, `temperature.json`, this diagnostic, cost basis.

**P1 closed (2026-10-02), full numbers in the spec's *P1 results*:** wire re-audited
**231/231 strict-valid** on both runs; the long-context A/B (`context` 512 → 4096) measured
**1 flip each way — net 0 accuracy (82/231 both, I 8.4 → 8.4) against p95 333 ms → 2,700 ms
and Speed 84.8 → 77.6** → reverted (`4bb5a6a`), which is lesson **L-014**: information access
is not the binding constraint of a similarity encoder. Serving settles on **fast path off**
(end-to-end slower at these shapes; OOMs at raised context — 18.8 GiB of graph pools) and
never `expandable_segments:True` (20.5 GiB at warm-up, 24.5 GiB leaked on kill — GPU0 keeps
12.0 GiB free, the historical training envelope). Cost basis measured: **510 input
tokens/decision, $0.00510/1000 @ $0.01/M → Cost 78.8**.

**P2 executed and measured (2026-10-02) — direction proven, target missed.** Layers:
(a) real sources MultiNLI/BoolQ/Banking77 (11k records, pinned+sha256, AG News/SST-5
excluded on licence); (b) five rule-tree families (`long_policy`/`trap`/`multi_hop`/
`temporal_numeric`/`adequacy`, 3,540 records — every target re-executes from its shipped
facts+tree, label spaces audited per L-010, zero public-item overlap); mixture 35,540.
Two arms one factor apart, both `exit=0`, fits green, evaluated in one harness. **Full
table: `.specs/features/jevbench/tasks.md` JB-7.** Verdict: all B-5 gates pass on every
arm; **Intelligence 8.4 (published) → 9.0 (control) → 15.0 (treatment)**, bench score
0.43 → 0.51 → **1.59** — the treatment's +6.0 is 10× the control's +0.6 variance move, so
the gain is the data's (L-006/L-012). **Cycle gate I ≥ 50 NOT met (15.0).** Calibration
is 0 for all three arms as shipped (**L-015**: the in-domain refit sharpens to T=0.05 and
zeroes the axis off-domain; an earlier pass that read 65.3/48.4 for the fresh arms had
served them *without* the asset). Structural leftovers: `noul` remains rubric-blind
(`hard/trap` 0.000), `hard/temporal_numeric` regressed.

**JB-8 executed same day — adopted by merit.** Both adoption conditions held (bench **1.59
vs 0.43** as-shipped, all gates PASS; treatment vs fresh control **1.59 vs 0.51**).
`checkpoints/en_jev_bank` promoted to `checkpoints/en` (old build kept at
`checkpoints/en_prev_pub0.9637`), calibration shipped as `temperature_calibration.json`.
Full sweep one set: support **1.000** / five-domain **1.000** / gate **1.000**; probes
XNLI **0.566**, typed-decisions **0.367**, MASSIVE **0.051**; fast path **2.543× / 0
flips**; head-to-head **re-run** (the English adapter moved — their turf **0.288** with
`banking77` 0.000 → 0.219, home **1.000** across all three primitives); Hub revision
`f28103bf` (six files sha256-verified, no stale files); report/benchmarks/model-card/
compare/huggingface/README/roadmap/CHANGELOG one set, `mkdocs build --strict` green.
Full record: `.specs/features/jevbench/tasks.md` JB-8.

**Acceptance.**
- [x] P0: baseline measured and recorded (231 public items, one harness, L-013-compliant
      explicit-adapter serve).
- [x] P1: re-run keeps 231/231 strict-valid; long-context A/B recorded with latency cost
      (measured, reverted, documented).
- [x] P2: two-arm run + full pipeline executed; **public diagnostic I 8.4 → 15.0 with all
      B-5 gates PASS** and public items evaluation-only for ever — **but the recorded
      target I ≥ 50 was not met (15.0)**; the data axis is proven (control +0.6 vs
      treatment +6.0), the remaining distance is a follow-up cycle, not a re-run.
- [ ] P3: **NOT met (2026-10-02)** — `calibration(ece)` **52.6** (ECE 0.2369 > 0.15) on the
      single public run after the assets were frozen. The in-domain half is met exactly
      (accuracy byte-identical at 0.999867, per-primitive ECE 0.0259 / 0.000004 / 0.000366,
      each ≤ 0.05) and Intelligence is invariant at 15.0, so the adoption is sound: Brier
      1.109 → 0.745, composite 1.59 → **4.28**. Two causes, both recorded rather than
      smoothed: (a) the pre-registered estimate "T=1 → ECE 0.086" was an artifact of
      reconstructing a pre-temperature distribution from *saturated* `T=0.05` output —
      measured properly, natural gives 0.2376, i.e. the fitted assets are as good as
      natural (lesson **L-016**); (b) no legal fit set can see bench difficulty (0.84 vs
      0.36 at equal evidence), and the one legal slice in the bench's regime — the L-014
      truncation slice, accuracy 0.379 / strength 0.715 — is the one the 2026-10-02
      direction removed. Reaching ≤ 0.15 needs bench-difficulty legal data or the P2
      capability fix, not another fit. Full record: spec *P3 results*, tasks JB-13.
- [x] P4: **done (2026-10-02)** — [`fstandhartinger/jevbench#182`](https://github.com/fstandhartinger/jevbench/issues/182) filed (offline
      artifact: pinned weights `1c88ebef`, licences, inference command smoke-tested, asset
      sha256s, the public diagnostic, cost basis, benchmark-directed-data disclosure, method
      `bb05a335` + METHOD sha256); `docs/jevbench.md` and CHANGELOG carry the same set, and
      `.specs/features/jevbench/submission-issue.md` keeps the verbatim body.

**Risks / notes.** Hard tier (30% weight) may stay at chance for a similarity encoder — it
contributes 0 rather than negative (cc clipped), so the multiplier is decided by
easy+standard (42% weight) plus judge (28%, no public items — the sealed run decides it).
Never train or calibrate on public/sealed items (the >25 pp public-to-sealed gap is
penalised); never vendor the harness (AD-011). Submission queue is long (~30 open
`[bench request]` issues) — file early.

**Related.** `.specs/features/jevbench/spec.md`, `.specs/project/STATE.md` L-003/L-006/
L-013, `docs/compare.md` §3 (the probes said this already: XNLI ≈ chance, ECE raw 0.35–0.65),
`AGENTS.md` hard rules 1/5/6.

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

## B-14 · Recompose retrain (Fases 0–2b of the data-recipe plan) · **In progress — gates fixed 2026-10-06, before training**

**Why.** The 2026-10-05 data audit confirmed the structural defects the plan was written for:
Fase 0 (option and hard-negative strides aliased with the language stride — 24/30 incomplete
cells, every six-language distractor in `pt`), Fase 1 (the eval shares ~99% of templates with
training, so nothing below the ceiling was measurable) and Fase 2 (authored content covered
19–36% distinct states; volume alone simulated +554 `noul` texts at 8×). Fases 0/1/2a/2b landed
on `fix/sampler-stride` — **local only: the JevBench submission is evaluated from the remote
`164ed3b`, so no push, no Hub upload, no publication until the benchmark has run.** The
recomposed `data/eval_multi_domains.jsonl` has different rows, so B-5b's old-row anchors cannot
be reused verbatim: the baseline is **re-fixed on the same rows and the same harness the arms
will use, before any training** (B5B-4 discipline).

**Baseline (released `checkpoints/multi`, explicit adapter, calibration refit per arm,
recomposed rows — `benchmarks/results/multi_domains_recompose_baseline.json`, local):**

| cell | baseline (fixed before the run) |
| --- | ---: |
| overall | **0.8401** (ECE 0.0267; raw 0.3163 before refit) |
| `support` | **0.7887** (ECE 0.0853) ← no-regression anchor |
| `ecommerce` | 0.9500 (ECE 0.0997) |
| `voice` | 0.8500 (ECE 0.0412) |
| `documents` | 0.8080 (ECE 0.0601) |
| `agent_tools` | 0.8040 (ECE 0.0497) |
| worst `domain/lang` | `support/it` 0.7590 |
| per language | `fr` 0.8546 · `es` 0.8516 · `de` 0.8402 · `it` 0.8337 · `pt` 0.8310 · `nl` 0.8297 |
| seen / unseen text | 0.7870 / 0.8708 |

**Fixed gates for this cycle (recorded before the launch):**
1. `support` ≥ **0.7887** — same harness, same rows, baseline-first (B-5b's anchor rule).
2. worst new domain ≥ **0.70** — absolute, unchanged from B-5b.
3. per-domain ECE ≤ **0.05**, exceptions declared — unchanged.
4. gate strict = **1.000** (0 to shared, 0 wrong domain) over the `choice` rows — unchanged.
5. **no language regresses**: each of the six ≥ its baseline above (B-1's rule: gate on the
   worst language, not the average). Measured but **not** gated: per-language ECE ≤ 0.05 —
   B-1's open acceptance closes only with evidence, never by re-fixing a gate.

**Run.** Recipe unchanged (`training/configs/finetune_multi_domains.json`: 8 epochs, r=64,
`choice_rank` 128 → `checkpoints/multi_b5b`; the previous build is kept as
`checkpoints/multi_b5b_precompose`), then `fit_bank_multi_domains.json`, then the B5B-6
harness pattern (explicit adapter, calibration refit per arm) against these gates. Artifacts
stay local (`benchmarks/results/` and `data/preds_*` are gitignored); the `benchmarks/report.md`
re-render and any publication happen as **one set** after JevBench.

**English arm — same discipline, second L4 (gates fixed 2026-10-06, before its run).**
Baseline = the released `checkpoints/en` (the B-13 mixture artifact) on the recomposed
`data/eval_en_domains.jsonl`, same explicit-adapter harness with calibration refit
(`benchmarks/results/en_domains_recompose_baseline.json`, local):

| cell | baseline (fixed before the run) |
| --- | ---: |
| overall | **0.9616** (ECE 0.0190; raw 0.1370 before refit) |
| `support` | **0.9733** (ECE 0.0110) ← no-regression anchor |
| `ecommerce` | 0.9767 (ECE 0.0073) |
| `documents` | 0.9787 (ECE 0.0068) |
| `agent_tools` | 0.9520 (ECE 0.0219) |
| `voice` | 0.9273 (ECE 0.0527) ← already above 0.05, exception carried |
| seen / unseen text | 0.9649 / 0.9530 |
| gate strict | 1.0000 (0 shared, 0 wrong, n = 2,500) |

Fixed gates: `support` ≥ **0.9733** · worst new domain ≥ **0.70** · per-domain ECE ≤ **0.05**
with declared exceptions (baseline already posts `voice` 0.0527) · gate strict = **1.0000**.
Single-language arm, so there is no per-language gate; `seen`/`unseen` is reported, not gated.
Run: `finetune_en_domains.json` (run-5 recipe: r=16, `choice_rank` 128, 6 epochs →
`checkpoints/en_domains`, previous build kept as `checkpoints/en_domains_precompose`) on the
second L4 while the multi arm occupies the first; the `fit_bank_en_domains.json` trunk pointer
moves from `en_domains_r5` to the freshly trained `en_domains` when the fit runs.

**Results (2026-10-07) — both runs `exit=0` under the L-009 detached launch (the first,
harness-launched attempts died silently mid-run, exactly the documented reap), fits `exit=0`,
harness = predict → calibration refit → predict per arm:**

| arm | overall | ECE | `support` | worst new domain | gate strict | seen / unseen | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| multi baseline (released) | 0.8401 | 0.0267 | 0.7887 | 0.8040 | 1.000 | 0.787 / 0.871 | — |
| multi joint bank | 0.9796 | 0.0095 | 0.9300 | 0.9853 | 1.000 | 0.968 / 0.986 | PASS |
| **multi fitted bank (publish)** | **0.9863** | **0.0080** | **0.9413** | **0.9913** | **1.000** | 0.978 / 0.991 | **PASS** |
| multi fitted shared | 0.9863 | 0.0080 | 0.9413 | 0.9913 | n/a (no gate) | 0.978 / 0.991 | ties the bank — L-012, third replication |
| en baseline (released B-13) | 0.9616 | 0.0190 | 0.9733 | 0.9273 | 1.000 | 0.965 / 0.953 | — |
| en joint bank | 0.8209 | 0.0282 | **0.8200 ✗** | 0.7653 | 1.000 | 0.817 / 0.830 | fails the support anchor — not the publish lineage (val_loss 0.666 vs multi's 0.097; the B5B joint-exposure pattern) |
| **en fitted bank (publish)** | **0.9993** | **0.0003** | **0.9987** | **0.9980** | **1.000** | 1.000 / 0.998 | **PASS** |
| en fitted shared | 0.9988 | 0.0007 | 0.9980 | 0.9973 | n/a (no gate) | 1.000 / 0.996 | ties the bank |

**Gate by gate (publish arm):** multi — `support` 0.9413 ≥ 0.7887 ✓ · worst new domain
0.9913 ≥ 0.70 ✓ · per-domain ECE 0.0026–0.0531 → **one declared exception (`support`
0.0531)** ✓ · strict 1.000 (0 fell, 0 wrong) ✓ · every language ≥ baseline — `de` 1.0000,
`nl` 1.0000, `fr` 0.9984, `pt` 0.9976, `es` 0.9857, `it` 0.9357, against baselines 0.830–0.855
✓. en — `support` 0.9987 ≥ 0.9733 ✓ · worst new domain 0.9980 ≥ 0.70 ✓ · per-domain ECE
0.0003–0.0017, **zero exceptions** ✓ · strict 1.000 ✓. Shared-only arms show `strict` n/a
(no gate to route through — the B5B table's convention), not a failure.

**Measured, not gated (B-1):** per-language ECE on the multi publish arm — `pt` 0.0014,
`fr` 0.0016, `nl` 0.0029, `de` 0.0035, `es` 0.0058, **`it` 0.0548**: five of six ≤ 0.05 on
these rows (`nl` was 0.104 at B-1), `it` keeps B-1 open — and these are in-domain synthetic
rows either way. The honest yardstick is the Fase 1 holdout eval, which becomes actionable
only once the training configs adopt `template_split: "train"` **and the arm retrained on it**
(B-14 trained on `all`, so its provenance pins `all`; the flip is deferred with the next data
decision rather than silently invalidating this run).

**Holdout probe (2026-10-07) — how much of the ceiling survives genuinely unseen phrasing.**
Both publish arms and both released adapters ran on the Fase 1 holdout evals (same harness,
calibration refit; `benchmarks/results/{multi,en}_holdout_*.json`):

| adapter | all-split eval | **holdout eval** | Δ |
| --- | ---: | ---: | ---: |
| multi B-14 fitted bank (trained on `all`) | 0.9863 | **0.9876** | +0.001 |
| en B-14 fitted bank (trained on `all`) | 0.9993 | **0.9995** | +0.001 |
| multi released (never saw the expansion's phrases) | 0.8401 | **0.7623** | **−0.078** — worst new domain 0.638, *below* the 0.70 gate line |
| en released (never saw them either) | 0.9616 | **0.8508** | **−0.111** — `voice` 0.515, ECE 0.069 |

Two readings, both recorded. (1) **The 0.9863/0.9993 is recognition**: the holdout cannot bite
an arm trained on `all`, and indeed its score does not move — in-domain and holdout are the
same measurement for B-14. (2) An adapter that genuinely never saw those phrases pays **8–11
points for novel phrasing inside the same family** and would fail the worst-new-domain gate —
real generalization, mid-family: far above the external-probe floor (XNLI 0.566,
typed-decisions 0.367, peer's turf 0.288) and far below the in-domain ceiling. Caveat: for the
released adapters `text_seen` is computed against *today's* train file, not their actual
pre-expansion training data, so only their overall column is comparable.

**Provenance pin (before any regeneration):** B-14 trained on these exact bytes —
`train_en` `c4101fc8037bef85`, `train_en_domains` `9212e46d6a7994dc`, `train_multi`
`de1bcca1e81190a6`, `train_multi_domains` `97a3ca3d5e56644f`, evaluated on `eval_en_domains`
`934145c047375133` / `eval_multi_domains` `92b526f99fe60187` (sha256[:16], split `all`).
Any later flip to `template_split: "train"` must reproduce *these* bytes to re-verify B-14's
own run before replacing them.

**Estágio 2 (the clean number, not yet run):** flip the four training configs to
`template_split: "train"`, pin today's B-14 training sets by hash first (provenance — B-14
must keep reproducing its own bytes), regenerate, retrain both arms (one evening of the two
L4s), score holdout: that model has never seen the held-out phrases, so its holdout row is a
pure phrasing-generalization measurement.

**RESOLVED (2026-10-07): `noul` in `support`/`it` was inverted in the multi publish arm —
closed the same day by the interleaved joint continuation recorded below (all five
pre-registered gates green). Publication itself stays deferred to the post-JevBench cycle.**
(Found 2026-10-07 while detailing B-1.) The joint trunk **and** the fitted
bank score their own 1,192 training rows at **10% accuracy** (`P|1)` = 0.456 vs `P|0)` = 0.715);
`eval_multi` reports `noul`/`it` **0.0723** (every other language 1.000), dragging `support/it`
to 0.6787 / ECE 0.3065 — while `it` in the other four domains is 1.000 and `choice`/`score`
stay healthy, which is why the pre-registered gates still passed (they cover overall cells and
per-language accuracy, not per-language `noul`). Ruled out by direct test, each with a
measurement: labels (B-11 audit clean on the train file: neutral→0, request→1), split
membership (1,192 rows in train, zero duplicate ids), question-text parity between training and
runtime (identical cosines −0.1958 both paths), `noul` math (same cosine, positive temperature
— sign only), content age (inversion uniform across pre-expansion and expansion phrases and
entities), the fit/export path (joint trunk inverted identically), and epoch 1 (cosines healthy
— `it` request +0.9735, neutral +0.9431, same as `de`; no separation yet): **the flip happens
between epoch 1 and 8.** Question-side signature: the empty-state baseline of this cell alone
is 0.707, against 0.437–0.444 in the other 14 cells. The released adapter and the pre-B-14
trunk both score these rows **0.892 correct-direction** — so this is a regression of the B-14
training run, not of the data or of the era.

**Localized (2026-10-07, seed-42 snapshots): the cell learns correctly, then late-training
interference inverts it.**

| snapshot | `support/it` `noul` | control `support/de` | val_loss (aggregate) |
| --- | ---: | ---: | ---: |
| epoch 1 | 0.500 (everything uniform) | 0.500 | 0.839 |
| **epoch 2** | **1.000** (`P\|1` 0.727 / `P\|0` 0.422) | 1.000 | 0.713 |
| **epoch 4** | **0.793, degrading** (0.584 / 0.494 — its *own* gradient still points to recovery) | 1.000 | 0.620 |
| epoch 8 (B-14) | **0.100 inverted** (0.457 / 0.715) | 1.000 | **0.097** |

The cell was correct at epoch 2 and dies while its own gradient still points the right way —
the signature of **catastrophic interference through the shared trunk**, driven by the other
streams (choice/score still converging after epoch 4), not by anything in the cell's own data.
Aggregate `val_loss` improves throughout (this cell is only ~0.010 of the final 0.097) and the
epoch-4 model already reads `noul` **0.9884** / `score` 0.9928 on the full eval — but its joint
`choice` is 0.3016 (fit-dependent) and its val_loss 0.620, so "stop early" is not a free fix.
**Options recorded:** (a) seed rerun — is the flip seed-specific? (b) a short `noul`-only
continuation from the epoch-4 checkpoint (analogous to the fit's head-only step) to test
whether choice/score gradients are the driver, then re-fit the bank and re-gate; (c) a
per-`(domain, lang, primitive)` val monitor in the trainer so this class of cell collapse can
never hide behind the aggregate again. The multi arm stays unpublished until one of (a)–(c)
lands with the gates green.

**Driver test (b) run and CONCLUSIVE (2026-10-07): choice/score gradients are the driver.**
Continuing the epoch-4 adapter for four `noul`-only epochs (same split, same per-epoch shuffle
seeds, same loss math) yields **`support/it` = 1.000** (`P|1)` 0.731 / `P|0)` 0.269) with
**zero** cells below 0.95 — while the real run's choice/score stream drives the same cell to
0.100 over the same epochs. The cell not only survives without them, it fully recovers.
**Trap recorded (it invalidated the first attempt):** `PeftModel.from_pretrained` defaults to
`is_trainable=False`, so the first continuation trained only the fresh temperature parameter
and saved input-identical weights — caught by hashing `adapter_model.safetensors` after the
run (`sha256` identical to the input) and fixed with `is_trainable=True`. Rule for any
continuation/repair script: **hash the adapter before and after; identical hashes mean nothing
trained.** Open decision now: (1) `noul`-only repair from the final trunk + bank re-fit + full
re-gate (~2 h, artifact recipe gains a documented touch-up step), (2) clean retrain with a
per-cell monitor and/or schedule mitigation (~5 h), (3) seed rerun first.

**Path (1) executed and REJECTED by its own gates (2026-10-07).** The `noul`-only repair from
the final trunk (`checkpoints/multi_b5b_noulfix`) closed the cell (train `it`/`support`
0.100 → 1.000, every language's `noul` 1.000) but was zero-sum through the shared trunk: on
the same harness the fitted bank fell overall 0.9863 → 0.9391, `choice` 1.000 → 0.9056,
`score` 0.9956 → 0.9116, and per-domain ECE rose to 0.063–0.102 against the 0.05 gate — the
pre-registered gates caught the repair exactly as they were fixed to do
(`benchmarks/results/multi_b14fix.json`, local).

**Resolution — interleaved joint continuation: defect CLOSED, all five gates PASS
(2026-10-07).** The hypothesis the driver test left standing, stated precisely: the trainer's
epoch runs *sequential phases* (all `noul` → all `choice` → all `score`, `score` last), so
each primitive gets ~490 consecutive pure steps through the shared trunk; the fix is
round-robin batches (`noul`, `choice`, `score`, `noul`, …). The test continues the
**original inverted trunk** (`checkpoints/multi_b5b`) for two more epochs with the same loss
math, split and per-epoch shuffle seeds — resume with `is_trainable=True`, adapter sha256
`9f646f3e…` → `b8e9c1b6…` (so training provably moved), script
`/tmp/opencode/finetune_interleave.py`, checkpoint `checkpoints/tmp_il_orig`:

| axis | e8 trunk | interleaved (+2 ep) |
| --- | ---: | ---: |
| train cell `noul`/`it`/`support` | 0.0977 | **0.9977** |
| `eval_multi` `noul`/`it` (the 0.0723 defect term) | 0.0723 | **0.9880** |
| `eval_multi_domains` `noul`/`it` | 0.8145 | **0.9976** |
| `choice` (raw eval) | 0.9800 | 0.9968 |
| `score` (raw eval) | 0.9956 | 0.9996 |
| overall (raw eval) | 0.9796 | 0.9985 |

The interleaving fixed **both directions of the interference in one run**: the inverted cell
recovered while the healthy siblings not only survived but improved — so paths (a) seed
rerun and (b) further schedule work are no longer needed. Fit + calibration + harness on
both evals (`checkpoints/tmp_il_orig_fit_bank`, config retarget commit; reports
`benchmarks/results/multi_b14il{,_evalsupport}.json`, local):

1. `support` **0.9993** ≥ 0.7887 — PASS
2. worst new domain `voice` **0.9987** ≥ 0.70 — PASS
3. per-domain ECE **0.0001–0.0011** ≤ 0.05 — PASS, zero exceptions needed
4. gate strict **1.0000** (0 fell to shared, 0 wrong domain) — PASS
5. no language regresses: `de` 1.0000 · `es` 1.0000 · `fr` 1.0000 · `it` 0.9984 ·
   `nl` 0.9992 · `pt` 0.9992 — every one ≥ baseline (0.830–0.855) — PASS

overall **0.9995 / ECE 0.0004** (B-14's first fit: 0.9863 / 0.0080). The control variant
continued from the *repaired* trunk (`checkpoints/tmp_il_fix`) is also healthy but never
fully recovers its siblings (raw `choice` 0.9928) — kept as evidence, not a publish
candidate. **The multi publish arm is `checkpoints/tmp_il_orig` (+ its fitted bank); the en
arm is unaffected.**

**Remainder of this defect — both items LANDED (2026-10-07):**
(c) the per-`(domain, lang, primitive)` val monitor is now in the fine-tune loop: every epoch
buckets validation by `cell_of`, appends one `cell_monitor.jsonl` line to `out_dir`, prints the
worst cell of each primitive next to the train loss, and ships the last epoch's cells in
`report["val_cells"]` — smoke-verified end-to-end (commit `f57343f`, tests in
`tests/test_training_finetune.py`); and the continuation script lives in the repo as
`training/interleave_continue.py` (commit `e9d49e4`, tests in
`tests/test_training_interleave.py`) — resolved hyperparameters from CLI >
`finetune_config.json` > `adapter_config.json`, sha256 before/after with a loud failure when
nothing trained. Bitwise parity with the original `/tmp` run is impossible on this GPU (two
identical runs differ Δ0.028 in adapter weights): recorded as **L-018**, parity is behavioral.

**Deferred to the publication cycle (post-JevBench, one set — B5B-8's rule):** promotion to
`checkpoints/multi` / `checkpoints/en`, `benchmarks/report.md` re-render, Hub upload. Nothing
of the sort happens while the submission is being evaluated from the remote.

---

## B-15 · Estágio 2 — template-holdout retrain (the honest generalization number) · **Executed 2026-10-08 — every gate PASS except B-1's per-language ECE, recorded NOT met**

**Why.** B-14's arms trained on `template_split: "all"`, so their holdout scores (multi
**0.9876**, en **0.9995**, `benchmarks/results/multi_holdout_b14.json`) are **in-sample** —
those rows' templates were in their training. The only honest unseen-phrasing numbers today
are the released adapters (multi 0.7623 / `choice` 0.5604, en 0.8508), which predate the
recompose recipe. B-1's acceptance was written for exactly this yardstick and has stayed
blocked on it ("the holdout eval … needs a `train`-split retrain to be actionable").

**What runs.** Flip the four training configs to `template_split: "train"`, regenerate, and
retrain both arms with the B-14 recipe — en 36,000 × 6 ep; multi 52,200 × 8 ep **with the
per-cell monitor** plus the **unconditional** 2-epoch interleaved touch-up (the
`tmp_il_orig` lineage), so the only variable against B-14 is the data. Fit the banks, then
score **three evals per arm**: current (`all`, unchanged rows), in-template (`train`, new),
holdout (`holdout`). Decisions recorded before the run: touch-up unconditional; the
`noul`-per-language tripwire below; the in-template column is measurement-only.

**Provenance guard (executed 2026-10-07 before any regeneration — all OK).** B-14's bytes
re-verified from the day's configs, disk **and** regeneration agreeing: `train_en`
`c4101fc8037bef85`, `train_en_domains` `9212e46d6a7994dc`, `train_multi` `de1bcca1e81190a6`,
`train_multi_domains` `97a3ca3d5e56644f`, `eval_en_domains` `934145c047375133`,
`eval_multi_domains` `92b526f99fe60187`. Holdout evals pinned for the first time:
`eval_en_domains_holdout` **`3119e73c6d851e31`**, `eval_multi_domains_holdout`
**`982236ca2728705f`**. After the flip, B-14's datasets remain reproducible from the
pre-flip configs in git history.

**Fixed gates (recorded before the launch):**

*Current benchmark* — `eval_{en,multi}_domains.jsonl`, rows unchanged, so B-14's gates
transfer verbatim:
1. multi `support` ≥ **0.7887** · worst new domain ≥ **0.70** · per-domain ECE ≤ **0.05**
   (exceptions declared) · gate strict = **1.000** · every language ≥ baseline (`fr` 0.8546 ·
   `es` 0.8516 · `de` 0.8402 · `it` 0.8337 · `pt` 0.8310 · `nl` 0.8297).
2. en `support` ≥ **0.9733**.

*Holdout benchmark* — `eval_*_domains_holdout.jsonl`, rows never in any training:
3. **B-1 acceptance, verbatim**: multilingual `choice` ≥ **0.60** · per-language ECE ≤
   **0.05** for `choice`/`score` with numbers published · English overall ≥ **0.72**.
4. **Anti-inversion tripwire (the B-14 lesson)**: `noul` ≥ **0.60** in *every* language
   (en included) — an inversion lands near 0.07, baseline is ~0.5, so the floor fails loudly
   on a defect and never on sampling noise.
5. `wrong_domain` = **0** on every holdout report (misrouting is structural); `fell_to_shared`
   is published as a measurement — signature matching may miss unseen phrasing, and if it
   moves that is a finding, not a gate.

*In-template eval* — `eval_*_domains_train.jsonl`, new rows: **measurement only**. No anchor
exists on rows that were never benchmarked; publishing a floor invented after seeing the data
would re-fix a gate.

**Reference columns (already measured, labeled in the report):** B-14 arms on holdout
0.9876 / 0.9995 = **in-sample**; released adapters 0.7623 / 0.8508 = pre-recompose.

**Honest-failure rule.** If B-1's criteria or the tripwire fail, they are recorded as NOT met
with the numbers — never re-fixed (B-1's own rule). Deliverable: the per-axis table
`current vs in-template vs holdout` (overall, `choice`/`score`/`noul` per language,
`support` per language — accuracy + ECE), Δ = the generalization gap, per arm.

**Results (2026-10-08).** Full cycle `exit=0`: flip → both retrains (monitor green every
epoch — **the B-14 cell defect did not recur**; final `noul:*:it` cells 0.047–0.067) →
unconditional touch-up (sha `19681298…` → `f7969603…`) → fits → both harness chains
(18 artifacts). Arms: `checkpoints/{en_tt, multi_tt, multi_tt_il}` + fitted banks.

*Gate verdict:*
- **Current, multi — all five PASS**: `support` **0.9760** ≥ 0.7887 · worst domain `voice`
  **0.9747** ≥ 0.70 · per-domain ECE **0.0019–0.0202** ≤ 0.05 (**zero exceptions**) · strict
  **1.0000** (0 shared, 0 wrong) · every language ≥ baseline — `fr` 0.9960 · `es` 0.9841 ·
  `de` 0.9904 · `it` 0.9799 · `pt` 0.9865 · `nl` 0.9944. overall **0.9885 / ECE 0.0069**
  (B-14: 0.9863 / 0.0080 — the B-15 trunk is *better* raw: choice 0.9936 vs 0.9800).
- **Current, en — all four PASS** (`85a2f05`): `support` **1.0000** ≥ 0.9733 · `voice` 0.9900
  ≥ 0.70 · per-domain ECE ≤ **0.0082** · strict **1.0000**. overall **0.9980 / 0.0009**.
- **Holdout**: B-1 `choice` ≥ 0.60 → multi **0.9248** PASS; en overall ≥ 0.72 → **0.9821**
  PASS; tripwire `noul` ≥ 0.60/idioma → PASS everywhere (worst `it` 0.7807, `es` 0.7810, en
  0.9944); `wrong_domain` = 0 both arms with **`fell_to_shared` = 0** and strict 1.0000 —
  the signature gate survives unseen phrasing untouched.
- **Holdout, B-1 per-language ECE ≤ 0.05 for `choice`/`score`: multi NOT met** — choice `es`
  0.0753 · `it` 0.0968 · `pt` 0.1014; score `de` 0.0540 · `es` 0.0951 · `it` 0.1495 · `nl`
  0.0577 (the report's all-primitives-per-language reading agrees: `es` 0.0717 · `it` 0.1261
  · `pt` 0.0507). en PASS (choice 0.0155, score 0.0067). Recorded as NOT met per the rule —
  **B-1 stays open with honest numbers.** Metric defs mirror `training/evaluate.py` exactly
  and were validated against the report's `per_language`/`noul_per_language` to 4 decimals.

*Generalization tables (accuracy; calibrated reports):*

overall acc/ECE — multi **0.9885/0.0069 → 0.9997/0.0002 → 0.8892/0.0426**;
en **0.9980/0.0009 → 1.0000/0.0002 → 0.9821/0.0089** (current → in-template → holdout).
**Pure phrasing gap (in-template → holdout): multi −0.1105, en −0.0179.**

| eixo | idioma | multi atual | train | holdout | Δ | | en atual | train | holdout | Δ |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| choice | de/es/fr/it/nl/pt | .9928/.9929/1.0/.9928/.9976/.9857 | .9976/1.0/1.0/1.0/1.0/1.0 | **.9735/.8619/.9928/.8795/.9807/.8619** | −.019/−.131/−.007/−.113/−.017/−.124 | choice | en .9956 | 1.0 | **.9600** | −.036 |
| score | de/es/fr/it/nl/pt | .9928/.9905/.9952/.9880/.9928/.9929 | 1.0/1.0/1.0/1.0/1.0/1.0 | **.9157/.8619/.9325/.8699/.9205/.9357** | −.077/−.129/−.063/−.118/−.072/−.057 | score | en .9996 | 1.0 | **.9920** | −.008 |
| noul | de/es/fr/it/nl/pt | .9855/.9690/.9928/.9590/.9928/.9810 | 1.0/1.0/1.0/1.0/.9976/1.0 | **.8867/.7810/.9060/.7807/.8434/.8238** | −.099/−.188/−.087/−.178/−.149/−.157 | noul | en .9988 | 1.0 | **.9944** | −.004 |
| support | de/es/fr/it/nl/pt | .9759/.9643/.9880/.9438/.9839/1.0 | 1.0/1.0/1.0/1.0/1.0/1.0 | **.7952/.7738/.7831/.7068/.7671/1.0** | −.181/−.191/−.205/−.237/−.217/+0.0 | support | en 1.0 | 1.0 | **1.0000** | ±0 |

*Findings.* (1) **Language asymmetry**: `fr`/`nl`/`de` `choice` lose ≤0.02 on unseen phrasing
while `es`/`it`/`pt` lose ~0.11–0.13 — the phrasing-generalization gap is not uniform across
languages (B-1's worst-language rule keeps catching what averages hide). (2) **`support` is
the weakest domain holdout-side** (`support/it` 0.7068, −0.237) while `support/pt` is
unchanged at 1.0000. (3) `noul` drops most per language (`es`/`it` −0.19/−0.18) yet clears the
tripwire everywhere. (4) The multi trunk's aggregate val_loss rose vs B-14 (0.532 vs 0.097)
while its raw eval *improved* (choice 0.9936 vs 0.9800): the val `choice` cells were noisy,
the benchmark is the arbiter — the monitor's cost is real too (~+14 min/epoch on 90 multi
cells; en's 15 cells ≈ free). (5) Reference: B-14 arms on holdout are in-sample (0.9876 /
0.9995); released adapters 0.7623 / 0.8508 → **B-15 honest = +0.127 (multi) / +0.131 (en)
over released, −0.098 / −0.017 under the in-sample ceiling.**

**Publish-candidate decision (deferred to the publication cycle):** B-15 arms supersede B-14
as the promotable lineage — current-benchmark numbers hold (multi better, en −0.0013) *and*
their holdout number is claimable, which B-14's in-sample 0.9876 never was.

**L-019 (analysis trap):** eval ids are **not** unique — 1,500 ids × 5 rows, one per domain —
so a preds→domain join by `id` silently collapses domains (it produced an all-zero `support`
axis and, during B-14, an "it is all `voice`" artifact). `support` × language comes from the
report's `per_domain_language`; never join by id.

**Publication cycle (2026-10-08) — decisions, measured on the way to the push:**
- **Promotion**: `checkpoints/{en,multi}` now carry the B-15 fitted banks (the v0.8.0 published
  dirs are preserved as `checkpoints/{en,multi}_prev_pub_v080`). The P3 stack was **rebuilt on
  `en_tt`**: holdout regenerated against the new training file (public-item assertion intact),
  prototypes over 36,000 states, confidence with sha256 provenance, temperatures
  `choice 10.0 / noul 0.75 / score pinned 0.1`.
- **P3 in-domain acceptance (≤ 0.05) NOT met for the rebuilt asset.** The pooled fit pegged the
  top of the pre-registered `DEFAULT_GRID` (**T = 10.0**): training is `split: train` while the
  holdout keeps its hard rows, and the pooled objective accepts the trade — in-domain `choice`
  ECE reads **0.5243**, accuracy untouched (0.9956). Recorded, never re-fixed; the follow-up
  is a fit-basis redesign (slice-balanced pooling), not another fit.
- **Multi's served temperature is the pooled fit** over 15,000 in-template + holdout
  predictions (`choice 6.0 / noul 0.25 / score 0.25`) — deliberately *not* the in-sample
  support-split fit (whose `ece_before ≈ 0` made the grid peg at `choice 0.05`, the L-015 trap).
- **Report harness pins BOTH local checkpoints.** The router sends empty/no-signal states to
  the English checkpoint by design (**L-013**, ~14% of five-domain multilingual rows); with the
  Hub adapter as fallback the measurement mixes revisions, so the pair measured is the pair
  published. Numbers (routed, as served): multi five-domain **0.9156 / ECE 0.0186** (the
  multilingual checkpoint itself answers **0.9885** when it answers everything), holdout
  **0.8355 / 0.1032**, support **0.9107 / 0.0506** (noisy 0.9087); en support **1.0000**,
  five-domain **0.9983**, holdout **0.9825** (ECE ~0.19 — the T=10 disclosure above).
- External probes (XNLI/typed-decisions/MASSIVE/JevBench) and the fast-path figures remain
  **v0.8.0-artifact measurements** — pending re-measurement, recorded on the card.

**Executed 2026-10-08 as one set (B5B-8; the JevBench deferral was lifted by decision — see
STATE *Current Work*):** promotion → `benchmarks/report.md` re-render (six entries, holdout
included) → `chore(release): v0.9.0` + tag → push → Hub upload with per-file sha256 → GitHub
release.

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
