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

## B-5 — Multi-domain coverage (5 domains) · B-5a **both gates pass**, isolate closes 91% of the support gap, **published as `munod/tachyone-en`** (2026-10-01); probes/head-to-head re-run open

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
- [ ] Probe before/after table published (`B-7` harness) → **still open**: neither the public
  probes nor the head-to-head `choice` cells have been re-run against the new revision (the
  fast-path parity check *was* re-run: **0 flips**, `benchmarks/results/fast_path_en.json`).

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

**B-5b (the multilingual half) stays blocked** behind whichever of those is chosen.

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
