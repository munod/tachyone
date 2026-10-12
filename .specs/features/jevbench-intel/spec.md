# B-21 — JevBench Intelligence ≥ 50 without a System-2 handoff (corrected plan)

**Status:** Spec — adopted 2026-10-12 (owner) after code/evidence review. **Supersedes** the
uncommitted working note of 2026-10-12; the review corrections below are applied inline and
marked `[review]`. Gates are pre-registered in `.specs/project/BACKLOG.md` **B-21** before any
data edit or training (the project's discipline). Nothing here is trained yet.

**Hard constraint (owner, 2026-10-12):** the answer must come from the **encoder alone** —
single forward pass, offline, ~70 ms p50. **`handoff.py` is out of scope**: no cascade, no
System-2 fallback, no serving-time abstention. The teacher LLM is a **training-time
instrument only** (data + soft labels), never a serving path.

**Evidence base:** `STATE.md`, `BACKLOG.md` B-1/B-13/B-15/B-16/B-17/B-18/B-19/B-20,
`docs/jevbench.md`, `docs/training.md`, lessons L-003/L-004/L-007/L-012/L-013/L-014/L-015/
L-016, ADR-0005/0014/0015/0016, and the three LLM reference rows measured 2026-10-11/12
(`gemma4:26b`, `qwen3.6:35b` think-off, and the qwen think-on paired subset).

---

## 0. The target, as arithmetic (verified cell-by-cell against `composite_v13.py`)

```
I = (0.14·cc_easy + 0.28·cc_standard + 0.30·cc_hard) / 0.72
cc(tier) = 100·(acc − chance) / (1 − chance)
chance: easy 0.2840 · standard 0.3167 · hard 0.3364
```

`I ≥ 50` ⇔ `0.14·cc_easy + 0.28·cc_standard + 0.30·cc_hard ≥ 36`.

| Scenario | cc easy | cc std | cc hard | acc easy | acc std | acc hard | **I** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **shipped v0.13.0** | 18.5 | 16.7 | 0.8 | 0.417 | 0.431 | 0.342 | **10.5** |
| **v0.8.0 mixture** | 47.6 | 14.6 | 0.0 | 0.625 | 0.417 | 0.333 | **15.0** |
| balanced floor | 50 | 50 | 50 | 0.642 | 0.658 | 0.668 | **50.0** |
| **recommended target** | 80 | 65 | 30 | 0.857 | 0.761 | 0.535 | **53.3** |
| stretch | 90 | 75 | 45 | 0.928 | 0.829 | 0.635 | **65.4** |

The mountain is **not only `hard`**: the shipped encoder is at chance on `easy` *and*
`standard` too, and the v0.8.0 mixture (same architecture, one dataset) already had `easy` at
cc 47.6. The first ~15 points are a **data** problem; the rest is data + architecture +
distillation. `[review]` With `cc_hard = 25`, `I = 50` still needs `easy ≈ 0.86` **and**
`standard ≈ 0.72` — the two natural-language tiers where **B-20 measured a regression**
(`standard` 0.431 → 0.361). That is the central hypothesis S1 must test.

## 1. Diagnosis (measured)

1. **Domain coverage is the binding constraint (L-007).** B-15 made training synthetic-only;
   `easy/intent` 0.750 → 0.250, XNLI 0.566 → 0.333. The public `easy`/`standard` tiers are
   natural-language intent/yes-no items.
2. **Architecture is a real but secondary ceiling.** `ChoiceScorer`
   (`training/finetune_rlcd.py`) scores each option with a low-rank residual over
   `cos(option,state) + cos(option,question)`; the option never attends to the state — the
   `trap`/`adversarial`/`temporal_numeric` signature.
3. **Context/arity are untested ceilings.** Inference truncates at the trained length (512);
   the hard tier averages ~1,100 tokens (measured est. 1,201). L-014: raising inference
   length alone bought nothing (+1/−1, p95 333 → 2,700 ms) — the fix is to **train** at
   length. `[review]` The public bench ships arity ≤ 6 (median 4–5), not 77 — 77 is the
   Banking77 class count and was conflated; the sealed set is unverifiable. Variable arity
   (2…~20) stays as a **robustness** axis (chance correction punishes positional bias), not
   as a response to a measured bench arity.
4. **Calibration is not the bottleneck.** Shipped Calibration 85.9; the miss is **hedging on
   hard items** (confidently wrong). Soft labels attack this directly.
5. **The teacher knows the answer.** Measured in-repo: `gemma4:26b` hard 0.964 (ECE 0.027);
   `qwen3.6:35b` reads 162/165 = 98.2% **with reasoning** vs 83.0% think-off (paired).
   `[review]` The `gpt-oss:120b-cloud` (I 92.99) and `gemma4:31b` figures come from the
   uncommitted companion note and are **unsourced in the repo** — measure or cite before any
   decision rests on them. **Extraction must use thinking mode** (+15 accuracy points, paired)
   and the cost model must name the model: gemma4 p50 ≈ 11 s/record, qwen think-on ≈ 120 s.

## 2. The four levers, ranked

| # | Lever | What it moves | Cost | Confidence |
| --- | --- | --- | --- | --- |
| **L1** | Cross-encoder pairwise per option | `hard` + `standard` accuracy | medium | high |
| **L2** | Data coverage (natural + reasoning + counterfactual + arity) | `easy` + `standard` + `hard` | low–medium | **highest** |
| **L3** | Soft-label distillation from the teacher | hedging/ECE + near-miss `hard` | medium (GPU) | medium |
| **L4** | Long-context **training** + better pooling | long-policy / multi-hop | medium | medium |

## 3. Workstreams

### W1 — Data coverage and reasoning shape (L2) — *first*

Recipe changes in `training/generate_data.py` + `training/data/domains/*.json`, measured
against the existing holdout eval. **`[review]` factored — one factor per arm (L-012):**

1. **Natural-text layer + curriculum (S1a).** MultiNLI/BoolQ/Banking77 (the "+6.0
   treatment") + **ANLI R2/R3** + **ContractNLI**; licences pinned in
   `training/data/jev_sources.lock.json`; **staged curriculum (natural → synthetic with
   replay) or a replay ratio — never naive concatenation** (B-20's lesson; the ordered recipe
   mechanism exists in `training/configs/jev_mixture_b20.json`). No JevBench item — public or
   sealed — ever enters training (asserted by every builder against `datasets/public/`).
2. **Counterfactual-rubric generator (S1b — its own arm).** Same state, minimally altered
   rubric → the previously correct option becomes invalid. Already validated by number
   (`hard/trap` 0.000 → 0.500 after B-11/B-12); highest data ROI — it teaches *reading the
   rubric*.
3. **Reasoning-shaped synthetic families (S1c).** `temporal_numeric`, `multi_hop`,
   `policy_trap`, `tradeoff` — text-derived labels only (ADR-0014/ADR-0015), never
   index-derived. `[review]` these families mirror the bench taxonomy — the **probes and the
   holdout eval are the external-validity gates** that separate capability from
   bench-shape memorisation (Risk R7).
4. **Variable arity + position permutation (S1c).** 2…~20 options and a many-option regime,
   permuted order. `[review]` justified as robustness/positional-debiasing (see §1.3).
5. **Hard-negative mining from the training split only** (never bench items), keeping
   `_HARD_NEGATIVE_RATE` (= 6) low (L-003: high distractor rates corrupt the term→option
   signal).
6. **Mixture discipline** — folded into S1a's curriculum; the B-20 ratios/one-set machinery
   are the starting point.

`[review]` **Expected (S1a+b+c together):** recovers `easy` toward cc ≈ 48 and lifts
`standard` toward cc 30–50 → **I ≈ 22–30**. The B-20 evidence (naive mixture: I 10.5 → 12.2,
`standard` down) is the lower bound; each factor's contribution must be attributable.

### W2 — Cross-encoder pairwise per option (L1) — *the accuracy lever*

`[review]` **Redirected: the cross-encoder is a training-time instrument; the serving path
stays single-pass.** Train a joint encoder over `[state ⊕ question ⊕ option]` (one logit per
option, softmaxed) and **distil its per-option logits into the existing `ChoiceScorer`/bank
head** with the W3 soft-target machinery. This keeps the hard constraint literally ("encoder
alone, single forward pass") and applies the plan's own principle symmetrically — a teacher is
training-time only, whether it is an LLM or a bigger encoder. The serving variant (K batched
forwards per question) remains an option **only** behind a numeric gate (p50 ≤ 150 ms, JevBench
Speed ≥ 70, arity cap) — decide at S2 with measured latency in hand.

- **Auxiliary tasks during training** (cheap, high value): an **NLI head**
  (entail/contradict/neutral over (state, option)) and a **"which option does the state
  contradict"** objective — dense gradients where answer-level signal is sparse
  (`adversarial`/`trap`).
- The seam is unchanged: `ChoiceHeadBank`/ADR-0016 shape kept; legacy assets must still load.

**Expected:** `hard` cc 0 → 20–35, `standard` +5–10.

### W3 — Soft-label distillation from the teacher (L3) — *the hedging lever*

1. **Extraction.** Per training record, ask the teacher for the **distribution over the exact
   rubric options** (`choice`) / levels (`score`) / the binary probability (`noul`) at a
   controlled temperature (T ≈ 2). The `llm` backend already returns these distributions.
   `[review]` **thinking mode on** (paired: +15 points); cost model names the model.
2. **Surgical scope (mandatory).** Only the hard/natural subset (counterfactual rubrics,
   ANLI, reasoning families, mined hard rows). `[review]` pin the pool size in the
   pre-registration (recorded pools: 50,540 en mixture + the multi recipe — the "88k" figure
   needs its source stated).
3. **Loss.** `noul_loss` already takes a `float` target; generalise `categorical_loss`'s
   `_one_hot` (`training/rlcd.py`) to accept a distribution. `[review]` simplify the blend —
   `α·Brier(one_hot) + (1−α)·Brier(teacher)` is linear in the target: fit
   `Brier(p, α·one_hot + (1−α)·p_teacher)` directly. Hard-label anchor α keeps the proper
   scoring identity (ADR-0005).
4. **Two-stage, controlled (S0 first).** Fit **only the head on the frozen trunk**
   (`training/fit_choice_bank.py`, ~10 min on an L4) with two arms differing by **exactly one
   factor** (L-012): control one-hot vs treatment soft. `[review]` the pre-registered metric
   is **accuracy and ECE on the hard/natural slice**, `score` excluded or gated separately; a
   "no" at S0 kills the *trunk* soft-label work but does **not** disprove distillation into a
   cross-encoder (W2) — the linear head is a weak probe of the thesis.
5. **Guards (unchanged — these are the two errors the project already paid for).**
   - `score` moves by construction (its answer *is* the expected value; temperature pinned at
     0.1 for that reason). **Keep `score` on hard labels in the first retrain.**
   - **Temperature must not double-apply**: the student learns `log_temperature`; serving
     applies `temperature_calibration.json`; measure ECE **as shipped** (L-015/L-016).

**Expected:** hard-tier ECE down (Calibration stable/up) + near-miss accuracy → **+3–8 I**.

### W4 — Long-context training + pooling (L4)

- **Train at 1024–2048** (the L-014 fix) with long-policy rows in the mixture. `[review]` the
  training envelope is now 2× L4 23 GB (not the 3060 of ADR-0005's era) — LoRA at 2048 is
  comfortable.
- Attention pooling / learned CLS instead of mean-pooling (L-004: mean-pool discards span
  structure).
- Re-check the fast path (CUDA-graph buckets per shape) and the ONNX export.

### W5 — Calibration / hedging stack

- Refit the B-16/B-18 `peakedness × strength` maps on the new predictions (served slices;
  never moves argmax). The B-20 attempt-3 lesson (fit-side redesign of the `noul` link) is the
  working method.
- **No improper loss** (focal) to chase hard examples — sample weights on the proper loss if
  needed.

## 4. Sequencing with kill-switches `[review: S1 factored]`

| Stage | Work | Cost | Decision gate (pre-registered in BACKLOG B-21) |
| --- | --- | --- | --- |
| **S0** | Frozen-trunk soft-vs-hard head fit (W3.4) | ~10 min L4 | soft ≥ control on hard-slice acc **and** ECE? If **no**, drop soft-label trunk work |
| **S1a** | Natural layer + curriculum (W1.1/1.6) | 1 retrain | `easy` cc recovers ≥ 40? Non-regression set green? |
| **S1b** | + counterfactual rubrics (W1.2) | 1 retrain | `hard/trap` ≥ 0.50 held/up; `standard` up vs S1a? |
| **S1c** | + reasoning families + arity (W1.3/1.4) | 1 retrain | per-family moves on the named families; probes held |
| **S2** | Cross-encoder distillation (+NLI aux) (W2) | medium + 1 retrain | `hard` cc ≥ 20? Speed gate held (p50 ≤ 150 ms) |
| **S3** | Long-context training + pooling (W4) | 1 retrain | long-policy/multi-hop up at fixed Speed budget |
| **S4** | Teacher pass + soft-label trunk retrain (W3) | GPU-h + 1 retrain | hard ECE down and I up vs S3 (paired) |
| **S5** | Calibration refit + one-set publication (W5) | hours | **all** gates green on one set; release/Hub/docs re-rendered together |

Every stage is a **one-set** change: regenerate, retrain, refit and re-render together —
never quote a pre-change number against a post-change one.

## 5. Is ≥ 50 reachable? (honest)

- **Yes on the public items if S1+S2 land** — `easy 0.86 / standard 0.76 / hard 0.53` →
  I ≈ 53. `[review]` the binding pair is `easy` + `standard`; B-20 showed `standard` can fall
  under a natural-text layer, so S1's whole job is to change that sign.
- **The stretch is `hard`** — realistic landing cc 20–35; the target weights `easy`/`standard`
  harder for that reason.
- **`judge` has no public items** — a public I = 53 is a *diagnostic*, not an official score.
- **No handoff means no free 90+** — the LLM rows (I 97.7 / 70.9 think-off) show the ceiling
  and the cost of the shortcut.

## 6. Risks

| Risk | Mitigation |
| --- | --- |
| Bench-directed data leaks a public item | Every builder asserts against `datasets/public/`; evaluation-only for ever |
| **`[review]` R7: taxonomy-shaped families read the bench without leaking items** | Probes (XNLI/MASSIVE/typed) + holdout eval are **mandatory non-regression gates** at every stage; per-family bench numbers are diagnostics, never the only gate |
| Architecture change breaks the frozen wire / legacy assets | `ChoiceHeadBank` shape and the ADR-0016 gate stay; legacy assets must load |
| Cross-encoder latency | **Distillation default** (W2 redirect); serving variant only behind the numeric Speed gate |
| Soft labels teach hedging but not reasoning | S0 kill-switch; gate accuracy and ECE separately |
| Teacher cost / quality | Surgical subset; **thinking on**; model named in the pre-registration |
| Synthetic↔natural interference (B-15) | Curriculum / replay ratio, measured on the holdout eval |
| `[review]` P3/holdout/probes regression (B-20's exact price) | The non-regression gate set (BACKLOG B-21) blocks any stage that pays for I with them |
| One-set discipline | All published numbers re-rendered together (report, model card, README, probes) |

## 7. Explicitly out of scope

- **`handoff.py` / System-2 cascade / serving-time abstention** — excluded by the owner.
- **Focal loss / improper objectives** — proper scoring is the confidence story (ADR-0005).
- **A larger backbone (3B+)** — separate programme; the levers above are the in-family path.
- **Training on JevBench items** — never.

## 8. Immediate next actions

1. **S0 (this week, ~10 min GPU):** frozen-trunk control-vs-soft-label head fit on a hard
   subset with a measured teacher (thinking on) — the cheapest experiment that de-risks the
   distillation thesis. Metric pre-registered (hard-slice acc + ECE).
2. **S1a design:** curriculum mechanism + ANLI/ContractNLI sources locked
   (`training/data/jev_sources.lock.json`), regenerate as one set, run the holdout eval.
3. Every stage: the **non-regression set** is read before any mover number is quoted.
