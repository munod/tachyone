# JevBench Preparation (B-13) — Tasks

**Spec:** `.specs/features/jevbench/spec.md` (P0/P1 results live there; the P3 design is
pre-registered in *P3 design*)
**Status:** P0 + P1 + P2 done; **P3 executed (JB-9 … JB-13) — adopted, its acceptance
criterion recorded as NOT met (52.6 < 60); JB-14 (sweep + publish) in progress**; P4 open

Commit-per-task, tests co-located (AGENTS.md hard rule 6), full gate before each commit:
`uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -m "not e2e"`.

Hard constraints for every task below (spec *Locked decisions*):
**the 231 public JevBench items are evaluation-only — never training or calibration
input**; English only; labels derived from text/executed rules, never from a loop index
(B-11/B-12/L-008 discipline).

---

## JB-1: Fetch the public sources with pinned revisions

**What:** `training/fetch_jev_sources.py` + `training/data/jev_sources.lock.json` — pinned
URLs/ revisions and sha256 for the three reviewed sources (**MultiNLI, BoolQ, Banking77**;
AG News `unknown` and SST-5 `unspecified` licences are **excluded**, recorded with the
reason). Downloads land in `data/sources/` (gitignored); the script verifies hashes and is
idempotent (refuses to clobber).
**Where:** `training/fetch_jev_sources.py`, `training/data/jev_sources.lock.json`.
**Depends on:** — · **Requirement:** P2 layer (a).
**Done when:** all three files cached + sha256-verified from a cold start; licences and
their source URLs recorded in the lock (tev1 `DATA_SOURCES.md` cross-checked).
**Tests:** `tests/test_training_jev_sources.py` (lock schema, hash format, excluded
sources documented; download itself network-gated/skipped).
**Gate:** full · **Commit:** `feat(training): fetch the JevBench-family sources with pinned hashes`
**Status:** **Done (2026-10-02), commit `bb1b113`.** Pinned revisions cross-checked against
tev1; all three files sha256-recorded (`1c1de036…`, `4f028e99…`, `3c648a31…`), 208 MB
cached under `data/sources/`; AG News and SST-5 excluded with their licence words
verbatim. 17 tests, `file://` downloads, `--check`/`--record` flows covered.

## JB-2: Convert the sources into Tachyone records

**What:** `training/build_jev_sources.py` → `data/train_jev_sources.jsonl`:
MultiNLI → `choice` {entailment, neutral, contradiction} (premise+hypothesis state),
BoolQ → `noul` (passage+question state, target from the answer), Banking77 → `choice`
over the 77 intents with authored one-line option descriptions (committed, not fetched).
Every record: `lang: "en"`, `source: <dataset>@<revision>`, `id` prefixed, per-record RNG
independent of any label (L-003).
**Where:** `training/build_jev_sources.py`, `training/data/jev_banking77_descriptions.json`.
**Depends on:** JB-1 · **Requirement:** P2 layer (a).
**Done when:** counts fixed (5,000 / 3,000 / 3,000), all records validate against the wire
primitives, spot-checked read-aloud (B5B-1 lesson), splits do not touch any public
JevBench item (asserted by normalized-text overlap check against the three public files).
**Tests:** `tests/test_training_jev_sources.py` (counts, shape, overlap assertion,
determinism) · **Gate:** full.
**Commit:** `feat(training): build Tachyone records from the pinned public sources`
**Status:** **Done (2026-10-02), commit `9901b6e`.** 11,000 records (MNLI 5k / BoolQ 3k /
Banking77 3k, seeded reservoir), MNLI state+instruction mirror `benchmarks.probes`' XNLI
mapping verbatim (single source of truth), Banking77 descriptions authored (77, label
order), **zero normalized-text overlap with the 231 public items** (build ran with
`--public-dir`), read-aloud spot-check green. One real bug caught by the tests: the pinned
card stores **one pair per row** (scalars), not two — `mnli_pairs` now accepts both
layouts.

## JB-3: Executable rule trees (pure python)

**What:** `training/rule_trees.py` — small rule-tree IR + executor: facts (dates,
durations, amounts, entity links, clause truth) → tree of `all/any/not/compare/lookup`
nodes → label. The generated record's `target` **is** the tree's execution result over the
emitted facts (never an index); the tree ships inside the record for audit (tev1 pattern).
**Where:** `training/rule_trees.py`.
**Depends on:** — · **Requirement:** P2 layer (b); B-11/B-12 label discipline.
**Done when:** executor is deterministic, raises on unbound facts, and round-trips
(tree → label) for the four family shapes (policy clause chain, multi-hop chain,
temporal/numeric predicate, trap-with-distractor).
**Tests:** `tests/test_training_rule_trees.py` (≥10 cases incl. negation, sublimit,
2-hop chain, off-by-one date boundary, trap distractor) · **Gate:** full.
**Commit:** `feat(training): add executable rule trees for family labels`
**Status:** **Done (2026-10-02), commit `60da2e2`.** Ops `const/fact/all/any/not/cmp/select`
with strict failures (unbound fact, mixed kinds, unknown op/comparison), `as_target`
validates choice outputs against the option set, `reexecute(record)` is the audit hook. 22
tests: every comparison op, the 30-day inclusive boundary, leap-year dates, multi-hop
chains, and the trap shape (the distractor is a fact but never a tree node). One test bug
found by the gate itself (a "boundary" date that was inside the window) — fixed and
amended before push.

## JB-4: Committed family content + generators

**What:** committed English content under `training/data/jev/` for the five targeted
families — `long_policy`, `trap` (shares policy machinery), `multi_hop`,
`temporal_numeric`, `adequacy` (judge-tier proxy) — clause banks, entity/relation pools,
date/amount pools, adequacy rubric pairs, option sets in the bench's vocabulary shape
(`pay_subject_to_N_sublimit`-style keys with readable descriptions).
`training/build_jev_families.py` composes content + rule trees into records
(~500–1,000 per family, seeded per record, states with controlled length including
**≤512-token states** — inference truncates at 512, L-014: keep decisive text visible).
**Where:** `training/data/jev/*.json`, `training/build_jev_families.py`.
**Depends on:** JB-3 · **Requirement:** P2 layer (b).
**Done when:** every generated record's target equals an independent re-execution of its
own tree; all five families produce their count; a read-aloud sample per family passes
(B5B-1: read the records, not the tables).
**Tests:** `tests/test_training_jev_families.py` (tree-label identity over the whole
generated set, counts, determinism, ≤512-token decisive states) · **Gate:** full.
**Commit:** `feat(training): generate the JevBench-family records from rule trees`
**Status:** **Done (2026-10-02), commit `a4ad754`.** 3,540 records (long_policy 800 /
trap 640 / multi_hop 700 / temporal_numeric 700 / adequacy 700), **every target
re-executes from its shipped facts+tree** (tested over the whole set), state window
enforced by char cap (1,800; measured 3.87 chars/token) **and** by the real tokenizer
(≤512 tokens, 3,540 states), determinism hash-tested, **zero public-item collisions**
(the overlap check caught a question I had copied verbatim from a bench trap item —
rewritten). Label distributions audited per label space (L-010): four skew bugs found and
fixed during the build — routing bands now uniform (85/70/81/78/86), `decline` scenario
added, duty-desk reachable only via the override, adequacy label drawn first (48/52).
Read-aloud found and fixed: coverage titles,9.4-before-9.3 numbering, seat article,
claim-ref = cert-ref.

## JB-5: Data configs + the mixed training set

**What:** `training/configs/data_jev_en.json` (sources + families, seed pinned) →
`data/train_jev_en.jsonl`; document the mixture in the config and spec. Counts and
composition recorded.
**Where:** `training/configs/`, `data/` (gitignored).
**Depends on:** JB-2, JB-4 · **Requirement:** reproducibility (L-011: config = recipe).
**Done when:** the file regenerates from committed inputs; mixture table recorded in the
spec.
**Tests:** co-located regeneration test · **Gate:** full.
**Commit:** `feat(training): add the JevBench-family mixture config`
**Status:** **Done (2026-10-02), commit `2c2d433`.** `training/build_jev_mixture.py` +
`training/configs/jev_mixture.json` → **35,540 records** (domains 21,000 + sources
11,000 + families 3,540; choice 16,780 / noul 11,760 / score 7,000). Cross-part id
uniqueness enforced (ids may repeat *inside* the incumbent domain part — per-type
counters — which the tests pin explicitly); 8 tests.

## JB-6: Two-arm training + full candidate pipeline

**What:** `training/configs/finetune_en_jev_ctrl.json` (21k `train_en_domains`, the run-5
recipe) and `finetune_en_jev.json` (same recipe + `train_jev_en` mixed, same seed/epochs —
arms differ **only** in the added data). Launch per L-009 (`setsid nohup`, `exit=$?`,
marker); reconcile `finetune_config.json` (L-011) before anything quotes it. Then the
published construction for each arm: `fit_choice_bank` on the frozen trunk + calibration
refit → candidate `checkpoints/en_jev{,_ctrl}`.
**Where:** `training/configs/`, run logs in `/tmp/opencode/`.
**Depends on:** JB-5 · **Requirement:** L-012 attribution (control arm exists so the gain
can be credited to the data and not to run variance).
**Done when:** both arms complete without collapse (L-011 loss-shape check) and each
artifact carries its recipe.
**Tests:** existing trainer tests (bank writer path) · **Gate:** full.
**Commit:** `feat(training): add the two-arm JevBench-family training configs`
**Status:** **In progress (2026-10-02).** Configs written and committed (`ffed949`):
`finetune_en_jev_ctrl.json` (run-5 recipe, `train_en_domains`, out
`checkpoints/en_jev_ctrl`) vs `finetune_en_jev.json` (identical — same seed 2, same
epochs 6, same LoRA/head ranks — only `data_path`/`out_dir` differ) plus the two fit
configs (`fit_bank_en_jev*.json`, lr 1e-4 / 8 epochs per the B-5a lesson). Both dry-runs
green (21,000 vs 35,540; committed-domain validation passes). **Both arms trained
concurrently and finished `exit=0`, 6/6 epochs, no L-011 collapse signature** (control
`choice` loss 1.154 → 0.0047, treatment 0.527 → 0.0695 — the mixture's harder data keeps
its final loss higher, expected). Artifacts reconcile with their launch configs (the only
`finetune_config.json` delta is the materialised `choice_init_std` default), each carries
the keyed bank (`shared` + `domains`) plus `temperature.json`. **Frozen-trunk fits
launched** (`b13_fit.sh`, L-009,15 s sampler): control on GPU0 (7,000 `choice` records),
treatment on GPU1 (16,780), logs `/tmp/opencode/b13_fit_{ctrl,treat}.log`.

## JB-7: Gates — bench diagnostic + in-domain non-regression

**What:** serve each arm explicitly (L-013), run the 231 public items through the jevbench
harness, compute the four axes; **and** `training.predict` on `eval_en_domains` (B-5 gates
must not regress: support ≥ 0.85, worst new domain ≥ 0.70, published reference 0.964).
Verdict: candidate vs published (P0, already measured) vs fresh control. Targets recorded
in the spec: easy ≥ 0.95, standard ≥ 0.73, **I ≥ 50**, ECE not worse than P0's 0.543.
**Where:** `/tmp/opencode/jevbench_runs_*`, `.specs/`.
**Depends on:** JB-6 · **Requirement:** B-13 acceptance.
**Done when:** the candidate table (published / control / treatment) is one comparable set
and the verdict is explicit.
**Tests:** — (measurement) · **Gate:** full.
**Commit:** `docs(specs): record the B-13 P2 measurement`
**Status:** **Done (2026-10-02).** One comparable set, explicit adapter per arm (L-013),
calibration refit per arm, bench = the 231 public items:

| arm | in-domain overall | gates B-5 | bench score | Intelligence | Calibration | Speed | Cost |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| published (P0 reference) | 0.9637 | all PASS | 0.43 | 8.4 | 0.0 | 85.0 | 78.8 |
| control (fresh run-5) | 0.9999 | all PASS | 0.51 | 9.0 | 0.0 | 84.6 | 78.8 |
| **treatment (mixture)** | **0.9999** | **all PASS** | **1.59** | **15.0** | 0.0 | 84.3 | 78.8 |

Attribution (L-006/L-012): the control arm — identical recipe, one factor apart — moves
Intelligence **+0.6**; the treatment moves **+6.0**, so the gain rides on the data.
Biggest movers: `standard/ordinal` 0.25 → 0.667, `score` 0.222 → 0.611, `hard/multi_hop`
0.056 → 0.333, `easy/intent` 0.583 → 0.750, `hard/adversarial` 0.333 → 0.667.
**Calibration is 0 for every arm as shipped (L-015):** the per-arm refit landed at
T=0.05–0.1 (sharpening on saturated in-domain data, ECE → 0.000012 at home) and the same
asset puts off-domain ECE ≥ 0.5 — an earlier pass measured ctrl 65.3 / treat 48.4 only
because those serves ran **without** the asset (silent fallback = T=1), which is an
instrumentation inconsistency, not a calibration difference; all three bench passes were
re-run as shipped. Argmax never moved (Intelligence identical across passes — temperature
cannot change it). P3 owns the real fix.
**Cycle target NOT met: I ≥ 50 landed at 15.0** (multiplier 0.09). Structural leftovers
recorded honestly: `noul` stays rubric-blind (0.473 → 0.487), so `hard/trap` is still
0.000 and `standard/policy` ≈ chance; `hard/temporal_numeric` regressed (0.267 → 0.133).
Both B-5 gates pass on all three arms with zero ECE exceptions.

## JB-8: Adopt, publish-or-keep, and close P2

**What:** promote the better arm only if it beats the published artifact on the bench
diagnostic **without** breaking the B-5 gates; regenerate the benchmark surfaces it moves
(`benchmarks/report.md` probes row re-run if weights change); BACKLOG/STATE/spec final
pass; CHANGELOG `[Unreleased]`.
**Where:** `checkpoints/`, `benchmarks/`, `.specs/`, `CHANGELOG.md`.
**Depends on:** JB-7 · **Requirement:** B-13 acceptance (I ≥ 50 or an explicit record of
what stopped it).
**Done when:** every number-bearing doc quotes one set; feature status updated.
**Tests:** `tests/test_docs_site.py` · **Gate:** full.
**Commit:** `feat(release): adopt the JevBench-family checkpoint` + `docs: …`
**Status:** **Done (2026-10-02) — adopted by merit.** Both adoption conditions held (bench
**1.59 vs 0.43** as-shipped with all B-5 gates PASS; treatment beats its fresh control 1.59
vs 0.51). Executed: `checkpoints/en_jev_bank` → **`checkpoints/en`** (previous build kept as
`checkpoints/en_prev_pub0.9637`), the per-arm refit shipped as `temperature_calibration.json`
(L-015 — every bench pass re-run *as shipped*). Measurement sweep on the new weights, one set
everywhere: support view **1.000** (noisy 0.997), five-domain **1.000**, gate strict
**1.000**; probes **XNLI 0.566 / typed-decisions 0.367 / MASSIVE 0.051** (all re-run;
`benchmarks/probes.md` re-rendered without leaking backups); fast path **2.543× / 0 flips**;
head-to-head **re-run, not merely re-checked** (the English adapter moved): their turf
**0.233 → 0.288** (`banking77` 0.000 → 0.219 — the real Banking77 data shows up), home turf
**0.958 → 1.000** (all three primitives), peers' artifacts byte-identical. Published:
**`munod/tachyone-en` revision [`f28103bf`](https://huggingface.co/munod/tachyone-en/commit/f28103bf4a85bacd10df90c21125c84522c82993)**
— six files **sha256-verified one by one**, no stale remote files. Docs one set:
`benchmarks/report.md` (entries + reproduce commands), `docs/benchmarks.md`,
`docs/model-card.md`, `docs/compare.md` §3, `docs/huggingface.md`, README, roadmap,
CHANGELOG `[Unreleased]`; `mkdocs build --strict` green. **Still open:** P3 (calibration
shrinkage — the Calibration axis reads 0 for every arm as shipped, L-015) and P4 (the
`[bench request]` issue itself).

---

## P3 — Calibration (JB-9 … JB-14)

Hard constraints (spec *Locked decisions* + *P3 design*): **the 231 public items and the
308 sealed items are never a fit input**; every answer (choice argmax, noul direction,
score expected value) must come out **byte-identical** — this phase moves confidence only;
the fit protocol is pre-registered in the spec and the public run is measured **once**
after the assets are frozen.

### JB-9: The legal calibration holdout

**What:** `training/build_calibration_holdout.py` + `training/configs/calibration_holdout.json`
→ `data/calibration_holdout.jsonl` with four slices, each recorded in the run report:
(a) fresh in-domain eval (new seed, never trained on), (b) **source remainder** — MultiNLI +
BoolQ train rows disjoint from `data/train_jev_sources.jsonl` (id *and* normalized-state
asserted), (c) fresh families (seed 7, existing builder), (d) **truncation slice** — (c)
behind a per-record document preamble so the decisive clause falls past the 512-token window.
**Where:** `training/`, `data/` (gitignored), `tests/test_training_calibration_holdout.py`.
**Depends on:** — · **Requirement:** P3 fit input.
**Done when:** all four slices present with counts + accuracy/strength recorded; **zero**
overlap with any trained record and **zero** normalized-text overlap with the 231 public
items (the existing `--public-dir` assertion reused); composition documented in the spec.
**Tests:** slice counts, disjointness assertions, determinism, overlap guard · **Gate:** full.
**Commit:** `feat(training): build the P3 calibration holdout from legal slices`
**Status:** **Done (2026-10-02), commit `570ab1b`.** Two slices, not the four planned: the
2026-10-02 direction fixed the basis on `data/train_en_domains.jsonl` and ruled out the
generated slices, so the anchor is a stride sample of the real training set (600 per
primitive, ids `cal-trn-*`) and the off-domain slice is 600 MultiNLI + 600 BoolQ train rows
the trainer never saw — **296 rows dropped as already trained, counted rather than
re-sampled**. Byte-deterministic; zero public-item overlap asserted on both. The
families/truncation slices built during exploration were deleted with that direction, and
JB-13 records what their absence cost.

### JB-10: The prototype bank asset

**What:** `training/build_prototypes.py` → `checkpoints/en/state_prototypes.json`: K=32
k-means centroids of the checkpoint's **own training state embeddings** (encoded with the
adapter loaded — same space as inference), unit-normalized, seeded, with provenance (data
file, record count, model, seed, K). Built once per checkpoint and rebuilt whenever the
checkpoint changes (documented next to the asset).
**Where:** `training/build_prototypes.py`, `checkpoints/en/state_prototypes.json`,
`tests/test_training_prototypes.py`.
**Depends on:** JB-9 (nothing) · **Requirement:** the `noul` strength signal.
**Done when:** deterministic across runs (hash), centroids unit-norm within 1e-6, and the
bank reproduces the measured separation (in-domain strength ≥ 0.95, public ≤ 0.95 — the
*AUC 1.000* result from the spec) on a 100-row sample.
**Tests:** determinism, shape/norm, provenance fields, separation smoke test (small sample)
· **Gate:** full. **Commit:** `feat(training): build the training-state prototype bank`
**Status:** **Done (2026-10-02), commits `9882951` + `18bdc27`.** K=32 over the **21,000**
states of `train_en_domains` encoded through the runtime loader: `state_prototypes.json`,
dim 1024, 406 KB, basis sha256 `2f90fe14…`. The first build burned **30 minutes of CPU in
pure Python k-means**, so the fit was vectorized (numpy fast path, Python fallback kept for
the base install, the two paths asserted equal to 1e-9) and the seeding moved from uniform
— which landed two starts in one cluster at seed 0 — to farthest-point after one seeded
draw.

### JB-11: Runtime — evidence-conditioned `noul` confidence

**What:** `src/tachyone/backends/encoder.py` gains `load_state_prototypes()` (same
absent-asset-silent / corrupt-asset-warn discipline as `load_temperatures`, B-8) and
`EncoderModel` gains the prototype bank + `confidence map`: per request compute
`strength = max_k cos(state_embedding, centroid_k)` once, then for `noul`
`p = g(strength)` (yes) / `1 − g(strength)` (no) — **direction from the answer, magnitude
from the evidence**. `choice`/`score` keep the temperature path; a missing map or missing
bank falls back to today's behaviour exactly.
**Where:** `src/tachyone/backends/encoder.py`, `src/tachyone/calibration.py` (the map
lookup), `tests/test_encoder.py`.
**Depends on:** JB-10 · **Requirement:** P3 mechanism.
**Done when:** the direction is provably unchanged for every input (test over edge cases:
p just above/below 0.5, strength outside the knot range → clamped, empty/degenerate
distribution), absent assets are silent, corrupt assets warn (B-8).
**Tests:** co-located in `tests/test_encoder.py` (≥6 cases) · **Gate:** full —
`tests/test_contract_wire.py` **must stay untouched** (no wire change ⇒ hard rule 1 idle).
**Commit:** `feat(backends): condition noul confidence on the training-state evidence`
**Status:** **Done (2026-10-02), commit `9d31c3e`.** `ConfidenceCalibration` (bank + map,
one file, B-8 discipline: absent silent, unusable warns by name and degrades), 14
validation rejections, `prototype_strength` shared by runtime and fitter, and the invariant
tested over edge states: **the noul direction never moves**, `choice`/`score` answers come
out byte-identical, a bank of the wrong width is refused. `tests/test_contract_wire.py`
untouched ✓.

### JB-12: The fitter

**What:** `training/predict.py` emits `strength` per row when `--prototypes` is given;
`training/fit_confidence.py` reads the natural predictions of JB-9 and writes
`checkpoints/en/confidence_calibration.json`: per primitive `{choice: temperature}` from
`fit_temperature` (DEFAULT_GRID, pooled rows), `{score: pinned 0.1 + reason}`, and
`{noul: g}` — accuracy per strength bin, monotone-enforced, piecewise-linear knots.
**Where:** `training/predict.py`, `training/fit_confidence.py`,
`tests/test_training_fit_confidence.py`.
**Depends on:** JB-9, JB-10 · **Requirement:** pre-registered fit protocol.
**Done when:** the report records ECE before/after per slice (in-domain, remainder,
families, truncation) and per primitive, the fitted `choice` `T` and the `noul` knots are
written with their provenance; `score` is pinned with the EV reason in the file itself.
**Tests:** monotone map, clamping, grid fit on a synthetic fixture, score pin honoured
· **Gate:** full. **Commit:** `feat(training): fit confidence on the legal P3 holdout`
**Status:** **Done (2026-10-02), commits `31fb754` + `2c2d5ae`.** Fitted on the 3,000-row
holdout (11 min including predictions): **choice T 0.05 → 1.5**, **noul T → 0.25** (kept
only as the no-map fallback), **score pinned at 0.1** with the reason in the file, and the
noul map as 10 knots from strength 0.410 → 0.9966 (confidence 0.567 → 1.0, monotone).
Holdout diagnostics, both mechanisms: **noul map 0.0076 vs noul temperature 0.0467**,
choice 0.0193, score 7e-06. Provenance carries the sha256 of the predictions and of the
bank, and the bank is embedded in the asset so the pair cannot drift.

### JB-13: Gates — one public measurement

**What:** (a) in-domain: `training.predict` on `eval_en_domains` with the new assets —
**accuracy must be byte-identical** to the published report and ECE per primitive ≤ 0.05;
(b) holdout ECE per slice (diagnostic); (c) serve `checkpoints/en` explicitly (L-013) and
run the 231 public items **once**: `calibration(ece)` ≥ 60 (ECE ≤ 0.15), Intelligence still
**15.0**, 231/231 strict-valid.
**Where:** `/tmp/opencode/jevbench_runs_p3`, `benchmarks/results/`, `.specs/`.
**Depends on:** JB-12 · **Requirement:** P3 acceptance.
**Done when:** one comparable table (published / P3) with all four axes and both ECE
surfaces, and an explicit verdict (adopt or record the miss).
**Tests:** — (measurement) · **Gate:** full.
**Commit:** `docs(specs): record the P3 calibration measurement`
**Status:** **Done (2026-10-02) — the in-domain gate PASSES, the P3 gate does NOT.**

| surface | published (as shipped) | **P3** | gate |
| --- | ---: | ---: | --- |
| in-domain accuracy (`eval_en_domains`, n=7,500) | 0.999867 | **0.999867** (byte-identical, every primitive and domain) | must not move ✓ |
| in-domain ECE choice / noul / score | 0.000007 / 0.000012 / 0.000366 | **0.025886 / 0.000004 / 0.000366** | each ≤ 0.05 ✓ |
| public ECE (231 items) | 0.5393 | **0.2369** | ≤ 0.15 ✗ |
| **`calibration(ece)`** | **0.0** | **52.6** | **≥ 60 ✗** |
| Intelligence | 15.0 | **15.0** | unchanged ✓ |
| Speed / Cost | 84.3 / 78.8 | 84.9 / 78.8 | — |
| **composite** | 1.59 | **4.28** | — |

Wire 231/231, **0 failed**; Brier 1.109 → **0.745**; `mkdocs` untouched by the run.
Contribution to the 0.2369: **`choice` 0.1421** (conf 0.553 vs acc 0.360), **`noul` 0.0953**
(map plateau 0.777 vs acc 0.486), **`score` 0.0201** (pinned — right call: natural score is
worse, 0.392). Two findings recorded rather than smoothed over:

1. **The design's own estimate was wrong, and why.** The pre-registration quoted "T=1 gives
   public ECE 0.086" from a *reconstruction* of the pre-temperature distribution out of the
   shipped `T=0.05` probabilities — precision is destroyed exactly there, so the simulated
   "natural" distribution was nearly uniform. Measured properly from the P3 run, the natural
   distribution gives **0.2376**, i.e. the fitted assets are as good as natural (52.6 vs
   52.5), not 3× better. Lesson **L-016**.
2. **The holdout cannot see bench difficulty.** At the same evidence level the model scores
   **0.84** on never-trained MultiNLI and **0.36** on the bench, so no function of our
   features separates "hard for us" from "easy off-domain" — the `noul` plateau (0.777) and
   `choice`'s T (1.5, pooled optimum of a holdout whose own accuracy is 0.806) are both
   *correct for the data we may legally fit on*. The one legal slice that does sit in the
   bench's regime (the L-014 truncation slice: accuracy 0.379, strength 0.715) is the one
   the 2026-10-02 direction removed. Reaching ≤ 0.15 needs either bench-difficulty legal
   data or a capability fix (P2's leftover), not a different fit.
**Verdict: adopt** — Intelligence is invariant, home is now honestly calibrated instead of
in-sample-sharpened, Brier halves and the board composite goes 1.59 → 4.28; the recorded
acceptance (≥ 60) stands **NOT met**.

### JB-14: Sweep, publish, close P3

**What:** every confidence-bearing surface re-measured on the adopted assets — in-domain
report, probes (`Conf`/`ECE`/`Brier` move, accuracy must not), fast path (0 flips),
head-to-head (ECE columns) — then one consistent set across `benchmarks/report.md`,
`docs/benchmarks.md`, `docs/model-card.md`, `docs/compare.md`, `docs/huggingface.md`
(the two new asset files), README, roadmap, CHANGELOG `[Unreleased]`; Hub revision with
every file sha256-verified; `mkdocs build --strict`; BACKLOG/STATE/spec closed for P3;
new lesson recorded if the measurements earned one.
**Where:** `benchmarks/`, `docs/`, `CHANGELOG.md`, `.specs/`, Hub.
**Depends on:** JB-13 · **Requirement:** hard rule 5 (one set everywhere).
**Done when:** no number-bearing doc quotes a pre-P3 confidence, the P3 acceptance box is
ticked, and P4 is the only open phase.
**Tests:** `tests/test_docs_site.py` · **Gate:** full.
**Commit:** `docs: publish the P3 calibration set`
**Status:** **Done (2026-10-02).** One set everywhere, every number re-measured on the
adopted assets (accuracy byte-identical, only confidence moved):

| surface | pre-P3 | **P3** |
| --- | ---: | ---: |
| in-domain support / five-domain ECE | 0.000 / 0.000 (in-sample) | **0.006 / 0.009** |
| probes `ECE raw` (typed / MASSIVE / XNLI) | 0.578 / 0.583 / 0.426 | **0.416 / 0.296 / 0.264** |
| head-to-head their turf `ECE raw` / `Conf` | 0.569 / 0.857 | **0.259 / 0.535** (accuracy 0.288 unchanged) |
| head-to-head home `ECE raw` / `Conf` | 0.000 / 1.000 | **0.005 / 0.995** (accuracy 1.000 unchanged) |
| fast path | 2.543× / 0 flips | **2.504× / 0 flips** |

Rendered `benchmarks/report.md` (English entries now quote `en_p3.json` + the re-run support
split) and `benchmarks/probes.md`; `docs/{benchmarks,compare,model-card,huggingface,training,
architecture,roadmap}.md`, README and CHANGELOG carry the same set; `training/package_hf.py`
ships the two new assets (tested). Published: **`munod/tachyone-en` [`1c88ebef`](https://huggingface.co/munod/tachyone-en/commit/1c88ebef8f15f68e8a6583c564d636221f22c29f) (8 files)** and
**`munod/tachyone-multi` [`3693def1`](https://huggingface.co/munod/tachyone-multi/commit/3693def1bd04d8208216f05459c5b83388ff9fd6)** (model card) — **every file sha256-verified
against the local build, no stale remote file**; `mkdocs build --strict` green; full gate
green (696 passed).

**Deferred (recorded, not scheduled):** `probability`, `ambiguous`, `tradeoff`,
`adversarial`, `routing_hard` synthetic families (public n ≤ 10 each); source data for
`easy`-tier only if JB-7 shows easy < 0.95 after the sources layer; a longer training
`max_len` paired with a re-run of the P1b context A/B (L-014).
