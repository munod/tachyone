# JevBench Preparation (B-13) — Tasks

**Spec:** `.specs/features/jevbench/spec.md` (P0/P1 results live there)
**Status:** P0 + P1 done; **P2 in progress**

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

**Deferred (recorded, not scheduled):** `probability`, `ambiguous`, `tradeoff`,
`adversarial`, `routing_hard` synthetic families (public n ≤ 10 each); source data for
`easy`-tier only if JB-7 shows easy < 0.95 after the sources layer; a longer training
`max_len` paired with a re-run of the P1b context A/B (L-014).
