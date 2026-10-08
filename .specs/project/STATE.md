# State

**Last Updated:** 2026-10-07
**Current Work:** **Fases 0–2a da correção das receitas de dados — stride do sampler, eval
leave-one-template-out e expansão de conteúdo — executadas localmente 2026-10-06 na branch
`fix/sampler-stride`.** ~~Nothing leaves this machine~~ **Deferral LIFTED 2026-10-08 by
explicit decision:** the JevBench run had not finished, but its maintainer changed the
submission methodology; the owner decided to set the benchmark aside and continue TachyOne
development, so the "no `git push` / no Hub upload / no republication until JevBench runs"
rule (which applied to every correction of this cycle) is **no longer in force** — the
publication cycle proceeds as one set (promotion → `benchmarks/report.md` re-render →
`chore(release): v0.9.0` + tag → push → Hub upload with per-file sha256 → GitHub release).
The historical deferral text stays below as recorded.

**Publication executed 2026-10-08 (the one-set cycle):** P3 stack rebuilt on `en_tt` → both
promoted dirs carry B-15 (v0.8.0 dirs kept as `*_prev_pub_v080`) → multi served temperature =
pooled fit (never in-sample, L-015) → report re-rendered with six entries pinning **both**
local checkpoints (router L-013: empty/no-signal rows go to English, so the measured pair must
be the published pair) → release gates green: ruff/pyright clean, **721 passed**, `mkdocs
build --strict`, `uv sync --locked --group docs --extra serve --extra train`, e2e 2 passed /
0 skipped. **NOT met, recorded on the card**: P3 in-domain ≤ 0.05 for the rebuilt asset
(`choice` 0.5243, pooled fit pegged the grid at T=10) and B-1 per-language ECE on holdout
(both disclosed, never re-fixed). Numbers: multi routed 0.9156 / holdout 0.8355 (checkpoint
alone 0.9885), en 0.9983 / 0.9825. Details in BACKLOG **B-15**.
**Fase 0 (stride)** — option and hard-negative cycles decoupled from the language stride
(`index // len(languages)`); goldens recaptured for `en_pt`/`noisy` only — `en` is
byte-identical, so the English arm and its published numbers did not move. Regenerated
locally: the five six-language datasets (`train_multi`, `train_multi_noisy`, `eval_multi`,
`train_multi_domains`, `eval_multi_domains`). Gates on every committed recipe: 0 incomplete
`(domain, language)` cells (was 6/6 and 24/30), `other` share 0.277–0.298 per language in
`support` (was 0.527 vs 0.053), distractor ≈ 1/6 in **every** language (was 100% `pt`),
`noul` contradictions 0, volume per language unchanged. Record: **L-017**.

**Fase 1 (eval leave-one-template-out)** — `DataConfig.template_split` (`all`/`train`/
`holdout`; default `all`, every shipped dataset byte-identical) plus two holdout eval configs
→ `data/eval_{en,multi}_domains_holdout.jsonl` (7,500 rows each, regenerated locally). Gate
`test_holdout_eval_shares_almost_no_template_with_training`: **0.00%** of non-empty holdout
rows share a `(kind, template)` with the `train` split — overall and per language — against
the ~99% the seed-drawn evals carry (89.5%/61.0% identical rows, declared a ceiling in the
model card). The split also drops sentences that are hold-outs in *another* bank of the same
language: 21 rows leaked through cross-domain identical phrasing before that rule, now 0.
Current evals are untouched, and training adopts the `train` split only when the data is next
recomposed **and the arms are retrained on it** (B-14 trained on `all` — its provenance pins
`all`, so the flip waits for the next data decision instead of invalidating the run);
`benchmarks/report.md` is re-rendered as one set at publication (a holdout run against
adapters trained on all phrases would be in-sample).
**Fase 2a (conteúdo, zero GPU)** — every domain file expanded ×3: **entities 20 → 60** and
phrase banks **request/neutral/calm/urgent/distractor = 11/9/9/9/8** per language, all seven
languages, authored (`support`) and drafted+reviewed (`ecommerce`, `agent_tools`, `documents`,
`voice`) under one validator: placeholders, no phrase in two banks of a language, no
cross-domain sentence in two different tones, determiner/gender conventions per language
(one caught defect: `documents`/`nl` entities carried `de ` while its phrases add it too —
stripped before injection). Domain content feeds every record, so **all three goldens and all
eleven datasets regenerated as one set** (en included this time — recomposition, as planned).
Measured against the projections: en 21,000 rows **3,981 → 9,787 distinct (46.6%)** (`noul`
1,131 → 3,855 vs projected 3,858; `score` 1,852 → 4,839 vs 4,840), multi 30,000 rows
**10,688 → 20,447 (68.2%)** (`noul` 76%, `score` 84% — the projected 76%/84%). Full suite
**706 passed**; ruff/pyright/`mkdocs --strict` clean. Holdout sets regenerated on the new last
phrases (the positional hold-out moves with the content — by design).

**Fase 2b (volume 1,7×)** — configs raised in lockstep so the incumbent row-invariants hold:
en support 3,000 → **4,000** per primitive and `per_type` 1,000 → **2,000** (`train_en`
12,000, `train_en_domains` **36,000**); multi support 6,000 → **7,800** and `per_type` 1,000 →
**2,400** (`train_multi` 23,400, `train_multi_domains` **52,200**). Evals stay 7,500 — they
needed construction, not size (Fase 1). Sizing rule corrected against the plan: exact per-cell
balance needs `count % (languages × options) == 0` → **÷24** for four-option domains, **÷30**
for `ecommerce`, not ÷12 (the chosen numbers already satisfied it; the written rule did not).
Measured: en **13,334 distinct (37.0%)** vs projected ~13k, multi **33,551 (64.3%)** vs ~30k;
per-language `choice` volume exact **2,900** (was 1,664–1,668); `other` share 0.289–0.290 in
all six languages (the `en` reference band); B5B-02 multi incumbent green and `train_en`
support rows byte-identical to `train_en_domains`' (4,000). AGENTS examples updated. Suite
**706 passed**. Left untouched by design: `data_noisy` and legacy `data.json` (outside the
plan's table); `preds_*`/reports still describe pre-2b datasets — recomputed with the retrain.

**B-14 (recompose retrain) — complete 2026-10-07, all gates PASS on both publish arms.**
Gates were committed before any training (`d8e94fc` multi, `85a2f05` en, baselines re-fixed on
the recomposed rows). The first launches died silently mid-run — the documented **L-009**
harness reap (no traceback, no `exit=`) — and were relaunched `setsid nohup` with per-run
session, RAM/VRAM sampling and `.done` markers: multi `exit=0` at 22:35 (52,200 records, 8
epochs), en `exit=0` at 21:05 (36,000, 6 epochs); `fit_choice_bank` `exit=0` both arms; the
B5B-6 harness (predict → calibration refit → predict per arm) produced six candidate reports.
**Publish arm = fitted bank: multi 0.9863 / ECE 0.0080 / `support` 0.9413 ≥ 0.7887 / worst new
0.9913 / strict 1.000 / every language ≥ baseline (0.936–1.000 vs 0.830–0.855); en 0.9993 /
ECE 0.0003 / `support` 0.9987 ≥ 0.9733 / worst new 0.9980 / strict 1.000 / zero ECE
exceptions.** Declared: one multi ECE exception (`support` 0.0531). Findings recorded rather
than smoothed: the **en joint trunk fails the support anchor (0.8200 < 0.9733, val_loss 0.666)**
and is not the publish lineage (B5B's joint-exposure pattern, L-012 third replication via the
fitted-bank/shared tie); per-language ECE 5/6 ≤ 0.05 (`it` 0.0548) keeps **B-1 open on the
honest yardstick** — the holdout eval, which needs a `train`-split retrain to be actionable.
Full tables in BACKLOG **B-14**. Everything local: no promotion, no re-render, no upload until
JevBench runs. Next: en/multi publication cycle is deferred; open decision = Fase 3 teacher
pilot (the second L4 makes the Qwen3.5:35b Q4 fit plausible) vs the cheap B-1 competitors.
Next: pre-register the gates and run the GPU retrain, or Fase 3's teacher pilot — both need a
decision on the GPU window (5–10 h per arm).

**B-14 open defect — CLOSED 2026-10-07, replacement arm green on all five gates.** The multi
publish arm's `noul`/`support`/`it` cell was inverted (train **0.100**, `eval_multi`
**0.0723**) and blocked publication. Trail: seed-42 snapshots localized the flip (healthy
1.000 at epoch 2 → 0.793 at epoch 4 → inverted 0.100 at epoch 8); the driver test proved
choice/score gradients through the shared trunk are the driver (`noul`-only continuation
recovers to 1.000); the noul-only repair closed the cell but **failed its own pre-registered
gates** (choice 1.000 → 0.9056, score → 0.9116, per-domain ECE 0.063–0.102); the final fix is
a **joint continuation with round-robin primitive batches from the original inverted trunk**
(`checkpoints/tmp_il_orig`, 2 epochs, same loss math/split/seeds): train cell 0.0977 →
**0.9977**, `eval_multi` `noul`/`it` 0.0723 → **0.9880**, and every sibling axis *improved*
(raw overall 0.9796 → 0.9985). Fit + calibration + harness: **`support` 0.9993 ≥ 0.7887, worst
new domain 0.9987 ≥ 0.70, per-domain ECE 0.0001–0.0011 ≤ 0.05 (zero exceptions), strict 1.000,
every language ≥ baseline; overall 0.9995 / ECE 0.0004** — all five PASS. **Publish arm =
`checkpoints/tmp_il_orig` + its fitted bank; en unaffected.** Both remainders landed the same
day: the trainer's per-cell val monitor (`cell_monitor.jsonl` + worst-cell line per epoch,
commit `f57343f`) and the recipe script `training/interleave_continue.py` (commit `e9d49e4`);
GPU parity lesson **L-018**. Full trail: BACKLOG **B-14**. Everything still local.

**B-15 (Estágio 2 — template-holdout retrain) — executed 2026-10-08; the honest
generalization number exists.** Order and evidence: provenance guard first (all six B-14 pins
verified on disk *and* by regeneration; holdout evals pinned for the first time
`3119e73c6d851e31` / `982236ca2728705f`) → gates pre-registered (`32b8d71`) → flip
(`3472dc9`: the four training configs to `template_split: "train"`, volumes unchanged
12k/36k/23.4k/52.2k, new pins `991dd971`/`2fc2486a`/`f1c7d2cf`/`97d6e9ea`, two in-template
evals added with a mirror gate test, 721 passed) → retrain both arms with the per-cell
monitor (**no defect recurrence** — final `noul:*:it` cells 0.047–0.067) + the unconditional
interleaved touch-up (multi; sha `19681298…`→`f7969603…`) → fits retargeted (`092bfb3`) →
harness (raw → fit_calibration → cal) on **three evals × two arms** — every stage `exit=0`,
18 artifacts. **Verdict: current benchmark — all gates PASS** (multi **0.9885 / ECE 0.0069**,
support 0.9760, worst domain 0.9747, **zero ECE exceptions**, strict 1.000, 6/6 languages ≥
baseline; en **0.9980**, all four `85a2f05` gates PASS). **Holdout — B-1 `choice` 0.9248 ✓,
en overall 0.9821 ✓, `noul` tripwire ≥0.60 ✓ everywhere (worst `it` 0.7807), `wrong_domain`=0
with `fell_to_shared`=0; B-1 per-language ECE NOT met on multi** (choice `es`/`it`/`pt`, score
`de`/`es`/`it`/`nl`; worst `score/it` 0.1495; en passes) → recorded, **B-1 stays open on
criterion 3**. **Generalization gap (in-template → holdout): multi −0.1105, en −0.0179**;
honest holdout = **+0.127 / +0.131 over released**, −0.098 / −0.017 under the B-14 in-sample
ceiling. Publish-candidate call recorded for the publication cycle: **the B-15 arms supersede
B-14** (current holds, holdout claimable). Analysis `/tmp/opencode/b15_tables.py` (metric
defs validated against the report to 4 decimals; **L-019** for the id trap). Full tables:
BACKLOG **B-15**. Still local — no push, no upload until JevBench finishes.
**Previous:** **B-13 P3 — calibration — executed, adopted and closed 2026-10-02, its
acceptance criterion recorded as NOT met.** The design was pre-registered from measurement
(spec *P3 design*), executed as JB-9…JB-13, and the assets are now in `checkpoints/en`:
**`state_prototypes.json`** (K=32 spherical k-means centroids of the 21,000 states of
`data/train_en_domains.jsonl` — the right training basis per the 2026-10-02 direction —
encoded through the runtime loader, 406 KB) and **`confidence_calibration.json`** (the bank
embedded with the fitted `noul` map so the pair cannot drift). What changed: `noul` keeps
its own direction and takes its *magnitude* from the evidence
(`p = g(strength)` / `1 - g(strength)`, clamped to [0.5, 1]); `choice`'s temperature went
0.05 → **1.5**; `score` is **pinned at 0.1** because its answer is the distribution's
expected value (a temperature change would move Intelligence). One public run, after the
assets were frozen:

| | published | **P3** | gate |
| --- | ---: | ---: | --- |
| in-domain accuracy (7,500, every primitive/domain) | 0.999867 | **0.999867** (byte-identical) | must not move ✓ |
| in-domain ECE choice / noul / score | 0.000007 / 0.000012 / 0.000366 | **0.025886 / 0.000004 / 0.000366** | each ≤ 0.05 ✓ |
| public ECE (231) | 0.5393 | **0.2369** | ≤ 0.15 ✗ |
| **Calibration axis** | **0.0** | **52.6** | **≥ 60 ✗** |
| Intelligence / Brier / composite | 15.0 / 1.109 / 1.59 | **15.0 / 0.745 / 4.28** | invariant ✓ |

231/231, 0 failed. Two findings are recorded rather than smoothed: the design's estimate
came from **inverting a saturated transform** (natural is 0.2376, not 0.086 — **L-016**),
and **no legal fit set can see bench difficulty** (0.84 vs 0.36 at equal evidence; the one
legal slice in the bench's regime, the L-014 truncation slice, is the one the same day's
"nothing generated, basis = `train_en_domains`" direction removed). Full record: spec
*P3 results*, tasks JB-9…JB-13, BACKLOG **B-13** (acceptance box unticked).

**Sweep + publish (JB-14, same day).** Every confidence-bearing surface re-measured on the
adopted assets — accuracy byte-identical, only confidence moved: in-domain support/five-domain
ECE **0.000-in-sample → 0.006 / 0.009**; probes `ECE raw` **0.578/0.583/0.426 → 0.416/0.296/0.264**
(typed / MASSIVE / XNLI) with `Brier` better on all three; head-to-head their turf **0.569 → 0.259**
(`Conf` 0.857 → 0.535, accuracy 0.288 unchanged) and home **0.000 → 0.005** (1.000 unchanged);
fast path **2.504× / 0 flips**. Rendered `benchmarks/report.md` + `benchmarks/probes.md`, docs one
set (`benchmarks`, `compare`, `model-card`, `huggingface`, `training`, `architecture`, `roadmap`,
README, CHANGELOG). Published **`munod/tachyone-en` [`1c88ebef`](https://huggingface.co/munod/tachyone-en/commit/1c88ebef8f15f68e8a6583c564d636221f22c29f)** (8 files) and
**`munod/tachyone-multi` [`3693def1`](https://huggingface.co/munod/tachyone-multi/commit/3693def1bd04d8208216f05459c5b83388ff9fd6)**
(model card) — every file sha256-verified, no stale remote file; `mkdocs build --strict` green,
full gate green. **P4 closed the same day:** `docs/jevbench.md` (nav + README) and the issue —
**[`fstandhartinger/jevbench#182`](https://github.com/fstandhartinger/jevbench/issues/182)**, offline artifact following the `#158` pattern,
with the no-benchmark-items disclosure and the benchmark-directed-data disclosure, method pin
`bb05a335` + `METHOD-v1.5.md` sha256, body kept verbatim in
`.specs/features/jevbench/submission-issue.md`. **B-13 is complete.**

**B-13 record (same day, earlier):** P0 baseline **I 8.4 / Calibration 0.0 / composite
0.43** on the 231 public items with the wire 231/231 strict-valid; P1 hardening (long
context measured and reverted, **L-014**; serve fast-off; cost basis 510 tokens/decision);
P2 family-shaped data (35,540 records) took **I 8.4 → 15.0** (control +0.6 vs treatment
+6.0, so the gain rides on the data), and JB-8 adopted it: `checkpoints/en` promoted,
published as **`munod/tachyone-en` [`f28103bf`](https://huggingface.co/munod/tachyone-en/commit/f28103bf4a85bacd10df90c21125c84522c82993)** with the full sweep re-measured
(probes XNLI 0.566 / typed-decisions 0.367 / MASSIVE 0.051, head-to-head 0.288 / 1.000),
docs one set, `mkdocs build --strict` green, **v0.7.0** released. **Open: P4** (the
`[bench request]` issue) — and the cycle's own **I ≥ 50 stands recorded as NOT met (15.0)**.

**B-5b remains closed** — see below; no change to its published numbers.

**Previous (2026-10-01/02): Post-M6 backlog.** **B-5's structural decision (ADR-0016) executed and measured —
the isolate closes 91% of the support gap (2026-10-01).** The per-domain `choice`-head bank, its
deterministic gate, the additive `choice_head` request hint, the bank trainer and the frozen-trunk
fitter shipped (`.specs/features/choice-head-bank`, tasks B1–B7), and the isolate ran against a
**frozen run-5 trunk**: baseline **0.879 → 0.964** overall, `support` **0.964 ≥ 0.85**, worst new
domain **0.963 ≥ 0.70**, gate **strict 1.000 / 0 to shared / 0 wrong domain** over 2,500 `choice`
rows, per-domain ECE **0.021–0.027** (all five under the 0.05 target for the first time), and the
gap to the released adapter's `support` cell **0.086 → 0.008**. The number that matters most:
**the shared-head control (0.9633) sits 3 of 2,500 `choice` rows behind the full bank (0.9637)**
— the axis that closed the gap was refitting the head on B-12-corrected labels, not per-domain
capacity, so **which arm to publish (bank vs control) is open**. Full table, artifacts and
reproduction in `.specs/project/BACKLOG.md` B-5; new lesson **L-012**. Remaining at that point:
**B8** (docs + changelog) and the publish/republish decision — both closed the same day (below).

**Published (2026-10-01).** The **bank** was chosen (ADR-0016's structure; never worse than the
control, gate measured 1.000, format ready for B-5b), promoted to `checkpoints/en` with a
calibration refitted on `eval_en`, and uploaded as **`munod/tachyone-en` commit `c00c174d`** (six
files sha256-verified, stale `finetune_config.json` deleted) plus **`munod/tachyone-multi`
`d3f64cc0`** (model card). `benchmarks/report.md` (three entries incl. the gate row),
`docs/benchmarks.md`, `docs/model-card.md`, `README.md`, `docs/roadmap.md`,
`docs/huggingface.md` §3 and `CHANGELOG.md` carry the new set; the fast path was re-checked
(**0 flips**). The support trade is published, not summarized: **0.972 → 0.964** overall with
`choice` **0.946 → 1.000** and `noul`/`score` **0.992/0.978 → 0.946/0.946**, while the five-domain
split goes **0.511 → 0.964** (worst domain 0.328 → 0.963). Latency columns were re-measured on
this L4 box and are labelled box-bound (the previous adapter measures 51.8 ms beside the new
54.4 ms on the same path). **Then open, now closed:** only the GitHub release tag — the public
probes and the head-to-head `choice` cells were re-run against the new revision the same day
(both recorded below), and **`v0.5.0`** was tagged, pushed and published with them.
**B-5b scheduled (2026-10-01) and measured through the isolate (2026-10-02):** spec/design/tasks
at `.specs/features/multilingual-five-domains/` with four locked decisions (context.md):
bank keyed by **domain** (five keys, signatures in the six training languages), **baseline-first
gates** (released multi measured on the new eval set fixes the numbers before training), **30k
volume** (support's 18k kept, not the 9k en mirror that would halve the incumbent), and the
**full B-5a publication pattern**. Landed: lexicons localized (3 review passes, 14k-combination
audit, `19de6b9`), 30k/7.5k datasets with support halves byte-identical (`367aa88`), the
`per_domain_language` cross cell (`ea24e73`), baseline + gates fixed (`9d94818`), joint run +
fit config (`cc74225`). **Two lessons recorded: L-013** (the published multi 0.743 is a routed
number — 13–15% of multilingual rows fall through to `tachyone-en`, whose 2026-10-01 republish
moved it to 0.736; the explicit harness gives 0.8413) and the **joint bank deficit** (bank
loses 5.6 points to its own shared head in joint training — optimization exposure, not
structure). **Verdict (B5B-7): every gate passes on the fitted bank** — overall **0.9975**,
support **1.0000**, worst new domain **0.993**, per-domain ECE **0.0005–0.0038 (zero
exceptions)**, gate strict **1.000**, unseen text 0.9957; fitted bank vs fitted shared tie at
**1 row** (L-012 replicated). **Published (B5B-8, 2026-10-02): `munod/tachyone-multi` revision
[`9a3ef5a5`](https://huggingface.co/munod/tachyone-multi/commit/9a3ef5a5d38972ffe119cdfe6b4b6c7bb14da9d1)**
— six files sha256-verified, stale `finetune_config.json` deleted (the upload fought back twice:
a fine-grained token without Write, then PR-only scope — the `pass create_pr=1` hint named it).
One consistent set across report (4 entries, reproduction commands fixed), `docs/benchmarks.md`,
model card, README, `huggingface.md`, roadmap, CHANGELOG, `training.md`; sweep: **MASSIVE 0.011
→ 0.039** (every language above chance; `ECE raw` 0.354 → 0.518 — sharp off-domain), fast path
**3.66× / 0 flips**, head-to-head **not** re-run (both tables English-only → untouched
`tachyone-en`). **Remaining: B5B-9** (docs site strict build + cycle close + release decision).

**Probes re-run (2026-10-01).** The three public probes were re-measured against `c00c174d` and
`benchmarks/probes.md` re-rendered (previous artifacts kept as `probe_*_preb5.json`):
typed-decisions **0.330 → 0.269** (best config 0.412 → 0.344), XNLI **0.333 → 0.341**, MASSIVE
flat at **0.011** (chance 0.017). Accuracy is flat and near chance; what moved is **confidence**
off-domain — `Conf` 0.399 → 0.811, 0.354 → **0.991**, 0.021 → 0.365, so `ECE raw` worsens to
0.650 on XNLI while every fit still pins T=20.0. The B-5 head is sharp in domain and confident out
of it; `docs/benchmarks.md` § Public probes and the CHANGELOG both say so.

**Head-to-head re-scored (2026-10-01).** `docs/compare.md` §3 now carries B-5 numbers for
Tachyone with the peers and the gold labels untouched (artifacts `compare_tachyone_b5.json` and
`compare_home_tachyone_b5.json`, rendered as `compare_table_{a,b}_b5.md`): their turf **0.236 →
0.233** (`ECE raw` 0.337 → 0.561, `Conf` 0.268 → 0.355 — confident off-domain), our turf **0.974 →
0.958** with `choice` **1.000** and `noul`/`score` 1.000/0.984 → 0.938/0.938, per-task cells and
method rows updated. **Released (2026-10-01): `v0.5.0`** — cut from `[Unreleased]`, versions
bumped (`pyproject` / `uv.lock` / `__init__`), lock checked, full gates green **including e2e**
(544 passed), tag pushed to `origin` (remote resolves `v0.5.0 → 7b2658e9`), all five CI checks
green (lint/types/tests · docs build · offline install · **GitHub Pages deployed**) and the GitHub
release published as *Latest* with the CHANGELOG section as its notes. **Nothing open in this
cycle;** the next items are `B-5b` (multilingual bank keys) and `B-1` (multilingual `choice`).

**Previous (2026-09-30):** Post-M6 backlog. **B-12 (`score`/`choice` labels from the text) done and the
published set re-chosen by merit (2026-09-29, ADR-0015).** The `index % 13` "near-tie" (7.8% of
`score` rows, capping the primitive at 0.888 where the released checkpoints sat) and the
index-derived labels of the **empty boundary state** (5.4% of *every* primitive — `score` kept the
tone drawn before the state was emptied, `choice` kept `options[index % 4]`) are gone: `score`
takes the tone's level with empty → middle level, `choice` takes the option the state names with
empty → catch-all, and `ecommerce` gained the `other` option every other domain shipped. Seven
datasets regenerated (golden hashes recaptured, `noul` untouched), one acceptance test per
primitive across five domains × seven languages, and — because `finetune_en_domains.json` still
carried **run 6's** recipe instead of the designated run 5 — **all three checkpoints retrained in
one cycle** (the wrong-recipe attempt collapsed at 0.338; config restored, lesson **L-011**).
The cycle's finding, measured three times: once the near-tie noise is gone, `noul` and `score`
become the *same tone detector* (both → **1.000**) and the shared trunk starves `choice` — English
**control** of the identical recipe 0.841, five-domain retrain `choice` **0.303** with `support`
**0.763 ✗**. So the published set is the **best measured checkpoint per adapter**:

| checkpoint | published weights | on B-12 labels | on B-11 labels | on pre-B-11 labels |
| --- | --- | ---: | ---: | ---: |
| English | B-11 weights | **0.972** (`choice` 0.946, `score` 0.978) | 0.945 | 0.859 |
| Multilingual | **B-12 retrain**, 8 epochs | **0.743** (`choice` 0.468) | 0.718 | 0.853 |
| five-domain run 5 | B-11 weights | **0.879** — support **0.886 ✓**, worst new **0.815 ✓** | 0.861 / 0.807 ✓ | 0.785 ✗ / 0.739 ✓ |

Everything else re-run on that set: fast path **2.65×** (8.97 → 3.39 ms, parity 0.000559,
0 flips), probes 0.330 / **0.011** / 0.333, head-to-head table A **0.236** (peers untouched) and
table B **0.974** at home with all four engines re-scored on the B-12 gold labels; every
`noul` audit reports **0 contradictory rows**. Both adapters were republished to the Hub
(**`796e6899`** / **`5f688052`**, every file sha256-verified against the local build) and
`docs/huggingface.md`, the model card, README, landing page, `benchmarks/report.md`,
`benchmarks/probes.md`, `docs/benchmarks.md` and `docs/compare.md` §3 all carry the new set.
**B-5's two cheap options — relabel (B-11) and
retrain (B-12) — are now measured and closed**; the structural decision (per-domain adapters /
two-stage / scoring rule) is what remains, against a baseline of 0.972.

**2026-09-30 — structural decision: `ADR-0016` (Accepted).** The chosen family is
**per-domain `choice` heads + a deterministic gate** (caller hint → lexical signature → shared
head fallback), trained in the existing run, with the asset `choice_head.json` growing to
`{shared, domains}` and the legacy format loading as shared-only. The scoring-rule change is
the sequenced fallback (it would move `choice` *and* `score`, i.e. the whole surface);
two-stage training and per-domain LoRA adapters stay rejected per the ADR's alternatives
section. First experiment decided: fit only the head bank on the frozen run-5 trunk to isolate
the head axis before any joint retrain. **Execution deferred — not started 2026-09-30**; B-5's
cycle gates get their numbers in `BACKLOG.md` when the cycle is scheduled.

**Previous cycle — B-11 (`noul` labels from the text), 2026-09-28, ADR-0014.** `_noul_record`
no longer takes its label from the loop index: the rule
is now `target = 1` iff the state was drawn from the `request` phrase bank (`neutral`/empty → 0),
asserted against each state's own phrase bank across all five domains × seven languages **and**
against the shipped eval sets. Only `noul.target` bytes moved (states, questions, `choice`/`score`
byte-identical), so the seven datasets were regenerated and the golden hashes recaptured.
`training/evaluate.py` now emits `report["noul_per_language"]` beside `report["noul_labels"]` —
per-language `noul` accuracy next to the contradictory-label rate — and `benchmarks/report.py`
renders both in one table. Measured outcome of the label fix, published as one set:

| | pre-B-11 labels | on the corrected labels (old weights) | published (retrained) |
| --- | ---: | ---: | ---: |
| English | 0.859 (`noul` 0.718) | 0.935 (`noul` 0.946) | **0.945** (`noul` **0.992**) |
| Multilingual | 0.853 (`noul` 0.960) | 0.714 (`noul` 0.614) | **0.718** (`noul` **0.832**, 8 epochs) |

The English row is the fix proving itself (the old weights already read the text); the
multilingual row needed a four-run sweep — a control (0.663 vs 0.657) showed the 4-epoch drop was
systematic, 8 epochs recovered `score` (0.648 → 0.854) and won on overall **and** ECE (0.042 vs
0.076), and `choice_rank` 128 overfit (train loss 0.085, eval 0.490). Its residual cost, `choice`
0.468 against 0.674 for the old weights with `choice` labels never having moved, is published
rather than hidden — and the external probes see it (MASSIVE 0.033 → **0.013**, below its 0.017
chance; typed-decisions 0.323 → 0.330; XNLI 0.333). Also re-measured on the new adapters: fast
path **2.72×** (9.19 → 3.37 ms, parity 0.000771, 0 flips), the head-to-head (table A Tachyone only
0.229 → **0.241**; table B **all four engines re-scored** because the gold labels changed —
Tachyone **0.953** at home vs peer 0.599), and **B-5a's run 5, which now passes both gates**
(support 0.785 → **0.861 ≥ 0.85**, worst new domain 0.739 → **0.807 ≥ 0.70**; `choice`/`score`
unchanged, so only `noul` moved 0.720 → 0.946). Both adapters were republished to the Hub
(`224c8a74` / `b7747756`, sha256-verified file by file); CHANGELOG records it under
`[Unreleased]`; new lesson **L-010** (a constant label is a shortcut the model wins with) and
backlog item **B-12** (the analogous `score` near-tie label) — which the cycle above closed the
next day.
**Everything below this paragraph was measured on pre-B-11 labels** (and the tables above on
pre-B-12 ones) and is kept as the record of
how each milestone was measured at the time; the current numbers are the three cells above and the
tables they link to.
**B-2 (fast path) done**: `TACHYONE_FAST` wires
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
**B-5b stays blocked** behind it (**resolved 2026-10-01**: the bank was chosen and published,
B-5b is unblocked — see ADR-0016 and BACKLOG B-5). **B-6** remains an Idea.
**Re-measured on the B-11 labels (2026-09-28): both gates pass** — support **0.861 ✓** (0.785 ✗
before), worst new domain **0.807 ✓** (`voice`), overall 0.804 → **0.880**, because only `noul`
moved (0.720 → 0.946) while `choice` 0.789 and `score` 0.905 are literally the same measurement
(`benchmarks/results/en_domains_r5_postb11.json`). Two facts reframe the structural decision: the
released support-only adapter now scores **0.945** on `support`, so run 5 clears the *absolute*
gate while sitting 0.084 below the released one; and run 5 itself was **trained on pre-B-11
labels**, so retraining it on corrected data is the cheap untried option that comes before any of
the three (BACKLOG B-5 carries this).

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

> **Resolved by B-11 (2026-09-28)** — the label now comes from the emitted text (ADR-0014), the
> audit that would have caught this ships in `training/evaluate.py`, and the per-language spread it
> explained (`nl` vs `es`) is gone from the label side. The follow-on finding is **L-010**.

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

### L-011: The config file is not the artifact's recipe — the artifact is

**Context:** B-12. Retraining the designated B-5a artifact (run 5: `choice_rank` 128, 6 epochs)
from `training/configs/finetune_en_domains.json`.
**Problem:** the config still carried **run 6's** settings (`choice_rank` 256, 8 epochs) — the
last experiment of the series had edited it in place and nothing tied the file to the artifact it
was supposed to produce. The retrain therefore ran a different, known-bad experiment and
**collapsed outright**: identical per-epoch losses from epoch 6 to 8 (0.7572/0.7498 = constant
output), eval at chance across all five domains, 0.338 overall — an hour and fifty minutes of GPU
before the numbers said "wrong recipe". Nothing in the config said which run it described.
**Solution:** before retraining an artifact, read `<artifact>/finetune_config.json` — the trainer
writes exactly what it ran (hyper-parameters, seed, data path, record counts) and it is the ground
truth for provenance; reconcile it with the config file, and when they disagree, fix the config
(`aedb064`) so the file documents the artifact it produces. A collapsed run is diagnosed by the
*shape* of the losses (identical across epochs), not by the final `val_loss` alone.
**Prevents:** an hour-plus of GPU training the wrong experiment, and a published config that has
silently stopped describing the published checkpoint.

---

### L-010: A constant label is a shortcut the model wins with, and the spread it leaves looks like capability

**Context:** B-11. The pre-B-11 `noul` rule was `target = 1 iff even index and request tone`, and
the eval sets interleave language by `index % len(languages)` — so in `eval_multi.jsonl` the odd
languages (`es`, `de`, `nl`) got **only label 0**, while the even ones (`pt`, `fr`, `it`) came out
tone-consistent by accident. The published multilingual `noul` accuracy was **0.960** and the
per-language spread (`nl` 0.663 vs `es` 0.956) read like a capability gap.
**Problem:** measured against the corrected labels, the *same* weights score `noul` **0.614**
(`de` 0.373, `nl` 0.217) — the multilingual model had learned language → label, not text → label.
The English weights, trained on the same parity noise but with a single language (no shortcut to
find), scored **0.946** on the corrected labels: they had to read the text. A high accuracy bought
by a constant label is indistinguishable from a well-trained model until the labels are fixed.
**Solution:** derive the label from the emitted text (ADR-0014); reconstruct the *old* rule over
the regenerated states — the states are byte-identical, so the before side is recoverable — and
publish both sides as one set; and audit the label distribution **per language** before believing
a per-language spread (`training/evaluate.py` now emits `noul_labels` beside `noul_per_language`).
**Prevents:** shipping a per-language "capability gap" that is a dataset artifact, and quoting an
accuracy that a constant label pays for.

---

### L-012: Arms that share the fixed factor cannot attribute the gain to it

**Context:** ADR-0016's first experiment — fit *only* the `choice` heads on a frozen run-5 trunk,
with a shared-head **control** arm beside the per-domain bank, to isolate the head-capacity /
gradient-share axis before any joint retrain.
**Problem:** both arms beat the baseline by **+0.085 overall** while differing from *each other*
by **3 of 2,500 `choice` rows** (0.9633 vs 0.9637). The design isolated the *structure* correctly
— control vs treatment is the estimate, and it says ≈ nothing — but both arms also re-learned the
head on B-12-corrected labels, which the baseline's head never saw (it was trained jointly with
the trunk *before* B-12). So the large delta is shared by both arms and cannot be credited to the
structural hypothesis that motivated the run: quoting "per-domain heads closed the gap" would have
been false, and only the control arm makes that claim falsifiable.
**Solution:** make the control differ from the treatment by **exactly one factor**, read
*treatment vs control* as that factor's effect, and report *treatment vs baseline* separately as
what the whole intervention bought. When the baseline also differs in data vintage or optimization
history, say so in the same table (L-006 / L-008 discipline: name what moved).
**Prevents:** publishing a structural ADR's mechanism as proven when the measurement only proves
the intervention worked.

---

### L-013: A routed evaluation measures the other checkpoint too

**Context:** B5B-4, fixing the B-5b gates. The released multilingual adapter had to be
measured on the new five-domain eval set, and the obvious anchor was the published overall
**0.743**.
**Problem:** two shipped harnesses disagree by **10.5 points** on identical rows and adapter:
`training.evaluate --backend encoder` (the runtime path) gives **0.736**, `training.predict`
(explicit adapter, the B-5a harness) gives **0.8413**. The runtime routes each record by
detected language, and on multilingual rows **13–15% fall through to `tachyone-en`** — empty
boundary states (5.4%, `detect_language` → `None`), short and loanword-heavy states detected
as `en`. Those rows are answered by the English checkpoint, whose weights changed on
2026-10-01 (the B-5 republish), which is why 0.743 no longer reproduces as 0.736: the number
was always `85% multi + 15% whatever-en-is`, and the second factor silently moved.
**Solution:** gate a checkpoint's quality only on an explicit-adapter harness (same
instrument for baseline and arms), and treat routed numbers as *product-level* measurements
that mix checkpoints by design. When a published number refuses to reproduce, diff the two
harnesses record-by-record before blaming the weights, and record which checkpoint answered
which rows.
**Prevents:** publishing a gate the runtime does not deliver, "losing" 0.7 points to someone
else's republish, and reading a routing artifact as a capability change.

---

### L-014: Seeing the input is not the same as being able to use it

**Context:** B-13 P1b. The JevBench hard tier's states average 1,079 tokens (max 3,746)
while English inference truncated at 512, so the obvious structural read was "the deciding
facts of a long policy never reach the encoder".
**Problem:** raising the runtime context to 4096 and re-running all 231 public items moved
**one** item right and **one** item wrong — 82/231 both runs, Intelligence 8.4 → 8.4 —
while raw p95 went 333 ms → 2,700 ms and the Speed axis 84.8 → 77.6. The extra visibility
bought nothing because a mean-pooled cosine scorer trained at ≤512 has learned no behavior
for *finding* a decisive clause among 4,096 tokens: the constraint was what the model can
do with text, not how much of it it sees. The intuition was plausible, cheap to test, and
wrong — which is why it was tested before it was shipped.
**Solution:** A/B a structural lever on the real measurement before keeping it, tie
inference length to *training* length, and record the negative result together with the
cost it carried (the speed points) rather than reverting silently.
**Prevents:** shipping a "reads long documents now" change with zero accuracy behind it,
paying a latency tax on every p95 for an unmeasured intuition, and confusing input access
with capability.

### L-015: A temperature fitted where the model is saturated zeroes the axis off-domain

**Context:** B-13 JB-7. The per-arm calibration refit (the B-5 pattern: `fit_calibration` on the
arm's own `eval_en_domains` predictions) ran where both fresh arms score 0.9999 — i.e. on data the
model has essentially memorised (the known train/eval row collision).
**Problem:** the fit landed at **T=0.05–0.1** (sharpening: in-domain ECE → 0.000012), and the
*same shipped asset* pushed off-domain ECE to ≥ 0.5 on the public items — **Calibration axis 0 for
every arm, including the published one** (ECE 0.543). A first pass had read ctrl 65.3 / treat 48.4
because those two serves ran **without** `temperature_calibration.json` (the silent, documented
fallback = T=1): two arms measured with the asset, two without — an instrumentation inconsistency
that looked like a calibration difference. Intelligence never moved between passes (temperature
cannot change argmax — the invariant that exposed the inconsistency).
**Solution:** measure every arm **as shipped** — the asset belongs to the instrument, not just the
adapter (L-013 applied to calibration); fit temperatures on data where the model is *not*
saturated, or make the temperature depend on input plausibility (the P3 shrinkage that B-13 keeps
deferring to); when an asset can legitimately be absent, publish both numbers instead of one.
**Prevents:** quoting a Calibration axis bought by an in-sample fit, reading an absent asset as
"better calibrated", and comparing arms whose serving configuration differs.

---

### L-016: Inverting a saturating transform measures the quantization, not the signal

**Context:** B-13 P3 design (2026-10-02). To sweep temperatures without re-running the
server, the pre-temperature scores were reconstructed from the *shipped* probabilities:
`scores = T_asset * log(p_shipped)` with `T_asset = 0.05`.
**Problem:** at `T=0.05` the shipped probabilities are already saturated (0.99…, often
`1 - 1e-9` or exactly `1.0` after JSON), so `log(p)` is either rounding noise or a clamped
floor and the recovered "natural" distribution came out nearly uniform. The sweep then
reported "public choice ECE 0.069, overall 0.086 at T=1", and the whole P3 design — the
mechanism, the fit protocol, the acceptance estimate — was pre-registered on it. The real
measurement (de-tempered from an *unsaturated* T=1.5 output) is **0.2376 natural vs 0.2369
fitted**: the apparent 3x headroom never existed, and the ≤ 0.15 gate was unreachable with
that plan. Caught only because the gate was finally measured for real (JB-13).
**Solution:** measure a distribution where it is produced (`training.predict` with no
`--temperature`, or a serve with the asset absent *and recorded as absent*, L-015's rule),
never invert a transform whose output has already been quantized — and when a reconstruction
is unavoidable, prove the round trip first: re-apply the known temperature to the
reconstruction and compare it with the source before quoting any number derived from it.
**Prevents:** committing a design to an artifact of float precision, and quoting headroom
that lives only in the far end of a saturating transform.

---

### L-017: Two cycles over one loop index are aliased by gcd — and the eval is built from the same sampler

**Context:** data audit 2026-10-05 (external review) + Fase 0 of the data-recipe plan,
executed 2026-10-06 on branch `fix/sampler-stride` (local only — see *Current Work*).
**Problem:** `language = languages[index % len(languages)]` already owns `index % L`, so a
second stride over the same `index` is aliased by `gcd`. `option = options[index % len(options)]`
with six languages and four options (`gcd = 2`) left **every** multilingual `(domain, language)`
cell training only a subset of the options as labels — `de`/`es`/`nl` never saw
`billing`/`sales` in `support`, `pt`/`fr`/`it` never saw `technical` (24/30 incomplete cells in
`data_multi_domains` *and* in its eval). The measured prior: `other` was 52.7% of `support`
`choice` rows in `de`/`es`/`nl` against 5.3% in `pt`/`fr`/`it` (eval majority baseline 0.47 vs
0.095). The eval could not see it — 0 of 2,500 eval `choice` rows named a label that cell never
trained — because it is drawn by the same sampler with the same strides. The sibling nobody had
checked: `_HARD_NEGATIVE_RATE = 6 == len(languages)` put **every** distractor row of every
six-language recipe in `pt` alone (94.7% of `pt`'s `choice` rows, 0% of the other five): five
languages never trained distractor rejection, and the one that did carried 6× the intended
ambiguity rate on the clause the generator keeps rare on purpose.
**Solution:** option and hard-negative cycles run on `index // len(languages)` — the record's
position inside its own language's stream — while language keeps `index % len(languages)` and
the label still comes from the emitted text (B-12/ADR-0014: the stride decides which text is
emitted, never what it is called). One-language configs keep `block == index`, so the English
arm, its datasets and its published numbers are byte-identical. Post-fix gates on every
committed recipe: 0 incomplete cells, `other` 0.277–0.298 per language, distractor ≈ 1/6 per
language, `noul` contradictions 0, per-language volume unchanged; pinned by
`test_every_language_covers_every_option` and
`test_hard_negative_distractors_reach_every_language`.
**Prevents:** assuming a second cyclic assignment is independent of the first — and assuming
the eval would have caught it. Any future stride is checked against `gcd(stride,
len(languages))`; an eval drawn by the same sampler can only expose a defect the sampler
hides symmetrically, which is why Fase 1 (leave-one-template-out eval) precedes every
measurement that matters.

---

### L-018: Two identical GPU runs are not the same experiment — parity is behavioral, never bitwise

**Context:** B-14 defect closure (2026-10-07): the interleaved-continuation script that fixed
the `support/it` cell existed only in `/tmp`; hard rule 6 moved it into
`training/interleave_continue.py`, and the refactored module had to be proven to reproduce the
original experiment.
**Problem:** the recipe claims "same loss math, same split, same shuffle seeds" — so the
re-run should match `checkpoints/tmp_il_orig` epoch 8 (`noul=0.0144 choice=0.0260
score=0.0312`). It did not: the module's run gave `choice=0.0058`, adapter weights off by max
**Δ0.0360** from the original. A normalized diff of the setup, `encode` and `batch_loss`
showed zero logic difference — and a second run *of the same module* differed from the first
by max **Δ0.0279** (losses `choice=0.0058` vs `0.0072`): the run-to-run floor of the L4 with
bf16 autocast accounts for the original gap. Two scripts cannot be compared bitwise on this
hardware; only a behavioral comparison (loss scale, cell probe, gates) means anything.
**Solution:** parity for `training/interleave_continue.py` is asserted as (a) a mechanical
diff of the loss math, (b) identical resolved hyperparameters (CLI > the trunk's
`finetune_config.json` > PEFT's `adapter_config.json` — never the directory name: the
`multi`-substring trap), (c) identical seeds and batch order (unit-tested),
(d) the sha256 before/after guard proving training happened, and (e) behavioral agreement
within the observed nondeterminism band. A trained arm's provenance is the artifact's own
hashes plus the gates, never the run log.
**Prevents:** burning hours chasing a "logic bug" that is GPU noise; claiming bitwise
reproducibility of a training run; comparing two training runs by loss deltas smaller than
the measured nondeterminism floor (≈Δ0.03 in adapter weights after one epoch here).

---

### L-019: An eval id is not a row key — domain cells come from the report, never an id join

**Context:** B-15 analysis (2026-10-08); the same trap had already produced a wrong answer
during B-14's defect trail.
**Problem:** `eval_*_domains.jsonl` carries **1,500 unique ids × 5 rows, one per domain** —
the id names the draw, not the row. Joining predictions back to `domain` by `id` keeps
whichever domain came last and silently fabricates data: the B-15 `support × language` axis
came out all zeros, and during B-14 the same collision produced "the `it` `noul` eval rows
are all `voice`" (false — `it` has 83 `noul` rows in *every* domain). Neither output crashed;
both looked plausible, and only a cross-check against report cells exposed them.
**Solution:** the calibrated report already ships `per_domain_language` (domain × language
accuracy + ECE) — that is the source for domain cells. Primitive × language cells need only
the `type` and `lang` fields the predictions do carry, and every derived metric was validated
by mirroring `training/evaluate.py`'s definitions and reproducing the report's `per_language`
and `noul_per_language` to four decimals *before* any new cell was trusted.
**Prevents:** plausible-looking zeroed or collapsed aggregates from key collisions — check a
key's cardinality (`rows per id`) before joining anything to it, and cross-check every
derived cell against an existing report cell before publishing it.

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
| B-11 | `noul` labels from the emitted text: generator rule + acceptance tests + label audit, 7 datasets regenerated, both adapters retrained, whole measured surface republished (Hub `224c8a74` / `b7747756`) | 2026-09-28 | `fix(training): derive the noul label from the emitted text (B-11)` | ✅ |
| B-12 | `score`/`choice` labels from the emitted text (near-tie removed, empty-state defaults, `ecommerce` gains `other`), all three checkpoints retrained, published set re-chosen by merit (EN 0.972 / multi 0.743 / run 5 0.879 both gates) | 2026-09-29 | `fix(training): derive score and choice labels from the text too (B-12)` | ✅ |
| B-8 | calibration assets that cannot be loaded now warn by name instead of degrading silently | 2026-09-27 | `fix(encoder): warn when a calibration asset cannot be loaded` | ✅ |
| B-7 | public probes: `benchmarks/probes.py` loaders for typed-decisions/MASSIVE/XNLI + `benchmarks/probes.md` | 2026-09-27 | `feat(benchmarks): add the public probe loaders and report` | ✅ |
| R-T1..T3 | multilingual LoRA rank 16 → 64 (accuracy 0.702 → 0.853, `es` ECE 0.170 → 0.038) | 2026-09-25 | `chore(training): raise multilingual LoRA rank to 64` | ✅ |
| B1..B7 | ADR-0016 choice-head bank: keyed asset + gate, `choice_head` hint, bank training, frozen-trunk fitter, gate/oracle report, four-arm isolate (support 0.886 → **0.964**, gate strict **1.000**) | 2026-10-01 | see `.specs/features/choice-head-bank/tasks.md` | ✅ |

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
**B-6** contrastive pre-fine-tuning (Idea). **B-10** was added 2026-09-26, **B-11** (the `noul`
label rule) closed **Done** on 2026-09-28 and **B-12** (the `score`/`choice` label rule, added
2026-09-28) closed **Done** on 2026-09-29 after it swallowed the empty-boundary states and
`ecommerce`'s missing catch-all too.
`BACKLOG.md` also
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

Verified authenticated as `munod` (2026-09-27); used it to refresh the model cards for `v0.4.0`,
to publish the **B-11 revision** of both adapters (2026-09-28, commits `224c8a74` / `b7747756`)
and the **B-12 revision** (2026-09-29, commits `796e6899` / `5f688052`) — each time every uploaded
file was sha256-verified against the local build.
Never echo it, never commit it (rule 4).
