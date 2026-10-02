# Multilingual Five-Domain Coverage (B-5b) — Tasks

**Spec:** `.specs/features/multilingual-five-domains/spec.md` · **Design:** `design.md`
**Status:** Not started

Commit-per-task, tests co-located (AGENTS.md hard rule 6), full gate before each commit:
`uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -m "not e2e"`.

---

## B5B-1: Localize the four new domain lexicons

**What:** Add `languages.{pt,es,fr,de,it,nl}` tables to `training/data/domains/{ecommerce,
agent_tools,documents,voice}.json`, mirroring each domain's `en` table (entities ≥ 8, 3
instructions, 4 levels, noul_criteria, option_terms ≥ 4 per option, option_descriptions ≥ 20
chars, 5 phrase tones with `{entity}`/`{distractor}` placeholders preserved). English tables
byte-untouched. Strengthen `test_every_committed_domain_file_is_complete` to iterate
`DEFAULT_LANGUAGES` instead of `en` only.
**Where:** `training/data/domains/*.json`, `tests/test_training_generate.py`.
**Depends on:** — · **Requirement:** B5B-01, context.md (keying: localized signatures).
**Done when:** completeness green for all 5 domains × 7 languages; all existing golden /
config-vs-dataset tests still pass (English byte-identity); one generated record per
(domain, language) read aloud as a localization spot-check.
**Tests:** `tests/test_training_generate.py` (extended completeness) · **Gate:** full.
**Commit:** `feat(training): localize the four new domain lexicons to the training languages`
**Status:** **Done (2026-10-01), commit `19de6b9`.** Four domain files × 6 languages authored
(2.9k inserted lines, `en` byte-preserved in all four, verified against `git show HEAD:`).
Three review passes were needed: the first `voice` pass skipped read-aloud review and shipped
visible agreement errors in all five languages (`el luces`, `du serrure`, `das Kamera`,
`del scena luci`) — a dedicated revision applied the gender-normalization strategy (voice);
a final consolidated pass rendered the **full template × filler cross product (14,082
combinations, 0 flagged)** for `de`/`it`/`pt` after live spot-checks caught `meinem
Rückerstattung`, `del mio fattura` and `da teste unitário`. Lesson: **read the generated
records, not just the tables** — a table that parses and passes counts can still produce
ungrammatical sentences in 40–50% of composed records. The residual (singular frames vs
plural fillers, e.g. `This payment terms needs attention` in `en` itself) is the shipped
parity class and is declared here. Completeness test now iterates `DEFAULT_LANGUAGES`.

## B5B-2: Multi five-domain dataset configs + byte-identity tests

**What:** `training/configs/data_multi_domains.json` (domains ×5, langs ×6,
`per_domain {support: 6000}`, `per_type 1000`, seed 1) and `data_eval_multi_domains.json`
(`per_type 500`, seed 2); generate `data/train_multi_domains.jsonl` (30,000) and
`data/eval_multi_domains.jsonl` (7,500). Tests: support halves ≡ `train_multi`/`eval_multi`
modulo the `domain` key, counts, six-language interleave, English datasets untouched
(existing config-vs-dataset suite).
**Where:** `training/configs/`, `tests/test_training_generate.py`.
**Depends on:** B5B-1 · **Requirement:** B5B-02.
**Done when:** both datasets reproduce from their configs and the two identity invariants hold.
**Tests:** `tests/test_training_generate.py` (identity + counts) · **Gate:** full.
**Commit:** `feat(training): add the multilingual five-domain datasets`
**Status:** **Done (2026-10-01), commit `367aa88`.** Train 30,000 (support 18,000 +
4 × 3,000) and eval 7,500 (1,500 per domain); both support halves equal the incumbent
`train_multi`/`eval_multi` **byte-for-byte modulo `domain`** (asserted); all 90
(domain, primitive, language) slices present (min 166 rows); `_SHIPPED` hash reproduction
covers both configs.

## B5B-3: `per_domain_language` cross cell in the report `[P]`

**What:** `training/evaluate.py` emits `report["per_domain_language"]` (same `_metrics`
bucketing as `per_domain`/`per_language`, keyed `(domain, lang)`); `benchmarks/report.py`
renders the cross table worst-cell-first. Legacy no-domain reports unchanged.
**Where:** `training/evaluate.py`, `benchmarks/report.py`.
**Depends on:** — · **Requirement:** B5B-05.
**Done when:** a domain×language report carries the cell beside `per_domain`/`per_language`; a
legacy report is byte-identical to before.
**Tests:** `tests/test_training_evaluate.py`, `tests/test_benchmark_report.py` · **Gate:** full.
**Commit:** `feat(evaluate): report accuracy and ECE per domain and language`
**Status:** **Done (2026-10-01), commit `ea24e73`.** `report["per_domain_language"]` keys
`domain/lang`, sorted, emitted **only when the set carries more than one domain** (legacy
reports byte-identical — covered by tests); the renderer prints the cross table worst-cell
first with a `Worst cell (accuracy)` line.

## B5B-4: Baseline measurement — fix the gates before training

**What:** Evaluate the released `checkpoints/multi` (the `d3f64cc0` build) on
`data/eval_multi_domains.jsonl` through `training/predict` — overall / per-domain /
per-primitive / per-language / cross cell, calibration as shipped. Write
`benchmarks/results/multi_domains_baseline.json`; record the fixed gate numbers (support ≥
cell, worst new domain ≥ 0.70, per-domain ECE ≤ 0.05 w/ exceptions) in BACKLOG **B-5b** before
any trained arm exists. The support cell is checked against `eval_multi`'s rows (identity
proven by B5B-2's test), **not** against the published 0.743 — L-013: that is a routed number.
**Where:** `benchmarks/results/multi_domains_baseline.json`, `.specs/project/BACKLOG.md`.
**Depends on:** B5B-2 (eval set) · **Requirement:** B5B-04 (gates fixed first).
**Done when:** the baseline JSON exists and BACKLOG quotes the gate numbers sourced from it.
**Tests:** — (measurement) · **Gate:** full.
**Commit:** `docs(specs): fix the B-5b gates from the released-multilingual baseline`
**Status:** **Done (2026-10-01).** Baseline: overall **0.5609**, `support` **0.8413** (the
no-regression anchor), worst new domain `agent_tools` **0.4380**, worst cell `agent_tools/nl`
**0.3012**, seen/unseen 0.595/0.510. Gates fixed in BACKLOG B-5b before training. The
anchor check exposed **L-013**: the published 0.743 comes from the routed harness, where
13–15% of multilingual rows fall through to `tachyone-en` — controls
(`/tmp/opencode/control_eval_multi.json` 0.8413 vs `control_eval_backend.json` 0.736) pinned
the 0.743 → 0.736 drift on the 2026-10-01 English republish.

## B5B-5: Training config + the joint run

**What:** `training/configs/finetune_multi_domains.json` exactly as design.md §3 (mmBERT,
lora 64/128, `choice_rank 128`, `epochs 8`, seed 42, lr 1e-4, `data_path` → `train_multi_domains`,
`out_dir checkpoints/multi_b5b`). `--dry-run` validates config + data; then launch per L-009
(`setsid nohup … < /dev/null &`, `exit=$?` + marker in the log, RAM/VRAM sampled from inside).
Reconcile `<artifact>/finetune_config.json` with the launch config before anything quotes it.
**Where:** `training/configs/finetune_multi_domains.json`, run log in `/tmp/opencode/`.
**Depends on:** B5B-2 · **Requirement:** B5B-03, B5B-07, design.md §3.
**Done when:** checkpoint dir carries adapter + keyed `choice_head.json` (five domain keys,
six-language signatures) + `temperature.json` + matching `finetune_config.json`.
**Tests:** `tests/test_training_finetune.py` (bank writer path, already covered) · **Gate:** full.
**Commit:** `feat(training): add the multilingual five-domain training config`

## B5B-6: Evaluate the three arms in one harness

**What:** Same command shape, calibration refit per arm, `--train-data` held-out split:
(1) **bank + gate** headline, (2) **oracle** `--head-hint <domain>`, (3) **shared-only** derived
asset (`domains` stripped → legacy). Publish `choice_gate`, `per_domain`, `per_language`,
`per_domain_language`, `text_seen` splits →
`benchmarks/results/multi_domains_{bank_gate,bank_oracle,bank_shared}.json` + preds.
**Where:** `benchmarks/results/`, `data/preds_multi_domains_*.jsonl`.
**Depends on:** B5B-3, B5B-5 (asset), B5B-4 (baseline already in the same harness).
**Requirement:** B5B-04.
**Done when:** three arms + baseline are one comparable set; L-012 attribution note drafted
with the table (what moved, what is shared).
**Tests:** — (measurement) · **Gate:** full.
**Commit:** `docs(specs): record the B-5b measurement in STATE and BACKLOG B-5b`
**Status:** **Done (2026-10-02).** Three arms + baseline in one harness (calibration refit per
arm, held-out split). All cycle gates PASS on the bank+gate arm (support 0.9453 ≥ 0.8413,
worst new domain 0.8467 ≥ 0.70, gate strict 1.0000; two declared ECE exceptions). The
**shared-only control beats the bank in every domain** (overall 0.9295 vs 0.8732, `choice`
0.7884 vs 0.6196) — first joint bank run ever measured; trainer math re-verified (each head
standalone, no path bug). Full table: BACKLOG B-5b.

## B5B-6b: The isolate on the frozen trained trunk (decides the publish arm)

**What:** `training/configs/fit_bank_multi_domains.json` (`trunk_adapter checkpoints/multi_b5b`,
`data_path data/train_multi_domains.jsonl`, model mmBERT, `max_len 1024`, `choice_rank 128`,
seed 42, lr 1e-4) → fit shared-refit control + bank from **one cached encode pass** over the
10k `choice` records, then evaluate both fitted assets through `training.predict` (same
harness, same calibration flow) against the joint arms already recorded.
**Where:** `training/configs/fit_bank_multi_domains.json`, `benchmarks/results/multi_domains_fit_*.json`.
**Depends on:** B5B-6 · **Requirement:** ADR-0016 §5, L-012 (attribute structure under equal
optimization).
**Done when:** the candidate table has all five arms (baseline, joint bank, joint shared,
fitted bank, fitted shared) and BACKLOG names the publish decision.
**Tests:** — (measurement) · **Gate:** full.
**Commit:** `docs(specs): record the B-5b isolate and the publish decision`
**Status:** **Done (2026-10-02).** Fitted bank **0.9975** / fitted shared 0.9976 (1-row tie —
L-012 replicates: structure neutral under equal optimization; the joint bank's deficit was
optimization exposure). Publish arm decided: **fitted bank** — all gates pass with zero ECE
exceptions, gate strict 1.000, unseen-text 0.9957. Artifacts:
`benchmarks/results/multi_domains_fit_{bank,ctrl}.json`, config
`training/configs/fit_bank_multi_domains.json`, loop `/tmp/opencode/b5b_fit{,_eval}.sh`.

## B5B-7: Gates verdict

**What:** Read the arms against the B5B-4 gate numbers: support ≥ baseline cell, worst new
domain ≥ 0.70, per-domain ECE ≤ 0.05 (exceptions declared), gate strict published. Record
verdict, full table and artifacts in BACKLOG B-5b + STATE (Current Work); update the spec's
Goals checkboxes. A failed gate routes to diagnosis (per L-006: control before tuning) — no
silent re-runs.
**Where:** `.specs/project/BACKLOG.md`, `.specs/project/STATE.md`, feature `spec.md`.
**Depends on:** B5B-6 · **Requirement:** B5B-04.
**Done when:** STATE/BACKLOG quote only same-harness numbers and the verdict is explicit.
**Tests:** — · **Gate:** full.
**Commit:** `docs(specs): record the B-5b verdict`

## B5B-8: Publish `munod/tachyone-multi` + the measurement sweep

**What:** Promote `checkpoints/multi_b5b` → `checkpoints/multi` (previous build kept
recoverable); upload Hub revision with per-file sha256 verification + stale-file deletion;
regenerate `benchmarks/report.md` (new entry incl. gate row), `docs/benchmarks.md`,
`docs/model-card.md`, README, `docs/huggingface.md` §3; re-run MASSIVE probe and fast-path
parity on the multi path (0 flips expected); re-run the head-to-head only if a quoted cell uses
the multilingual engine; CHANGELOG `[Unreleased]`.
**Where:** Hub, `benchmarks/`, `docs/`, `README.md`, `CHANGELOG.md`.
**Depends on:** B5B-7 (only a passing verdict publishes) · **Requirement:** B5B-06.
**Done when:** every number-bearing doc quotes the same set and the sha256 log matches.
**Tests:** `tests/test_docs_site.py` · **Gate:** full.
**Commit:** `feat(release): publish the multilingual five-domain adapter` + docs commits

## B5B-9: Docs site + cycle close

**What:** `docs/training.md` (multi five-domain generation/training/eval commands, the three
arms and the cross table), `mkdocs build --strict`, spec status → Done, STATE/BACKLOG final
pass, release-tag decision recorded (B-5 pattern: after the numbers).
**Where:** `docs/training.md`, feature spec/tasks, `.specs/project/*`.
**Depends on:** B5B-8 · **Requirement:** release hygiene.
**Done when:** docs site strict-builds and the feature's Cycle status paragraph is written.
**Tests:** `tests/test_docs_site.py` · **Gate:** full.
**Commit:** `docs: document the multilingual five-domain cycle`

**Cycle status:** Not started. Order: B5B-1 → B5B-2 → (B5B-3 ∥ B5B-4) → B5B-5 → B5B-6 →
B5B-7 → B5B-8 → B5B-9. Gates fixed in B5B-4 **before** B5B-5 launches.
