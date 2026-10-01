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

## B5B-4: Baseline measurement — fix the gates before training

**What:** Evaluate the released `checkpoints/multi` (the `d3f64cc0` build) on
`data/eval_multi_domains.jsonl` through `training/predict` — overall / per-domain /
per-primitive / per-language / cross cell, calibration as shipped. Write
`benchmarks/results/multi_domains_baseline.json`; record the fixed gate numbers (support ≥
cell, worst new domain ≥ 0.70, per-domain ECE ≤ 0.05 w/ exceptions) in BACKLOG **B-5b** before
any trained arm exists. Confirm the support cell reproduces the 0.743 anchor.
**Where:** `benchmarks/results/multi_domains_baseline.json`, `.specs/project/BACKLOG.md`.
**Depends on:** B5B-2 (eval set) · **Requirement:** B5B-04 (gates fixed first).
**Done when:** the baseline JSON exists and BACKLOG quotes the gate numbers sourced from it.
**Tests:** — (measurement) · **Gate:** full.
**Commit:** `docs(specs): fix the B-5b gates from the released-multilingual baseline`

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
