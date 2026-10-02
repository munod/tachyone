# Multilingual Five-Domain Coverage (B-5b) Specification

**Phase:** Post-M6 (B-5 second half, `.specs/project/BACKLOG.md` B-5)
**Status:** **Spec accepted (2026-10-01)** — not started
**Context (locked decisions):** `.specs/features/multilingual-five-domains/context.md`
**Related:** `.specs/features/choice-head-bank/spec.md` (B-5a), `.specs/features/multilingual-quality/spec.md`
(B-1), `docs/adr/ADR-0016-per-domain-choice-heads.md` §4, AGENTS.md hard rules 1, 5, 6

## Problem Statement

B-5 proved five-domain coverage in English and published it with `v0.5.0`, but the multilingual
checkpoint (`munod/tachyone-multi`) is still support-only: `train_multi.jsonl` carries no
`domain` field and four of the five domain lexicons ship `en` only, so for the six training
languages (`pt es fr de it nl`) the four new domains are entirely zero-shot — their records fall
back to English text under a non-English `lang` tag, which is a label artifact of exactly the
kind L-008/L-010 forbid. ADR-0016 §4 unblocked this cycle explicitly: the bank's
`{shared, domains}` keys are plain strings that reuse the decided asset format with no new ADR.

## Goals

- [x] **Localized content:** all five committed domains ship complete language tables for
      `DEFAULT_LANGUAGES` (7), with English tables byte-untouched (golden hashes hold).
- [x] **Datasets:** `data/train_multi_domains.jsonl` (30k) and `data/eval_multi_domains.jsonl`
      (7.5k) reproduce from committed configs, with the support halves identical to
      `train_multi`/`eval_multi` modulo the `domain` key.
- [x] **Trained bank:** one joint mmBERT run writes a keyed `choice_head.json` (shared + five
      domain heads, signatures in the six training languages) that loads and answers through
      the unchanged ADR-0016 gate — **and the frozen-trunk fitter (ADR-0016 §5 isolate)
      produced the publish arm: `checkpoints/multi_b5b_fit_bank`.**
- [x] **Gates (baseline first):** released-multilingual baseline measured on the new eval
      set before training fixed the numbers (`support` ≥ 0.8413, worst new domain ≥ 0.70,
      per-domain ECE ≤ 0.05 with exceptions declared, gate strict published) — **all pass on
      the publish arm with zero ECE exceptions (0.0005–0.0038), gate strict 1.000.**
- [ ] **Published:** `munod/tachyone-multi` republished sha256-verified; report, model card,
      README, benchmarks docs and CHANGELOG carry the new set; MASSIVE probe and fast-path
      parity re-run (0 flips expected). → **B5B-8, next.**

## Out of Scope

| Item | Reason |
| --- | --- |
| English checkpoint / datasets / numbers | Untouched; golden-hash tests prove it (no regression = byte-identity) |
| Wire or gate changes | ADR-0016 gate, hint and asset format ship already (hard rule 1 not triggered) |
| Scoring-rule change, two-stage training, per-domain LoRA | ADR-0016 sequenced fallbacks / rejected alternatives — own ADRs |
| B-1's per-language ECE ≤ 0.05 gate (NFR-C06) | Published in the cross table, but B-1 keeps its own acceptance |
| Per-language head keys | Rejected in context.md; measured axis is domain (L-012 discipline) |
| Release tag | Decided after the numbers, as in B-5 (context.md) |

---

## User Stories

### P1: Every domain speaks every training language ⭐ MVP

**User Story:** As a data author, I want the four new domain lexicons localized to the six
training languages so that a `pt` `ecommerce` record is Portuguese text, not English text under
a `pt` tag.

**Why P1:** without it the multilingual five-domain dataset reproduces L-008's artifact
(labels/tags the text cannot support) and the cycle measures nothing new.

**Acceptance Criteria:**

1. WHEN a domain file is loaded for any language in `DEFAULT_LANGUAGES` THEN it SHALL return
   that language's table — no English fallback for any shipped domain.
2. WHEN a language table is inspected THEN it SHALL carry all seven fields with the same shape
   as `en` (entities ≥ 8, levels = 4, instructions/noul_criteria keys complete, option terms ≥ 4
   and descriptions ≥ 20 chars per option key, phrase tones complete with `{entity}` /
   `{distractor}` placeholders preserved).
3. WHEN the English tables are compared against the committed bytes THEN they SHALL be
   byte-identical — every existing golden hash and config-vs-dataset test still passes.
4. WHEN a localized phrase is formatted THEN it SHALL be grammatical in its language with the
   placeholders substituted (no English tokens leaking into non-English records).

**Independent Test:** `tests/test_training_generate.py` completeness test parameterized over
`DEFAULT_LANGUAGES`; generate one record per (domain, language) and inspect.

---

### P1: The multilingual five-domain datasets preserve the incumbent byte-for-byte

**User Story:** As an evaluator, I want the support halves of the new datasets to be the old
multilingual records, so the released adapter's numbers stay comparable on the same rows.

**Why P1:** the no-regression gate is only meaningful if `support` rows are identical to what
the published adapter was measured on (the AD-009 / B-5a pattern).

**Acceptance Criteria:**

1. WHEN `data/train_multi_domains.jsonl` is filtered to `domain == "support"` and the `domain`
   key is removed THEN the records SHALL equal `train_multi.jsonl` record-for-record.
2. WHEN `data/eval_multi_domains.jsonl` is filtered the same way THEN the records SHALL equal
   `eval_multi.jsonl` record-for-record (the baseline's support cell is therefore measured on
   exactly the rows the released adapter was published against; the same-harness anchor is
   **0.8413** — the published 0.743 is a *routed* number that does not reproduce, BACKLOG B-5b /
   L-013).
3. WHEN counts are checked THEN train SHALL be 30,000 (support 18,000 + 4 × 3,000) and eval
   7,500 (1,500 per domain), six languages interleaved.
4. WHEN any committed config is re-run THEN it SHALL reproduce its shipped dataset (existing
   config-vs-dataset tests keep passing, including the English ones).

**Independent Test:** the byte-identity assertions in `tests/test_training_generate.py`.

---

### P1: The joint run produces a bank that answers through the gate

**User Story:** As a runtime user, I want the multilingual checkpoint to ship the ADR-0016 bank
keyed by domain with signatures in all six training languages, so each `choice` question is
routed by the deterministic gate with the shared head as fallback.

**Acceptance Criteria:**

1. WHEN the run completes THEN `choice_head.json` SHALL carry `shared` + five domain keys, each
   with signatures built from the six languages present in the training records.
2. WHEN a question's localized instructions/options match a domain's localized signatures THEN
   the gate SHALL select that head; on tie/no match it SHALL fall back to `shared`.
3. WHEN a legacy (single-scorer) asset is loaded THEN behaviour SHALL be unchanged — the
   existing golden tests keep passing untouched.
4. WHEN the config is compared with `<artifact>/finetune_config.json` after the run THEN they
   SHALL match (L-011 — the config documents the artifact it produces).

**Independent Test:** bank-load and gate tests in `tests/test_encoder.py` (existing) +
`training/predict.py` gate report on the new asset.

---

### P1: Baseline-first gates are measured and recorded

**User Story:** As an evaluator, I want the released multilingual adapter measured on
`eval_multi_domains.jsonl` **before** training fixes the gate numbers, and the trained arms
measured in the same harness afterwards, so the verdict compares like with like.

**Why P1:** gates set after seeing the trained result are not gates (the B-5 lesson: fix the
number first, publish the whole table).

**Acceptance Criteria:**

1. WHEN the baseline runs THEN `benchmarks/results/` SHALL carry the released adapter's table
   (overall, per-domain, per-primitive, per-language, gate absent) on the new eval set, and
   BACKLOG B-5b SHALL record the fixed gate numbers from it.
2. WHEN the trained arms are evaluated THEN `support` SHALL be ≥ the baseline support cell,
   worst new domain ≥ 0.70, and per-domain ECE ≤ 0.05 with exceptions declared.
3. WHEN the asset ships a bank THEN `report["choice_gate"]` (strict / fell-to-shared /
   wrong-domain) SHALL be published beside `per_domain`, with the oracle arm measured from the
   same command shape.
4. WHEN rows are split by whether their text appears in training (`--train-data`) THEN the
   held-out-text numbers SHALL be published beside the headline (the B-7 81%-collision
   discipline).

**Independent Test:** the four result JSONs in `benchmarks/results/` + the BACKLOG table.

---

### P2: The domain × language cross table exists in the report

**User Story:** As an evaluator, I want one report cell per (domain, language), so the
project's worst-cell convention can be read directly instead of from a one-off script (the
B-5a workaround).

**Acceptance Criteria:**

1. WHEN the eval set carries both fields THEN `report["per_domain_language"]` SHALL carry
   accuracy/ECE per (domain, language) cell beside `per_domain` and `per_language`.
2. WHEN a report is rendered THEN `benchmarks/report.py` SHALL print the cross table (or its
   worst cell) without breaking single-domain/legacy reports.
3. WHEN the eval set has no domain field THEN the report SHALL behave exactly as before.

**Independent Test:** `tests/test_training_evaluate.py` + `tests/test_benchmark_report.py`.

---

### P2: The full B-5a publication sweep

**User Story:** As a project owner, I want `munod/tachyone-multi` republished and every
number-bearing document regenerated, so the published surface is one consistent set.

**Acceptance Criteria:**

1. WHEN the Hub upload completes THEN every file SHALL be sha256-verified against the local
   build and stale files deleted (the `c00c174d` pattern).
2. WHEN docs are regenerated THEN `benchmarks/report.md`, `docs/benchmarks.md`,
   `docs/model-card.md`, README, `docs/huggingface.md` and `CHANGELOG.md` SHALL all quote the
   same numbers.
3. WHEN the adapter feeds a public probe or the fast path THEN MASSIVE and fast-path parity
   SHALL be re-run and published (0 top-label flips expected); the head-to-head SHALL be
   re-run only if a quoted cell uses the multilingual engine.
4. WHEN the docs site builds THEN `uv run mkdocs build --strict` SHALL pass.

**Independent Test:** `tests/test_docs_site.py` + `mkdocs build --strict` + sha256 log.

---

### P3: One detached, observable training run

**User Story:** As a trainer, I want the run launched in its own session with resource
sampling, so an hour-plus of GPU cannot die silently (L-009).

**Acceptance Criteria:**

1. WHEN training starts THEN it SHALL run `setsid nohup … < /dev/null &` with `exit=$?` and a
   completion marker appended to its log.
2. WHEN the run ends THEN `finetune_config.json`, the checkpoint files and the report SHALL
   exist and the config SHALL match the launch config (L-011).

**Independent Test:** the run log + artifact directory listing.

---

## Edge Cases

- WHEN a language table is missing from a domain THEN `DomainData` falls back to English (existing
  behaviour) — but the completeness test forbids that state for shipped domains.
- WHEN localized signatures of `support` and `ecommerce` overlap (`reembolso`/`refund` vocabulary)
  THEN the gate may misroute — measured by `choice_gate` wrong-domain counts; the oracle arm
  separates routing from head quality; fallback is `shared` (never a crash).
- WHEN a record has no `domain` key THEN it trains only the shared head (ADR-0016 §1) and legacy
  configs keep their exact RNG stream.
- WHEN `eval_multi` (no domain field) is evaluated THEN no report cell may change (legacy path).
- WHEN two heads tie on signature score THEN the shared head answers (existing gate rule).

## Requirement Traceability

| ID | Story | Where |
| --- | --- | --- |
| B5B-01 | P1 localized content | `training/data/domains/*.json`, `tests/test_training_generate.py` |
| B5B-02 | P1 datasets + incumbent identity | `training/configs/data_*_multi_domains.json`, `tests/test_training_generate.py` |
| B5B-03 | P1 joint bank run | `training/configs/finetune_multi_domains.json`, `training/finetune_rlcd.py`, `training/predict.py` |
| B5B-04 | P1 baseline-first gates | `benchmarks/results/multi_domains_*.json`, `.specs/project/BACKLOG.md` B-5b |
| B5B-05 | P2 cross table | `training/evaluate.py`, `benchmarks/report.py` |
| B5B-06 | P2 publication sweep | Hub, `benchmarks/report.md`, `docs/*`, `CHANGELOG.md` |
| B5B-07 | P3 detached run | run script + log (L-009) |

**Coverage:** 7 IDs, 7 mapped to tasks, 0 unmapped ✓ (hard rule 1 not triggered: no public API
change in this cycle).

## Success Criteria

- [ ] All five domains × seven languages complete and validated; English byte-identity holds.
- [ ] 30k/7.5k datasets reproduce from committed configs; support halves byte-identical.
- [ ] Baseline measured first; gates fixed in BACKLOG before the trained arms are read.
- [ ] `support` ≥ baseline, worst new domain ≥ 0.70, per-domain ECE ≤ 0.05 (exceptions declared),
      gate strict published.
- [ ] `munod/tachyone-multi` republished and verified; docs/report/probes/fast-path one set.
- [ ] Full gate green per commit (ruff, format, pyright, pytest `-m "not e2e"`), conventional
      commits, spec status updated at cycle close.
