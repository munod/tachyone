# ADR-0014: Fix the `noul` label everywhere and retrain

**Status:** Accepted
**Date:** 2026-09-27

## Context

`_noul_record` derived its label from the loop index — `positive = (index % 2 == 0)`, forced to
`False` when the phrase bank was `neutral` — so a record was positive iff it had an even index
*and* a request tone. Half of every request-toned set therefore carried `target = 0` while reading
like a request, and half of what the model could not get right was not the model. Measured on the
shipped eval sets (L-008):

| Set | request-toned `noul` records | of which labeled 0 |
| --- | ---: | ---: |
| `eval_en.jsonl` | 241 | 121 (**50%**) |
| `eval_multi.jsonl` → `fr`, `it`, `pt` | ~40 each | **0** (0%) |
| `eval_multi.jsonl` → `de`, `es`, `nl` | ~33–46 each | **40–55%** |

Two consequences made this a decision rather than a bugfix:

1. **`noul` was label-bound, not model-bound.** The Bayes-optimal accuracy given the tone is
   **0.746** on `eval_en.jsonl`; the published model scores **0.718** — it was already at the
   ceiling the labels imposed.
2. **The per-language spread was an artifact.** `nl` 0.663 against `es` 0.956 tracks how the
   per-language RNG happens to correlate tone with parity, i.e. data, not capability (NFR-C06
   territory).

Fixing the rule changes every `noul` label, so every dataset, checkpoint and published number
moves with it. Backlog **B-11** explicitly left the scope open, and the alternative — fix only the
five domains B-5a just added — was refused during B-5a itself: it would leave `support` labelled
one way and the new domains another inside the *same* dataset, hidden behind a config flag.

## Decision

1. **The label comes from the text just emitted**, never from the index: `target = 1` iff the
   state was drawn from the `request` phrase bank, `0` for the `neutral` bank and for the empty
   boundary state (which reads as no request and is therefore deterministically `0`).
   `tests/test_training_generate.py` recovers each state's phrase bank from the text itself —
   across all five committed domains and seven languages — so the rule is asserted against what a
   reader would see.
2. **The scope is everything.** All seven shipped datasets are regenerated, both published
   adapters (`tachyone-en`, `tachyone-multi`) are retrained with their existing recipes, the
   temperatures are refit, and the whole measured surface — `benchmarks/report.md`,
   `docs/benchmarks.md`, `docs/compare.md` §3, `benchmarks/probes.md`, the model card, README and
   CHANGELOG — is **republished as one set**. The pre-B-11 numbers stay valid for the data that
   produced them; they are simply not comparable row-by-row with the corrected ones (the AD-009
   pattern).
3. **The old checkpoints are measured against the corrected eval sets before they are replaced**
   (L-005), so the report can say what the label fix alone changed about the *measurement*,
   independent of the retrain.
4. **Accuracy and label quality are published together.** `training.evaluate` writes
   `report["noul_per_language"]` beside `report["noul_labels"]` (the per-language
   contradictory-label rate, judged against the same phrase banks; states no bank explains —
   B-4's surface noise — count as `unknown`, never as contradictions), and
   `benchmarks/report.py` renders both in one table, so the two can no longer be confused.

## Consequences

**Positive:** labels a reader can verify from the text; `noul` stops being label-bound and its
per-language spread stops masquerading as a capability gap; the audit makes any future label
regression loud instead of silent.

**Negative:** every published number moves as a set, at the cost of two GPU retrains plus a full
benchmark/probe re-run (B-002); the pre- and post-B-11 tables cannot be read as a before/after of
the *model* — the dataset changed too; historical verdicts measured on pre-B-11 labels (notably
the B-5a gate verdict) have to be re-measured to stay reproducible.

**Neutral / follow-ups:** the golden hashes in `tests/test_training_generate.py` are recaptured
(only `noul.target` bytes move — states, questions and every `choice`/`score` record stay
byte-identical); `score` has an analogous index-derived quirk
(`index % 13 == 0` lowers a near-tie level) that is **deliberately left alone** and tracked
separately in the backlog, because it is a documented ambiguity rather than a contradiction.
