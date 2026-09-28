# ADR-0015: `score` and `choice` labels come from the text

**Status:** Accepted
**Date:** 2026-09-28

## Context

ADR-0014 established the rule for `noul` — a label must be recoverable from the emitted text —
and left two sibling artifacts alone, tracked as backlog **B-12**. Both are index-derived, and
both price out as measured ceilings rather than model errors:

1. **The `score` near-tie.** `_score_record` took the level from the tone and then, on
   `index % 13 == 0`, lowered it one level as an "ambiguous near-tie label". That is 7.8% of rows
   (39/500 in both `eval_en.jsonl` and `eval_multi.jsonl`, 195/2500 in `eval_en_domains.jsonl`)
   whose text reads level *k* while carrying *k−1*. A model that reads the tone perfectly and
   always predicts the tone's level therefore tops out at **0.888** on the shipped eval sets —
   which is where the released checkpoints sit (`score` 0.882 English, 0.854 multilingual).
2. **The empty boundary state, in every primitive.** `_boundary_state` blanks one record in 19
   (5.4% of `noul`, `choice` **and** `score` rows). For `noul` ADR-0014 already gave it a default
   (`""` → 0, "reads as no request"). For `score` the label kept the tone that was drawn *before*
   the state was emptied — text no signal, label yes — and for `choice` it kept
   `options[index % len(options)]`, an option the empty text names not at all: ~4 points of
   ceiling on `score`, ~1.4 on `choice`.

A third fact made the fix awkward in one place: **`ecommerce` was the only domain without a
catch-all option** (`shipping`, `returns`, `payments`, `catalog` — the other four ship `other`),
so "an empty input means *none of the above*" had no target to point at in that domain.

The backlog already said what this costs: every `score`/`choice` label moves, so datasets,
checkpoints and published numbers move with it — which is why B-12 was written to be **bundled
with the next retrain** rather than paid for on its own. The next retrain was already scheduled
(the B-5 run-5 retrain on B-11 labels), so the two became one cycle.

## Decision

1. **`score` takes the level of the tone the state was written in.** The `index % 13` downgrade is
   removed, not authored: deliberate ambiguity the text cannot express is label noise, and L-008 /
   L-010 had already priced what that costs. An **empty** `score` state takes the **middle level**
   (`_EMPTY_SCORE_TARGET = 1`) — no signal, neither calm nor urgent — the same "absence reads as
   the default" rule ADR-0014 applied to `noul` empty → 0.
2. **`choice` takes the option the state names, and an empty state takes the catch-all.**
   `_default_option()` returns `"other"` — which **every** committed domain now ships — falling
   back to the first option for any domain that ever ships none.
3. **`ecommerce` gains `other`** as its fifth option, with eight terms and a description written
   in the domain's own voice. Its options re-cycle 4 → 5, so its `choice` rows move with it; the
   other four domains are untouched.
4. **Scope: everything, in one cycle** — the same choice ADR-0014 made for `noul`: regenerate all
   seven datasets (golden hashes recaptured), retrain **all three** checkpoints (English,
   multilingual, and the B-5 run-5 artifact) under their existing recipes, evaluate the B-11
   checkpoints against the corrected eval sets *before* replacing them (L-005), then re-run the
   probes, the head-to-head, the fast path and the B-5a comparison, and republish the Hub
   adapters and every table as one set.
5. **Coverage is preserved where it is meaningful.** Empty and very-long boundary cases remain in
   all three primitives, now with a learnable default in each; the near-tie device (a label made
   ambiguous on purpose) is gone from the docs' coverage list, and genuine ambiguity still exists
   where the text carries it — `choice`'s hard-negative distractor clauses.

## Consequences

**Positive:** every label in all three primitives is now checkable against its own text, with a
test per primitive across five domains and seven languages; `score`'s ceiling moves from 0.888 to
**0.960** (≈1.0 with the empty default) and `choice` gains the ~1.4 points the empty rows were
throwing away; `ecommerce`'s option set is uniform with the rest of the domain files, so the
catch-all rule needs no per-domain branch.

**Negative:** the whole measured surface moves *again* one cycle after ADR-0014 — datasets,
three retrains (≈5 h of GPU on the RTX 3060, B-002), a re-run of the public probes, the
head-to-head (all four engines on table B, whose gold labels move), the fast path and the B-5a
gate verdict, plus another republish. Numbers produced between ADR-0014 and this ADR are
superseded the same way pre-B-11 numbers were.

**Neutral / follow-ups:** the deliberate-ambiguity coverage claim in `docs/training.md` was
rewritten rather than kept as a promise the generator no longer keeps; ADR-0014's follow-up note
("the `score` near-tie is deliberately left alone") is superseded by this ADR — recorded in
`docs/adr/README.md` rather than by editing ADR-0014 in place.
