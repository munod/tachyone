# Multilingual Five-Domain Coverage (B-5b) Context

**Gathered:** 2026-10-01
**Spec:** `.specs/features/multilingual-five-domains/spec.md`
**Status:** Ready for design

---

## Feature Boundary

Deliver B-5's second half: the five-domain coverage that shipped for English (`v0.5.0`) to the
multilingual checkpoint — localize the four new domain lexicons to the six training languages,
generate the multilingual five-domain datasets, train the joint run with the ADR-0016 head bank,
measure against a baseline-first gate set, and republish `munod/tachyone-multi` with the full
B-5a measurement sweep.

---

## Implementation Decisions

### Keying axis of the multilingual bank (discussed: ADR-0016 §4's "language keys" is ambiguous)

- **Decision: domain keys — the same five keys as English** (`support`, `ecommerce`,
  `agent_tools`, `documents`, `voice`), with signatures built over the six training languages
  (`pt es fr de it nl`) by the existing `domain_signatures(name, languages)`.
- Rationale: the measured structural risk B-5a fixed (one shared `choice` head trading domains
  against each other) is the same axis here; per-language calibration is already covered by the
  per-`(primitive, language)` temperature, and per-language head capacity was never measured as
  the bottleneck. Language keys and domain×language keys (30 heads, 1/30 data slices, L-012
  overfit risk) were rejected; measuring both axes as parallel arms was rejected as ~2× the
  cycle for an axis with no evidence behind it.

### Cycle gates: baseline first, then mirrored from B-5a

- **Decision:** measure the released `munod/tachyone-multi` on the new
  `eval_multi_domains.jsonl` **before** training, fix the gates from that table, then gate:
  - `support` ≥ its released-baseline cell (no regression — same harness, same rows);
  - worst new domain ≥ 0.70;
  - per-domain ECE ≤ 0.05, published with exceptions declared;
  - gate strict accuracy published beside per-domain accuracy (ADR-0016 consequence; not a
    pass/fail gate — the oracle arm separates head quality from routing).
- Absolute B-5a gates (0.85 support) were rejected: the released multilingual adapter posts
  0.743 overall, and 6-language quality has never reached 0.85.
- Folding B-1's open NFR-C06 (per-language ECE ≤ 0.05: `nl` 0.104, `de` 0.063, `it` 0.059,
  `fr` 0.051) into this cycle as a pass/fail gate was rejected — the full domain × language
  cross table is published, but B-1 keeps its own gate.

### Data volume: keep the incumbent's 18k, total 30k

- **Decision:** `per_domain: {support: 6000}` (its current volume in `train_multi.jsonl` —
  6k/type, 3k/language), new domains `per_type: 1000` (1k/type each) → **30,000 records**,
  six languages in round-robin.
- Rationale (correction during discuss): mirroring English's 9k support would **halve** the
  multilingual incumbent's volume — B-5a measured that cutting support volume costs up to 22
  points, and English never suffered that cut (it kept its published 9k through run 5).
  30k × 8 epochs ≈ 1.67× the published B-12 multilingual run's record-epochs; on this box
  (2× L4 23 GB) that is an afternoon run, not a budget risk.

### Publication scope: the full B-5a pattern

- **Decision:** gates → republish `munod/tachyone-multi` (sha256-verified file by file, stale
  files deleted) → regenerate `benchmarks/report.md`, `docs/benchmarks.md`, `docs/model-card.md`,
  README, `docs/huggingface.md`, `CHANGELOG.md` → re-run what this adapter feeds (MASSIVE probe,
  fast-path parity — 0 top-label flips expected; head-to-head only if a quoted cell uses the
  multilingual engine). A release tag is decided **after** the numbers, as in B-5.

---

## Agent's Discretion

- Exact phrase/term wording of the localized lexicons (bounded by the completeness test:
  counts, placeholders, option-key coverage).
- Training config details not fixed above (recorded with rationale in `design.md`).
- Whether the head-to-head tables need re-running (decide by checking which engine feeds the
  quoted cells).

---

## Specific References

- "Support records are identical inside and outside a five-domain run" — verified: the support
  halves match `train_multi`/`eval_multi` **modulo the added `domain` key** (the same is true
  of the English pair), and that is the invariant the data tests assert.
- Mirrors `support.json`'s seven-field language table structure exactly (entities,
  instructions, levels, noul_criteria, option_terms, option_descriptions, phrases × 5 tones).
- L-009 (detached `setsid nohup` training launch), L-011 (config must match the artifact),
  L-012 (arms differ by exactly one factor when attributing).

---

## Deferred Ideas

- Per-language head keys / per-language ECE gate — stays in **B-1** (its own acceptance).
- Scoring-rule change and two-stage training — sequenced fallbacks of ADR-0016, own ADRs.
- Presets / `docs/use-cases.md` expansion for the new domains — unchanged from B-5a (out of
  this cycle's gates; noted if the publish review finds stale text).
