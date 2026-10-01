# Per-domain Choice Head Bank (ADR-0016) Specification

**Phase:** Post-M6 (B-5 structural decision, `.specs/project/BACKLOG.md`)
**Status:** In progress. ADR-0016 Accepted (2026-09-30): per-domain `choice` heads behind a
deterministic gate, first experiment = fit only the bank on the frozen run-5 trunk.
**B1–B6 done (2026-10-01)** — asset + gate, wire hint, bank training, the frozen-trunk fit
(two loadable arms, `lr 1e-4` after `1e-3` destroyed the domain heads) and the gate/oracle
report row. **Remaining: B7 (record the four arms' numbers, incl. a held-out-text split) and
B8 (docs).**
**Related docs:** `docs/adr/ADR-0016-per-domain-choice-heads.md`, `.specs/project/BACKLOG.md` B-5,
`docs/protocol.md` (additive extensions), `AGENTS.md` hard rules 1–3.

## Problem Statement

B-5a is structurally stuck: the designated run-5 artifact clears both written gates on the B-12
labels (overall **0.879**, `support` **0.886 ✓**, worst new domain **0.815 ✓**) but sits **0.086
below** the released support-only adapter (0.972), and the entire residual is `choice` (0.745 vs
0.946). One shared low-rank head serves five domains and trades one against another — zeroing it
leaves `support` unchanged (fact 1), rank bumps move the leak between runs instead of closing it
(fact 2), every miss falls into the catch-all whose description names the other options (fact 3),
and once B-12 corrected the labels the shared trunk starved `choice` outright (0.303, fact 4).
Both cheap remedies (relabel = B-11/B-12, retrain = measured and lost) are spent.

## Goals

- [x] **Bank asset:** `choice_head.json` grows to `{shared, domains: {name: {head, signatures}}}`;
      the legacy single-scorer format loads as shared-only, so every published adapter keeps
      loading (B-8 warn-and-degrade extends to the keyed form).
- [x] **Deterministic gate:** exactly one head per `choice` question, priority order — caller hint
      (optional additive request field, honored only when the asset ships that key) → lexical
      signature match (pure Python, no encode, offline) → shared head. A wrong gate can at worst
      reproduce today's numbers.
- [x] **Training:** `finetune_rlcd.py` learns a bank — each record updates its own domain's head,
      the shared head keeps training on every record; records with no `domain` field train only
      the shared head, so legacy configs reproduce today's artifact exactly.
- [ ] **First experiment (the isolate):** fit only the bank on the **frozen run-5 trunk** and
      measure whether the head-capacity / gradient-share axis closes the gap, before any joint
      retrain is considered. *(fitted in B5 — B7 records the four arms' numbers)*
- [x] **Gate accuracy published beside per-domain accuracy** — the new failure mode (misrouting)
      must be visible in every report that quotes the bank.

## Out of Scope

| Item | Reason |
| --- | --- |
| Scoring-rule change (`cos(criterion, question)` rebalance) | Sequenced fallback — moves `choice` *and* `score`, forces a full re-benchmark (ADR-0016 §3) |
| Two-stage training / per-domain LoRA adapters | Rejected in ADR-0016 alternatives (L-004 rule) |
| Joint retrain on corrected labels | Follows only if the isolate says the trunk axis is also in play (ADR-0016 §5) |
| `score` / `noul` heads, their temperatures and probes | Untouched by this ADR — only `choice` cells move |
| Multilingual (B-5b) bank keys | Blocked behind this cycle; the asset format is already forward-compatible |

---

## User Stories

### P1: The bank loads everywhere today's head does ⭐ MVP

**User Story:** As a runtime user, I want a keyed `choice_head.json` to load and answer through
the gate, while every published (legacy) adapter keeps answering bit-for-bit as before.

**Acceptance Criteria:**

1. WHEN the asset is the legacy `{rank, w1, w2}` THEN it loads as a shared-only bank and every
   `choice` answer is unchanged (no gate keys, no residual difference).
2. WHEN the asset is keyed THEN `shared` and every `domains` entry load; a corrupt entry warns by
   name (file + key) and degrades — corrupt `shared` → baseline for unmatched questions, corrupt
   domain → that head dropped, its questions fall through — never a crash (B-8 contract).
3. WHEN no head matches and no hint is given THEN the shared head answers — bit-for-bit today's
   behaviour.
4. IF the adapter ships no bank THEN behaviour is exactly as before ADR-0016.

### P1: The gate is deterministic, offline and testable

**User Story:** As a runtime user, I want my question routed to the right domain's head without a
model call, and to the shared head whenever the evidence is ambiguous.

**Acceptance Criteria:**

1. WHEN a caller hint names a head the asset ships THEN that head is used for every `choice`
   question in the request; an unknown hint falls through to the lexical rule (never errors).
2. WHEN no hint is given THEN signature terms (committed option labels, terms, descriptions) are
   matched against the question's instructions, option labels and descriptions — the strictly
   highest-scoring domain wins; a tie or a zero score selects the shared head.
3. WHEN evaluated on the shipped `eval_en_domains.jsonl` choice rows THEN gate accuracy (strict:
   selected key == record domain) is published as `report["choice_gate"]` beside `per_domain`.
4. THE gate is pure Python over strings — no encode, no torch, deterministic, offline (ROUTE-03).

### P2: The caller hint is an additive wire field

**User Story:** As a client author, I want to name the head explicitly when I know my question's
domain, without breaking Jev compatibility.

**Acceptance Criteria:**

1. THE request field is optional and additive — `tests/test_contract_wire.py` and
   `docs/protocol.md` are updated **in the same commit** (AGENTS.md hard rule 1).
2. WHEN absent THEN the gate decides; WHEN present but unknown THEN the gate decides (ADR-0016
   §2.1).
3. THE response shape is untouched (hard rule 3) — no echo, no new required field.
4. THE hint is exposed on the CLI (`--choice-head`) and the SDK (`system_one(..., choice_head=)`).

### P2: Training learns a bank without disturbing legacy runs

**User Story:** As a trainer, I want one joint run producing `shared` + per-domain heads, and my
single-domain configs to reproduce today's artifact exactly.

**Acceptance Criteria:**

1. IF the training records carry `domain` THEN each `choice` record trains its domain head **and**
   the shared head; IF they do not THEN only the shared head trains and `choice_head.json` is
   written in the **legacy format, byte-identical** to today's writer.
2. THE keyed output embeds signatures built from the committed domain data for the languages the
   records actually used.
3. THE head bank, its init order and the RNG stream of legacy runs are unchanged when no domains
   exist (L-003/L-011 discipline: nothing hidden in a config).

### P1: The isolate experiment answers the head-axis question cheaply

**User Story:** As an evaluator, I want to know whether per-domain heads close the 0.086 gap
*without* retraining the trunk — in minutes, not hours.

**Acceptance Criteria:**

1. THE fit script freezes the run-5 trunk (encode once, no grads), fits the shared control arm and
   the bank arm from the same encoding pass, and writes loadable adapter dirs (adapter files +
   new `choice_head.json` + the fit's own recipe — L-011).
2. THE experiment reports, per arm: overall / per-domain / per-primitive accuracy, `choice` cells,
   gate accuracy, and the oracle upper bound (bank with the record's own domain forced).
3. THE baseline (run-5 as shipped, 0.8788 / `choice` 0.7448) is re-measured in the same harness
   before any comparison is quoted.
4. RESULTS land in `benchmarks/results/` and the cycle gates get their numbers in
   `BACKLOG.md` B-5.

---

## Requirements Traceability

| Story | Requirement | Where |
| --- | --- | --- |
| P1 load | EXT-02 (additive), NFR-C05 | `src/tachyone/backends/encoder.py`, `tests/test_encoder.py` |
| P1 gate | ROUTE-03 (model-free), ADR-0016 §2.2 | `tests/test_encoder.py`, gate-accuracy test |
| P2 hint | ADR-0001 + hard rule 1 | `tests/test_contract_wire.py`, `docs/protocol.md` |
| P2 train | TRAIN-06 (reproducible configs), ADR-0016 §1 | `training/finetune_rlcd.py`, `tests/test_training_finetune.py` |
| P1 isolate | OPS-06 (measured), B-5 gates | `training/fit_choice_bank.py`, `benchmarks/results/` |
