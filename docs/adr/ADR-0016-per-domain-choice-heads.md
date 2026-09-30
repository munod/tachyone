# ADR-0016: Per-domain `choice` heads behind a deterministic gate

**Status:** Accepted
**Date:** 2026-09-30

## Context

B-5a is measured and structurally stuck. On the B-12 labels the designated artifact (run 5)
clears both written gates — overall **0.879**, `support` **0.886 ✓**, worst new domain
**0.815 ✓** — while the released support-only adapter scores **0.972** on the same eval set:
the five-domain checkpoint sits **0.086 below** it, and the entire residual is `choice`
(run-5 `choice` **0.745** vs the released **0.946**). Both cheap remedies are already spent:
relabeling was B-11/B-12 (ADR-0014 / ADR-0015), and the retrain on corrected labels **lost** —
`noul` and `score` both reached 1.000 while `choice` collapsed to **0.303** (train loss 0.027,
eval at chance), dragging `support` to 0.763 ✗; the English control of the identical recipe
posts 0.841. BACKLOG B-5 therefore asks for a structural decision, and four measurements say
where the structure must change:

1. **Zeroing the head** (run-4 checkpoint): global `choice` 0.557 → 0.371 but `support`
   unchanged (0.655 → 0.655) — the single shared low-rank head learned to help the four new
   domains and *not* support.
2. **The leak migrates between runs, not away.** Run 5 leaks `billing → other` (82/125) and
   `sales → other` (48/125) on support; run 6 (`choice_rank` 128 → 256, 6 → 8 epochs) fixes
   that and leaks `billing → sales` (88/125) there while `ecommerce` grows `returns → catalog`
   (117). Raising capacity made the headlines worse (`choice_rank` 256: 0.804 → 0.755; LoRA
   r=64: 0.630 → 0.540). Which domain the shared head under-fits is run-to-run variance
   (L-006 / AD-009 territory), not a capacity curve.
3. **The mechanism of the leak:** every miss falls into the catch-all, whose description names
   the other options, so the per-domain constant `cos(criterion, question)` beats a
   `cos(criterion, state)` term the shared head never had enough capacity or gradient share to
   strengthen.
4. **Gradient competition across primitives.** Once B-12's corrected labels made `noul` and
   `score` trivially learnable (both → 1.000), the shared trunk starved `choice` outright
   (0.303): one head, five domains, three primitives, one fixed gradient budget.

The forces around any fix: the `/v1/systemone` request has **no domain field** and the wire is
frozen (ADR-0001, hard rules 1–3) — anything new must be an additive optional field or an
extension; the `choice` head is deliberately **plain JSON lists** that run without torch and
sit *after* the encode, so ONNX export and the CUDA-graph fast path (which wrap only the
encode) are untouched by whatever we do to it; the router's house style is **model-free, pure
Python, microseconds, offline** (ROUTE-03); and L-004 already deferred *MoE of LoRA adapters*
with a standing rule: revisit only with a new ADR and measured evidence.

## Decision

1. **The `choice` head becomes a bank of low-rank heads keyed by domain**, trained in the same
   run: each record updates its own domain's head, and one **shared head** — today's head —
   keeps training on every record as the fallback. A record with no `domain` field (every
   legacy single-domain config) trains only the shared head, so existing configs reproduce
   today's artifact exactly.
2. **Inference picks exactly one head per `choice` question** (per question, not per request)
   through a deterministic gate, in priority order:
   1. an **explicit caller hint** — an optional additive request field (CLI/SDK flag too) that
      is honoured only when it names a head the loaded asset actually ships; unknown name falls
      through to the next rule. The exact spelling is fixed in the same commit that updates
      `tests/test_contract_wire.py` (hard rule 1);
   2. a **lexical signature gate**: the selected head carries its own signature terms (the
      committed per-domain option labels, terms and descriptions), matched against the
      question's instructions, option labels and descriptions — pure Python, no encode, no
      model, deterministic, offline;
   3. **no match or ambiguity → the shared head**, which is bit-for-bit today's behaviour: a
      wrong gate can at worst reproduce the current numbers, never invent a new failure.

   The gate therefore ships **inside the adapter asset** — `choice_head.json` grows from a
   single scorer to `{shared, domains: {name: {head, signatures}}}`. The legacy single-scorer
   format loads as shared-only, so every published adapter keeps loading, and B-8's
   warn-and-degrade contract extends to the keyed form.
3. **`score` and `noul` are untouched, and so is the cosine base formula** — the per-domain
   constant `cos(criterion, question)` stays. The scoring-rule change (BACKLOG B-5 option 3) is
   recorded as the **sequenced fallback**: if the per-domain bank measurably fails to close the
   gap, that is the next ADR — and it is the more expensive one, because the base formula feeds
   `score` too and would force a full re-benchmark of both primitives.
4. **Scope: five domain keys for the English checkpoint now.** Keys are plain strings, so
   B-5b's language keys reuse the same asset format without needing a new ADR; only what a
   cycle actually trains ships.
5. **Training stays one joint run** (trunk + bank + shared head + temperature). The first
   experiment is the cheap isolate: fit **only the bank on the frozen run-5 trunk**, which
   tests the head-capacity / gradient-share axis (facts 1–3) in minutes instead of hours; the
   joint retrain on corrected labels follows only if that result says the trunk axis (fact 4)
   is also in play. Per-domain heads each see ~1/5 of the records, so the shared head — trained
   on all of them — doubles as the overfit guard as well as the fallback.

**Alternatives evaluated.** (a) *Per-domain LoRA adapters / adapter routing* — the L-004 item,
rejected again: the measured need is head capacity and gradient share (facts 1–3), which a bank
of tiny heads buys at a fraction of the cost of five trunks, and adapter switching is exactly
what complicates ONNX export and the CUDA-graph path. (b) *Two-stage training* (five domains
first, support polish after) — it re-arranges the gradient budget instead of partitioning it,
so it does not remove the head competition, and it invalidates the recipe behind published
numbers; it needs its own ADR if ever adopted. (c) *A learned / dynamic gating network* — a
second trained artifact whose errors are neither auditable nor reproducible offline; the
lexical gate matches the router's model-free contract and is unit-testable against the shipped
eval sets. (d) *Blending several heads per question* — MoE in miniature: strictly more moving
parts than single selection, with no measurement yet that selection itself is the bottleneck.

## Consequences

**Positive:** the intervention lands exactly where the four measurements point — support gets
its own head capacity and gradient share with no other domain trading against it (fact 1), the
"which domain loses this run" variance disappears by construction (fact 2), and the bank can be
trained on a frozen trunk to isolate the head axis cheaply (facts 3–4); heads stay plain JSON
lists computed after the encode, so the torch-free runtime, the ONNX story and the CUDA-graph
fast path are all unchanged; the gate is deterministic and offline (ROUTE-03 style) and its
accuracy is a directly measurable number on the shipped eval sets; the wire moves only through
an optional additive hint (ADR-0001 intact); only `choice` cells move — `noul`/`score`, their
temperatures and their probes do not — so the republish is a smaller cycle than B-12's
full-surface sweep; and B-5b is unblocked, because language keys reuse the decided asset
format.

**Negative:** a new failure mode appears — gate misrouting. It cannot corrupt today's numbers
(the fallback is the shared head) but it can leave the gap open, so gate accuracy must be
published beside per-domain accuracy. `choice` numbers move *again*: one more re-run of the
probes, the head-to-head `choice` cells and the fast-path parity check (0 top-label flips
expected — only the head's inputs change, not the encode). The asset format grows a keyed form
that must stay backward-compatible forever and now carries the signature terms, coupling the
committed data files to a runtime asset. Per-domain heads cut each head's training slice ~5×,
an overfit risk on the smaller domains. And `training/finetune_rlcd.py` gains a head bank in
the optimizer and the loss.

**Neutral / follow-ups:** ONNX currently builds a bare `EncoderCheckpoint` — no temperatures
and no `choice` head at all (`OnnxBackend.from_config`) — so this ADR neither helps nor worsens
it, but wiring calibration assets into that loader is an orthogonal gap worth a backlog item.
If the bank misses its gates, the sequenced fallback is the scoring-rule ADR (B-5 option 3),
which moves `choice` *and* `score` and therefore the whole measured surface; two-stage training
stays on the table behind its own ADR. The exact cycle gates — closing most of the 0.086,
support ≥ 0.85, worst new domain ≥ 0.70, and the gate-accuracy number — are set in BACKLOG B-5
when the cycle is scheduled. The hint field lands together with `docs/protocol.md`'s extension
table and the contract test, in the same commit (hard rule 1).
