# Per-domain Choice Head Bank (ADR-0016) — Tasks

**Spec:** `.specs/features/choice-head-bank/spec.md`
**Status:** In progress

Commit-per-task, tests co-located (AGENTS.md hard rule 6), full gate before each commit:
`uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest -m "not e2e"`.

---

## B1: Bank asset + deterministic gate (runtime, model-free)

**What:** `ChoiceHeadBank` in `backends/encoder.py`: `shared: ChoiceScorer | None`,
`domains: {name: {head, signatures}}`. `load_choice_head()` parses both formats (legacy →
shared-only), extends the B-8 warn-and-degrade to the keyed form (warn by name: file + key, then
degrade — never raise). Gate `select_key(question, hint)` = hint (only if shipped) → strictly
highest signature-match count → tie/zero → shared. Signatures matched as normalized token-sequence
substrings against instructions + option labels + descriptions.
**Where:** `src/tachyone/backends/encoder.py`.
**Depends on:** — · **Requirement:** ADR-0016 §2, EXT-02, ROUTE-03, NFR-C05.
**Done when:** legacy adapters answer unchanged (golden), keyed asset answers through the gate,
corrupt keyed entries warn and degrade; gate is pure Python (no encode call on its path).
**Tests:** `tests/test_encoder.py` (bank load ×4, gate ×7, corrupt keyed ×3) · **Gate:** full.
**Commit:** `5b0092b feat(encoder): add the keyed choice-head bank and its deterministic gate`

## B2: `EncoderModel` selects a head per choice question + hint plumbing to the model

**What:** `answer_state` asks the bank for the scorer per `choice` question (optional
`choice_head` kwarg, default `None`); `EncoderCheckpoint`/`load_checkpoint`/`EncoderBackend`
thread it; `training/predict.py` builds the model from the bank.
**Where:** `src/tachyone/backends/encoder.py`, `training/predict.py`.
**Depends on:** B1 · **Requirement:** ADR-0016 §2.
**Done when:** one request with two `choice` questions from different domains gets each domain's
head; a question the gate cannot place gets the shared head; `noul`/`score` untouched.
**Tests:** `tests/test_encoder.py::test_model_selects_one_head_per_choice_question`,
`test_noul_and_score_ignore_the_head_bank` · **Gate:** full.
**Commit:** `5b0092b feat(encoder): add the keyed choice-head bank and its deterministic gate`
(B1 and B2 are one cohesive runtime change)

## B3: The caller hint — wire field, CLI, SDK, contract test (same commit)

**What:** Optional additive `SystemOneRequest.choice_head: str | None = None` (spelling fixed
here); threaded to the encoder backend as a kwarg on `Backend.predict` (`head_hint`) — additive
keyword with default, no response change; CLI `--choice-head`; SDK `system_one(..., choice_head=)`;
`docs/protocol.md` extension table row + request table row.
**Where:** `src/tachyone/wire.py`, `backends/base.py`, `backends/{fake,llm,encoder}.py`,
`cli.py`, `client.py`, `tests/test_contract_wire.py`, `docs/protocol.md`.
**Depends on:** B2 · **Requirement:** ADR-0016 §2.1, hard rule 1, hard rule 3.
**Done when:** contract suite updated in the same commit; unknown hint falls through without
error; response shape byte-stable with and without the field.
**Tests:** `tests/test_contract_wire.py`, `tests/test_wire.py`, `tests/test_cli.py`,
`tests/test_client.py` · **Gate:** full.
**Commit:** `feat(wire): add the optional choice_head hint (ADR-0016)`

## B4: Signature builder + head bank in the trainer

**What:** `domain_signatures(name, languages)` in `training/generate_data.py` (option labels +
terms + descriptions from `training/data/domains/*.json`, deduped, deterministic order).
`finetune_rlcd.py` learns `shared` + per-domain heads: each `choice` record trains its domain head
**and** the shared head; records without `domain` train only shared and write the **legacy format
byte-identically** (init order keeps the RNG stream identical). Keyed output embeds signatures.
**Where:** `training/generate_data.py`, `training/finetune_rlcd.py`.
**Depends on:** B1 (format) · **Requirement:** ADR-0016 §1, TRAIN-06.
**Done when:** a domains run writes a loadable keyed asset; a legacy run's `choice_head.json` is
byte-identical to the pre-change writer; both paths covered.
**Tests:** `tests/test_training_generate.py`, `tests/test_training_finetune.py` · **Gate:** full.
**Commit:** `feat(training): train a bank of per-domain choice heads`

## B5: The isolate — fit the bank on the frozen run-5 trunk

**What:** `training/fit_choice_bank.py`: load the run-5 encoder frozen, encode the `choice`
records **once**, fit two arms from that pass — shared-only control and the bank (shared + five
domain heads, warm-started from run-5's shared head) — and write two loadable adapter dirs
(adapter files + new `choice_head.json` + `choice_bank_fit.json` recipe, L-011). `--dry-run`
validates config + data without torch.
**Where:** `training/fit_choice_bank.py`, `training/configs/fit_bank_en_domains.json`.
**Depends on:** B1, B4 · **Requirement:** ADR-0016 §5, B-5 gates.
**Done when:** both dirs load through `training.predict` and answer through the gate.
**Tests:** `tests/test_training_fit_choice_bank.py` (config, dry-run, arm outputs, determinism of
the pure parts) · **Gate:** full.
**Commit:** `447a306 feat(training): fit the choice-head bank on a frozen trunk` +
`bc9f464 fix(training): fit the choice-head bank at the trainer's learning rate`
**Status:** **Done (2026-10-01).** The isolate ran on the L4 (one encode pass over 7,000
`choice` records ≈ 9 min, 2 arms × 8 epochs). `lr` first shipped at 1e-3 and **destroyed the
domain heads** (combined loss 0.37 → 1.48, accuracy 0.77 → 0.30: an Adam step of 1e-3 is ~10%
of the head weights' RMS and each domain head sees only ~1/5 of the batches) — at the
trainer's **1e-4** both arms converge (control train/val 0.0042 / 0.0057, bank 0.0064 /
0.0048) and the domain heads land at 0.89–0.95 on their own domains. Both dirs then load
through `training.predict` and answer through the gate: on the first 4,000 eval rows the
trunk posts `choice` 0.7713, the control **0.9980** and the bank **1.0000** (same harness,
preliminary). **Measurement caveat for B7:** 81% of the eval `choice` rows are byte-identical
to a training row (the generator's phrase pools collide across train/eval — it is true of the
legacy support pair too), so B7 must publish a held-out-text split beside the headline rather
than quote 1.0 as the isolate's answer.

## B6: Gate accuracy + oracle in the evaluation report

**What:** `training/predict.py` reports `report["choice_gate"]` (strict accuracy, fell-to-shared
and wrong-domain counts, per-domain) beside `per_domain` whenever the asset ships a bank, and gains
`--head-hint domain` (oracle: force each record's own domain head) so head quality separates from
gate quality. `benchmarks/report.py` renders the gate row.
**Where:** `training/predict.py`, `benchmarks/report.py`.
**Depends on:** B5 · **Requirement:** ADR-0016 consequences (gate accuracy published beside
per-domain accuracy).
**Done when:** a bank report carries the gate row; the oracle arm runs from the same command
shape.
**Tests:** `tests/test_training_evaluate.py` / `tests/test_benchmark_report.py` · **Gate:** full.
**Commit:** `9daf136 feat(evaluate): judge the choice gate beside per-domain accuracy` +
`bd6eaf0 feat(benchmarks): publish choice-gate accuracy beside per-domain accuracy`
**Status:** **Done (2026-10-01).** `report["choice_gate"]` carries strict / fell-to-shared /
wrong-domain counts (overall and per domain) plus `answers_routed_by`; the oracle arm runs
from the same command shape (`--head-hint domain`, or a shipped key, with the run failing
fast on a typo — the *wire* hint keeps its ADR-0016 fall-through) and the renderer prints the
gate row above the `noul` audit. Measured on the first 4,000 eval rows with the bank asset:
**strict 1.000, 0 to shared, 0 wrong domain** over 1,500 `choice` records (the oracle arm
gives the same accuracy, so on this slice the gate is not the bottleneck); the control arm's
legacy asset ships no keys, so forcing a head on it exits with a message naming the adapter.

## B7: Run the isolate and record the numbers

**What:** Regenerate/verify data (done — golden 45/45), fit both arms on the L4, evaluate:
baseline (re-measure), shared control, bank+gate, bank+oracle on `data/eval_en_domains.jsonl`;
refit calibration for the arms before quoting ECE. Record per-domain/per-primitive/gate numbers.
**Where:** `benchmarks/results/en_domains_bank*.json`, `data/preds_*`.
**Depends on:** B5, B6 · **Requirement:** B-5 cycle gates.
**Done when:** the four arms are one comparable set; STATE/BACKLOG quote only same-harness numbers.
**Tests:** — (measurement) · **Gate:** full.
**Commit:** `d4580b2 feat(evaluate): split accuracy by whether the row's input text was trained on` +
`docs(specs): record the ADR-0016 isolate in STATE and the B-5 gates`
**Status:** **Done (2026-10-01).** Four arms + the released reference on **one harness**
(`training.predict`, all 7,500 rows, calibration refit per arm, and a new `--train-data` split so
the 81% train/eval row collision cannot hide behind a headline). Baseline 0.879 → **0.964**
overall; `support` **0.964 ≥ 0.85 ✓**; worst new domain **0.963 ≥ 0.70 ✓**; gate **strict 1.000 /
0 shared / 0 wrong**; per-domain ECE **0.021–0.027** (all five under 0.05); gap to the released
adapter's `support` cell **0.086 → 0.008 (91% closed)**; on the 473 never-seen `choice` rows
0.892 → **0.998**. **The isolate's real answer: the shared-head control (0.9633) is 3 rows
behind the bank (0.9637)** — labels, not per-domain capacity, closed the gap (L-012), so *which
arm to publish* stays open. Numbers and artifacts: BACKLOG B-5; loop: `/tmp/opencode/b7_run.sh`.

## B8: Docs + changelog

**What:** `docs/huggingface.md` §4 (keyed asset, legacy fallback, warn-by-name),
`docs/training.md` (bank training + fit script + reproduction commands), `CHANGELOG.md`
`[Unreleased]`, `docs/architecture.md` note where the backend seam gained the keyword.
**Where:** docs as listed.
**Depends on:** B1–B7 · **Requirement:** release hygiene.
**Done when:** every doc that names `choice_head.json` describes both formats.
**Tests:** `tests/test_docs_site.py` · **Gate:** full.
**Commit:** `78e5761 docs: document the keyed choice-head bank and its gate`
**Status:** **Done (2026-10-01).** `docs/huggingface.md` §4 now carries both asset shapes, the
gate's priority order and the B-8 warn-by-name contract; `docs/training.md` documents bank
training, the fitter (with its `--dry-run` and fit commands) and the new `choice_gate` /
`text_seen` report rows; `docs/architecture.md` records the additive `choice_head` keyword on the
seam and the bank inside `encoder.py`; `CHANGELOG.md` `[Unreleased]` has an **Added** section for
the feature and a **Changed** entry for the isolate's gates. `mkdocs build --strict` passes.

**Cycle status:** **B1–B8 complete, published and released as `v0.5.0` (2026-10-01)** — the bank
is `checkpoints/en` and `munod/tachyone-en` (`c00c174d`); docs, report, model card and CHANGELOG
updated; fast path (0 flips), probes and head-to-head all re-run against the new revision; tag
pushed, CI green, GitHub release published as *Latest*. Next items: **B-5b** (multilingual bank
keys, unblocked by the format) and **B-1** (multilingual `choice`).
