# Data & Training Strategy

**Status:** Implemented (M4) and executed full-scale on the RTX 3060 12GB (English + multilingual
LoRA, 6k train / 1.5k eval, temperature calibration; measured numbers in `benchmarks/report.md`).
**Related:** `docs/adr/ADR-0005-encoder-rlcd.md`, `.specs/features/training-calibration/spec.md`.

---

## Objective

Produce a local encoder backend that answers `noul` / `choice` / `score` in a single forward
pass, in 100+ languages, with **calibrated** probabilities — trainable and reproducible on a
single **RTX 3060 12GB**.

---

## Pipeline overview

```mermaid
graph LR
    A["generate_data.py<br/>synthetic JSONL"] --> B["finetune_rlcd.py<br/>LoRA/QLoRA + RLCD"]
    B --> C["fit_calibration.py<br/>temperature / ECE"]
    C --> D["evaluate.py<br/>accuracy, ECE, latency"]
    D --> E["checkpoint + model card"]
```

Every stage is a committed script with a committed config and a fixed seed.

---

## 1. Data generation (`training/generate_data.py`)

**Goal:** deterministic, streamed synthetic supervision for the three primitives.

- **Determinism:** same seed + config → byte-identical output.
- **Streaming:** write JSONL incrementally; never hold the full dataset in memory.
- **Coverage:** per primitive, per language group, with hard negatives and boundary cases
  (empty and very long inputs). Labels are never ambiguous by construction: the near-tie /
  index-parity devices that made a label unreadable from the text were removed in B-11/B-12.
- **Localization (B-1):** the `state`, `instructions`, `criteria`, `score` levels, and
  `noul`/`score` entities are all authored per language, so the multilingual checkpoint does not
  learn an English template.
- **`noul` labels come from the text (B-11):** `target = 1` iff the emitted state was drawn from
  the `request` phrase bank, `0` for the `neutral` bank and for the empty boundary state (which
  reads as no request). The label never depends on the loop index, so no request-toned record can
  carry `0` — the old index-parity rule contradicted 50% of `eval_en.jsonl`'s request-toned rows
  and made `noul` accuracy label-bound (L-008). Only `noul.target` bytes move against the
  pre-B-11 datasets; states and questions are unchanged.
- **`score` and `choice` labels come from the text too (B-12):** `score` takes the level of the
  tone the state was written in — the old `index % 13` "near-tie" downgrade contradicted 7.8% of
  rows and capped the primitive at ~0.888 whatever the model learned — and an **empty** `score`
  state takes the middle level (no signal). `choice` takes the option the state names, and an
  **empty** `choice` state answers to the domain's catch-all `other` (every committed domain
  ships one; `ecommerce` gained it in B-12, which re-cycled its four options into five). The empty
  boundary case therefore exists in all three primitives with a learnable default instead of an
  index-derived label.
- **Domains (B-5):** every domain lives in its own committed file,
  `training/data/domains/<domain>.json`, holding its option labels, option terms and
  descriptions, entities, phrase banks, `score` levels, `noul` criteria and per-primitive
  instructions — one shape for every domain, per language. `--domains a,b,c` selects the domains
  to emit (default `support`, the original four-team triage); `DataConfig.domains` is the field.
  Two rules keep the existing datasets reproducible:
  - `support` keeps the **legacy record seed** (`seed:kind:index:language`), so its records are
    byte-identical whether generated alone or inside a five-domain run;
  - the `domain` field is written only for non-default runs, so `data.json`, `data_multi.json`
    and `data_noisy.json` regenerate byte-for-byte.
  `tests/test_training_generate.py` pins both rules with golden hashes and checks each committed
  data config against the dataset it documents.
- **Localization completeness (B-5b):** every committed domain ships a full table for each tag in
  `DEFAULT_LANGUAGES` (`en, pt, es, fr, de, it, nl`) — entities, instructions, levels, criteria,
  ≥ 4 option terms and ≥ 20-char descriptions per option, and all five phrase tones with their
  `{entity}`/`{distractor}` placeholders. The completeness test iterates the seven languages
  rather than `en` alone, because a table that only ships English produces English sentences
  under a non-`en` `lang` tag — the L-008 artifact in its second form. Runtime fallback to the
  English table still exists (`DomainData._lang`), but shipped domains may not rely on it.
- **Input-noise augmentation (B-4):** `--noise-rate r` (config field `noise_rate`) applies one
  deterministic surface edit — char swap/delete, accent strip, casing flip, or terminal-punctuation
  drop — to `r` of records' `state` only (labels/questions untouched). `r=0` reproduces the clean
  dataset byte-for-byte; each record draws from its own RNG (`(seed[, domain], kind, index,
  language, noise)`) so noise stays independent of the cyclic label (L-003). Evaluate the noisy
  view separately: the harness reports it under `report["noisy"]` when `--noise-rate` is passed.

### JSONL record format

```json
{"id": "noul-000001", "type": "noul", "state": "…", "instructions": "…", "criteria": {"true": "…", "false": "…"}, "target": 1, "lang": "en", "source": "synthetic"}
{"id": "choice-000001", "type": "choice", "state": "…", "instructions": "…", "criteria": {"a": null, "b": "…"}, "target": "b", "lang": "pt", "source": "synthetic"}
{"id": "score-000001", "type": "score", "state": "…", "instructions": "…", "criteria": ["poor","fair","good"], "target": 2, "lang": "es", "source": "synthetic", "domain": "voice"}
```

| Field | Meaning |
| --- | --- |
| `id` | Stable id, unique **within** a domain (a multi-domain run repeats ids across domains — pair it with `domain`) |
| `type` | `noul` / `choice` / `score` |
| `state` | The judged subject |
| `instructions` | The atomic question |
| `criteria` | Primitive-specific options/levels |
| `target` | Label (`0/1`, option key, or level index) |
| `lang` | Language tag for routing/stratification |
| `source` | Provenance (`synthetic`, `public`, `human`) |
| `domain` | Domain tag. Absent in single-domain `support` runs (byte-identity); present otherwise, and records without it evaluate as `support` |

### Data sources

- Synthetic generation (primary, deterministic).
- Public datasets for evaluation probes (MASSIVE, XNLI, typed-decisions) — **evaluation only**,
  never training, to keep benchmark claims honest.
- Optional human-curated seed sets for hard cases.

---

## 2. Fine-tuning (`training/finetune_rlcd.py`)

**Goal:** adapt ModernBERT-class (English) and mmBERT-base (multilingual) encoders with three
task heads, within 12GB VRAM.

### Architecture

- Shared encoder trunk.
- Three heads: `noul` (binary logit), `choice` (logits over options), `score` (logits over levels).
- Heads are trained jointly; the trunk is shared.

### The `choice` head: shared, or a per-domain bank (ADR-0016)

`choice` is the one primitive with a trained scorer (`ChoiceScorer`: a low-rank residual on top of
the cosine base). Since ADR-0016 the trainer can learn **a bank**: when the training records carry
a `domain` field, every `choice` record updates **its own domain's head *and* the shared head**,
the shared head keeps training on all records as the fallback, and inference picks exactly one
head per question through a deterministic gate (hint → lexical signature → shared).

- The asset grows to `{shared, domains: {name: {head, signatures}}}`; each head ships the
  committed option labels, terms and descriptions from `training/data/domains/<name>.json` as its
  signatures, so the runtime gate matches questions without the data files.
- Records **without** a `domain` field — every legacy single-domain config — train only the shared
  head and write the plain `{rank, w1, w2}` shape **byte-identically** to the pre-ADR-0016 writer
  (same init order, same RNG stream), so existing configs reproduce today's artifact.
- Config knobs are unchanged: `choice_rank` / `choice_init_std` apply to every head in the bank,
  and `domains` are validated against the committed files before training starts (L-011).

### The frozen-trunk isolate (`training/fit_choice_bank.py`)

To test whether the **head** axis — not the trunk — is what a five-domain checkpoint is missing,
the fitter freezes an existing trunk, encodes every `choice` record **once** through the runtime
encoder, and fits two arms from that single cache:

| arm | asset written | question it answers |
| --- | --- | --- |
| control | shared head only, refit | "would *any* refit do this?" |
| bank | shared + one head per domain, warm-started from the trunk's own head | "does per-domain capacity add anything?" |

Both arms write a loadable adapter dir — the trunk's adapter files copied bit-for-bit plus their
own `choice_head.json` and a `choice_bank_fit.json` recipe (the artifact carries its own recipe:
hyper-parameters, seed, record counts — L-011).

```bash
# validate config + data without torch:
uv run python -m training.fit_choice_bank --config training/configs/fit_bank_en_domains.json --dry-run
# fit both arms (one encode pass, ~10 min on an L4):
uv run python -m training.fit_choice_bank --config training/configs/fit_bank_en_domains.json
# the multilingual counterpart (B-5b): trunk = the joint five-domain run's adapter
uv run python -m training.fit_choice_bank --config training/configs/fit_bank_multi_domains.json
```

The measured outcome — four arms on one harness — is in `.specs/project/BACKLOG.md` **B-5**:
refitting only the head took five-domain accuracy 0.879 → **0.964** on a frozen trunk, but the
**control sits 3 of 2,500 `choice` rows behind the bank**, so the corrected labels, not
per-domain capacity, closed the gap (lesson L-012). Keep the trainer's learning rate (**1e-4**):
at 1e-3 an Adam step is ~10% of these heads' weight RMS and the domain heads — which see only
~1/5 of the batches — degrade while the shared head still looks fine.

**B-5b is the same finding on the other checkpoint, twice.** Under joint training the
per-domain bank **lost** to its own shared head by 5.6 points overall (0.8732 vs 0.9295 — the
first joint bank run ever measured), while the frozen-trunk fit produced **0.9975 with bank and
shared tied at 1 row**: the structure is neutral under equal optimization (L-012 replicated),
joint-training exposure is what made the bank look bad, and the fitted bank is what shipped.

### Parameter-efficient training

| Technique | Why |
| --- | --- |
| LoRA (or QLoRA with 4-bit base) | Fits 12GB; small trainable footprint |
| Gradient checkpointing | Trades compute for memory |
| Gradient accumulation | Effective batch size without VRAM blowup |
| Mixed precision (bf16/fp16) | Speed + memory |
| Length-sorted batching | Fewer padding tokens |

### VRAM budget notes (RTX 3060 12GB)

- Full fine-tuning of large encoders is **not** feasible; LoRA/QLoRA is required.
- Two checkpoints are trained separately (English, multilingual), not simultaneously.
- If a config OOMs, reduce per-device batch first, then sequence length, then LoRA rank.
- Exact ceilings are measured and documented in M4-T6 (see B-002).

---

## 3. RLCD proper-scoring calibration (`training/finetune_rlcd.py`)

**Goal:** calibrated probabilities, not just correct labels.

- Train against a **strictly proper scoring rule** (e.g., log loss / Brier) over the primitive
  distributions. A proper scoring rule is minimized in expectation by reporting the true
  probability — this is what makes `confidence` meaningful.
- RLCD (reinforcement learning against the scoring rule) shapes the distribution beyond
  accuracy-only supervision.
- Probabilities are normalized before leaving the backend.

**Why proper scoring:** accuracy-only training produces overconfident, miscalibrated
distributions. Proper scoring directly optimizes the quantity users consume (`confidence`).

---

## 4. Calibration fitting (`training/fit_calibration.py`)

**Goal:** minimize ECE on a held-out calibration split.

- Fit a single temperature (or per-primitive temperatures) on a **calibration split**.
- **Per-(primitive, language) temperatures (B-1, implemented):** `fit_calibration.py` keeps the
  per-primitive fit and adds a per-language fit under `by_language`; the runtime selects
  `kind:lang` → `kind` → global, detecting the language from the state text. A language below
  `min_samples` warns and falls back to the per-primitive value.
- **Never** fit on the test split.
- Report ECE before/after; target is provisional (NFR-C06, `ECE ≤ 0.05` **[open]**).
- Persist the fitted temperature with the checkpoint.

### Evidence-conditioned confidence (P3)

One temperature cannot serve two regimes: fitted where the model is saturated it sharpens to
`T=0.05` and zeroes the axis off-domain (L-015), fitted where it is not, home calibration
suffers. P3 therefore gives `noul` a confidence that depends on the *evidence*, in three
committed steps — in this order, because the map is fitted **against** the bank:

```bash
# 1. the holdout: a stride sample of the real training set + never-trained source rows
uv run python -m training.build_calibration_holdout --public-dir <jevbench>/datasets/public

# 2. the evidence signal: k-means centroids of the checkpoint's own training states
uv run python -m training.build_prototypes --data data/train_en_domains.jsonl \
  --out checkpoints/en/state_prototypes.json --k 32 --adapter checkpoints/en

# 3. natural predictions (no map) -> both assets
uv run python -m training.predict --data data/calibration_holdout.jsonl \
  --adapter checkpoints/en --prototypes checkpoints/en/state_prototypes.json \
  --no-confidence --out-predictions data/calibration_holdout_preds.jsonl
uv run python -m training.fit_confidence --predictions data/calibration_holdout_preds.jsonl \
  --prototypes checkpoints/en/state_prototypes.json --out-dir checkpoints/en
```

`fit_confidence` writes `temperature_calibration.json` (the per-primitive grid fit, with
**`score` pinned** — its answer *is* the expected value, so a temperature change would move
the answer) and `confidence_calibration.json` (the bank embedded with the monotone `noul`
knots). Rebuilding after a retrain means rerunning all three: the bank describes the states
the checkpoint was trained on, and the map describes that bank.

---

## 5. Evaluation (`benchmarks/` + `training/evaluate.py`)

Report per primitive and per language group:

| Metric | Definition |
| --- | --- |
| Accuracy | Correct label rate |
| ECE | Expected calibration error over confidence bins |
| Latency | p50 / p95, batch=1 and batched |
| Throughput | requests/s vs batch size |
| Coverage | Languages/scripts exercised |
| `noul` accuracy per language | `report["noul_per_language"]` — that primitive's rows broken out per language, so it can be read against the label audit below (B-11) |
| Contradictory-label rate | `report["noul_labels"]` — every `noul` label judged against its own text (request → 1, neutral/empty → 0); states no phrase bank explains (surface noise) count as `unknown` and are never judged |
| `choice` gate accuracy | `report["choice_gate"]` — how often the bank's gate picked the record's **own** domain, with `fell_to_shared` (the designed fallback) and `wrong_domain` (the only harmful outcome) counted separately, per domain (ADR-0016). Present only when the loaded asset ships a bank |
| Seen-text split | `report["text_seen"]` — accuracy on rows whose **input text** occurs in the training set versus rows it never does, from `--train-data`. The shipped eval sets collide with their training rows on most `choice` rows, so this is the only line that separates memorization from generalization |
| Domain × language cells | `report["per_domain_language"]` — accuracy/ECE per `domain/lang` cell, emitted only when the eval set carries more than one domain (legacy reports stay byte-identical) and rendered worst-cell-first. `per_domain` hides the language spread inside a domain and `per_language` hides the domain spread inside a language; the worst-cell gate needs both (B-5b) |

`training/predict --head-hint domain` runs the **oracle arm** — every record forced onto its own
domain head — so head quality separates from gate quality; `--head-hint <key>` forces one head,
and a typo exits before the run instead of publishing an unforced arm as forced (the *wire* hint
keeps its ADR-0016 fall-through).

**Which harness measures what (lesson L-013).** `training.predict --adapter <dir>` measures
*that checkpoint* explicitly; `training.evaluate --backend encoder` goes through the wire and
the language router, where on multilingual rows 13–15% fall through to `tachyone-en` (empty or
short states) — so the routed number depends on whatever the *English* adapter happens to be
today (the published multilingual 0.743 stopped reproducing the day the English adapter was
republished). Gate one checkpoint's quality only on the explicit harness; read routed numbers
as product-level measurements that mix checkpoints by design.

`benchmarks/report.py` renders the last two **in one table**, because the pre-B-11 generator made
`de`/`es`/`nl` look weak when the labels, not the model, were the problem (L-008).

Public probes: **MASSIVE**, **XNLI** and **typed-decisions** are published, with licences and
reproduction commands, in [`benchmarks/probes.md`](https://github.com/munod/tachyone/blob/main/benchmarks/probes.md)
(B-7); the external nine-family `pngwn/system-one-decisions` head-to-head is in `docs/compare.md` §3.

---

## 6. Reproducibility

- Every stage takes a config file (seed, hyperparameters, data paths) committed under
  `training/configs/`.
- Re-running with the same seed/config must land within documented tolerance.
- Artifacts record: git commit, hardware, library versions, seed, command.
- No secrets or private data in configs or artifacts.

---

## Limitations & risks

| Limitation | Impact | Mitigation |
| --- | --- | --- |
| RTX 3060 12GB | Bounds model size and batch | LoRA/QLoRA, checkpointing, accumulation |
| Synthetic data bias | Model inherits generator biases | Mix sources; human seed sets; eval on public probes |
| Small calibration sets | Noisy temperature fit | Stratify; warn on small N; hold out test |
| Multilingual imbalance | Weak languages | Stratify generation; per-language ECE reporting; per-language temperature (B-1) |
| Two checkpoints | Memory pressure | `max_loaded`, LRU eviction (ROUTE-04) |

---

## Open questions

- **OD-3:** weights distribution — RESOLVED by `docs/adr/ADR-0010` (HF Hub on demand + local cache;
  prefetch with `hf download` or a warm-up run, cache-only via `TACHYONE_OFFLINE=1`).
- Exact ECE target and binning scheme — resolved by the measured target (ECE ≤ 0.05,
  10 bins, `min_samples` 30) used in `benchmarks/report.md`.
- Whether the typed-decisions checkpoint is trained — deferred; tracked as `BACKLOG.md` B-7.