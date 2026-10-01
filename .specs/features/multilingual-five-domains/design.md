# Multilingual Five-Domain Coverage (B-5b) — Design

**Spec:** `.specs/features/multilingual-five-domains/spec.md` · **Context:** `context.md`
**Size:** Large (data content → datasets → training → measurement → publication), but every
component **reuses B-5a/B-1 machinery** — no new runtime code, no wire change.

## Components

```
training/data/domains/{ecommerce,agent_tools,documents,voice}.json   ← +6 language tables each
        │ (completeness test parameterized over DEFAULT_LANGUAGES)
        ▼
training/configs/data_multi_domains.json   → data/train_multi_domains.jsonl   (30,000)
training/configs/data_eval_multi_domains.json → data/eval_multi_domains.jsonl  (7,500)
        │ (support halves byte-identical to train_multi/eval_multi ⊖ `domain`)
        ├──────────────────────────────┐
        ▼                              ▼
training/finetune_multi_domains.json   baseline eval: released checkpoints/multi
        │ (joint run: trunk LoRA + shared head + 5 domain heads + temperature)
        ▼                              ▼
checkpoints/multi_b5b (bank asset)     benchmarks/results/multi_domains_baseline.json  ← gates fixed HERE
        │
        ▼
training/predict.py arms (same harness, calibration refit per arm, --train-data split):
  1. bank + gate   (headline)
  2. bank + oracle (--head-hint <domain>)
  3. shared only   (derived asset: `domains` stripped → legacy shared-only behaviour)
        ▼
benchmarks/report.md + docs/* + munod/tachyone-multi  (B-5a publication pattern)
```

## Key design points

### 1. Localization is data-only

`DomainData` already falls back per language and `domain_signatures(name, languages)` already
accumulates terms/descriptions per language — **zero generator or trainer code changes**. The
work is authoring `languages.{pt,es,fr,de,it,nl}` tables mirroring each domain's `en` table
(same seven fields, same option keys, `{entity}`/`{distractor}` placeholders preserved), bounded
by the completeness test (strengthened to iterate `DEFAULT_LANGUAGES` instead of `en` only).

### 2. Dataset configs (byte-identity by construction)

| config | shape | invariant |
| --- | --- | --- |
| `data_multi_domains.json` | domains ×5, langs ×6, `per_domain {support: 6000}`, `per_type 1000`, seed 1 | support records ≡ `train_multi.jsonl` ⊖ `domain` (legacy seed prefix rule, `_seed_base`) |
| `data_eval_multi_domains.json` | domains ×5, langs ×6, `per_type 500`, seed 2 | support records ≡ `eval_multi.jsonl` ⊖ `domain` → the baseline's support cell measures the published support rows (anchor **0.8413** in the explicit-adapter harness; the routed 0.743 does not reproduce — L-013) |

English configs and datasets are not touched; the existing 45 golden hashes and
`test_committed_config_reproduces_its_shipped_dataset` prove it every gate.

### 3. Training recipe (`finetune_multi_domains.json`)

Base = the **published multilingual recipe** (B-12's `finetune_multi_b12.json`, the measured
winner) with the two fields the five-domain structure needs:

| field | value | rationale |
| --- | --- | --- |
| `model_id` / `lora_rank` / `lora_alpha` | mmBERT-base / **64** / 128 | published multi (rank-bump experiment, adopted) |
| `learning_rate` | 1e-4 | ADR-0016 fitter lesson: 1e-3 destroyed domain heads |
| `choice_rank` | **128** | mirrors the published en bank (`fit_bank_en_domains.json`); 32 was chosen when multi had *one* head — with five heads each sees ~1/5 of `choice` records, and en measured 128 as the fitting capacity (256 regressed) |
| `epochs` | **8** | the measured multi setting — the B-11/B-12 sweep showed 4 epochs drops `score` (0.648 → 0.854 at 8) |
| `seed` | 42 | pinned in the published multi recipe |
| `data_path` / `out_dir` | `data/train_multi_domains.jsonl` / `checkpoints/multi_b5b` | L-011: trainer writes `finetune_config.json`; reconcile before quoting |
| batch 8 × grad_accum 4, `max_len` 1024, bf16, grad-ckpt, `val_split` 0.1 | unchanged | published multi |

**Cost estimate:** 30k × 8 epochs = 240k record-epochs ≈ 1.67× the B-12 multi run
(18k × 8). L-011 timestamps put a comparable 21k × 8 run at ≈ 1 h 50; extrapolated **≈ 2.5–3.5 h
on one L4**. Confirmed with `--dry-run` before launch; if it exceeds ~4 h, `epochs 6` is the
documented fallback (recorded in BACKLOG before running, not after).

### 4. Measurement arms (L-012 discipline)

All three arms come from **one run** and differ by exactly one factor:

- **bank + gate** — what ships;
- **oracle** (`--head-hint <domain>`) — same asset, forced routing → isolates gate quality;
- **shared only** — the bank payload with `domains` stripped (loads as legacy shared-only) →
  measures what the bank buys over its own fallback, no second training run.

Attribution caveat published with the table: all three share the joint retrain (data + trunk +
head refit), so *bank vs shared* is the structural estimate and *each vs baseline* is the whole
intervention — the L-012 reading rule.

The **baseline** (released `checkpoints/multi`) is re-measured in the same harness on the same
rows before the arms are read; gate numbers are written to BACKLOG at that point.

**Instrument rule (learned in B5B-4, L-013):** every number in this cycle comes from
`training.predict` with the **explicit adapter** — never the routed runtime path, which answers
13–15% of multilingual rows with `tachyone-en` (empty/short states) and therefore changes
whenever *that* checkpoint is republished.

### 5. Report additions

`training/evaluate.py` gains `report["per_domain_language"]` — the same `_metrics` bucketing
already used by `per_domain`/`per_language`, keyed `(domain, lang)`. `benchmarks/report.py`
renders it (worst cell first). Legacy reports without a `domain` field are unaffected.

### 6. Publication

Promote `checkpoints/multi_b5b` → `checkpoints/multi` (keeping the previous build recoverable),
upload to `munod/tachyone-multi` with per-file sha256 verification and stale-file deletion,
regenerate the number-bearing docs from the result JSONs, re-run MASSIVE + fast-path parity
(multi path, 0 flips expected), decide the head-to-head by checking whether its quoted cells use
the multilingual engine, then CHANGELOG + spec status + release decision.

## Risks

| Risk | Mitigation |
| --- | --- |
| Support ↔ ecommerce lexical overlap misroutes the localized gate | `choice_gate` wrong-domain counts published; oracle arm separates; fallback = shared (bounded failure, ADR-0016) |
| Cutting nothing but adding 4 domains dilutes support in a fixed-capacity LoRA | support ≥ baseline gate; volume kept at 18k (context.md) |
| Training runs > 4 h | `--dry-run` + documented epochs-6 fallback before launch |
| Localization quality (agent-authored translations) | completeness test (counts/placeholders) + L-008-style audit: generate one record per (domain, language) and read it; native-level review of instructions/phrases by the author |
| 81% train/eval text collision (known generator property) | `--train-data` held-out split published beside the headline (B-7 discipline) |
| Config ≠ artifact (L-011) | reconcile `finetune_config.json` before quoting anything |

## Execution order

`B5B-1 (localize)` → `B5B-2 (datasets)` → `B5B-3 (cross cell)` → `B5B-4 (baseline + gates)` →
`B5B-5 (train)` → `B5B-6 (arms)` → `B5B-7 (verdict)` → `B5B-8 (publish)` → `B5B-9 (docs/close)`.

B5B-3 is independent of B5B-1/2 and can run in parallel; the baseline eval (B5B-4) needs only
B5B-2 and runs while the dataset settles — gates are fixed **before** B5B-5 launches.
