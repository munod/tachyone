# Public probes

**Externally comparable numbers.** Everything here is measured on public datasets
Tachyone did not train on, through the same `benchmarks/compare.py` metrics used in
[`docs/compare.md`](../docs/compare.md). Evaluation only: no row on this page has ever
reached `training/`, which is exactly what makes the numbers mean something.

> **Do not read these next to `benchmarks/report.md` as if they were the same test.**
> That report is the *in-sample synthetic* split (it shares states with the training
> data — `B-9`, lesson L-005); these are public distributions. The table at the bottom
> shows both, labelled, and makes no regression claim across them.

### How to read this page

- **Judge each number against that probe's chance level**, stated in its section — not
  against the synthetic 1.000. Different tasks, different option counts,
  no shared distribution with the training data.
- **Temperature fits (`typed-decisions` T=20, `massive` T=1.75, `xnli` T=20).** A fit pinned at the grid ceiling (T=20.0)
  flattens the distribution, so a low `ECE cal` is *bought* with confidence: read it next
  to `Conf` and `Brier`, and treat `ECE raw` as what the adapter actually ships (same
  caveat as [`docs/compare.md` §3](../docs/compare.md#3-why-not-another-open-system-one-scorer)).
- **These are the released adapters, zero-shot.** Both now cover five domains — the English one since B-5 (2026-10-01) and the multilingual one since B-5b (2026-10-02, support plus four new domains across six languages). None of these rows was ever in `training/`.

## Method

| | |
| --- | --- |
| Hardware | one GPU per run, recorded in each artifact's `environment.gpu` — the 2026-10-01/02 re-runs were a **single NVIDIA L4**, the 2026-09-29 numbers an RTX 3060 · Python 3.12 · stock encoder forward (no `fast`) |
| Engine | `encoder` backend through `tachyone.wire.answer`, released adapters (`munod/tachyone-en`, `munod/tachyone-multi`) selected per row by the language router |
| Metrics | one implementation of accuracy / 10-bin ECE / Brier; a failed answer counts as **wrong** |
| Temperature | fitted per probe on that probe's own development split, never on the split being reported |
| Caps | rows are taken deterministically (dataset order) up to `--per-task` per task; the exact cap is in each section |

## `typed-decisions` — LocalLLaMA/typed-decisions (Apache-2.0)

- **Licence:** Apache-2.0 — dataset card front-matter `license: apache-2.0`.
- **Size:** 4 configs + `all`, n<1K rows each (3.15 MB).
- **Chance level:** mixed — `noul` 0.50, `choice` 0.25-0.50, `score` 0.20-0.33.
- **How it is mapped:** Already in our wire shape. Multi-question rows are flattened to **one row per question** — deliberately conservative, because it ignores the `choice` head's ability to score several options of one question in a single pass. `task` is the dataset's own config (workflow), and each question keeps its native `noul` / `choice` / `score` type. Two `noul` questions ship without `criteria` and are answered as bare true/false (`criteria: null`), which the wire allows.

### tachyone (encoder)

2000 rows · `test` split · 500 per task · temperature fitted on train (capped) (T=20.0) · NVIDIA L4 · Python 3.12.3

#### Quality

| Engine | n | answered | Accuracy | ECE raw | ECE cal | Brier | Conf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tachyone (encoder) | 2000 | 2000 | 0.306 | 0.096 | 0.018 | 0.685 | 0.323 |

#### Performance

| Engine | p50 (ms) | p95 (ms) | items/s | JSON ok | RSS (MiB) | VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tachyone (encoder) | 121.43 | 290.80 | 6.7 | 1.000 | 2483 | 1831 |

#### Per task: accuracy and calibration

| Task | Engine | n | Accuracy | ECE raw | ECE cal | Conf |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| agent_trace_observability | tachyone (encoder) | 500 | 0.256 | 0.126 | 0.073 | 0.309 |
| customer_service | tachyone (encoder) | 500 | 0.260 | 0.146 | 0.042 | 0.286 |
| invoice_processing | tachyone (encoder) | 500 | 0.446 | 0.194 | 0.091 | 0.355 |
| security_incidents | tachyone (encoder) | 500 | 0.262 | 0.156 | 0.082 | 0.344 |

## `massive` — AmazonScience/massive (CC-BY-4.0)

- **Licence:** CC-BY-4.0 — `LICENSE` in the dataset repo and in the official S3 tarball.
- **Size:** 1M utterances, 60 intents, 51 locales (v1.1 tarball is 40 MB).
- **Chance level:** 0.017 (1 of 60 intents).
- **How it is mapped:** `state` is the utterance, `options` the fixed 60-intent vocabulary, `task` the **language** — which is what makes the per-language accuracy/ECE table that `B-1` needs come out of the same run. Rows are routed to the English checkpoint for `en` and to the multilingual checkpoint for the other six.

### tachyone (encoder)

3584 rows · `test` split · 512 per language · temperature fitted on dev (capped) (T=1.75) · NVIDIA L4 · Python 3.12.3

#### Quality

| Engine | n | answered | Accuracy | ECE raw | ECE cal | Brier | Conf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tachyone (encoder) | 3584 | 3584 | 0.074 | 0.042 | 0.036 | 0.967 | 0.067 |

#### Performance

| Engine | p50 (ms) | p95 (ms) | items/s | JSON ok | RSS (MiB) | VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tachyone (encoder) | 319.98 | 452.68 | 2.8 | 1.000 | 3119 | 2960 |

#### Per task: accuracy and calibration

| Task | Engine | n | Accuracy | ECE raw | ECE cal | Conf |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| de | tachyone (encoder) | 512 | 0.104 | 0.041 | 0.039 | 0.075 |
| en | tachyone (encoder) | 512 | 0.080 | 0.049 | 0.056 | 0.024 |
| es | tachyone (encoder) | 512 | 0.074 | 0.026 | 0.043 | 0.059 |
| fr | tachyone (encoder) | 512 | 0.076 | 0.115 | 0.077 | 0.120 |
| it | tachyone (encoder) | 512 | 0.057 | 0.060 | 0.054 | 0.075 |
| nl | tachyone (encoder) | 512 | 0.064 | 0.054 | 0.014 | 0.062 |
| pt | tachyone (encoder) | 512 | 0.061 | 0.037 | 0.032 | 0.051 |

## `xnli` — facebook/xnli (CC BY-NC 4.0)

- **Licence:** CC BY-NC 4.0 — `LICENSE` in `facebookresearch/XNLI` (the HF card carries **no** license tag and its Licensing Information section is a placeholder).
- **Size:** en: 5,010 test / 2,490 validation pairs.
- **Chance level:** 0.333 (3 labels).
- **How it is mapped:** `state` is `Premise: …\nHypothesis: …`, the options are `{entailment, neutral, contradiction}` in the dataset's label order, and the question is a fixed instruction. Scored as `choice`, **not** as the alternative `noul` formulation (a design decision: the harness scores a full 3-way distribution directly).

### tachyone (encoder)

5010 rows · `test` split · 6000 per task · temperature fitted on validation (capped) (T=20.0) · NVIDIA L4 · Python 3.12.3

#### Quality

| Engine | n | answered | Accuracy | ECE raw | ECE cal | Brier | Conf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tachyone (encoder) | 5010 | 5010 | 0.333 | 0.068 | 0.004 | 0.667 | 0.337 |

#### Performance

| Engine | p50 (ms) | p95 (ms) | items/s | JSON ok | RSS (MiB) | VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tachyone (encoder) | 63.04 | 80.91 | 15.1 | 1.000 | 2965 | 2829 |

#### Per task: accuracy and calibration

| Task | Engine | n | Accuracy | ECE raw | ECE cal | Conf |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| en | tachyone (encoder) | 5010 | 0.333 | 0.068 | 0.004 | 0.337 |

## Synthetic vs public

| Evaluation set | Nature | n | Accuracy | ECE |
| --- | --- | ---: | ---: | ---: |
| in-sample synthetic eval (`benchmarks/report.md`) | synthetic, shares states with training data | 1500 | 1.000 | 0.188 |
| typed-decisions `test` | public, Apache-2.0 | 2000 | 0.306 | 0.096 |
| massive `test` | public, CC-BY-4.0 | 3584 | 0.074 | 0.042 |
| xnli `test` | public, CC BY-NC 4.0 | 5010 | 0.333 | 0.068 |

Raw (uncalibrated) ECE is what the engine shipped with; see
[`docs/compare.md`](../docs/compare.md#3-why-not-another-open-system-one-scorer) for why ECE alone is not a quality metric.

## Reproduce

```bash
OUT=benchmarks/results
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('LocalLLaMA/typed-decisions', repo_type='dataset', local_dir='data/typed-decisions')"
curl -L -o data/massive/raw/amazon-massive-dataset-1.1.tar.gz https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
tar -xzf data/massive/raw/amazon-massive-dataset-1.1.tar.gz -C data/massive --wildcards '1.1/data/en-US.jsonl' '1.1/data/pt-PT.jsonl' '1.1/data/es-ES.jsonl' '1.1/data/fr-FR.jsonl' '1.1/data/de-DE.jsonl' '1.1/data/it-IT.jsonl' '1.1/data/nl-NL.jsonl'
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/xnli', repo_type='dataset', allow_patterns=['en/*', 'README.md'], local_dir='data/xnli')"

uv run python -m benchmarks.compare run --engine tachyone --format probe \
  --probe typed-decisions --data data/typed-decisions --per-task 500 \
  --out $OUT/probe_typed_decisions.json
uv run python -m benchmarks.compare run --engine tachyone --format probe \
  --probe massive --data data/massive --per-task 512 \
  --languages en,pt,es,fr,de,it,nl --out $OUT/probe_massive.json
uv run python -m benchmarks.compare run --engine tachyone --format probe \
  --probe xnli --data data/xnli --per-task 6000 \
  --out $OUT/probe_xnli.json
uv run python -m benchmarks.probes render --out benchmarks/probes.md \
  --synthetic benchmarks/results/en.json $OUT/probe_*.json
```

Artifacts live under `benchmarks/results/` (gitignored); each records its own command,
hardware, fitted temperature and per-task breakdown in `environment.command`.

## Citations and licences

- **LocalLLaMA/typed-decisions** — Apache-2.0. LocalLLaMA, *Typed Decisions* (Hugging Face dataset), Apache-2.0.
- **AmazonScience/massive** — CC-BY-4.0. FitzGerald et al., *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages*, arXiv:2204.08582 (2022).
- **facebook/xnli** — CC BY-NC 4.0. Conneau et al., *XNLI: Evaluating Cross-lingual Sentence Representations*, EMNLP 2018, arXiv:1809.05053. CC BY-NC 4.0.

XNLI is **non-commercial**; the tables above are derived metrics (our model's scores),
not a redistribution of the corpus — the data is never committed to this repository.
