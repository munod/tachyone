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
  against the synthetic 0.945. Different tasks, different option counts,
  no shared distribution with the training data.
- **All three temperature fits landed on the grid ceiling (T=20.0).** That flattens the
  distribution, so a low `ECE cal` is *bought* with confidence: read it next to `Conf`
  and `Brier`, and treat `ECE raw` as what the adapter actually ships (same caveat as
  [`docs/compare.md` §3](../docs/compare.md#3-why-not-another-open-system-one-scorer)).
- **These are the released adapters, zero-shot**, trained on support tickets with four
  team labels. Broadening what they know is `B-5`; these numbers are the evidence for it.

## Method

| | |
| --- | --- |
| Hardware | single RTX 3060 12GB · Python 3.12 · stock encoder forward (no `fast`) |
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

2000 rows · `test` split · 500 per task · temperature fitted on train (capped) (T=20.0) · NVIDIA GeForce RTX 3060 · Python 3.12.13

#### Quality

| Engine | n | answered | Accuracy | ECE raw | ECE cal | Brier | Conf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tachyone (encoder) | 2000 | 2000 | 0.330 | 0.498 | 0.066 | 0.700 | 0.393 |

#### Performance

| Engine | p50 (ms) | p95 (ms) | items/s | JSON ok | RSS (MiB) | VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tachyone (encoder) | 123.35 | 343.18 | 5.6 | 1.000 | 2560 | 1823 |

#### Per task: accuracy and calibration

| Task | Engine | n | Accuracy | ECE raw | ECE cal | Conf |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| agent_trace_observability | tachyone (encoder) | 500 | 0.250 | 0.537 | 0.095 | 0.345 |
| customer_service | tachyone (encoder) | 500 | 0.370 | 0.470 | 0.041 | 0.357 |
| invoice_processing | tachyone (encoder) | 500 | 0.288 | 0.572 | 0.216 | 0.458 |
| security_incidents | tachyone (encoder) | 500 | 0.412 | 0.411 | 0.052 | 0.410 |

## `massive` — AmazonScience/massive (CC-BY-4.0)

- **Licence:** CC-BY-4.0 — `LICENSE` in the dataset repo and in the official S3 tarball.
- **Size:** 1M utterances, 60 intents, 51 locales (v1.1 tarball is 40 MB).
- **Chance level:** 0.017 (1 of 60 intents).
- **How it is mapped:** `state` is the utterance, `options` the fixed 60-intent vocabulary, `task` the **language** — which is what makes the per-language accuracy/ECE table that `B-1` needs come out of the same run. Rows are routed to the English checkpoint for `en` and to the multilingual checkpoint for the other six.

### tachyone (encoder)

3584 rows · `test` split · 512 per language · temperature fitted on dev (capped) (T=18.15) · NVIDIA GeForce RTX 3060 · Python 3.12.13

#### Quality

| Engine | n | answered | Accuracy | ECE raw | ECE cal | Brier | Conf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tachyone (encoder) | 3584 | 3584 | 0.013 | 0.136 | 0.008 | 0.983 | 0.020 |

#### Performance

| Engine | p50 (ms) | p95 (ms) | items/s | JSON ok | RSS (MiB) | VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tachyone (encoder) | 96.80 | 191.47 | 8.2 | 1.000 | 3212 | 2951 |

#### Per task: accuracy and calibration

| Task | Engine | n | Accuracy | ECE raw | ECE cal | Conf |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| de | tachyone (encoder) | 512 | 0.014 | 0.106 | 0.006 | 0.020 |
| en | tachyone (encoder) | 512 | 0.018 | 0.267 | 0.007 | 0.025 |
| es | tachyone (encoder) | 512 | 0.006 | 0.134 | 0.013 | 0.019 |
| fr | tachyone (encoder) | 512 | 0.010 | 0.068 | 0.008 | 0.018 |
| it | tachyone (encoder) | 512 | 0.010 | 0.116 | 0.010 | 0.020 |
| nl | tachyone (encoder) | 512 | 0.023 | 0.176 | 0.002 | 0.022 |
| pt | tachyone (encoder) | 512 | 0.008 | 0.086 | 0.011 | 0.019 |

## `xnli` — facebook/xnli (CC BY-NC 4.0)

- **Licence:** CC BY-NC 4.0 — `LICENSE` in `facebookresearch/XNLI` (the HF card carries **no** license tag and its Licensing Information section is a placeholder).
- **Size:** en: 5,010 test / 2,490 validation pairs.
- **Chance level:** 0.333 (3 labels).
- **How it is mapped:** `state` is `Premise: …\nHypothesis: …`, the options are `{entailment, neutral, contradiction}` in the dataset's label order, and the question is a fixed instruction. Scored as `choice`, **not** as the alternative `noul` formulation (a design decision: the harness scores a full 3-way distribution directly).

### tachyone (encoder)

5010 rows · `test` split · 6000 per task · temperature fitted on validation (capped) (T=20.0) · NVIDIA GeForce RTX 3060 · Python 3.12.13

#### Quality

| Engine | n | answered | Accuracy | ECE raw | ECE cal | Brier | Conf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tachyone (encoder) | 5010 | 5010 | 0.333 | 0.269 | 0.021 | 0.670 | 0.354 |

#### Performance

| Engine | p50 (ms) | p95 (ms) | items/s | JSON ok | RSS (MiB) | VRAM (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tachyone (encoder) | 46.03 | 65.70 | 21.9 | 1.000 | 2993 | 2821 |

#### Per task: accuracy and calibration

| Task | Engine | n | Accuracy | ECE raw | ECE cal | Conf |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| en | tachyone (encoder) | 5010 | 0.333 | 0.269 | 0.021 | 0.354 |

## Synthetic vs public

| Evaluation set | Nature | n | Accuracy | ECE |
| --- | --- | ---: | ---: | ---: |
| in-sample synthetic eval (`benchmarks/report.md`) | synthetic, shares states with training data | 1500 | 0.945 | 0.034 |
| typed-decisions `test` | public, Apache-2.0 | 2000 | 0.330 | 0.498 |
| massive `test` | public, CC-BY-4.0 | 3584 | 0.013 | 0.136 |
| xnli `test` | public, CC BY-NC 4.0 | 5010 | 0.333 | 0.269 |

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
