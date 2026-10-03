# P4 submission record

Filed **2026-10-02** as [`fstandhartinger/jevbench#182`](https://github.com/fstandhartinger/jevbench/issues/182) — title and body below,
verbatim (the body is what GitHub received).

---

# [bench request]: Tachyone (offline Hugging Face artifact, native TypeSafe probabilities)

Please evaluate **Tachyone** under the current JevBench v1.5 protocol and add it to the board
if it qualifies.

## Candidate

| Field | Value |
| --- | --- |
| Public model name | Tachyone (English checkpoint `tachyone-en`) |
| Weights | [munod/tachyone-en](https://huggingface.co/munod/tachyone-en) at pinned revision [`1c88ebef8f15f68e8a6583c564d636221f22c29f`](https://huggingface.co/munod/tachyone-en/tree/1c88ebef8f15f68e8a6583c564d636221f22c29f) |
| Base and adaptation | `answerdotai/ModernBERT-large` (Apache-2.0); LoRA r=16 on the trunk plus a low-rank `choice` head and a per-domain head bank. Single encoder forward pass — **no generation, no sampling, no autoregressive decoding** |
| License | Apache-2.0 (code and adapter); base Apache-2.0. Training sources: MultiNLI (mixed CC-BY-3.0 / CC-BY-SA-3.0 / MIT / other), BoolQ (CC-BY-SA-3.0), Banking77 (CC-BY-4.0), each pinned by git revision and sha256 in [`training/data/jev_sources.lock.json`](https://github.com/munod/tachyone/blob/main/training/data/jev_sources.lock.json) |
| Protocol | TypeSafe-compatible structured responses for `choice`, `score`, and `noul` on `POST /v1/systemone` |
| Probability source | Native structured distributions: cosine scores over the declared options/levels plus the trained low-rank head, softmaxed over exactly the declared labels (`noul` is a sigmoid over the question–state cosine). Nothing is generated, so every number is a model output, not parsed text |
| Calibration | `temperature_calibration.json` (`choice` T=1.5, `score` **pinned at 0.1**) and `confidence_calibration.json`, a `noul` map from training-state similarity to confidence. **Neither changes a decision**: the `choice` argmax and the `score` expected value are preserved by construction, so the accuracy below is identical with and without the calibration assets |

This is an **offline artifact submission**. We are not providing an online endpoint or API
credential — please download the pinned Hugging Face revision and run the sealed evaluation in
your own environment.

## Inference (offline, no network after the download)

```bash
git clone https://github.com/munod/tachyone && cd tachyone
uv sync --extra train                     # or: pip install "tachyone[train] @ git+https://github.com/munod/tachyone"

# serve with an explicit local adapter (no Hub lookup at request time)
TACHYONE_BACKEND=encoder TACHYONE_PORT=8756 \
  TACHYONE_ADAPTERS="tachyone-en=$PWD/checkpoints/en" TACHYONE_PRELOAD=tachyone-en \
  uv run tachyone-serve
```

A single decision, verifiable without the harness:

```bash
curl -s -X POST http://127.0.0.1:8756/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"Section 7.2: refunds are not permitted after the cooling-off period ends.",
       "model":"tachyone-latest",
       "questions":{"decision":{"type":"noul",
         "instructions":"Is the requested action permitted?",
         "criteria":{"false":"A condition is missing or a prohibition applies.",
                     "true":"Every required condition is established."}}}}'
```

The response carries `noul`, `choice`/`score` with `probabilities` over exactly the declared
labels, and `confidence`. Your `typesafe` adapter needs no changes — it consumed our
`/v1/systemone` endpoint as-is (231/231 strict-valid, 0 renormalised, 0 failed).

## Artifact verification at the pinned revision

```
adapter_model.safetensors   9046aeb739e4ccdc0e788411e7d142c8f8dcacf9bb17a0ab918b41cdbe3ccaef
confidence_calibration.json ae96187f52fa762dc664b652524a50cf4b70788e4d4aee8ee185846016eacd73
state_prototypes.json       7cb1d45cfd29de6f96623197e564e6bd49c340a23d81cccfdad4a88ce531bbc6
temperature_calibration.json 6463e272b3f6f9fb8aa2dcf87f3049b3502ae89bc92999fefce9101cbcf043db
README.md                   836451260c5f1801cd367d9d49d83a8baa22c99d87728b567ade623d7464bfc6
```

Inference source: [`munod/tachyone`](https://github.com/munod/tachyone) at commit
`538ac68` (Apache-2.0); the encoder math lives in
[`src/tachyone/backends/encoder.py`](https://github.com/munod/tachyone/blob/main/src/tachyone/backends/encoder.py)
and the wire in [`src/tachyone/wire.py`](https://github.com/munod/tachyone/blob/main/src/tachyone/wire.py).

## Public diagnostic (not an official v1.5 score)

Our own run over the **231 published items** (`datasets/public/`: easy 48, standard 72,
hard 111), through your `typesafe` adapter against a local serve, on one NVIDIA L4. These are
our local numbers only — please use the sealed evaluation for the official result.

| Tier | n | Accuracy | Chance | Chance-corrected |
| --- | ---: | ---: | ---: | ---: |
| easy | 48 | 0.625 | 0.284 | 47.6 |
| standard | 72 | 0.417 | 0.317 | 14.6 |
| hard | 111 | 0.333 | 0.336 | 0.0 |
| judge | 0 | — | 0.292 | no public items |

| Primitive | n | Accuracy |
| --- | ---: | ---: |
| `noul` | 74 | 0.487 |
| `choice` | 139 | 0.360 |
| `score` | 18 | 0.611 (ordinal MAE 0.575) |

Public-item axes under the frozen method's published-item view
(`jevbench/composite_v13.py` — v1.4/v1.5 fold in the sealed accuracy, which only you have):

| Axis | Value | Basis |
| --- | ---: | --- |
| Intelligence | **15.0** | judge renormalised over three tiers |
| Calibration | **52.6** | ECE-only (`100·(1 − ECE/0.5)`); ECE **0.2369**, Brier **0.7451** |
| Speed | **84.9** | raw p50 70.0 ms / p95 487.9 ms → adjusted 0.290 s / 1.126 s (`×2 + 0.15 s`) |
| Cost | **78.8** | 510 input tokens/decision → **$0.00510 per 1,000 decisions** at $0.01/M input, $0 output (encoder size class) |
| **Composite** | **4.28** | `(I/50)²` multiplier 0.089 |

Schema: **231/231 strict-valid, 0 renormalised, 0 failed.** We record two misses on our own
side rather than omit them: our internal target was Intelligence ≥ 50 (landed 15.0) and a
calibration target of ≥ 60 (landed 52.6).

## Disclosure

- **No JevBench item — public or sealed — was used for training or calibration.** Every data
  builder asserts zero normalized-text overlap with `datasets/public/` before writing a file,
  and the public items have been evaluation-only from the start. The sealed items were never
  seen, requested or approximated.
- **Benchmark-directed data authoring:** our training mixture does follow the family taxonomy
  your hard tier publishes (`long_policy`, `multi_hop`, `temporal_numeric`, `trap`, …) with
  records we generated ourselves from committed content and executable rule trees. No item
  text, wording or answer key from this repository is in the training set; the overlap audit
  above is what checks it.
- The public diagnostic is a **measurement**, not a target we tuned against: the calibration
  assets are fitted on a holdout of never-trained records, and the accuracy above is
  byte-identical with those assets removed (they move confidence only).

## Reproducibility and requested run

The requested method is the frozen JevBench v1.5 method file at upstream commit
[`bb05a335bc809e61b20c0f745d25499a82b326fc`](https://github.com/fstandhartinger/jevbench/tree/bb05a335bc809e61b20c0f745d25499a82b326fc).
The SHA-256 of `docs/METHOD-v1.5.md` at that commit is:

```
c25d3d8b8512e4d93370a9e0c99705d19b2a9389956ca33b8a4bd2b0ec501c07
```

Please run the maintainer-controlled v1.5 sealed evaluation from the pinned artifact, report
the standard accuracy/calibration/latency fields, and list the model on the current board if
accepted. No JevBench sealed questions, answers, raw requests, or credentials are included in
this issue.

Thank you for maintaining the benchmark.
