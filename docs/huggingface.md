# Publishing to the Hugging Face Hub

Tachyone ships **PEFT LoRA adapters**, not full checkpoints. Each adapter's
`adapter_config.json` records `base_model_name_or_path` (ModernBERT-large or mmBERT-base), so
consumers load the base trunk plus the adapter. Weights are fetched/cached locally at runtime
(ADR-0010); publishing is only about distribution.

> **Published (verified):**
> - <https://huggingface.co/munod/tachyone-en> (ModernBERT-large adapter, LoRA r=16)
> - <https://huggingface.co/munod/tachyone-multi> (mmBERT-base adapter, LoRA r=64)
>
> Both load via `PeftModel.from_pretrained(base, "munod/tachyone-en")` and predict.
>
> **Latest revision: 2026-10-02 (B-5b)** — commit [`9a3ef5a5`](https://huggingface.co/munod/tachyone-multi/commit/9a3ef5a5d38972ffe119cdfe6b4b6c7bb14da9d1)
> (`tachyone-multi`). The multilingual adapter now ships the **five-domain artifact**: an mmBERT
> trunk joint-trained on 30,000 multilingual five-domain records (support keeps its 18,000, four
> new domains 3,000 each) with its `choice` heads re-fitted on the **frozen** trunk — the recipe
> ships beside the weights as `choice_bank_fit.json`, and the stale `finetune_config.json` was
> deleted (it described the weights this replaces). On the five-domain split: **0.9975** overall
> (previous adapter zero-shot 0.561), `choice` 0.992, `noul`/`score` 1.000, per-domain 0.993–1.000,
> per-language ECE 0.001–0.004 (all six ≤ 0.05), gate strict **1.000** (0 to shared, 0 wrong
> domain), held-out text 0.996; on the support-only *routed* split the product-level number moves
> 0.743 → **0.895** (13–15% of multilingual rows route to `tachyone-en`, L-013). Fast path
> re-checked on this adapter: **3.66× p50, 0 top-label flips**. All six uploaded files were
> sha256-verified against the local build.
>
> Previous revision **2026-10-01 (B-5, ADR-0016)** — commits [`c00c174d`](https://huggingface.co/munod/tachyone-en/commit/c00c174dde0208b186cb503abfa0e15bdde0400d)
> (`tachyone-en`) and [`d3f64cc0`](https://huggingface.co/munod/tachyone-multi/commit/d3f64cc0704e67d8a18cf4906dc3a3ece6be8a6a)
> (`tachyone-multi`, model card only). `tachyone-en` now ships the **five-domain artifact**: run
> 5's trunk kept frozen while its `choice` head was re-fitted as a **bank** — shared + one head per
> domain behind the deterministic gate, keyed `choice_head.json` `{shared, domains}` plus its
> `choice_bank_fit.json` recipe and a calibration refitted on `eval_en` (the stale
> `finetune_config.json` was deleted from the repo: it described the weights this replaces). On
> the support split: **0.964** overall / **1.000** `choice` / 0.946 `noul` / 0.946 `score`
> (previous 0.972 / 0.946 / 0.992 / 0.978 — the trade is published, not summarized); on the
> five-domain split **0.964** against the previous adapter's **0.511**, gate strict **1.000**
> (0 to shared, 0 wrong domain), fast path **0 top-label flips**. All six uploaded files were
> sha256-verified against the local build. The keyed shape is read by this revision onward: an
> older checkout warns and answers through the baseline (B-8) or drops the head entirely.
>
> Previous revision **2026-09-29 (B-12, ADR-0015)**: commits [`796e6899`](https://huggingface.co/munod/tachyone-en/commit/796e6899f12f02a6075feffb20d04aa307b6d525)
> (`tachyone-en`) and [`5f688052`](https://huggingface.co/munod/tachyone-multi/commit/5f68805251e6c51ab519b81ce98bbae492202cf2)
> (`tachyone-multi`) — text-derived labels for all three primitives, datasets regenerated, and the
> published set re-chosen **by merit**: `tachyone-en` shipped the B-11 English weights at **0.972**
> / 0.992 `noul` / 0.978 `score` (retraining on corrected labels cost `choice`, control 0.841),
> `tachyone-multi` the B-12 retrain at **0.743** / `choice` 0.468 / 8 epochs.
>
> Before that, **2026-09-28 (B-11, ADR-0014)**: commits [`224c8a74`](https://huggingface.co/munod/tachyone-en/commit/224c8a746861b0882b83c92abea6dbd65f171af6)
> and [`b7747756`](https://huggingface.co/munod/tachyone-multi/commit/b7747756f95bdd9f72c28c01d1d5751d130130e0)
> — the `noul` label fix, English 0.945 / multilingual 0.718. Earlier still, the multilingual
> adapter was published at **LoRA rank 64** (commit `599df58`), whose numbers (0.702 → 0.853, `es`
> ECE 0.170 → 0.038) were measured on pre-B-11 labels and are **not** comparable with the table
> above — see `.specs/project/BACKLOG.md` **B-11**/**B-12**/**B-5**.

## 0. Prerequisites

- A Hugging Face account. Two upload paths: a **write token** (`HF_TOKEN`) over HTTPS, or your
  **SSH key** over git (`ssh -T git@hf.co` → `Hi <user>, welcome to Hugging Face.`).
- **Git LFS** for the SSH/git path: `adapter_model.safetensors` is ~29 MB, above the Hub's
  10 MiB git limit, so the repo's `.gitattributes` routes it through LFS. Install `git-lfs` and
  run `git lfs install --local` in the clone before adding files. The `hf upload` (token) path
  handles LFS itself and needs no git-lfs.
- The `train` extra installed (provides the `hf` CLI and `huggingface_hub`):

  ```bash
  uv sync --extra train
  ```

- Trained adapters under `checkpoints/en` and `checkpoints/multi` (see `docs/release.md`) and
  a fitted `temperature_calibration.json` in each.

> **Network note:** pushing over SSH needs outbound access to `hf.co:22` (not
> `huggingface.co`, whose SSH port times out on some networks). If port 22 is blocked, use
> `ssh -T -p 443 git@hf.co` or the token/HTTPS path below (it always works).

## 1. Create the model repos (once)

Web UI: create `<user>/tachyone-en` and `<user>/tachyone-multi` as **Model** repos. Or with a token:

```bash
uv run hf repo create <user>/tachyone-en   --repo-type model
uv run hf repo create <user>/tachyone-multi --repo-type model
```

## 2. Package the adapters

```bash
uv run python -m training.package_hf --adapter checkpoints/en    --out dist/hf/en    --name en
uv run python -m training.package_hf --adapter checkpoints/multi --out dist/hf/multi --name multi
```

Each folder contains the adapter files, the fitted temperature, and a `README.md` model card
(HF uses `README.md` as the model page).

## 3. Upload

### Option A — token over HTTPS (recommended, always works)

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
uv run hf upload <user>/tachyone-en    dist/hf/en    --repo-type model
uv run hf upload <user>/tachyone-multi dist/hf/multi --repo-type model
```

### Option B — git over SSH (your configured key)

Requires SSH access to `hf.co:22`. Verify with `ssh -T git@hf.co` (you should see
`Hi <user>, welcome to Hugging Face.`). If port 22 is blocked, use port 443 instead:
`ssh -T -p 443 git@hf.co` (SSH config `Hostname hf.co`, `Port 443`).

```bash
git clone git@hf.co:<user>/tachyone-en     hf-tachyone-en
cd hf-tachyone-en && git lfs install --local   # required: safetensors > 10 MiB
cp -r ../dist/hf/en/.  .
git add -A
git commit -m "Add tachyone-en adapter, temperature, and model card"
git push
# repeat for tachyone-multi
```

If SSH is blocked, clone over HTTPS with the token instead:
`git clone https://<user>:${HF_TOKEN}@huggingface.co/<user>/tachyone-en`.

## 4. Load a published adapter

```python
from peft import PeftModel
from transformers import AutoModel
base = AutoModel.from_pretrained("answerdotai/ModernBERT-large")
model = PeftModel.from_pretrained(base, "<user>/tachyone-en")
```

The runtime `tachyone` encoder backend loads these adapters whenever `TACHYONE_BACKEND=encoder` is
selected:
`CheckpointInfo` for each checkpoint carries `base_model` + `adapter`, so the backend
loads ModernBERT-large/mmBERT-base and applies `munod/tachyone-en` / `munod/tachyone-multi` (and their
per-primitive fitted temperature) on first use, cached under `TACHYONE_MODELS_DIR`.

```bash
uv run tachyone --predict --preset triage --backend encoder "Quero cancelar minha assinatura agora"

# point a checkpoint at another adapter, or disable it (empty value):
TACHYONE_ADAPTERS="tachyone-en=acme/tuned-en,tachyone-multi=" uv run tachyone --predict --backend encoder "..."
TACHYONE_OFFLINE=1 uv run tachyone --predict --backend encoder "..."   # cache-only, no network
```

!!! note "Air-gapped installs: prefetch first"
    `TACHYONE_OFFLINE=1` sets `local_files_only` on every Hub call, so it **fails** on a machine that
    has never downloaded the base encoder and the adapter. Prefetch once while online, then go
    offline:

    ```bash
    hf download munod/tachyone-multi --local-dir ~/.cache/tachyone/models/munod/tachyone-multi
    hf download munod/tachyone-en    --local-dir ~/.cache/tachyone/models/munod/tachyone-en
    # or simply run one warm-up prediction online:
    uv run tachyone --predict --preset triage --backend encoder "warm-up"
    TACHYONE_OFFLINE=1 uv run tachyone --predict --preset triage --backend encoder "..."
    ```

    There is **no `tachyone download` subcommand** (ADR-0010 named one that was never implemented —
    see [ADR notes](adr/README.md#implementation-notes-post-acceptance)). Note also that `TACHYONE_MODELS_DIR` must
    point at the directory layout `huggingface_hub` expects; if the cache is only partially
    populated, prefer a fresh warm-up run over `TACHYONE_OFFLINE=1`.

    The encoder backend no longer degrades quietly in that situation (B-8). When an adapter asset
    cannot be used — absent from a partially populated cache, or present but unreadable — it logs
    a WARNING naming the asset and where it looked, then answers with the uncalibrated baseline:

    ```text
    tachyone: cannot load temperature_calibration.json from munod/tachyone-en
    (LocalEntryNotFoundError: ...); continuing without it
    ```

    An adapter that simply does not ship `temperature_calibration.json` / `choice_head.json` in
    normal (online) operation stays silent — that is a documented state, not a fault.

### The `choice_head.json` asset has two shapes (ADR-0016)

Next to the weights, an adapter ships a `choice` scorer. Since ADR-0016 that file is either:

| Shape | Content | Loads as |
| --- | --- | --- |
| **legacy** (everything published so far) | `{rank, w1, w2}` — one shared scorer | a shared-only bank: every `choice` question answers exactly as before |
| **keyed** (the per-domain bank) | `{shared, domains: {name: {head, signatures}}}` | the shared head **plus** one head per domain, routed by the gate |

The gate picks exactly one head per `choice` question, in priority order: the optional
`choice_head` request field (honoured only when the asset ships that key), then a lexical
signature match against the question's instructions and option labels — pure Python, no encode,
offline — and falls back to the shared head when nothing matches or the scores tie. A wrong gate
can therefore *at worst* reproduce today's numbers. A **corrupt** entry follows the B-8 contract:
it warns by name (file + key) and drops only itself — a broken `shared` keeps the domain heads, a
broken domain keeps the others.

```bash
# the gate's own accuracy, beside the per-domain rows it routes (B-6):
uv run python -m training.predict --data data/eval_en_domains.jsonl \
  --adapter checkpoints/en_domains_bank --out-report report.json
```

No migration: `load_choice_head` accepts both shapes, and an adapter that ships no bank at all
remains a normal, silent state.

## 5. After a full-scale run

- Refresh the metrics in `docs/model-card.md` and `benchmarks/report.md` from the new report.
- Re-package and re-upload (step 2–3); HF keeps history.
- Tag the GitHub release from `CHANGELOG.md` (see `docs/release.md`).
