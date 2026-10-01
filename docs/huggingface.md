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
> **Latest revision: 2026-09-29 (B-12, ADR-0015)** — commits [`796e6899`](https://huggingface.co/munod/tachyone-en/commit/796e6899f12f02a6075feffb20d04aa307b6d525)
> (`tachyone-en`) and [`5f688052`](https://huggingface.co/munod/tachyone-multi/commit/5f68805251e6c51ab519b81ce98bbae492202cf2)
> (`tachyone-multi`). All three primitives now carry text-derived labels (the `score` near-tie and
> the empty-state defaults went in with ADR-0015), every dataset was regenerated and the published
> set was re-chosen **by merit**: `tachyone-en` ships the **B-11 English weights re-evaluated on
> the corrected labels** (**0.972** overall / **0.992** `noul` / **0.978** `score`) because
> retraining on the corrected labels cost `choice` (a control of the identical recipe landed at
> 0.841), while `tachyone-multi` ships the **B-12 retrain** (**0.743** / `choice` 0.468 / 8
> epochs), which won on merit. Every uploaded file was verified against the local build (sha256
> match on the adapter, temperature, choice head, `finetune_config` and README).
>
> Previous revision **2026-09-28 (B-11, ADR-0014)**: commits [`224c8a74`](https://huggingface.co/munod/tachyone-en/commit/224c8a746861b0882b83c92abea6dbd65f171af6)
> and [`b7747756`](https://huggingface.co/munod/tachyone-multi/commit/b7747756f95bdd9f72c28c01d1d5751d130130e0)
> — the `noul` label fix, English 0.945 / multilingual 0.718. Before that, the multilingual
> adapter was published at **LoRA rank 64** (commit `599df58`); its numbers (0.702 → 0.853, `es`
> ECE 0.170 → 0.038) were measured on the pre-B-11 labels and are **not** comparable with the
> table above — see `.specs/project/BACKLOG.md` **B-11**/**B-12**.

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
