# Publishing to the Hugging Face Hub

jeba ships **PEFT LoRA adapters**, not full checkpoints. Each adapter's
`adapter_config.json` records `base_model_name_or_path` (ModernBERT-large or mmBERT-base), so
consumers load the base trunk plus the adapter. Weights are fetched/cached locally at runtime
(ADR-0010); publishing is only about distribution.

> **Published (verified):**
> - <https://huggingface.co/munod/jeba-en> (ModernBERT-large adapter)
> - <https://huggingface.co/munod/jeba-multi> (mmBERT-base adapter)
>
> Both load via `PeftModel.from_pretrained(base, "munod/jeba-en")` and predict.

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

Web UI: create `<user>/jeba-en` and `<user>/jeba-multi` as **Model** repos. Or with a token:

```bash
uv run hf repo create <user>/jeba-en   --repo-type model
uv run hf repo create <user>/jeba-multi --repo-type model
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
uv run hf upload <user>/jeba-en    dist/hf/en    --repo-type model
uv run hf upload <user>/jeba-multi dist/hf/multi --repo-type model
```

### Option B — git over SSH (your configured key)

Requires SSH access to `hf.co:22`. Verify with `ssh -T git@hf.co` (you should see
`Hi <user>, welcome to Hugging Face.`). If port 22 is blocked, use port 443 instead:
`ssh -T -p 443 git@hf.co` (SSH config `Hostname hf.co`, `Port 443`).

```bash
git clone git@hf.co:<user>/jeba-en     hf-jeba-en
cd hf-jeba-en && git lfs install --local   # required: safetensors > 10 MiB
cp -r ../dist/hf/en/.  .
git add -A
git commit -m "Add jeba-en adapter, temperature, and model card"
git push
# repeat for jeba-multi
```

If SSH is blocked, clone over HTTPS with the token instead:
`git clone https://<user>:${HF_TOKEN}@huggingface.co/<user>/jeba-en`.

## 4. Load a published adapter

```python
from peft import PeftModel
from transformers import AutoModel
base = AutoModel.from_pretrained("answerdotai/ModernBERT-large")
model = PeftModel.from_pretrained(base, "<user>/jeba-en")
```

The runtime `jeba` encoder backend loads these adapters **by default**:
`CheckpointInfo` for each checkpoint carries `base_model` + `adapter`, so `JEBA_BACKEND=encoder`
loads ModernBERT-large/mmBERT-base and applies `munod/jeba-en` / `munod/jeba-multi` (and their
per-primitive fitted temperature) on first use, cached under `JEBA_MODELS_DIR`.

```bash
uv run jeba --predict --preset triage --backend encoder "Quero cancelar minha assinatura agora"

# point a checkpoint at another adapter, or disable it (empty value):
JEBA_ADAPTERS="jeba-en=acme/tuned-en,jeba-multi=" uv run jeba --predict --backend encoder "..."
JEBA_OFFLINE=1 uv run jeba --predict --backend encoder "..."   # cache-only, no network
```

## 5. After a full-scale run

- Refresh the metrics in `docs/model-card.md` and `benchmarks/report.md` from the new report.
- Re-package and re-upload (step 2–3); HF keeps history.
- Tag the GitHub release from `CHANGELOG.md` (see `docs/release.md`).
