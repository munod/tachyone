# Publishing to the Hugging Face Hub

jeba ships **PEFT LoRA adapters**, not full checkpoints. Each adapter's
`adapter_config.json` records `base_model_name_or_path` (ModernBERT-large or mmBERT-base), so
consumers load the base trunk plus the adapter. Weights are fetched/cached locally at runtime
(ADR-0010); publishing is only about distribution.

## 0. Prerequisites

- A Hugging Face account and (for pushing) a **write token** from
  <https://huggingface.co/settings/tokens>, exported as `HF_TOKEN`.
- The `train` extra installed (provides the `hf` CLI and `huggingface_hub`):

  ```bash
  uv sync --extra train
  ```

- Trained adapters under `checkpoints/en` and `checkpoints/multi` (see `docs/release.md`) and
  a fitted `temperature_calibration.json` in each.

> **Network note:** pushing over SSH needs outbound access to `huggingface.co:22`. If port 22 is
> blocked, use the token/HTTPS path below (it always works). Test with
> `ssh -T git@huggingface.co`.

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

Requires SSH access to `huggingface.co:22`.

```bash
git clone git@huggingface.co:<user>/jeba-en     hf-jeba-en
cp -r dist/hf/en/.  hf-jeba-en/
git -C hf-jeba-en add -A
git -C hf-jeba-en commit -m "Add jeba-en adapter, temperature, and model card"
git -C hf-jeba-en push
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

The runtime `jeba` encoder backend will load published checkpoints from the hub on first use
once their ids are registered in `jeba.backends.encoder.MODEL_IDS` / configured via
`JEBA_MODELS_DIR`.

## 5. After a full-scale run

- Replace the placeholder metrics in `docs/model-card.md` and `benchmarks/report.md`.
- Re-package and re-upload (step 2–3); HF keeps history.
- Tag the GitHub release from `CHANGELOG.md` (see `docs/release.md`).
