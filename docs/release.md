# Release process

Checklist for publishing a Tachyone release. Release artifacts that depend on trained weights wait
for the RTX 3060 training run (`uv sync --extra train`).

## 1. Preconditions

- [ ] All gates green: `uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest`.
- [ ] `uv sync --locked --extra serve` passes; integration/e2e tests not skipped.
- [ ] `uv run mkdocs build --strict` succeeds.
- [ ] `CHANGELOG.md` updated; version bumped in `pyproject.toml` and `src/tachyone/__init__.py`.
- [ ] `tests/test_contract_wire.py` unchanged for the release (or updated with the wire change).

## 2. Train and calibrate (GPU)

```bash
uv sync --extra train
uv run python -m training.generate_data --languages en --per-type 2000 --out data/train_en.jsonl
uv run python -m training.finetune_rlcd --config training/configs/finetune_en.json
uv run python -m training.finetune_rlcd --config training/configs/finetune_multi.json
uv run python -m training.fit_calibration --calibration data/preds_en.jsonl --out temperature.json
```

## 3. Benchmarks

```bash
uv run python -m training.evaluate --data data/eval_en.jsonl --out benchmarks/results/encoder.json --backend encoder
uv run python -m benchmarks.report --entry encoder=benchmarks/results/encoder.json --out benchmarks/report.md
```

## 4. Publish

- [ ] Package the adapters: `uv run python -m training.package_hf --adapter checkpoints/en --out dist/hf/en --name en` (and `multi`).
- [ ] Upload to the Hugging Face Hub — full guide in [`huggingface.md`](huggingface.md).
- [ ] Publish the model card (`docs/model-card.md`) beside the weights (included by packaging).
- [ ] Tag the release (`git tag -a vX.Y.Z`) and create the GitHub release from `CHANGELOG.md`.
- [ ] Verify the docs site build artifact is attached or deployed.
