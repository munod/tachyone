"""LoRA/QLoRA fine-tuning of the encoder with an RLCD proper-scoring objective.

The training head mirrors the inference math in ``jeba.backends.encoder`` (cosine similarity
between the state, question, and criterion embeddings, softmaxed with a learned temperature),
so fine-tuning the shared encoder directly improves the local backend without a separate
head format. Torch/transformers/peft are imported lazily (the ``train`` extra); a ``--dry-run``
validates the config and data without them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from training.rlcd import categorical_loss

_TRAIN_HINT = "the train extra is required for fine-tuning: uv sync --extra train"


@dataclass(frozen=True, slots=True)
class FinetuneConfig:
    """Hyperparameters and paths for one fine-tuning run."""

    model_id: str = "answerdotai/ModernBERT-large"
    data_path: str = "data/train.jsonl"
    out_dir: str = "checkpoints/en"
    seed: int = 42
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 4
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_len: int = 512
    val_split: float = 0.1
    bf16: bool = True
    gradient_checkpointing: bool = True
    use_4bit: bool = False
    max_records: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinetuneConfig:
        known = {field.name for field in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> FinetuneConfig:
    """Load a :class:`FinetuneConfig` from a JSON file."""
    return FinetuneConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def read_records(path: str | Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield JSONL records, optionally capped at ``limit``."""
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize_dataset(path: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    """Count records per primitive and language (pure; no torch)."""
    per_type: dict[str, int] = {}
    per_lang: dict[str, int] = {}
    total = 0
    for record in read_records(path, limit=limit):
        total += 1
        per_type[record["type"]] = per_type.get(record["type"], 0) + 1
        per_lang[record["lang"]] = per_lang.get(record["lang"], 0) + 1
    return {"total": total, "per_type": per_type, "per_lang": per_lang}


def _ground_truth_loss(record: dict[str, Any], probabilities: Any) -> float:
    """Pure-python reference loss used to sanity-check the torch path."""
    if record["type"] == "noul":
        return (float(probabilities) - float(record["target"])) ** 2
    return categorical_loss(probabilities, record["target"])


def _train(config: FinetuneConfig, report: dict[str, Any]) -> dict[str, Any]:
    """Run LoRA/QLoRA training. Imports torch lazily; requires the ``train`` extra."""
    try:
        import torch  # pyright: ignore[reportMissingImports]
        from peft import LoraConfig, get_peft_model  # pyright: ignore[reportMissingImports]
        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoModel,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_TRAIN_HINT) from exc

    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    encoder = AutoModel.from_pretrained(config.model_id)
    if config.gradient_checkpointing:
        encoder.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=["query", "key", "value", "dense"],
    )
    model = get_peft_model(encoder, lora)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    log_temperature = torch.zeros((), requires_grad=True, device=device)
    optimizer = torch.optim.AdamW([*model.parameters(), log_temperature], lr=config.learning_rate)

    records = list(read_records(config.data_path, limit=config.max_records))
    split = max(1, int(len(records) * (1.0 - config.val_split)))
    train_records, val_records = records[:split], records[split:]

    def encode(texts: list[str]) -> Any:
        batch = tokenizer(
            texts, padding=True, truncation=True, max_length=config.max_len, return_tensors="pt"
        ).to(device)
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def loss_for(record: dict[str, Any]) -> Any:
        temperature = torch.exp(log_temperature) + 1e-3
        state_embedding = encode([record["state"]])[0]
        question_embedding = encode([_question_text(record)])[0]
        criterion_texts = _criterion_texts(record)
        if record["type"] == "noul":
            logit = torch.dot(question_embedding, state_embedding) / temperature
            probability = torch.sigmoid(logit)
            target = torch.tensor(float(record["target"]), device=device)
            return (probability - target) ** 2
        criteria = encode(criterion_texts)
        logits = (criteria @ state_embedding + criteria @ question_embedding) / temperature
        probabilities = torch.softmax(logits, dim=0)
        keys = list(range(len(criterion_texts)))
        target_index = keys[record["target"]] if isinstance(record["target"], int) else None
        if target_index is None:  # choice: target is an option key
            target_index = _choice_index(record)
        one_hot = torch.zeros_like(probabilities)
        one_hot[target_index] = 1.0
        return torch.sum((probabilities - one_hot) ** 2)

    model.zero_grad(set_to_none=True)
    for step, record in enumerate(train_records):
        loss = loss_for(record) / config.grad_accum
        loss.backward()
        if (step + 1) % config.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    optimizer.step()

    output_dir = Path(config.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    (output_dir / "finetune_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "temperature.json").write_text(
        json.dumps({"temperature": float(torch.exp(log_temperature).item())}, indent=2),
        encoding="utf-8",
    )
    report["train_records"] = len(train_records)
    report["val_records"] = len(val_records)
    report["checkpoint"] = str(output_dir)
    report["temperature"] = float(torch.exp(log_temperature).item())
    return report


def _question_text(record: dict[str, Any]) -> str:
    instructions = record["instructions"]
    if isinstance(instructions, str):
        return instructions
    return json.dumps(instructions, ensure_ascii=False)


def _criterion_texts(record: dict[str, Any]) -> list[str]:
    criteria = record["criteria"]
    if isinstance(criteria, dict):
        return [str(key) for key in criteria]
    return [str(level) for level in criteria]


def _choice_index(record: dict[str, Any]) -> int:
    keys = list(record["criteria"])
    return keys.index(record["target"])


def run(config: FinetuneConfig, *, dry_run: bool = False) -> dict[str, Any]:
    """Validate data and either summarize (dry run) or train (needs the ``train`` extra)."""
    report: dict[str, Any] = {
        "config": config.to_dict(),
        "dataset": summarize_dataset(config.data_path),
    }
    if dry_run:
        report["mode"] = "dry-run"
        return report
    report["mode"] = "train"
    return _train(config, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jeba-finetune", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a JSON FinetuneConfig")
    parser.add_argument("--dry-run", action="store_true", help="validate config and data only")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = run(load_config(args.config), dry_run=args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
