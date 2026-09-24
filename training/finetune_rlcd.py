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

#: Candidate attention/MLP projections, tried in order. ModernBERT uses Wqkv/Wo/Wi;
#: BERT/mmBERT-style trunks use query/key/value/dense.
_LORA_CANDIDATES: tuple[str, ...] = (
    "Wqkv",
    "Wo",
    "Wi",
    "query",
    "key",
    "value",
    "dense",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)


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
    """Run LoRA/QLoRA training with batched forward passes.

    The head math mirrors the runtime encoder (cosine similarity, softmaxed with a learned
    temperature), so only the shared trunk is adapted. Batches are grouped by primitive and the
    encoder runs once per batch; imports are lazy behind the ``train`` extra.
    """
    try:
        import torch  # pyright: ignore[reportMissingImports]
        import torch.nn.functional as F  # pyright: ignore[reportMissingImports]
        from peft import LoraConfig, get_peft_model  # pyright: ignore[reportMissingImports]
        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoModel,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_TRAIN_HINT) from exc

    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = config.bf16 and device == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    encoder = AutoModel.from_pretrained(config.model_id)
    if config.gradient_checkpointing:
        encoder.gradient_checkpointing_enable()
    linear_names = {
        name.split(".")[-1]
        for name, module in encoder.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    target_modules = [candidate for candidate in _LORA_CANDIDATES if candidate in linear_names]
    if not target_modules:  # last resort: adapt every linear projection
        target_modules = sorted(linear_names)
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(encoder, lora)
    if config.gradient_checkpointing:
        model.enable_input_require_grads()  # type: ignore[attr-defined]
    model.to(device)

    log_temperature = torch.zeros((), requires_grad=True, device=device)
    optimizer = torch.optim.AdamW([*model.parameters(), log_temperature], lr=config.learning_rate)

    records = list(read_records(config.data_path, limit=config.max_records))
    split = max(1, int(len(records) * (1.0 - config.val_split)))
    train_records, val_records = records[:split], records[split:]
    grouped: dict[str, list[dict[str, Any]]] = {kind: [] for kind in ("noul", "choice", "score")}
    for record in train_records:
        grouped[record["type"]].append(record)

    def encode(texts: list[str]) -> Any:
        batch = tokenizer(
            texts, padding=True, truncation=True, max_length=config.max_len, return_tensors="pt"
        ).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return pooled.float()

    def batch_loss(kind: str, batch: list[dict[str, Any]]) -> Any:
        temperature = torch.exp(log_temperature) + 1e-3
        states = encode([record["state"] for record in batch])
        questions = encode([_question_text(record) for record in batch])
        if kind == "noul":
            similarity = F.cosine_similarity(questions, states, dim=-1)
            probabilities = torch.sigmoid(similarity / temperature)
            targets = torch.tensor([float(record["target"]) for record in batch], device=device)
            return ((probabilities - targets) ** 2).mean()
        criterion_texts = [text for record in batch for text in _criterion_texts(record)]
        criteria = encode(criterion_texts)
        total = torch.zeros((), device=device)
        offset = 0
        for index, record in enumerate(batch):
            count = len(_criterion_texts(record))
            block = criteria[offset : offset + count]
            offset += count
            state = states[index].expand(count, -1)
            question = questions[index].expand(count, -1)
            logits = (
                F.cosine_similarity(block, state, dim=-1)
                + F.cosine_similarity(block, question, dim=-1)
            ) / temperature
            probabilities = torch.softmax(logits, dim=0)
            target_index = (
                int(record["target"])
                if isinstance(record["target"], int)
                else _choice_index(record)
            )
            one_hot = torch.zeros_like(probabilities)
            one_hot[target_index] = 1.0
            total = total + ((probabilities - one_hot) ** 2).sum()
        return total / len(batch)

    def evaluate_loss(items: list[dict[str, Any]]) -> float:
        if not items:
            return float("nan")
        model.eval()  # type: ignore[attr-defined]
        total, count = 0.0, 0
        with torch.no_grad():
            for kind in ("noul", "choice", "score"):
                subset = [record for record in items if record["type"] == kind]
                for start in range(0, len(subset), config.batch_size):
                    chunk = subset[start : start + config.batch_size]
                    total += float(batch_loss(kind, chunk).item()) * len(chunk)
                    count += len(chunk)
        return total / count if count else float("nan")

    epochs_run = 0
    for _epoch in range(config.epochs):
        model.train()  # type: ignore[attr-defined]
        optimizer.zero_grad(set_to_none=True)
        accumulations = 0
        for kind in ("noul", "choice", "score"):
            items = grouped[kind]
            for start in range(0, len(items), config.batch_size):
                chunk = items[start : start + config.batch_size]
                loss = batch_loss(kind, chunk) / config.grad_accum
                loss.backward()
                accumulations += 1
                if accumulations % config.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        if accumulations % config.grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        epochs_run += 1

    output_dir = Path(config.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    (output_dir / "finetune_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    fitted_temperature = float(torch.exp(log_temperature).item())
    (output_dir / "temperature.json").write_text(
        json.dumps(
            {"temperature": fitted_temperature, "val_loss": evaluate_loss(val_records)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report["train_records"] = len(train_records)
    report["val_records"] = len(val_records)
    report["epochs"] = epochs_run
    report["batch_size"] = config.batch_size
    report["target_modules"] = target_modules
    report["checkpoint"] = str(output_dir)
    report["temperature"] = fitted_temperature
    report["val_loss"] = evaluate_loss(val_records)
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
