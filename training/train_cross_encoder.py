"""Train the S2 cross-encoder teacher (B-21 W2) — a **training-time** instrument only.

The teacher scores every option of a ``choice`` record jointly: one forward pass over
``[state ⊕ question ⊕ option]`` per option, one logit each, softmaxed over the record's own
options (listwise log loss — the proper scoring rule for the categorical). An auxiliary NLI
head (entail/neutral/contradict over the premise/hypothesis pair the arm's NLI rows carry in
their state) supplies dense gradients, and the distractor contrast ("which option does the
state contradict") is implicit in the listwise softmax. The serving path stays single-pass
(W2's redirect): this artefact is never served by default —
``training/distil_cross_encoder.py`` distils its per-option distributions into the
single-pass bank head.

    uv run python -m training.train_cross_encoder --config training/configs/cross_encoder_s2.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.finetune_rlcd import (
    GRAD_CLIP_MAX_NORM,
    criterion_texts,
    question_text,
    read_records,
    shuffled,
    split_records,
)

_TRAIN_HINT = "the train extra is required for fine-tuning: uv sync --extra train"

#: The arm's NLI rows carry the premise/hypothesis pair in their state (build_jev_sources).
_PREMISE_PREFIX = "Premise: "
_HYPOTHESIS_MARK = "\nHypothesis: "

#: The three NLI relations, in the head's class order.
NLI_LABELS: tuple[str, ...] = ("entailment", "neutral", "contradiction")

_CONFIG_KEYS = {
    "model_id",
    "data_path",
    "out_dir",
    "epochs",
    "learning_rate",
    "batch_size",
    "grad_accum",
    "max_len",
    "seed",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "val_split",
    "nli_weight",
    "max_records",
    "gradient_checkpointing",
}


@dataclass(frozen=True)
class TeacherConfig:
    model_id: str = "answerdotai/ModernBERT-large"
    data_path: str = "data/train_en_s1c_union.jsonl"
    out_dir: str = "checkpoints/en_s2_teacher"
    epochs: int = 2
    learning_rate: float = 2e-5
    batch_size: int = 8
    grad_accum: int = 8
    max_len: int = 512
    seed: int = 2
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    val_split: float = 0.1
    nli_weight: float = 0.5
    max_records: int | None = None
    gradient_checkpointing: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TeacherConfig:
        data = dict(data)
        data.pop("_comment", None)
        unknown = set(data) - _CONFIG_KEYS
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in sorted(_CONFIG_KEYS)}


def load_config(path: str | Path) -> TeacherConfig:
    return TeacherConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def split_nli(state: str) -> tuple[str, str] | None:
    """The (premise, hypothesis) pair an NLI row carries, or None when it carries none."""
    if not state.startswith(_PREMISE_PREFIX) or _HYPOTHESIS_MARK not in state:
        return None
    premise, hypothesis = state[len(_PREMISE_PREFIX) :].split(_HYPOTHESIS_MARK, 1)
    return premise.strip(), hypothesis.strip()


def is_nli_record(record: dict[str, Any]) -> bool:
    """The arm's NLI rows: 3-option choice rows over the relation labels (build_jev_sources)."""
    return (
        record.get("type") == "choice"
        and set(record.get("criteria", {})) == set(NLI_LABELS)
        and split_nli(str(record["state"])) is not None
    )


def nli_class_index(record: dict[str, Any]) -> int:
    return NLI_LABELS.index(str(record["target"]))


def soft_targets_from_logits(
    logits: Sequence[float],
    *,
    target_index: int,
    temperature: float = 2.0,
    blend: float = 0.5,
) -> list[float]:
    """The distillation target (B-21 W3): `blend·one_hot + (1-blend)·softmax(logits/T)`.

    Pure maths — the S0-pinned recipe (T=2 flattening, 0.5 blend). The result is a proper
    distribution aligned with the record's criterion order; `target_index` names the gold
    option so the hard-label anchor keeps the proper scoring identity (ADR-0005).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0, 1], got {blend}")
    if not logits:
        raise ValueError("logits must be non-empty")
    if not 0 <= target_index < len(logits):
        raise ValueError(f"target_index {target_index} out of range for {len(logits)} logits")
    scaled = [float(value) / temperature for value in logits]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = sum(weights)
    teacher = [weight / total for weight in weights]
    return [
        (blend if index == target_index else 0.0) + (1.0 - blend) * probability
        for index, probability in enumerate(teacher)
    ]


def _mean_pool(last_hidden: Any, attention_mask: Any) -> Any:
    """Masked mean pooling — the same pooling the runtime encoder uses (L-004)."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def _train(config: TeacherConfig) -> dict[str, Any]:
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

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    encoder = AutoModel.from_pretrained(config.model_id)
    if config.gradient_checkpointing:
        encoder.gradient_checkpointing_enable()
    linear_names = {
        name.split(".")[-1]
        for name, module in encoder.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    from training.finetune_rlcd import (
        _LORA_CANDIDATES,
    )  # reuse of the trainer's LoRA candidate list

    target_modules = [c for c in _LORA_CANDIDATES if c in linear_names] or sorted(linear_names)
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
    model.train()

    hidden_size = int(encoder.config.hidden_size)
    score_head = torch.nn.Linear(hidden_size, 1).to(device)
    nli_head = torch.nn.Linear(hidden_size, len(NLI_LABELS)).to(device)

    records = list(read_records(config.data_path, limit=config.max_records))
    choice_records = [r for r in records if r.get("type") == "choice"]
    train_records, val_records = split_records(
        choice_records, val_fraction=config.val_split, seed=config.seed
    )
    parameters = [
        *model.parameters(),
        *score_head.parameters(),
        *nli_head.parameters(),
    ]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)

    def encode_pairs(pairs: list[tuple[str, str]]) -> Any:
        batch = tokenizer(
            [left for left, _ in pairs],
            [right for _, right in pairs],
            truncation=True,
            max_length=config.max_len,
            padding=True,
            return_tensors="pt",
        ).to(device)
        hidden = model(**batch).last_hidden_state
        return _mean_pool(hidden, batch["attention_mask"])

    def encode_texts(texts: list[str]) -> Any:
        batch = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_len,
            padding=True,
            return_tensors="pt",
        ).to(device)
        hidden = model(**batch).last_hidden_state
        return _mean_pool(hidden, batch["attention_mask"])

    def option_logits(record: dict[str, Any]) -> Any:
        options = criterion_texts(record)
        question = question_text(record)
        pairs = [(str(record["state"]), f"{question} {option}") for option in options]
        pooled = encode_pairs(pairs)
        return score_head(pooled).squeeze(-1)

    def gold_index(record: dict[str, Any]) -> int:
        target = record["target"]
        if isinstance(target, int) and not isinstance(target, bool):
            return int(target)
        return list(record["criteria"]).index(str(target))

    def record_loss(record: dict[str, Any]) -> Any:
        # One record per forward: arities run to 20 options, so the batch is the *gradient*
        # accumulation (batch_size x grad_accum records), never the forward (OOM otherwise).
        logits = option_logits(record)
        gold = torch.tensor([gold_index(record)], device=device)
        loss = F.cross_entropy(logits.unsqueeze(0), gold)
        if is_nli_record(record):
            nli_logits = nli_head(encode_texts([str(record["state"])]))
            nli_target = torch.tensor([nli_class_index(record)], device=device)
            loss = loss + config.nli_weight * F.cross_entropy(nli_logits, nli_target)
        return loss

    def evaluate(records_batch: Sequence[dict[str, Any]]) -> dict[str, float]:
        correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for record in records_batch:
                logits = option_logits(record)
                gold = gold_index(record)
                total_loss += float(
                    F.cross_entropy(logits.unsqueeze(0), torch.tensor([gold], device=device))
                )
                correct += int(int(logits.argmax()) == gold)
        return {
            "loss": total_loss / max(1, len(records_batch)),
            "accuracy": correct / max(1, len(records_batch)),
        }

    report: dict[str, Any] = {
        "config": config.to_dict(),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "nli_rows": sum(1 for r in train_records if is_nli_record(r)),
        "train_loss": [],
        "val": [],
    }
    per_step = config.batch_size * config.grad_accum  # the effective batch, in records
    for epoch in range(config.epochs):
        order = shuffled(train_records, seed=f"{config.seed}:{epoch}")
        losses = []
        optimizer.zero_grad()
        for index, record in enumerate(order):
            loss = record_loss(record) / per_step
            loss.backward()
            losses.append(float(loss) * per_step)
            if (index + 1) % per_step == 0 or index + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP_MAX_NORM)
                optimizer.step()
                optimizer.zero_grad()
        val = evaluate(val_records)
        report["train_loss"].append({"epoch": epoch, "loss": sum(losses) / max(1, len(losses))})
        report["val"].append({"epoch": epoch, **val})
        # L-011 loss-shape check: collapse or divergence fails loudly, not after the fit.
        if not math.isfinite(val["loss"]) or val["accuracy"] < 0.2:
            raise RuntimeError(
                f"epoch {epoch}: teacher collapsed (val {val}) — check the data/recipe"
            )

    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    torch.save(score_head.state_dict(), out / "score_head.pt")
    torch.save(nli_head.state_dict(), out / "nli_head.pt")
    (out / "teacher_config.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-train-cross-encoder", description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = _train(load_config(args.config))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
