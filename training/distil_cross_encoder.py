"""Distil the S2 cross-encoder teacher into single-pass soft targets (B-21 W2/W3).

For every ``choice`` record of the input pool the teacher's per-option logits become the
distillation target: ``blend·one_hot + (1-blend)·softmax(logits/T)`` (the S0-pinned recipe
— T=2 flattening, 0.5 hard-label anchor; ADR-0005 keeps the proper scoring identity). The
output rows are the input rows with ``target`` replaced by that vector, aligned with
``criterion_texts`` order — exactly the ``target_mode: soft`` format
``training/fit_choice_bank`` consumes. ``noul``/``score`` rows pass through untouched (the
W3 guard: ``score`` keeps hard labels).

    uv run python -m training.distil_cross_encoder \
        --teacher checkpoints/en_s2_teacher \
        --data data/train_en_s1c_union.jsonl \
        --out data/train_en_s1c_distilled.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from training.finetune_rlcd import criterion_texts, question_text, read_records
from training.train_cross_encoder import _mean_pool, soft_targets_from_logits

_TRAIN_HINT = "the train extra is required for fine-tuning: uv sync --extra train"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-distil-cross-encoder", description=__doc__)
    parser.add_argument("--teacher", required=True, help="the trained teacher directory")
    parser.add_argument("--data", required=True, help="the choice pool to score")
    parser.add_argument("--out", required=True, help="the distilled data file to write")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=512)
    return parser


def distil(
    *,
    teacher: str | Path,
    data: str | Path,
    out: str | Path,
    temperature: float = 2.0,
    blend: float = 0.5,
    batch_size: int = 16,
    max_len: int = 512,
) -> dict[str, Any]:
    try:
        import torch  # pyright: ignore[reportMissingImports]
        from peft import PeftModel  # pyright: ignore[reportMissingImports]
        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoModel,
            AutoTokenizer,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_TRAIN_HINT) from exc

    teacher_dir = Path(teacher)
    recipe = json.loads((teacher_dir / "teacher_config.json").read_text(encoding="utf-8"))
    model_id = str(recipe["config"]["model_id"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoder = AutoModel.from_pretrained(model_id)
    model = PeftModel.from_pretrained(encoder, teacher_dir, is_trainable=False)
    model.to(device)
    model.eval()
    hidden_size = int(encoder.config.hidden_size)
    score_head = torch.nn.Linear(hidden_size, 1)
    score_head.load_state_dict(torch.load(teacher_dir / "score_head.pt", map_location=device))
    score_head.to(device)
    score_head.eval()

    records = list(read_records(data))
    distilled = 0
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            choice = [record for record in chunk if record.get("type") == "choice"]
            soft: dict[str, list[float]] = {}
            if choice:
                pairs: list[tuple[str, str]] = []
                spans: list[tuple[int, int]] = []
                for record in choice:
                    options = criterion_texts(record)
                    question = question_text(record)
                    idx = len(pairs)
                    pairs.extend(
                        (str(record["state"]), f"{question} {option}") for option in options
                    )
                    spans.append((idx, len(pairs)))
                batch = tokenizer(
                    [left for left, _ in pairs],
                    [right for _, right in pairs],
                    truncation=True,
                    max_length=max_len,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    hidden = model(**batch).last_hidden_state
                    pooled = _mean_pool(hidden, batch["attention_mask"])
                    logits = score_head(pooled).squeeze(-1)
                for record, (a, b) in zip(choice, spans, strict=True):
                    keys = list(record["criteria"])
                    target = record["target"]
                    gold = (
                        int(target)
                        if isinstance(target, int) and not isinstance(target, bool)
                        else keys.index(str(target))
                    )
                    soft[str(record["id"])] = soft_targets_from_logits(
                        [float(value) for value in logits[a:b]],
                        target_index=gold,
                        temperature=temperature,
                        blend=blend,
                    )
                    distilled += 1
            for record in chunk:
                if str(record["id"]) in soft:
                    record = dict(record)
                    record["target"] = soft[str(record["id"])]
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    return {
        "teacher": str(teacher_dir),
        "data": str(data),
        "out": str(out_path),
        "records": len(records),
        "distilled": distilled,
        "temperature": temperature,
        "blend": blend,
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = distil(
        teacher=args.teacher,
        data=args.data,
        out=args.out,
        temperature=args.temperature,
        blend=args.blend,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
