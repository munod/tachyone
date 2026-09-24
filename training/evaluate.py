"""Evaluation harness: accuracy, ECE, and latency per primitive and language.

A ``predictor`` is any callable ``(state, question) -> Answer``; the harness converts JSONL
records to primitives, times each prediction, and aggregates metrics (TRAIN-05). Public
probes (MASSIVE/XNLI/typed-decisions) can feed the same harness for reproducible artifacts.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jeba.calibration import confidence, expected_calibration_error
from jeba.primitives import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    State,
)

#: A predictor answers a single (state, question) pair.
type Predictor = Callable[[State, Question], Answer]


@dataclass(frozen=True, slots=True)
class EvalExample:
    """One labeled record: primitive, state, question, target, language."""

    id: str
    type: str
    state: State
    question: Question
    target: int | str
    lang: str


def record_to_example(record: dict[str, Any]) -> EvalExample:
    """Convert a JSONL training/eval record into an :class:`EvalExample`."""
    kind = record["type"]
    instructions = record["instructions"]
    criteria = record["criteria"]
    if kind == "noul":
        question: Question = NoulQuestion.model_validate(
            {"type": "noul", "instructions": instructions, "criteria": criteria}
        )
    elif kind == "choice":
        question = ChoiceQuestion.model_validate(
            {"type": "choice", "instructions": instructions, "criteria": criteria}
        )
    elif kind == "score":
        question = ScoreQuestion.model_validate(
            {"type": "score", "instructions": instructions, "criteria": criteria}
        )
    else:
        raise ValueError(f"unknown record type: {kind!r}")
    return EvalExample(
        id=record["id"],
        type=kind,
        state=record["state"],
        question=question,
        target=record["target"],
        lang=record["lang"],
    )


def load_examples(path: str | Path, *, limit: int | None = None) -> list[EvalExample]:
    """Load evaluation examples from JSONL."""
    examples: list[EvalExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, raw in enumerate(handle):
            if limit is not None and index >= limit:
                break
            line = raw.strip()
            if line:
                examples.append(record_to_example(json.loads(line)))
    return examples


def _predicted_and_confidence(example: EvalExample, answer: Answer) -> tuple[Any, float, bool]:
    if example.type == "noul":
        assert isinstance(answer, NoulAnswer)
        predicted = 1 if answer.noul >= 0.5 else 0
        conf = max(answer.noul, 1.0 - answer.noul)
        return predicted, conf, predicted == int(example.target)
    if example.type == "choice":
        assert isinstance(answer, ChoiceAnswer)
        distribution: dict[str, float] = dict(answer.probabilities)
        predicted = max(distribution, key=distribution.__getitem__)
        conf = confidence(distribution)
        return predicted, conf, predicted == example.target
    assert isinstance(answer, ScoreAnswer)
    level_distribution: dict[int, float] = dict(answer.probabilities)
    predicted = max(level_distribution, key=level_distribution.__getitem__)
    conf = confidence(level_distribution)
    return predicted, conf, predicted == int(example.target)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _metrics(rows: list[tuple[bool, float, float]], bins: int) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": 0.0, "ece": 0.0, "latency_ms": {"p50": 0.0, "p95": 0.0}}
    correct = [row[0] for row in rows]
    confidences = [row[1] for row in rows]
    latencies = [row[2] for row in rows]
    return {
        "n": len(rows),
        "accuracy": sum(correct) / len(correct),
        "ece": expected_calibration_error(confidences, correct, bins=bins),
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 4),
            "p95": round(_percentile(latencies, 0.95), 4),
        },
    }


def evaluate(
    examples: Sequence[EvalExample], predictor: Predictor, *, bins: int = 10
) -> dict[str, Any]:
    """Run ``predictor`` over ``examples`` and aggregate per-primitive/language metrics."""
    per_primitive: dict[str, list[tuple[bool, float, float]]] = {}
    per_language: dict[str, list[tuple[bool, float, float]]] = {}
    overall: list[tuple[bool, float, float]] = []
    for example in examples:
        start = time.perf_counter()
        answer = predictor(example.state, example.question)
        elapsed_ms = (time.perf_counter() - start) * 1000
        _, conf, correct = _predicted_and_confidence(example, answer)
        row = (correct, conf, elapsed_ms)
        per_primitive.setdefault(example.type, []).append(row)
        per_language.setdefault(example.lang, []).append(row)
        overall.append(row)
    return {
        "bins": bins,
        "overall": _metrics(overall, bins),
        "per_primitive": {kind: _metrics(rows, bins) for kind, rows in per_primitive.items()},
        "per_language": {lang: _metrics(rows, bins) for lang, rows in per_language.items()},
    }


def save_report(report: dict[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _backend_predictor(backend_name: str, models_dir: str) -> Predictor:
    import asyncio

    from jeba.backends import build_backend
    from jeba.config import Config
    from jeba.wire import SystemOneRequest, answer

    config = Config.from_env({"JEBA_BACKEND": backend_name, "JEBA_MODELS_DIR": models_dir})
    backend = build_backend(config)

    def predict(state: State, question: Question) -> Answer:
        request = SystemOneRequest(state=state, model="jeba-latest", questions={"q": question})
        response = asyncio.run(answer(request, backend))
        return response.answers["q"]

    return predict


def _iter_examples(examples: Sequence[EvalExample]) -> Iterator[EvalExample]:
    yield from examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jeba-evaluate", description=__doc__)
    parser.add_argument("--data", required=True, help="JSONL evaluation records")
    parser.add_argument("--out", required=True, help="output report JSON")
    parser.add_argument("--backend", default="fake", help="fake / encoder / llm")
    parser.add_argument("--models-dir", default=".cache/jeba/models")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    examples = load_examples(args.data, limit=args.limit)
    report = evaluate(examples, _backend_predictor(args.backend, args.models_dir), bins=args.bins)
    save_report(report, args.out)
    print(json.dumps(report["overall"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
