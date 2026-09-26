"""Evaluation harness: accuracy, ECE, and latency per primitive and language.

A ``predictor`` is any callable ``(state, question) -> Answer``; the harness converts JSONL
records to primitives, times each prediction, and aggregates metrics (TRAIN-05). Public
probes (MASSIVE/XNLI/typed-decisions) can feed the same harness for reproducible artifacts.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tachyone.calibration import confidence, expected_calibration_error
from tachyone.primitives import (
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
from training.generate_data import _perturb_state

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


def _run(examples: Sequence[EvalExample], predictor: Predictor, bins: int) -> dict[str, Any]:
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


def add_state_noise(
    examples: Sequence[EvalExample], rate: float, *, seed: int = 42
) -> list[EvalExample]:
    """Return ``examples`` with one deterministic surface-noise edit applied to ``rate`` of them.

    Used to evaluate robustness on a *noisy* view of the same labelled set without changing the
    labels or questions (B-4). The draw is seeded per ``(seed, example.id)`` so the noisy split
    is reproducible and independent of the clean split.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError("noise rate must be within [0, 1]")
    noisy: list[EvalExample] = []
    for example in examples:
        rng = random.Random(f"{seed}:{example.id}:noise")
        if rate > 0.0 and rng.random() < rate and isinstance(example.state, str):
            noisy.append(replace(example, state=_perturb_state(example.state, rng)))
        else:
            noisy.append(example)
    return noisy


def evaluate(
    examples: Sequence[EvalExample],
    predictor: Predictor,
    *,
    bins: int = 10,
    noise_rate: float = 0.0,
    noise_seed: int = 42,
) -> dict[str, Any]:
    """Run ``predictor`` over ``examples`` and aggregate per-primitive/language metrics.

    When ``noise_rate > 0`` a second, noisy view of the same examples is evaluated and returned
    under the ``"noisy"`` key, so clean accuracy and robustness are never conflated (B-4).
    """
    report = _run(examples, predictor, bins)
    if noise_rate > 0.0:
        report["noise_rate"] = noise_rate
        report["noisy"] = _run(
            add_state_noise(examples, noise_rate, seed=noise_seed), predictor, bins
        )
    return report


def save_report(report: dict[str, Any], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _backend_predictor(backend_name: str, models_dir: str) -> Predictor:
    import asyncio
    import os

    from tachyone.backends import build_backend
    from tachyone.config import Config
    from tachyone.wire import SystemOneRequest, answer

    # Merge the process env with the explicit overrides so `TACHYONE_ADAPTERS` (and any other
    # setting) is honored; passing a literal dict replaces the whole environment.
    source = {**os.environ, "TACHYONE_BACKEND": backend_name, "TACHYONE_MODELS_DIR": models_dir}
    config = Config.from_env(source)
    backend = build_backend(config)

    def predict(state: State, question: Question) -> Answer:
        request = SystemOneRequest(state=state, model="tachyone-latest", questions={"q": question})
        response = asyncio.run(answer(request, backend))
        return response.answers["q"]

    return predict


def _iter_examples(examples: Sequence[EvalExample]) -> Iterator[EvalExample]:
    yield from examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-evaluate", description=__doc__)
    parser.add_argument("--data", required=True, help="JSONL evaluation records")
    parser.add_argument("--out", required=True, help="output report JSON")
    parser.add_argument("--backend", default="fake", help="fake / encoder / llm")
    parser.add_argument("--models-dir", default=".cache/tachyone/models")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.0,
        help="if > 0, also evaluate a noisy view of the same data under report['noisy']",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    examples = load_examples(args.data, limit=args.limit)
    report = evaluate(
        examples,
        _backend_predictor(args.backend, args.models_dir),
        bins=args.bins,
        noise_rate=args.noise_rate,
    )
    save_report(report, args.out)
    print(json.dumps(report["overall"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
