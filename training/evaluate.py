"""Evaluation harness: accuracy, ECE, and latency per primitive, language and domain.

A ``predictor`` is any callable ``(state, question) -> Answer``; the harness converts JSONL
records to primitives, times each prediction, and aggregates metrics (TRAIN-05). Public
probes (MASSIVE/XNLI/typed-decisions) can feed the same harness for reproducible artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tachyone.calibration import expected_calibration_error
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
from training.generate_data import DEFAULT_DOMAIN, _perturb_state, noul_tone

#: A predictor answers a single (state, question) pair. When the caller asks for an explicit
#: head (B6, ADR-0016 §2.1) it is also called with a keyword-only ``head_hint``, so a predictor
#: that does not take one keeps working as long as no hint is requested.
type Predictor = Callable[..., Answer]

#: A gate answers which bank key a ``choice`` record is routed to (``None`` → the shared head).
type GateFn = Callable[[EvalExample], str | None]


@dataclass(frozen=True, slots=True)
class EvalExample:
    """One labeled record: primitive, state, question, target, language, domain."""

    id: str
    type: str
    state: State
    question: Question
    target: int | str
    lang: str
    #: Domain the record came from; pre-B-5 records carry no ``domain`` field and are support.
    domain: str = DEFAULT_DOMAIN
    #: Which holdout slice the record came from (P3 diagnostics); empty for plain eval sets.
    slice: str = ""


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
        domain=str(record.get("domain", DEFAULT_DOMAIN)),
        slice=str(record.get("slice", "")),
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
        # What the answer reports (B-16): the fitted cell when the asset ships one, the
        # selected mass otherwise — the harness must measure what is served.
        conf = answer.confidence
        return predicted, conf, predicted == example.target
    assert isinstance(answer, ScoreAnswer)
    level_distribution: dict[int, float] = dict(answer.probabilities)
    predicted = max(level_distribution, key=level_distribution.__getitem__)
    conf = answer.confidence
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


def record_hint(example: EvalExample, head_hint: str) -> str:
    """Resolve a requested head hint for one record: ``domain`` is the oracle (its own head).

    Any other value is a literal head key (B6) — one rule, shared by the report's predictions
    and its gate row, so the two can never disagree about which head answered.
    """
    return example.domain if head_hint == "domain" else head_hint


def input_signature(example: EvalExample) -> str:
    """Fingerprint of a record's **input** text — state, instructions and criteria, no label.

    The shipped eval sets collide with their training sets on most rows (the generator draws
    from phrase pools shared by both), so a report that quotes one accuracy mixes memorization
    with generalization. Splitting on this fingerprint says how much of each (B7): two records
    that a model cannot tell apart must fingerprint alike, which is why the label is excluded —
    the same text with a different label is a *contradiction*, not a different question.
    """
    payload = json.dumps(
        {
            "state": example.state,
            "instructions": example.question.instructions,
            "criteria": example.question.criteria,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _run(
    examples: Sequence[EvalExample],
    predictor: Predictor,
    bins: int,
    head_hint: str | None = None,
    seen_inputs: Collection[str] | None = None,
) -> dict[str, Any]:
    per_primitive: dict[str, list[tuple[bool, float, float]]] = {}
    per_language: dict[str, list[tuple[bool, float, float]]] = {}
    per_domain: dict[str, list[tuple[bool, float, float]]] = {}
    #: ``domain/lang`` cells: the project gates on the worst cell, never the average, and on a
    #: multilingual five-domain set neither ``per_domain`` nor ``per_language`` alone can show
    #: it (B-5b). Only emitted when the set actually carries more than one domain, so legacy
    #: single-domain reports stay byte-identical.
    per_domain_language: dict[str, list[tuple[bool, float, float]]] = {}
    #: ``noul`` alone per language: the primitive whose labels B-11 had to fix, published next
    #: to the contradictory-label rate so label noise and model error stay separable (L-008).
    noul_per_language: dict[str, list[tuple[bool, float, float]]] = {}
    #: ``primitive/lang`` cells: B-1's acceptance is per-language ECE for ``choice``/``score``
    #: (NFR-C06) and neither axis alone reads it — the gate table the B-16 cycle measures
    #: (previously reconstructed by a throwaway script; now emitted by the harness itself).
    per_primitive_language: dict[str, list[tuple[bool, float, float]]] = {}
    overall: list[tuple[bool, float, float]] = []
    #: Accuracy split by whether the row's input text also occurs in training (B7).
    text_seen: dict[str, list[tuple[bool, float, float]]] = {"seen": [], "unseen": []}
    for example in examples:
        start = time.perf_counter()
        # ``domain`` is the oracle arm: each record is forced onto its own domain's head, so
        # head quality separates from gate quality (B6). Any other value is a literal key.
        if head_hint is None:
            answer = predictor(example.state, example.question)
        else:
            answer = predictor(
                example.state, example.question, head_hint=record_hint(example, head_hint)
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        _, conf, correct = _predicted_and_confidence(example, answer)
        row = (correct, conf, elapsed_ms)
        per_primitive.setdefault(example.type, []).append(row)
        per_language.setdefault(example.lang, []).append(row)
        per_domain.setdefault(example.domain, []).append(row)
        per_domain_language.setdefault(f"{example.domain}/{example.lang}", []).append(row)
        per_primitive_language.setdefault(f"{example.type}/{example.lang}", []).append(row)
        if example.type == "noul":
            noul_per_language.setdefault(example.lang, []).append(row)
        if seen_inputs is not None:
            bucket = "seen" if input_signature(example) in seen_inputs else "unseen"
            text_seen[bucket].append(row)
        overall.append(row)
    report = {
        "bins": bins,
        "overall": _metrics(overall, bins),
        "per_primitive": {kind: _metrics(rows, bins) for kind, rows in per_primitive.items()},
        "per_language": {lang: _metrics(rows, bins) for lang, rows in per_language.items()},
        # Single-domain datasets (the pre-B-5 ones) collapse to one entry, so the key is always
        # present and the worst-domain gate in BACKLOG B-5 has something to read (B-5).
        "per_domain": {domain: _metrics(rows, bins) for domain, rows in per_domain.items()},
        "noul_per_language": {
            lang: _metrics(rows, bins) for lang, rows in noul_per_language.items()
        },
        # Flat ``kind/lang`` keys, the notation BACKLOG B-1/B-16 gate on (NFR-C06).
        "per_primitive_language": {
            cell: _metrics(rows, bins) for cell, rows in sorted(per_primitive_language.items())
        },
    }
    if len(per_domain) > 1:
        report["per_domain_language"] = {
            cell: _metrics(rows, bins) for cell, rows in sorted(per_domain_language.items())
        }
    if seen_inputs is not None:
        report["text_seen"] = {bucket: _metrics(rows, bins) for bucket, rows in text_seen.items()}
    return report


_NOUL_TONE_KEYS = ("request", "neutral", "empty", "unknown")
#: The label each recoverable tone demands; ``unknown`` states are not judged (B-11).
_EXPECTED_TARGET = {"request": 1, "neutral": 0, "empty": 0}


def noul_label_audit(examples: Sequence[EvalExample]) -> dict[str, Any]:
    """Judge every ``noul`` label against the text it accompanies (B-11, L-008).

    Published beside the per-language ``noul`` accuracy, so a contradictory label can no longer
    be read as a model mistake: the old index-parity rule labelled half of ``eval_en.jsonl``'s
    request-toned rows 0 and tracked the per-language RNG's tone/parity correlation (40-55% in
    ``de``/``es``/``nl``). ``contradictory_rate`` is over *judged* records — states no phrase
    bank explains (surface noise) count as ``unknown``, not as contradictions.
    """
    per_language: dict[str, dict[str, int]] = {}

    def cell(lang: str) -> dict[str, int]:
        row = per_language.setdefault(lang, {"n": 0, "positive": 0, "contradictory": 0})
        for key in _NOUL_TONE_KEYS:
            row.setdefault(key, 0)
        return row

    for example in examples:
        if example.type != "noul":
            continue
        state = example.state
        tone = (
            noul_tone(example.domain, example.lang, state) if isinstance(state, str) else "unknown"
        )
        row = cell(example.lang)
        row["n"] += 1
        row[tone] += 1
        target = int(example.target)
        row["positive"] += target
        expected = _EXPECTED_TARGET.get(tone)
        if expected is not None and expected != target:
            row["contradictory"] += 1

    def finish(row: dict[str, int]) -> dict[str, Any]:
        judged = row["n"] - row["unknown"]
        out: dict[str, Any] = dict(row)
        out["judged"] = judged
        out["positive_rate"] = round(row["positive"] / row["n"], 4) if row["n"] else 0.0
        out["contradictory_rate"] = round(row["contradictory"] / judged, 4) if judged else 0.0
        return out

    totals = {
        key: sum(row[key] for row in per_language.values())
        for key in ("n", "positive", "contradictory", *_NOUL_TONE_KEYS)
    }
    return {
        **finish(totals),
        "per_language": {lang: finish(row) for lang, row in sorted(per_language.items())},
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


def choice_gate_report(examples: Sequence[EvalExample], gate: GateFn) -> dict[str, Any]:
    """Judge the gate against each record's own domain (ADR-0016's new failure mode).

    Published beside ``per_domain`` so a misrouted question can never be mistaken for a weak
    head: ``strict`` is a right routing, ``fell_to_shared`` is the ADR's designed fallback
    (the worst it can do is today's numbers) and ``wrong_domain`` is the only harmful
    outcome — that question answered with *another* domain's head. Reports of a run without a
    bank simply omit the whole row.
    """

    def blank() -> dict[str, Any]:
        return {"n": 0, "strict": 0, "fell_to_shared": 0, "wrong_domain": 0}

    totals = blank()
    per_domain: dict[str, dict[str, Any]] = {}
    for example in examples:
        if example.type != "choice":
            continue
        row = per_domain.setdefault(example.domain, blank())
        selected = gate(example)
        for cell in (totals, row):
            cell["n"] += 1
            if selected is None:
                cell["fell_to_shared"] += 1
            elif selected == example.domain:
                cell["strict"] += 1
            else:
                cell["wrong_domain"] += 1

    def finish(row: dict[str, Any]) -> dict[str, Any]:
        count = row["n"]
        return {
            **row,
            "strict_accuracy": row["strict"] / count if count else 0.0,
        }

    return {
        **finish(totals),
        "per_domain": {domain: finish(row) for domain, row in sorted(per_domain.items())},
    }


def evaluate(
    examples: Sequence[EvalExample],
    predictor: Predictor,
    *,
    bins: int = 10,
    noise_rate: float = 0.0,
    noise_seed: int = 42,
    choice_gate: GateFn | None = None,
    head_hint: str | None = None,
    seen_inputs: Collection[str] | None = None,
) -> dict[str, Any]:
    """Run ``predictor`` over ``examples`` and aggregate per-primitive/language/domain metrics.

    ``report["noul_labels"]`` audits the `noul` labels against their own text (B-11), and
    ``report["noul_per_language"]`` reports that primitive per language — accuracy beside the
    contradictory-label rate, so the two can no longer be confused (L-008).

    When ``choice_gate`` is given — i.e. the loaded asset ships a bank — ``report["choice_gate"]``
    judges that gate beside the per-domain numbers it could otherwise hide behind, together with
    ``answers_routed_by``: ``"gate"`` (the default), ``"domain"`` (the oracle arm: every record
    forced onto its own head) or a literal head key (B6, ADR-0016 §2.1).

    When ``noise_rate > 0`` a second, noisy view of the same examples is evaluated and returned
    under the ``"noisy"`` key, so clean accuracy and robustness are never conflated (B-4).

    When ``seen_inputs`` — the :func:`input_signature` of a training set — is given,
    ``report["text_seen"]`` splits accuracy by whether the row's input text occurred in
    training, so a benchmark whose eval rows collide with its training rows can quote the
    generalization side on its own (B7).
    """
    report = _run(examples, predictor, bins, head_hint, seen_inputs)
    report["noul_labels"] = noul_label_audit(examples)
    if choice_gate is not None:
        report["choice_gate"] = {
            **choice_gate_report(examples, choice_gate),
            "answers_routed_by": head_hint or "gate",
        }
    if noise_rate > 0.0:
        report["noise_rate"] = noise_rate
        report["noisy"] = _run(
            add_state_noise(examples, noise_rate, seed=noise_seed),
            predictor,
            bins,
            head_hint,
            seen_inputs,
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
