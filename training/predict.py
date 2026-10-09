"""Run a (optionally fine-tuned) encoder over records and emit predictions + a metrics report.

This is the glue the release pipeline needs: it loads the base trunk plus a saved LoRA adapter,
computes answers with the same math as the runtime backend, writes a predictions JSONL for
``training.fit_calibration``, and an evaluation report for ``benchmarks.report``. A fitted
temperature file can be applied so the reported ECE reflects calibration.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tachyone.backends.encoder import (
    MODEL_IDS,
    EncoderModel,
    load_choice_head,
    load_confidence,
    load_encoder,
    prototype_strength,
)
from tachyone.calibration import parse_temperature_report
from tachyone.primitives import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    Question,
    ScoreAnswer,
    State,
)
from tachyone.router import CheckpointInfo
from training.evaluate import (
    EvalExample,
    GateFn,
    evaluate,
    input_signature,
    load_examples,
    record_hint,
    save_report,
)

_DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".cache", "tachyone", "models")


def _build_model(
    model_id: str,
    adapter_dir: str | None,
    *,
    device: str,
    max_len: int,
    confidence: bool = True,
    temperatures: dict[str, float] | None = None,
) -> EncoderModel:
    """Reuse the runtime loader so training and inference share one encode implementation.

    ``confidence=False`` skips the adapter's evidence-confidence asset so the emitted
    probabilities are the *natural* ones — that is what the fitter reads, because fitting a
    temperature on top of an already-mapped confidence would be fitting the wrong signal.
    Temperatures live **inside** the model (same as the serve path since B-16), so the
    confidence an answer carries is computed over the scaled distribution — the offline
    pipeline and the backend report the same numbers by construction.
    """
    info = CheckpointInfo(
        id="adhoc",
        languages=["*"],
        context=max_len,
        size_params=0,
        base_model=model_id,
        adapter=adapter_dir,
    )
    encode = load_encoder(info, models_dir=_DEFAULT_MODELS_DIR, device=device)
    choice_bank = load_choice_head(info, models_dir=_DEFAULT_MODELS_DIR)
    calibration = load_confidence(info, models_dir=_DEFAULT_MODELS_DIR) if confidence else None
    return EncoderModel(
        encode,
        temperatures=temperatures,
        choice_bank=choice_bank,
        confidence=calibration,
    )


def _load_temperatures(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_temperature_report(report)


def _prediction_row(example: EvalExample, answer: Answer) -> dict[str, Any]:
    if example.type == "noul":
        assert isinstance(answer, NoulAnswer)
        return {
            "id": example.id,
            "type": "noul",
            "lang": example.lang,
            "slice": example.slice,
            "probabilities": answer.noul,
            "target": int(example.target),
        }
    if example.type == "choice":
        assert isinstance(answer, ChoiceAnswer)
        return {
            "id": example.id,
            "type": "choice",
            "lang": example.lang,
            "slice": example.slice,
            "probabilities": {key: float(value) for key, value in answer.probabilities.items()},
            "target": example.target,
        }
    assert isinstance(answer, ScoreAnswer)
    return {
        "id": example.id,
        "type": "score",
        "lang": example.lang,
        "slice": example.slice,
        "probabilities": {
            str(index): float(value) for index, value in answer.probabilities.items()
        },
        "target": int(example.target),
    }


def _load_prototypes(path: str | Path) -> list[list[float]]:
    """Read a ``state_prototypes.json`` bank (JB-10) and validate it before measuring."""
    from training.build_prototypes import validate_asset

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_asset(payload)
    return [[float(value) for value in row] for row in payload["centroids"]]


def build_predictor(model: EncoderModel):
    """A ``(state, question)`` predictor that answers exactly as the serve path does.

    The language key, the temperature and the confidence map all live inside the model
    (B-16): no explicit ``lang`` is passed, so ``answer_state`` derives the serve key
    itself and the offline pipeline cannot drift from the backend.
    """

    def predict(state: State, question: Question, *, head_hint: str | None = None) -> Answer:
        return model.answer_state(state, {"q": question}, choice_head=head_hint)["q"]

    return predict


def build_gate(model: EncoderModel) -> GateFn | None:
    """The loaded bank's lexical gate, or ``None`` when the asset ships no bank (B6).

    The gate is judged on its own: ``select_key`` is what routes a question whether or not the
    run also forces a hint, so one report can carry both the arm's accuracy and the asset's
    routing quality (ADR-0016 consequences).
    """
    bank = model.choice_bank
    if bank is None:
        return None

    def gate(example: EvalExample) -> str | None:
        if example.type != "choice" or not isinstance(example.question, ChoiceQuestion):
            return None
        return bank.select_key(example.question)

    return gate


def resolve_head_hint(value: str | None, shipped: Iterable[str]) -> str | None:
    """Validate ``--head-hint`` before a run: ``domain`` (the oracle) or a shipped key.

    The *wire* hint falls through when it names nothing the asset ships (ADR-0016 §2.1); the
    evaluation CLI fails fast instead — a typo'd arm would otherwise publish a whole run
    labelled as forced when nothing was.
    """
    if value is None or value == "domain":
        return value
    keys = sorted(shipped)
    if value in keys:
        return value
    raise SystemExit(
        f"unknown head hint {value!r}; this asset ships: {', '.join(keys) or '(no bank keys)'}"
    )


def run(
    records_path: str | Path,
    *,
    model_id: str,
    adapter_dir: str | None = None,
    device: str = "auto",
    max_len: int = 512,
    limit: int | None = None,
    temperature_path: str | None = None,
    out_report: str | Path | None = None,
    out_predictions: str | Path | None = None,
    head_hint: str | None = None,
    train_data: str | Path | None = None,
    prototypes_path: str | Path | None = None,
    apply_confidence: bool = True,
) -> dict[str, Any]:
    examples = load_examples(records_path, limit=limit)
    model = _build_model(
        model_id,
        adapter_dir,
        device=device,
        max_len=max_len,
        confidence=apply_confidence,
        temperatures=_load_temperatures(temperature_path),
    )
    bank_keys = model.choice_bank.domains if model.choice_bank is not None else ()
    hint = resolve_head_hint(head_hint, bank_keys)
    if hint is not None and not bank_keys:
        raise SystemExit(
            f"--head-hint {hint!r} needs an adapter that ships a choice-head bank: {adapter_dir}"
        )
    predictor = build_predictor(model)

    bank = _load_prototypes(prototypes_path) if prototypes_path else None
    if out_predictions:
        path = Path(out_predictions)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                answer = predictor(
                    example.state,
                    example.question,
                    head_hint=None if hint is None else record_hint(example, hint),
                )
                row = _prediction_row(example, answer)
                if bank is not None:
                    # P3: emit the evidence signal beside the prediction the fit will read,
                    # measured on the same embedding the runtime will score (state_embedding).
                    strength = prototype_strength(model.state_embedding(example.state), bank)
                    if strength is None:
                        raise SystemExit(
                            f"prototype bank is not for this encoder ({adapter_dir or model_id})"
                        )
                    row["strength"] = round(strength, 6)
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    # The eval sets collide with their training sets on most rows, so accuracy is also split
    # by whether the row's input text occurred in training (B7) — same harness for every arm.
    seen_inputs = (
        {input_signature(example) for example in load_examples(train_data)} if train_data else None
    )
    report = evaluate(
        examples,
        predictor,
        choice_gate=build_gate(model),
        head_hint=hint,
        seen_inputs=seen_inputs,
    )
    if out_report:
        save_report(report, out_report)
    return report


def _default_model_id(adapter_dir: str) -> str:
    return MODEL_IDS["tachyone-multi"] if "multi" in adapter_dir else MODEL_IDS["tachyone-en"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-predict", description=__doc__)
    parser.add_argument("--data", required=True, help="records JSONL")
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory")
    parser.add_argument("--model-id", default=None, help="base model id (default: inferred)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", default=None, help="fitted temperature report JSON")
    parser.add_argument("--out-report", default=None, help="evaluation report JSON")
    parser.add_argument("--out-predictions", default=None, help="predictions JSONL for calibration")
    parser.add_argument(
        "--head-hint",
        default=None,
        metavar="domain|KEY",
        help="'domain' forces each record onto its own domain head (the oracle arm), a key "
        "forces that head for every record, and omitting it lets the asset's gate decide (B6)",
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="ignore the adapter's evidence-confidence asset: emit the natural probabilities "
        "the fit reads (training.fit_confidence's input), not the deployed confidence",
    )
    parser.add_argument(
        "--prototypes",
        default=None,
        help="state_prototypes.json (JB-10): emit a 'strength' per prediction row — the "
        "evidence signal training.fit_confidence maps to a confidence",
    )
    parser.add_argument(
        "--train-data",
        default=None,
        help="training records JSONL: splits accuracy into report['text_seen'] by whether "
        "each row's input text occurs in it (B7)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    model_id = args.model_id or _default_model_id(args.adapter or "")
    report = run(
        args.data,
        model_id=model_id,
        adapter_dir=args.adapter,
        device=args.device,
        max_len=args.max_len,
        limit=args.limit,
        temperature_path=args.temperature,
        out_report=args.out_report,
        out_predictions=args.out_predictions,
        head_hint=args.head_hint,
        train_data=args.train_data,
        prototypes_path=args.prototypes,
        apply_confidence=not args.no_confidence,
    )
    print(json.dumps(report["overall"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
