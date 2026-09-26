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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tachyone.backends.encoder import MODEL_IDS, EncoderModel, load_choice_head, load_encoder
from tachyone.calibration import apply_temperature, confidence, parse_temperature_report
from tachyone.primitives import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    ScoreAnswer,
    State,
)
from tachyone.router import CheckpointInfo, detect_language, detect_script, state_text
from training.evaluate import EvalExample, evaluate, load_examples, save_report

_DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".cache", "tachyone", "models")


def _build_model(
    model_id: str, adapter_dir: str | None, *, device: str, max_len: int
) -> EncoderModel:
    """Reuse the runtime loader so training and inference share one encode implementation."""
    info = CheckpointInfo(
        id="adhoc",
        languages=["*"],
        context=max_len,
        size_params=0,
        base_model=model_id,
        adapter=adapter_dir,
    )
    encode = load_encoder(info, models_dir=_DEFAULT_MODELS_DIR, device=device)
    choice_scorer = load_choice_head(info, models_dir=_DEFAULT_MODELS_DIR)
    return EncoderModel(encode, choice_scorer=choice_scorer)


def _load_temperatures(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_temperature_report(report)


def _apply_temperature(kind: str, answer: Answer, temperature: float) -> Answer:
    if temperature == 1.0:
        return answer
    if kind == "noul":
        assert isinstance(answer, NoulAnswer)
        p = apply_temperature({"true": answer.noul, "false": 1.0 - answer.noul}, temperature)
        return NoulAnswer(noul=p["true"])
    if kind == "choice":
        assert isinstance(answer, ChoiceAnswer)
        scaled = apply_temperature(dict(answer.probabilities), temperature)
        best = max(scaled, key=scaled.__getitem__)
        return ChoiceAnswer(choice=best, probabilities=scaled, confidence=confidence(scaled))
    assert isinstance(answer, ScoreAnswer)
    scaled_levels = apply_temperature(dict(answer.probabilities), temperature)
    expected = sum(index * probability for index, probability in scaled_levels.items())
    return ScoreAnswer(
        score=expected,
        legend=answer.legend,
        probabilities=scaled_levels,
        confidence=confidence(scaled_levels),
    )


def _prediction_row(example: EvalExample, answer: Answer) -> dict[str, Any]:
    if example.type == "noul":
        assert isinstance(answer, NoulAnswer)
        return {
            "id": example.id,
            "type": "noul",
            "lang": example.lang,
            "probabilities": answer.noul,
            "target": int(example.target),
        }
    if example.type == "choice":
        assert isinstance(answer, ChoiceAnswer)
        return {
            "id": example.id,
            "type": "choice",
            "lang": example.lang,
            "probabilities": {key: float(value) for key, value in answer.probabilities.items()},
            "target": example.target,
        }
    assert isinstance(answer, ScoreAnswer)
    return {
        "id": example.id,
        "type": "score",
        "lang": example.lang,
        "probabilities": {
            str(index): float(value) for index, value in answer.probabilities.items()
        },
        "target": int(example.target),
    }


def build_predictor(model: EncoderModel, temperatures: dict[str, float]):
    def predict(state: State, question: Question) -> Answer:
        text = state_text(state)
        lang = detect_language(text, detect_script(text))
        answer = model.answer_state(state, {"q": question}, lang=lang)["q"]
        temperature = temperatures.get(
            f"{question.type}:{lang}", temperatures.get(question.type, 1.0)
        )
        return _apply_temperature(question.type, answer, temperature)

    return predict


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
) -> dict[str, Any]:
    examples = load_examples(records_path, limit=limit)
    model = _build_model(model_id, adapter_dir, device=device, max_len=max_len)
    predictor = build_predictor(model, _load_temperatures(temperature_path))

    if out_predictions:
        path = Path(out_predictions)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for example in examples:
                answer = predictor(example.state, example.question)
                handle.write(json.dumps(_prediction_row(example, answer), sort_keys=True) + "\n")

    report = evaluate(examples, predictor)
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
    )
    print(json.dumps(report["overall"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
