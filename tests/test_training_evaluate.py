"""Tests for the evaluation harness (M4-T5, TRAIN-05)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tachyone.primitives import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from training.evaluate import (
    EvalExample,
    add_state_noise,
    evaluate,
    load_examples,
    record_to_example,
    save_report,
)
from training.generate_data import DataConfig, generate


def _examples(tmp_path: Path, per_type: int = 8) -> list[EvalExample]:
    path = tmp_path / "eval.jsonl"
    generate(DataConfig(seed=11, per_type=per_type, languages=("en", "pt")), path)
    return load_examples(path)


def _never_called(state: object, question: Question) -> Answer:
    raise AssertionError("predictor should not be called")


def _perfect(examples: list[EvalExample]):
    """Predict the known target with full confidence, keyed by question identity."""
    by_question = {id(example.question): example.target for example in examples}

    def predictor(state: object, question: Question) -> Answer:
        del state
        target = by_question[id(question)]
        if isinstance(question, NoulQuestion):
            return NoulAnswer(noul=1.0 if target == 1 else 0.0)
        keys = list(question.criteria)
        if isinstance(question, ChoiceQuestion):
            return ChoiceAnswer(
                choice=str(target),
                probabilities={key: (1.0 if key == target else 0.0) for key in keys},
                confidence=1.0,
            )
        assert isinstance(question, ScoreQuestion)
        return ScoreAnswer(
            score=float(target),
            legend=dict(enumerate(keys)),
            probabilities={index: (1.0 if index == target else 0.0) for index in range(len(keys))},
            confidence=1.0,
        )

    return predictor


def _biased(state: object, question: Question) -> Answer:
    del state
    if isinstance(question, NoulQuestion):
        return NoulAnswer(noul=0.5)
    keys = list(question.criteria)
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            choice=str(keys[0]),
            probabilities={key: (1.0 if key == keys[0] else 0.0) for key in keys},
            confidence=1.0,
        )
    assert isinstance(question, ScoreQuestion)
    return ScoreAnswer(
        score=0.0,
        legend=dict(enumerate(keys)),
        probabilities={index: (1.0 if index == 0 else 0.0) for index in range(len(keys))},
        confidence=1.0,
    )


def test_record_to_example_types() -> None:
    noul = record_to_example(
        {
            "id": "n",
            "type": "noul",
            "state": "x",
            "instructions": "q?",
            "criteria": {"true": "y", "false": "n"},
            "target": 1,
            "lang": "en",
        }
    )
    assert isinstance(noul.question, NoulQuestion)
    choice = record_to_example(
        {
            "id": "c",
            "type": "choice",
            "state": "x",
            "instructions": "q?",
            "criteria": {"a": None, "b": None},
            "target": "a",
            "lang": "en",
        }
    )
    assert isinstance(choice.question, ChoiceQuestion)
    score = record_to_example(
        {
            "id": "s",
            "type": "score",
            "state": "x",
            "instructions": "q?",
            "criteria": ["low", "high"],
            "target": 1,
            "lang": "en",
        }
    )
    assert isinstance(score.question, ScoreQuestion)


def test_evaluate_with_perfect_predictor(tmp_path: Path) -> None:
    examples = _examples(tmp_path)
    report = evaluate(examples, _perfect(examples))
    assert report["overall"]["accuracy"] == pytest.approx(1.0)
    assert report["overall"]["ece"] == pytest.approx(0.0, abs=1e-9)
    assert set(report["per_primitive"]) == {"noul", "choice", "score"}
    assert report["overall"]["latency_ms"]["p50"] <= report["overall"]["latency_ms"]["p95"]


def test_evaluate_reports_per_language(tmp_path: Path) -> None:
    examples = _examples(tmp_path)
    report = evaluate(examples, _biased)
    assert set(report["per_language"]) <= {"en", "pt"}
    assert report["overall"]["accuracy"] < 1.0


def test_save_report(tmp_path: Path) -> None:
    examples = _examples(tmp_path, per_type=3)
    out = tmp_path / "report.json"
    save_report(evaluate(examples, _biased), out)
    assert '"overall"' in out.read_text(encoding="utf-8")


def test_evaluate_empty() -> None:
    report = evaluate([], _never_called)
    assert report["overall"]["n"] == 0


def test_evaluate_noise_rate_adds_noisy_view(tmp_path: Path) -> None:
    examples = _examples(tmp_path)
    report = evaluate(examples, _biased, noise_rate=0.3)
    assert report["noise_rate"] == 0.3
    assert "noisy" in report
    assert set(report["noisy"]) == {"bins", "overall", "per_primitive", "per_language"}
    assert report["noisy"]["overall"]["n"] == report["overall"]["n"]


def test_evaluate_default_has_no_noisy_view(tmp_path: Path) -> None:
    report = evaluate(_examples(tmp_path), _biased)
    assert "noisy" not in report


def test_add_state_noise_is_deterministic_and_scoped() -> None:
    examples = [
        EvalExample(
            id="a",
            type="noul",
            state="café com açúcar",
            question=NoulQuestion(instructions="q?"),
            target=1,
            lang="pt",
        )
    ]
    one = add_state_noise(examples, rate=1.0, seed=7)
    two = add_state_noise(examples, rate=1.0, seed=7)
    assert [e.state for e in one] == [e.state for e in two]


def test_add_state_noise_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        add_state_noise([], rate=1.5)


def test_backend_predictor_honors_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: the predictor used to pass a literal env dict to Config.from_env, dropping
    # TACHYONE_ADAPTERS. It must merge the process env so a local adapter can be evaluated.
    from tachyone.config import Config

    monkeypatch.setenv("TACHYONE_ADAPTERS", "tachyone-multi=checkpoints/multi_noisy")
    merged = {
        **__import__("os").environ,
        "TACHYONE_BACKEND": "encoder",
        "TACHYONE_MODELS_DIR": ".cache/tachyone/models",
    }
    assert Config.from_env(merged).adapters == {"tachyone-multi": "checkpoints/multi_noisy"}
