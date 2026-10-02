"""Tests for the evaluation CLI's head-hint and gate helpers (B6, ADR-0016 §2.1)."""

from __future__ import annotations

import pytest

from tachyone.backends.encoder import ChoiceHeadBank, EncoderModel
from tachyone.primitives import ChoiceQuestion, NoulQuestion
from training.evaluate import EvalExample
from training.predict import build_gate, resolve_head_hint

_SHIPPED = ("support", "voice")


def _bank() -> ChoiceHeadBank:
    scorer = {"rank": 1, "w1": [[0.0] * 4], "w2": [[0.0] * 2]}
    asset = ChoiceHeadBank.from_dict(
        {
            "shared": scorer,
            "domains": {
                "support": {"head": scorer, "signatures": ["billing", "payments and refunds"]},
                "voice": {"head": scorer, "signatures": ["thermostat", "lighting"]},
            },
        }
    )
    assert asset is not None
    return asset


def _model(choice_bank: ChoiceHeadBank | None = None) -> EncoderModel:
    def encode(texts: list[str]) -> list[list[float]]:
        return [[0.0] * 2 for _ in texts]

    return EncoderModel(encode, choice_bank=choice_bank)


def test_resolve_head_hint_accepts_the_oracle_and_shipped_keys() -> None:
    assert resolve_head_hint(None, _SHIPPED) is None
    assert resolve_head_hint("domain", _SHIPPED) == "domain"
    assert resolve_head_hint("voice", _SHIPPED) == "voice"


def test_resolve_head_hint_fails_fast_on_a_typo() -> None:
    """The wire hint falls through (ADR-0016 §2.1); the evaluation CLI must not."""
    with pytest.raises(SystemExit, match="unknown head hint"):
        resolve_head_hint("voise", _SHIPPED)
    with pytest.raises(SystemExit, match="no bank keys"):
        resolve_head_hint("support", ())


def test_build_gate_is_none_without_a_bank() -> None:
    assert build_gate(_model()) is None


def test_build_gate_routes_lexically_and_ignores_other_primitives() -> None:
    gate = build_gate(_model(_bank()))
    assert gate is not None
    choice = EvalExample(
        id="c",
        type="choice",
        state="x",
        question=ChoiceQuestion.model_validate(
            {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": None, "other": None},
            }
        ),
        target="billing",
        lang="en",
        domain="support",
    )
    assert gate(choice) == "support"  # the asset's own vocabulary routes it

    voice = EvalExample(
        id="v",
        type="choice",
        state="x",
        question=ChoiceQuestion.model_validate(
            {"type": "choice", "instructions": "Which device?", "criteria": {"thermostat": None}}
        ),
        target="thermostat",
        lang="en",
        domain="voice",
    )
    assert gate(voice) == "voice"

    # No vocabulary match at all → the shared head, which is all a wrong gate may do.
    unrelated = EvalExample(
        id="u",
        type="choice",
        state="x",
        question=ChoiceQuestion.model_validate(
            {"type": "choice", "instructions": "qqq?", "criteria": {"zzz1": None}}
        ),
        target="zzz1",
        lang="en",
        domain="voice",
    )
    assert gate(unrelated) is None

    # A non-choice record is never judged, even if it carries a choice-shaped question.
    noul = EvalExample(
        id="n",
        type="noul",
        state="x",
        question=NoulQuestion(instructions="q?"),
        target=1,
        lang="en",
        domain="voice",
    )
    assert gate(noul) is None


def test_predict_cli_applies_confidence_by_default_and_can_opt_out() -> None:
    """The report a run publishes must carry the deployed confidence; the fitter must not."""
    from training.predict import build_parser

    default = build_parser().parse_args(["--data", "rows.jsonl"])
    assert default.no_confidence is False, "prediction reports match the runtime by default"
    fitting = build_parser().parse_args(["--data", "rows.jsonl", "--no-confidence"])
    assert fitting.no_confidence is True
