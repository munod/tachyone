"""Tests for the ONNX backend (M5-T1, BACK-04 / OPS-01).

The contract path is exercised with an injected encoder; the real ``onnxruntime`` load path is
covered only when the extra is installed.
"""

from __future__ import annotations

import importlib.util

import pytest

from tachyone.backends.encoder import EncoderBackend, EncoderCheckpoint
from tachyone.backends.onnx import OnnxBackend, load_onnx_encoder, onnx_model_path
from tachyone.primitives import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from tachyone.router import DEFAULT_CHECKPOINTS, ENGLISH, Router
from tachyone.wire import SystemOneRequest, answer

pytestmark = pytest.mark.contract


def _encode(texts: list[str]) -> list[list[float]]:
    return [
        [float(sum(map(ord, text)) % 83) / 83 + index * 0.01 for index in range(8)]
        for text in texts
    ]


_QUESTIONS: dict[str, Question] = {
    "urgent": NoulQuestion(instructions="Is it urgent?"),
    "team": ChoiceQuestion(
        instructions="Which team?", criteria={"billing": "refunds", "technical": "bugs"}
    ),
    "urgency": ScoreQuestion(instructions="How urgent?", criteria=["low", "medium", "high"]),
}


def _onnx_backend() -> OnnxBackend:
    router = Router(loader=lambda info: EncoderCheckpoint(info, _encode), max_loaded=2)
    return OnnxBackend(router)


@pytest.mark.asyncio
async def test_onnx_backend_answers_all_primitives() -> None:
    result = await _onnx_backend().predict(
        _QUESTIONS, state="please refund", model="tachyone-latest"
    )
    assert isinstance(result.answers["urgent"], NoulAnswer)
    assert isinstance(result.answers["team"], ChoiceAnswer)
    assert isinstance(result.answers["urgency"], ScoreAnswer)


@pytest.mark.asyncio
async def test_onnx_matches_encoder_contract() -> None:
    onnx_response = await answer(
        SystemOneRequest(state="x", model=ENGLISH, questions=_QUESTIONS), _onnx_backend()
    )
    encoder = EncoderBackend(
        Router(loader=lambda info: EncoderCheckpoint(info, _encode), max_loaded=2)
    )
    encoder_response = await answer(
        SystemOneRequest(state="x", model=ENGLISH, questions=_QUESTIONS), encoder
    )
    assert onnx_response.model_dump(mode="json") == encoder_response.model_dump(mode="json")
    assert _onnx_backend().name == "onnx"


def test_onnx_backend_from_config_with_injected_encoder() -> None:
    from tachyone.config import Config

    backend = OnnxBackend.from_config(Config.from_env({"TACHYONE_BACKEND": "onnx"}), encode=_encode)
    assert backend.name == "onnx"


def test_onnx_model_path() -> None:
    assert str(onnx_model_path("/models", ENGLISH)).endswith(f"{ENGLISH}/model.onnx")


def test_load_onnx_encoder_requires_extra() -> None:
    if importlib.util.find_spec("onnxruntime") is not None:
        pytest.skip("onnxruntime is installed; the real ONNX path is exercised elsewhere")
    with pytest.raises(RuntimeError, match="onnx extra"):
        load_onnx_encoder(DEFAULT_CHECKPOINTS[ENGLISH], models_dir="/tmp/tachyone-models")
