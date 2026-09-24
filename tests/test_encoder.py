"""Unit tests for the encoder backend (M3-T5, BACK-03 / BACK-06 / BACK-07).

The decision math is exercised with an injected deterministic encoder, so these tests need
no torch and no network.
"""

from __future__ import annotations

import importlib.util

import pytest

from jeba.backends.encoder import EncoderBackend, EncoderCheckpoint, load_encoder
from jeba.primitives import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from jeba.router import DEFAULT_CHECKPOINTS, ENGLISH, MULTILINGUAL, CheckpointInfo, Router
from jeba.wire import SystemOneRequest, answer

pytestmark = pytest.mark.contract


def _encode(texts: list[str]) -> list[list[float]]:
    return [
        [float(sum(map(ord, text)) % 97) / 97 + index * 0.01 for index in range(8)]
        for text in texts
    ]


_QUESTIONS: dict[str, Question] = {
    "is_urgent": NoulQuestion(instructions="Does this express urgency?"),
    "department": ChoiceQuestion(
        instructions="Which team?",
        criteria={"billing": "refunds", "technical": "bugs", "sales": "pricing"},
    ),
    "urgency": ScoreQuestion(instructions="How urgent?", criteria=["low", "medium", "high"]),
}


class _Loader:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    def __call__(self, info: CheckpointInfo) -> EncoderCheckpoint:
        self.loaded.append(info.id)
        return EncoderCheckpoint(info, _encode)


def _backend(loader: _Loader | None = None) -> tuple[EncoderBackend, _Loader]:
    used = loader or _Loader()
    return EncoderBackend(Router(loader=used, max_loaded=2)), used


@pytest.mark.asyncio
async def test_all_primitives_answered() -> None:
    backend, _ = _backend()
    result = await backend.predict(_QUESTIONS, state="please refund", model="jeba-latest")
    assert isinstance(result.answers["is_urgent"], NoulAnswer)
    assert isinstance(result.answers["department"], ChoiceAnswer)
    assert isinstance(result.answers["urgency"], ScoreAnswer)


@pytest.mark.asyncio
async def test_distributions_are_normalized_and_confident() -> None:
    backend, _ = _backend()
    result = await backend.predict(_QUESTIONS, state="please refund", model="jeba-latest")
    department = result.answers["department"]
    urgency = result.answers["urgency"]
    assert isinstance(department, ChoiceAnswer)
    assert isinstance(urgency, ScoreAnswer)
    assert abs(sum(department.probabilities.values()) - 1.0) < 1e-9
    assert abs(sum(urgency.probabilities.values()) - 1.0) < 1e-9
    assert 0.0 <= department.confidence <= 1.0
    assert 0.0 <= urgency.confidence <= 1.0
    assert urgency.legend == {0: "low", 1: "medium", 2: "high"}
    noul = result.answers["is_urgent"]
    assert isinstance(noul, NoulAnswer)
    assert 0.0 <= noul.noul <= 1.0


@pytest.mark.asyncio
async def test_empty_questions_short_circuits() -> None:
    backend, loader = _backend()
    result = await backend.predict({}, state="x", model="jeba-latest")
    assert result.answers == {}
    assert result.usage.input_tokens == 0
    assert loader.loaded == []


@pytest.mark.asyncio
async def test_routes_multilingual_for_non_latin() -> None:
    backend, loader = _backend()
    await backend.predict({"q": _QUESTIONS["is_urgent"]}, state="请退款", model="jeba-latest")
    assert loader.loaded == [MULTILINGUAL]


@pytest.mark.asyncio
async def test_explicit_model_overrides_routing() -> None:
    backend, loader = _backend()
    await backend.predict({"q": _QUESTIONS["is_urgent"]}, state="请退款", model=ENGLISH)
    assert loader.loaded == [ENGLISH]


@pytest.mark.asyncio
async def test_encoder_backend_passes_the_contract() -> None:
    backend, _ = _backend()
    request = SystemOneRequest(state="help", model="jeba-en", questions=_QUESTIONS)
    response = await answer(request, backend)
    assert set(response.answers) == set(_QUESTIONS)
    dumped = response.model_dump(mode="json")
    assert set(dumped) == {"model", "answers", "usage"}


def test_from_config_injects_encoder() -> None:
    from jeba.config import Config

    backend = EncoderBackend.from_config(
        Config.from_env({"JEBA_BACKEND": "encoder"}), encode=_encode
    )
    assert backend.name == "encoder"


def test_load_encoder_requires_train_extra() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed; the real encoder path is exercised elsewhere")
    with pytest.raises(RuntimeError, match="train extra"):
        load_encoder(DEFAULT_CHECKPOINTS[ENGLISH], models_dir="/tmp/jeba-models", device="cpu")
