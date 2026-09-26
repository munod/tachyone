"""Offline multilingual smoke test (M3-T8, ROUTE-05 / BACK-06 / NFR-S03 / NFR-S04).

Runs the encoder backend across many languages and scripts with an injected deterministic
encoder, so it needs no model and no network. A second test proves no socket is touched.
"""

from __future__ import annotations

import socket

import pytest

from tachyone.backends.encoder import EncoderBackend, EncoderCheckpoint
from tachyone.primitives import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)
from tachyone.router import ENGLISH, MULTILINGUAL, CheckpointInfo, Router

pytestmark = pytest.mark.contract

_QUESTIONS: dict[str, Question] = {
    "department": ChoiceQuestion(
        instructions="Which team?",
        criteria={"billing": "refunds", "technical": "bugs", "other": "else"},
    ),
    "urgency": ScoreQuestion(instructions="How urgent?", criteria=["low", "medium", "high"]),
    "churn": NoulQuestion(instructions="Threatens to cancel?"),
}

# (language label, state, expected checkpoint)
_SAMPLES: tuple[tuple[str, str, str], ...] = (
    ("en", "Please refund the duplicate charge today.", ENGLISH),
    ("pt", "Por favor, devolva a cobrança duplicada hoje.", MULTILINGUAL),
    ("es", "Por favor, devuelva el cargo duplicado hoy.", MULTILINGUAL),
    ("fr", "Merci de rembourser le doublon aujourd'hui.", MULTILINGUAL),
    ("de", "Bitte erstatten Sie die doppelte Abbuchung.", MULTILINGUAL),
    ("ru", "Пожалуйста, верните деньги за дубликат.", MULTILINGUAL),
    ("uk", "Будь ласка, поверніть кошти за дублікат.", MULTILINGUAL),
    ("ar", "من فضلك استرد المبلغ المكرر.", MULTILINGUAL),
    ("he", "אנא החזר את החיוב הכפול.", MULTILINGUAL),
    ("hi", "कृपया डुप्लिकेट शुल्क वापस करें।", MULTILINGUAL),
    ("bn", "অনুগ্রহ করে ডুপ্লিকেট চার্জ ফেরত দিন।", MULTILINGUAL),
    ("th", "กรุณาคืนเงินที่เรียกเก็บซ้ำ", MULTILINGUAL),
    ("ko", "중복 청구를 환불해 주세요.", MULTILINGUAL),
    ("ja", "重複請求を返金してください。", MULTILINGUAL),
    ("zh", "请退还重复收取的费用。", MULTILINGUAL),
    ("el", "Παρακαλώ επιστρέψτε τη διπλή χρέωση.", MULTILINGUAL),
)


def _encode(texts: list[str]) -> list[list[float]]:
    return [
        [float(sum(map(ord, text)) % 89) / 89 + index * 0.02 for index in range(8)]
        for text in texts
    ]


class _Loader:
    def __init__(self) -> None:
        self.loaded: list[str] = []

    def __call__(self, info: CheckpointInfo) -> EncoderCheckpoint:
        self.loaded.append(info.id)
        return EncoderCheckpoint(info, _encode)


@pytest.fixture
def backend() -> tuple[EncoderBackend, _Loader]:
    loader = _Loader()
    router = Router(loader=loader, max_loaded=2)
    return EncoderBackend(router), loader


def _assert_valid(result: object) -> None:
    answers = result.answers  # type: ignore[attr-defined]
    assert set(answers) == set(_QUESTIONS)
    assert isinstance(answers["department"], ChoiceAnswer)
    assert isinstance(answers["urgency"], ScoreAnswer)
    assert isinstance(answers["churn"], NoulAnswer)
    assert abs(sum(answers["department"].probabilities.values()) - 1.0) < 1e-9
    assert abs(sum(answers["urgency"].probabilities.values()) - 1.0) < 1e-9
    assert 0.0 <= answers["churn"].noul <= 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize(("label", "state", "expected"), _SAMPLES, ids=[s[0] for s in _SAMPLES])
async def test_multilingual_offline_answers(
    backend: tuple[EncoderBackend, _Loader], label: str, state: str, expected: str
) -> None:
    manager, loader = backend
    result = await manager.predict(_QUESTIONS, state=state, model="tachyone-latest")
    _assert_valid(result)
    assert loader.loaded[0] == expected, f"{label}: expected {expected}, loaded {loader.loaded}"


@pytest.mark.asyncio
async def test_runs_with_network_disabled(
    backend: tuple[EncoderBackend, _Loader], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", _no_socket)
    monkeypatch.setattr(socket, "create_connection", _no_socket)
    manager, _ = backend
    result = await manager.predict(_QUESTIONS, state="请退款", model="tachyone-latest")
    _assert_valid(result)
