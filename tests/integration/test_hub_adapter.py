"""Conditional integration: the runtime loads the published HF adapters.

Opt-in (network + ``train`` extra): set ``TACHYONE_TEST_HUB_ADAPTER=1``. Skipped by default so CI
without network stays green.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from tachyone.backends import build_backend
from tachyone.config import Config
from tachyone.primitives import NoulQuestion
from tachyone.wire import SystemOneRequest, answer

pytestmark = pytest.mark.contract


@pytest.mark.skipif(
    os.environ.get("TACHYONE_TEST_HUB_ADAPTER") != "1",
    reason="set TACHYONE_TEST_HUB_ADAPTER=1 to load adapters from the Hub",
)
def test_runtime_loads_published_adapter() -> None:
    pytest.importorskip("torch")
    backend = build_backend(Config.from_env({"TACHYONE_BACKEND": "encoder"}))
    request = SystemOneRequest(
        state="quero cancelar minha assinatura agora",
        model="tachyone-latest",
        questions={"q": NoulQuestion(instructions="Is the user threatening to leave?")},
    )
    response = asyncio.run(answer(request, backend))
    assert response.answers["q"].type == "noul"
