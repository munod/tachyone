"""Conditional integration: the runtime loads the published HF adapters.

Opt-in (network + ``train`` extra): set ``JEBA_TEST_HUB_ADAPTER=1``. Skipped by default so CI
without network stays green.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from jeba.backends import build_backend
from jeba.config import Config
from jeba.primitives import NoulQuestion
from jeba.wire import SystemOneRequest, answer

pytestmark = pytest.mark.contract


@pytest.mark.skipif(
    os.environ.get("JEBA_TEST_HUB_ADAPTER") != "1",
    reason="set JEBA_TEST_HUB_ADAPTER=1 to load adapters from the Hub",
)
def test_runtime_loads_published_adapter() -> None:
    pytest.importorskip("torch")
    backend = build_backend(Config.from_env({"JEBA_BACKEND": "encoder"}))
    request = SystemOneRequest(
        state="quero cancelar minha assinatura agora",
        model="jeba-latest",
        questions={"q": NoulQuestion(instructions="Is the user threatening to leave?")},
    )
    response = asyncio.run(answer(request, backend))
    assert response.answers["q"].type == "noul"
