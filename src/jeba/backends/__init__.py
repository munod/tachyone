"""Pluggable inference backends behind ``base.py``.

``build_backend`` resolves the configured backend id to a concrete :class:`Backend`. The
encoder/onnx ids are recognized but not implemented until later milestones and fail with an
actionable message rather than silently misbehaving.
"""

from __future__ import annotations

from jeba.backends.base import Backend, PredictionResult
from jeba.backends.fake import FakeBackend
from jeba.config import Config


def build_backend(config: Config) -> Backend:
    """Instantiate the backend selected by ``config.backend``."""
    if config.backend == "fake":
        return FakeBackend()
    if config.backend == "llm":
        from jeba.backends.llm import LLMBackend

        return LLMBackend.from_config(config)
    if config.backend == "encoder":
        raise NotImplementedError(
            "the encoder backend ships in M3; set JEBA_BACKEND=llm or fake for now"
        )
    if config.backend == "onnx":
        raise NotImplementedError(
            "the onnx backend ships in M5; set JEBA_BACKEND=llm or fake for now"
        )
    raise ValueError(f"unknown backend: {config.backend!r}")


__all__ = ["Backend", "PredictionResult", "build_backend"]
