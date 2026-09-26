"""Pluggable inference backends behind ``base.py``.

``build_backend`` resolves the configured backend id to a concrete :class:`Backend`. The
encoder/onnx ids are recognized but not implemented until later milestones and fail with an
actionable message rather than silently misbehaving.
"""

from __future__ import annotations

from tachyone.backends.base import Backend, PredictionResult
from tachyone.config import Config


def build_backend(config: Config) -> Backend:
    """Instantiate the backend selected by ``config.backend``."""
    if config.backend == "fake":
        from tachyone.backends.fake import FakeBackend

        return FakeBackend()
    if config.backend == "llm":
        from tachyone.backends.llm import LLMBackend

        return LLMBackend.from_config(config)
    if config.backend == "encoder":
        from tachyone.backends.encoder import EncoderBackend

        return EncoderBackend.from_config(config)
    if config.backend == "onnx":
        from tachyone.backends.onnx import OnnxBackend

        return OnnxBackend.from_config(config)
    raise ValueError(f"unknown backend: {config.backend!r}")


__all__ = ["Backend", "PredictionResult", "build_backend"]
