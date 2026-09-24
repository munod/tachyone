"""ONNX runtime backend.

Ships the same decision math as the encoder backend but executes the transformer through
``onnxruntime`` (CPU or CUDA providers), which is portable and a lightweight deployment
payload (BACK-04, OPS-01). The backend is a thin subclass of
:class:`~jeba.backends.encoder.EncoderBackend`, so it passes the identical contract; only the
embedding function differs. ``onnxruntime`` is imported lazily behind the ``onnx`` extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from jeba.backends.encoder import MODEL_IDS, EncodeFn, EncoderBackend, EncoderCheckpoint
from jeba.router import CheckpointInfo, Router

if TYPE_CHECKING:
    from jeba.config import Config
    from jeba.hooks import Hooks

_ONNX_HINT = "the onnx extra is required for the onnx backend: uv sync --extra onnx"


def _providers(device: str) -> list[str]:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_onnx_encoder(info: CheckpointInfo, *, models_dir: str, device: str = "auto") -> EncodeFn:
    """Load a tokenizer + ONNX session from ``models_dir`` and return an encoding function.

    The model is expected at ``<models_dir>/<checkpoint-id>/model.onnx`` with its tokenizer
    files beside it. Without the ``onnx`` extra this raises a :class:`RuntimeError`.
    """
    try:
        import onnxruntime  # pyright: ignore[reportMissingImports]
        from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(_ONNX_HINT) from exc

    model_dir = Path(models_dir).expanduser() / info.id
    session = onnxruntime.InferenceSession(
        str(model_dir / "model.onnx"), providers=_providers(device)
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    input_names = {entry.name for entry in session.get_inputs()}

    def encode(texts: list[str]) -> list[list[float]]:
        batch = tokenizer(
            texts, padding=True, truncation=True, max_length=info.context, return_tensors="np"
        )
        feed = {
            name: batch[name] for name in ("input_ids", "attention_mask") if name in input_names
        }
        hidden = session.run(None, feed)[0]
        mask = batch["attention_mask"][..., None].astype(hidden.dtype)
        pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-6)
        return cast("list[list[float]]", pooled.tolist())

    return encode


class OnnxBackend(EncoderBackend):
    """Contract-identical encoder backend executed by ``onnxruntime``."""

    name = "onnx"

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        hooks: Hooks | None = None,
        encode: EncodeFn | None = None,
    ) -> OnnxBackend:
        def loader(info: CheckpointInfo) -> EncoderCheckpoint:
            encoder = encode or load_onnx_encoder(
                info, models_dir=config.models_dir, device=config.device
            )
            return EncoderCheckpoint(info, encoder)

        router = Router(loader=loader, max_loaded=2, hooks=hooks)
        if config.preload:
            router.preload(list(config.preload))
        return cls(router, hooks=hooks)


def onnx_model_path(models_dir: str, checkpoint_id: str) -> Path:
    """Conventional path of the ONNX weights for a checkpoint."""
    return Path(models_dir).expanduser() / checkpoint_id / "model.onnx"


def onnx_model_ids() -> dict[str, str]:
    """Base checkpoint ids mapped to their source model (see ADR-0010)."""
    return dict(MODEL_IDS)


__all__ = ["OnnxBackend", "load_onnx_encoder", "onnx_model_ids", "onnx_model_path"]
