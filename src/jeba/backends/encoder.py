"""Local encoder backend: answer all primitives offline in one batched pass.

The decision math is pure Python over text embeddings, so it is fully testable without a
model: each state and each criterion is embedded, and answers come from cosine similarities
softmaxed over the declared options/levels (BACK-03, BACK-06). Embeddings for the real path
come from a Hugging Face encoder loaded lazily (the ``train`` extra); M4 replaces the
similarity baseline with trained task heads and calibrated temperatures.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from jeba.agent import Agent
from jeba.backends.base import PredictionResult
from jeba.calibration import confidence
from jeba.primitives import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    State,
)
from jeba.router import ENGLISH, MULTILINGUAL, CheckpointInfo, Router, state_text
from jeba.wire import Usage

if TYPE_CHECKING:
    from jeba.config import Config
    from jeba.hooks import Hooks

#: Encodes a batch of texts into dense vectors (one list of floats per text).
type EncodeFn = Callable[[list[str]], list[list[float]]]

#: Base checkpoints used until M4 fine-tunes jeba-owned weights (see ADR-0010).
MODEL_IDS: dict[str, str] = {
    ENGLISH: "answerdotai/ModernBERT-large",
    MULTILINGUAL: "jhu-clsp/mmBERT-base",
}

_TRAIN_HINT = "the train extra is required for the encoder backend: uv sync --extra train"


class EncoderCheckpointLike(Protocol):
    """What a loaded checkpoint must provide (the Agent's runner)."""

    def predict(
        self, states: list[State], questions: dict[str, Question]
    ) -> list[dict[str, Answer]]:
        """Answer every state against the same questions."""
        ...


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return _dot(left, right) / (left_norm * right_norm)


def _softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    if not scores:
        return []
    peak = max(scores)
    exps = [math.exp((score - peak) / temperature) for score in scores]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _instructions_text(instructions: Any) -> str:
    if isinstance(instructions, str):
        return instructions
    if isinstance(instructions, list):
        return " ".join(_instructions_text(item) for item in instructions)
    if isinstance(instructions, dict):
        return " ".join(
            f"{key}: {_instructions_text(value)}" for key, value in instructions.items()
        )
    return str(instructions)


def _question_text(question: Question) -> str:
    return _instructions_text(question.instructions)


def _criterion_texts(question: Question) -> list[str]:
    if isinstance(question, ChoiceQuestion):
        return [
            option if description is None else f"{option}: {_instructions_text(description)}"
            for option, description in question.criteria.items()
        ]
    if isinstance(question, ScoreQuestion):
        return list(question.criteria)
    return []


class EncoderModel:
    """Turns embeddings into typed answers with a similarity baseline."""

    def __init__(
        self,
        encode: EncodeFn,
        *,
        temperature: float = 1.0,
        temperatures: Mapping[str, float] | None = None,
    ) -> None:
        self._encode = encode
        self._temperature = temperature
        self._temperatures = dict(temperatures or {})

    def _temp(self, kind: str) -> float:
        return self._temperatures.get(kind, self._temperature)

    def answer_state(self, state: State, questions: dict[str, Question]) -> dict[str, Answer]:
        state_embedding_text = state_text(state)
        texts = [state_embedding_text]
        plan: list[tuple[str, Question, int, list[int]]] = []
        for question_id, question in questions.items():
            texts.append(_question_text(question))
            question_index = len(texts) - 1
            criterion_indices: list[int] = []
            for criterion in _criterion_texts(question):
                texts.append(criterion)
                criterion_indices.append(len(texts) - 1)
            plan.append((question_id, question, question_index, criterion_indices))

        embeddings = self._encode(texts)
        state_embedding = embeddings[0]
        answers: dict[str, Answer] = {}
        for question_id, question, question_index, criterion_indices in plan:
            question_embedding = embeddings[question_index]
            if isinstance(question, NoulQuestion):
                score = _cosine(question_embedding, state_embedding) / self._temp("noul")
                answers[question_id] = NoulAnswer(noul=min(1.0, max(0.0, _sigmoid(score))))
                continue
            scores = [
                _cosine(state_embedding, embeddings[index])
                + _cosine(question_embedding, embeddings[index])
                for index in criterion_indices
            ]
            if isinstance(question, ChoiceQuestion):
                probabilities = _softmax(scores, self._temp("choice"))
                options = list(question.criteria)
                distribution = {
                    option: probabilities[position] for position, option in enumerate(options)
                }
                best = max(distribution, key=lambda option: distribution[option])
                answers[question_id] = ChoiceAnswer(
                    choice=best, probabilities=distribution, confidence=confidence(distribution)
                )
            else:
                probabilities = _softmax(scores, self._temp("score"))
                level_distribution = {
                    position: probabilities[position] for position in range(len(question.criteria))
                }
                expected = sum(
                    position * probability for position, probability in level_distribution.items()
                )
                answers[question_id] = ScoreAnswer(
                    score=expected,
                    legend=dict(enumerate(question.criteria)),
                    probabilities=level_distribution,
                    confidence=confidence(level_distribution),
                )
        return answers


class EncoderCheckpoint:
    """A loaded checkpoint: an :class:`EncoderModel` plus its metadata."""

    def __init__(
        self,
        info: CheckpointInfo,
        encode: EncodeFn,
        *,
        temperature: float = 1.0,
        temperatures: Mapping[str, float] | None = None,
    ) -> None:
        self.info = info
        self.model = EncoderModel(encode, temperature=temperature, temperatures=temperatures)

    def predict(
        self, states: list[State], questions: dict[str, Question]
    ) -> list[dict[str, Answer]]:
        return [self.model.answer_state(state, questions) for state in states]


def _resolve_device(torch: Any, device: str) -> str:
    if device in {"cpu", "cuda", "mps"}:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(
    info: CheckpointInfo,
    *,
    models_dir: str,
    device: str = "auto",
    offline: bool = False,
) -> EncodeFn:
    """Load a Hugging Face encoder (+ optional LoRA adapter) and return an encoding function.

    The base trunk comes from ``info.base_model`` (falling back to :data:`MODEL_IDS`), and
    ``info.adapter`` (a Hub repo id or local directory) is applied with PEFT when set.
    Torch/transformers are imported lazily; without the ``train`` extra this raises a
    :class:`RuntimeError` naming the extra.
    """
    try:
        import torch  # pyright: ignore[reportMissingImports]
        from transformers import (  # pyright: ignore[reportMissingImports]
            AutoModel,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(_TRAIN_HINT) from exc

    model_id = info.base_model or MODEL_IDS.get(info.id, info.id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=models_dir, local_files_only=offline
    )
    model = AutoModel.from_pretrained(model_id, cache_dir=models_dir, local_files_only=offline)
    if info.adapter:
        try:
            from peft import PeftModel  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(_TRAIN_HINT) from exc
        model = PeftModel.from_pretrained(
            model, info.adapter, cache_dir=models_dir, local_files_only=offline
        )
    resolved = _resolve_device(torch, device)
    model = model.to(resolved)
    model.eval()

    def encode(texts: list[str]) -> list[list[float]]:
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=info.context,
            return_tensors="pt",
        ).to(resolved)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return cast("list[list[float]]", pooled.cpu().tolist())

    return encode


def load_temperatures(
    info: CheckpointInfo, *, models_dir: str, offline: bool = False
) -> dict[str, float]:
    """Load per-primitive fitted temperatures from the adapter repo/dir, if present."""
    if not info.adapter:
        return {}
    source: Path | None = None
    local = Path(os.path.expanduser(info.adapter))
    if local.is_dir():
        source = local / "temperature_calibration.json"
    else:
        try:
            from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]

            downloaded = hf_hub_download(
                repo_id=info.adapter,
                filename="temperature_calibration.json",
                cache_dir=models_dir,
                local_files_only=offline,
            )
            source = Path(downloaded)
        except Exception:
            return {}
    if source is None or not source.exists():
        return {}
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
        return {
            kind: float(entry["temperature"])
            for kind, entry in report.get("per_primitive", {}).items()
        }
    except (ValueError, KeyError, OSError):
        return {}


def apply_adapters(router: Router, adapters: Mapping[str, str | None]) -> Router:
    """Override each checkpoint's adapter from a config mapping (empty value disables it)."""
    for checkpoint_id, adapter in adapters.items():
        info = router.checkpoints.get(checkpoint_id)
        if info is not None:
            router.checkpoints[checkpoint_id] = info.model_copy(update={"adapter": adapter})
    return router


def load_checkpoint(
    info: CheckpointInfo,
    *,
    models_dir: str,
    device: str = "auto",
    offline: bool = False,
) -> EncoderCheckpoint:
    """Load a full checkpoint: encoder (+ adapter) and its fitted temperatures."""
    encode = load_encoder(info, models_dir=models_dir, device=device, offline=offline)
    temperatures = load_temperatures(info, models_dir=models_dir, offline=offline)
    return EncoderCheckpoint(info, encode, temperatures=temperatures)


def _usage(state: State, questions: dict[str, Question]) -> Usage:
    return Usage(
        input_tokens=max(1, len(state_text(state)) // 4),
        output_tokens=max(1, len(questions)),
    )


class EncoderBackend:
    """Offline :class:`~jeba.backends.base.Backend` backed by local encoder checkpoints."""

    name = "encoder"

    def __init__(
        self,
        router: Router,
        *,
        max_batch_size: int | None = None,
        hooks: Hooks | None = None,
    ) -> None:
        self._router = router
        self._max_batch_size = max_batch_size
        self._hooks = hooks

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        hooks: Hooks | None = None,
        encode: EncodeFn | None = None,
    ) -> EncoderBackend:
        """Build from configuration; ``encode`` injects a fake encoder for tests."""

        def loader(info: CheckpointInfo) -> EncoderCheckpoint:
            if encode is not None:
                return EncoderCheckpoint(info, encode)
            return load_checkpoint(
                info,
                models_dir=config.models_dir,
                device=config.device,
                offline=config.offline,
            )

        router = Router(loader=loader, max_loaded=2, hooks=hooks)
        apply_adapters(router, config.adapters)
        if config.preload:
            router.preload(list(config.preload))
        return cls(router, hooks=hooks)

    async def predict(
        self,
        questions: dict[str, Question],
        *,
        state: State,
        model: str,
        return_details: bool = False,
    ) -> PredictionResult:
        if not questions:
            return PredictionResult(answers={}, usage=Usage(input_tokens=0, output_tokens=0))
        checkpoint_id = (
            model if model in self._router.checkpoints else self._router.route(state).checkpoint_id
        )
        checkpoint = cast(EncoderCheckpointLike, self._router.get(checkpoint_id))
        agent = Agent(checkpoint.predict, max_batch_size=self._max_batch_size, hooks=self._hooks)
        answers = await asyncio.to_thread(agent.predict, state, questions)
        return PredictionResult(answers=answers, usage=_usage(state, questions))


def _fake_encode(texts: list[str]) -> list[list[float]]:
    """Deterministic embedding for tests and offline smoke runs (no model)."""
    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vectors.append([byte / 255.0 for byte in digest[:8]])
    return vectors


__all__ = [
    "MODEL_IDS",
    "EncodeFn",
    "EncoderBackend",
    "EncoderCheckpoint",
    "EncoderCheckpointLike",
    "EncoderModel",
    "apply_adapters",
    "load_checkpoint",
    "load_encoder",
    "load_temperatures",
]
