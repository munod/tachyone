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
from jeba.calibration import confidence, parse_temperature_report
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
from jeba.router import (
    ENGLISH,
    MULTILINGUAL,
    CheckpointInfo,
    Router,
    detect_language,
    detect_script,
    state_text,
)
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


class ChoiceScorer:
    """Low-rank residual scorer for ``choice``.

    The base score is the cosine baseline (``cos(state, option) + cos(question, option)``); the
    scorer adds a learned low-rank bilinear residual
    ``u = W1 [state; question]``, ``v_k = W2 c_k``, ``<u, v_k> / sqrt(rank)``. With the small
    initialization used in training the residual starts near zero, so the model begins exactly at
    the cosine baseline and learns a correction. It is permutation-equivariant across options and
    parameter-free in the number of options (1..255). Stored as plain lists so it runs without
    torch (and exports to ONNX).
    """

    def __init__(self, w1: list[list[float]], w2: list[list[float]]) -> None:
        self._w1 = w1
        self._w2 = w2
        self._rank = len(w1)
        self._scale = 1.0 / math.sqrt(self._rank) if self._rank else 0.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChoiceScorer:
        return cls(
            [[float(value) for value in row] for row in data["w1"]],
            [[float(value) for value in row] for row in data["w2"]],
        )

    def residual(
        self,
        state_embedding: list[float],
        question_embedding: list[float],
        criteria_embeddings: list[list[float]],
    ) -> list[float]:
        if not criteria_embeddings:
            return []
        context = state_embedding + question_embedding  # [2H]
        u = [_dot(row, context) for row in self._w1]
        residual: list[float] = []
        for criterion in criteria_embeddings:
            v = [_dot(row, criterion) for row in self._w2]
            residual.append(self._scale * sum(a * b for a, b in zip(u, v, strict=True)))
        return residual


class EncoderModel:
    """Turns embeddings into typed answers with a similarity baseline."""

    def __init__(
        self,
        encode: EncodeFn,
        *,
        temperature: float = 1.0,
        temperatures: Mapping[str, float] | None = None,
        choice_scorer: ChoiceScorer | None = None,
    ) -> None:
        self._encode = encode
        self._temperature = temperature
        self._temperatures = dict(temperatures or {})
        self._choice_scorer = choice_scorer

    def _temp(self, kind: str, lang: str | None = None) -> float:
        """Temperature for ``kind``, preferring a per-language fit over the per-primitive one."""
        if lang:
            specific = self._temperatures.get(f"{kind}:{lang}")
            if specific is not None:
                return specific
        return self._temperatures.get(kind, self._temperature)

    def answer_state(
        self, state: State, questions: dict[str, Question], *, lang: str | None = None
    ) -> dict[str, Answer]:
        state_embedding_text = state_text(state)
        if lang is None:
            lang = detect_language(state_embedding_text, detect_script(state_embedding_text))
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
                score = _cosine(question_embedding, state_embedding) / self._temp("noul", lang)
                answers[question_id] = NoulAnswer(noul=min(1.0, max(0.0, _sigmoid(score))))
                continue
            scores = [
                _cosine(state_embedding, embeddings[index])
                + _cosine(question_embedding, embeddings[index])
                for index in criterion_indices
            ]
            if isinstance(question, ChoiceQuestion):
                if self._choice_scorer is not None:
                    residual = self._choice_scorer.residual(
                        state_embedding,
                        question_embedding,
                        [embeddings[index] for index in criterion_indices],
                    )
                    scores = [base + extra for base, extra in zip(scores, residual, strict=True)]
                probabilities = _softmax(scores, self._temp("choice", lang))
                options = list(question.criteria)
                distribution = {
                    option: probabilities[position] for position, option in enumerate(options)
                }
                best = max(distribution, key=lambda option: distribution[option])
                answers[question_id] = ChoiceAnswer(
                    choice=best, probabilities=distribution, confidence=confidence(distribution)
                )
            else:
                probabilities = _softmax(scores, self._temp("score", lang))
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
        choice_scorer: ChoiceScorer | None = None,
    ) -> None:
        self.info = info
        self.model = EncoderModel(
            encode,
            temperature=temperature,
            temperatures=temperatures,
            choice_scorer=choice_scorer,
        )

    def predict(
        self, states: list[State], questions: dict[str, Question]
    ) -> list[dict[str, Answer]]:
        return [self.model.answer_state(state, questions) for state in states]


def _resolve_device(torch: Any, device: str) -> str:
    if device in {"cpu", "cuda", "mps"}:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


#: Maximum number of captured CUDA graphs kept per checkpoint (one per input shape). High enough
#: that a diverse batch of distinct lengths does not evict graphs and re-capture mid-run.
_MAX_CUDA_GRAPHS = 128


def _pool_hidden(hidden: Any, mask: Any) -> Any:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


def _cuda_graph_encode(model: Any, tokenize: Any, *, device: str) -> EncodeFn:
    """Wrap a forward pass in per-shape CUDA graphs to cut launch overhead (B-2).

    CUDA graphs require static shapes, so one graph is captured per ``(batch, length)`` bucket
    and replayed by copying the fresh inputs into the captured static tensors. Shapes that fail
    to capture fall back to eager execution, and the oldest graph is evicted past the cap.
    """
    import torch  # pyright: ignore[reportMissingImports]

    graphs: dict[tuple[int, int], tuple[Any, Any, Any, Any]] = {}
    eager_shapes: set[tuple[int, int]] = set()

    def _eager(input_ids: Any, attention_mask: Any) -> Any:
        with torch.no_grad():
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return _pool_hidden(hidden, attention_mask)

    def _capture(input_ids: Any, attention_mask: Any) -> tuple[Any, Any, Any, Any]:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.current_stream().wait_stream(side)
        static_ids = input_ids.clone()
        static_mask = attention_mask.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            static_hidden = model(
                input_ids=static_ids, attention_mask=static_mask
            ).last_hidden_state
        return graph, static_ids, static_mask, static_hidden

    def encode(texts: list[str]) -> list[list[float]]:
        input_ids, attention_mask = tokenize(texts)
        key = (int(input_ids.shape[0]), int(input_ids.shape[1]))
        entry = graphs.get(key)
        if entry is None and key not in eager_shapes:
            try:
                entry = _capture(input_ids, attention_mask)
                if len(graphs) >= _MAX_CUDA_GRAPHS:
                    graphs.pop(next(iter(graphs)))
                graphs[key] = entry
            except Exception:  # pragma: no cover - device/driver dependent
                eager_shapes.add(key)
        if entry is None:
            return cast("list[list[float]]", _eager(input_ids, attention_mask).cpu().tolist())
        graph, static_ids, static_mask, static_hidden = entry
        static_ids.copy_(input_ids)
        static_mask.copy_(attention_mask)
        graph.replay()
        return cast("list[list[float]]", _pool_hidden(static_hidden, static_mask).cpu().tolist())

    return encode


def load_encoder(
    info: CheckpointInfo,
    *,
    models_dir: str,
    device: str = "auto",
    offline: bool = False,
    fast: bool = False,
) -> EncodeFn:
    """Load a Hugging Face encoder (+ optional LoRA adapter) and return an encoding function.

    The base trunk comes from ``info.base_model`` (falling back to :data:`MODEL_IDS`), and
    ``info.adapter`` (a Hub repo id or local directory) is applied with PEFT when set.
    Torch/transformers are imported lazily; without the ``train`` extra this raises a
    :class:`RuntimeError` naming the extra. With ``fast`` and a CUDA device, the forward is
    wrapped in per-shape CUDA graphs (bf16 when supported) and otherwise falls back unchanged.
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

    if fast:
        from jeba.fast import maybe_accelerate

        def tokenize(texts: list[str]) -> tuple[Any, Any]:
            batch = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=info.context,
                return_tensors="pt",
            ).to(resolved)
            return batch["input_ids"], batch["attention_mask"]

        def builder() -> EncodeFn:
            if resolved == "cuda" and torch.cuda.is_bf16_supported():
                model.to(torch.bfloat16)
            return _cuda_graph_encode(model, tokenize, device=resolved)

        return cast("EncodeFn", maybe_accelerate(encode, device=resolved, builder=builder).target)

    return encode


def load_temperatures(
    info: CheckpointInfo, *, models_dir: str, offline: bool = False
) -> dict[str, float]:
    """Load fitted temperatures from the adapter repo/dir, if present.

    Keys are ``kind`` for the per-primitive fit and ``kind:lang`` for the per-language fit
    (B-1); :meth:`EncoderModel._temp` prefers the latter. A legacy per-primitive-only file still
    loads.
    """
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
    except (ValueError, OSError):
        return {}
    return parse_temperature_report(report)


def load_choice_head(
    info: CheckpointInfo, *, models_dir: str, offline: bool = False
) -> ChoiceScorer | None:
    """Load a trained ``choice`` scorer from the adapter repo/dir, if present."""
    if not info.adapter:
        return None
    source: Path | None = None
    local = Path(os.path.expanduser(info.adapter))
    if local.is_dir():
        candidate = local / "choice_head.json"
        if candidate.exists():
            source = candidate
    else:
        try:
            from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]

            source = Path(
                hf_hub_download(
                    repo_id=info.adapter,
                    filename="choice_head.json",
                    cache_dir=models_dir,
                    local_files_only=offline,
                )
            )
        except Exception:
            return None
    if source is None or not source.exists():
        return None
    try:
        return ChoiceScorer.from_dict(json.loads(source.read_text(encoding="utf-8")))
    except (ValueError, KeyError, OSError):
        return None


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
    fast: bool = False,
) -> EncoderCheckpoint:
    """Load a full checkpoint: encoder (+ adapter), fitted temperatures, and choice scorer."""
    encode = load_encoder(info, models_dir=models_dir, device=device, offline=offline, fast=fast)
    temperatures = load_temperatures(info, models_dir=models_dir, offline=offline)
    choice_scorer = load_choice_head(info, models_dir=models_dir, offline=offline)
    return EncoderCheckpoint(info, encode, temperatures=temperatures, choice_scorer=choice_scorer)


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
                fast=config.fast,
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
    "ChoiceScorer",
    "EncodeFn",
    "EncoderBackend",
    "EncoderCheckpoint",
    "EncoderCheckpointLike",
    "EncoderModel",
    "apply_adapters",
    "load_checkpoint",
    "load_choice_head",
    "load_encoder",
    "load_temperatures",
]
