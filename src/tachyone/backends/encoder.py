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
import logging
import math
import os
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from tachyone.agent import Agent
from tachyone.backends.base import PredictionResult
from tachyone.calibration import confidence, interpolate_confidence, parse_temperature_report
from tachyone.primitives import (
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
from tachyone.router import (
    ENGLISH,
    MULTILINGUAL,
    CheckpointInfo,
    Router,
    detect_language,
    detect_script,
    state_text,
)
from tachyone.wire import Usage

if TYPE_CHECKING:
    from tachyone.config import Config
    from tachyone.hooks import Hooks

#: Encodes a batch of texts into dense vectors (one list of floats per text).
type EncodeFn = Callable[[list[str]], list[list[float]]]

#: Base checkpoints used until M4 fine-tunes tachyone-owned weights (see ADR-0010).
MODEL_IDS: dict[str, str] = {
    ENGLISH: "answerdotai/ModernBERT-large",
    MULTILINGUAL: "jhu-clsp/mmBERT-base",
}

_TRAIN_HINT = "the train extra is required for the encoder backend: uv sync --extra train"


class EncoderCheckpointLike(Protocol):
    """What a loaded checkpoint must provide (the Agent's runner)."""

    def predict(
        self, states: list[State], questions: dict[str, Question], *, choice_head: str | None = None
    ) -> list[dict[str, Answer]]:
        """Answer every state against the same questions (optional ADR-0016 head hint)."""
        ...


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return _dot(left, right) / (left_norm * right_norm)


def prototype_strength(
    embedding: Sequence[float], centroids: Sequence[Sequence[float]]
) -> float | None:
    """Largest cosine between an embedding and a training-state prototype bank (P3).

    Returns ``None`` when the bank was built for a different embedding width — a mismatch
    is refused, never approximated. Shared by the runtime and the fitter so the signal the
    map is *fitted* on is bit-for-bit the signal it is *applied* with.
    """
    if not centroids or not embedding:
        return None
    dim = len(centroids[0])
    if len(embedding) != dim or any(len(centroid) != dim for centroid in centroids):
        return None
    return max(_cosine(embedding, centroid) for centroid in centroids)


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


def _gate_tokens(text: str) -> tuple[str, ...]:
    """Normalize signature/question text into comparison tokens (ADR-0016 §2.2).

    Diacritics fold to their base letters and everything non-alphanumeric splits, so
    ``"Réfunds"`` and ``"refunds"`` compare on equal footing. Pure string work: no
    encode, no model, deterministic and offline (ROUTE-03).
    """
    folded = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(char for char in folded if not unicodedata.combining(char))
    return tuple(re.findall(r"[a-z0-9]+", stripped.lower()))


def _gate_prefixes(tokens: Sequence[str]) -> set[str]:
    """Every prefix of every token, so one set lookup replaces a scan per signature term."""
    prefixes: set[str] = set()
    for token in tokens:
        for size in range(1, len(token) + 1):
            prefixes.add(token[:size])
    return prefixes


@dataclass(frozen=True, slots=True)
class DomainChoiceHead:
    """One domain-keyed head: its scorer plus the signature terms the gate matches."""

    head: ChoiceScorer
    signatures: tuple[str, ...] = ()


class ChoiceHeadBank:
    """Shared + per-domain ``choice`` heads behind a deterministic gate (ADR-0016).

    The bank ships *inside* ``choice_head.json``: ``{shared, domains: {name: {head,
    signatures}}}``. The legacy single-scorer format loads as a shared-only bank, so every
    published adapter keeps answering exactly as before. The gate picks one head per
    ``choice`` question, in priority order:

    1. the caller's ``choice_head`` hint, honoured only when it names a shipped domain key;
    2. the lexical signature gate — the strictly highest signature-match count wins;
    3. no match or ambiguity → the shared head (bit-for-bit today's behaviour, and the
       worst a wrong gate can do).

    Signature terms are parsed into tokens once at load time; per question the gate builds
    a prefix index of the question's tokens and counts matched entries — microseconds,
    fully offline.
    """

    def __init__(
        self,
        shared: ChoiceScorer | None,
        domains: Mapping[str, DomainChoiceHead] | None = None,
    ) -> None:
        self.shared = shared
        self.domains: dict[str, DomainChoiceHead] = dict(domains or {})
        self._terms: dict[str, tuple[tuple[str, ...], ...]] = {
            name: tuple(
                dict.fromkeys(  # dedupe signature terms, order-preserving
                    tokens
                    for tokens in (_gate_tokens(signature) for signature in head.signatures)
                    if tokens
                )
            )
            for name, head in self.domains.items()
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, warn: Callable[[str], None] | None = None
    ) -> ChoiceHeadBank | None:
        """Parse the keyed or legacy asset shape; ``None`` when nothing is usable.

        A corrupt piece degrades only itself (B-8): a broken ``shared`` keeps the domain
        heads, a broken domain keeps the others, a broken entry warns by name.
        """

        def _warn(message: str) -> None:
            if warn is not None:
                warn(message)

        if "shared" in data or "domains" in data:
            shared: ChoiceScorer | None = None
            if "shared" in data:
                try:
                    shared = ChoiceScorer.from_dict(data["shared"])
                except (ValueError, KeyError, TypeError) as exc:
                    _warn(f"invalid shared head, {type(exc).__name__}: {exc}")
            else:
                _warn("keyed asset ships no 'shared' head; unmatched questions use the baseline")
            raw_domains = data.get("domains", {})
            if not isinstance(raw_domains, Mapping):
                _warn(f"'domains' must be an object, found {type(raw_domains).__name__}")
                raw_domains = {}
            domains: dict[str, DomainChoiceHead] = {}
            for name, entry in raw_domains.items():
                if not isinstance(entry, Mapping) or "head" not in entry:
                    _warn(f"domain {name!r} ships no usable head; dropping it")
                    continue
                try:
                    head = ChoiceScorer.from_dict(entry["head"])
                except (ValueError, KeyError, TypeError) as exc:
                    _warn(f"domain {name!r}: invalid head, {type(exc).__name__}: {exc}")
                    continue
                signatures = entry.get("signatures")
                if signatures is None:
                    signatures = ()
                elif not isinstance(signatures, (list, tuple)) or not all(
                    isinstance(term, str) for term in signatures
                ):
                    _warn(f"domain {name!r}: invalid signatures; reachable only by the hint")
                    signatures = ()
                domains[str(name)] = DomainChoiceHead(head=head, signatures=tuple(signatures))
            if shared is None and not domains:
                return None
            return cls(shared=shared, domains=domains)
        try:
            return cls(shared=ChoiceScorer.from_dict(data))
        except (ValueError, KeyError, TypeError) as exc:
            _warn(f"invalid scorer, {type(exc).__name__}: {exc}")
            return None

    def _score(self, question: ChoiceQuestion) -> dict[str, int]:
        """Signature entries matched by the question's instructions, labels and descriptions."""
        text = " ".join([_question_text(question), *_criterion_texts(question)])
        tokens = _gate_tokens(text)
        if not tokens:
            return {}
        prefixes = _gate_prefixes(tokens)
        return {
            name: sum(1 for entry in self._terms[name] if all(token in prefixes for token in entry))
            for name in self.domains
        }

    def select_key(self, question: ChoiceQuestion, *, hint: str | None = None) -> str | None:
        """The domain key the gate selects, or ``None`` for the shared head.

        The hint wins when it names a shipped domain; an unknown hint falls through to the
        lexical gate (ADR-0016 §2.1). A tie or a zero score means ambiguity → shared.
        """
        if hint is not None and hint in self.domains:
            return hint
        if not self.domains:
            return None
        scores = self._score(question)
        if not scores:
            return None
        best = max(scores.values())
        if best <= 0:
            return None
        winners = [name for name, score in scores.items() if score == best]
        return winners[0] if len(winners) == 1 else None

    def select(self, question: ChoiceQuestion, *, hint: str | None = None) -> ChoiceScorer | None:
        """The scorer to apply (``None`` → no residual, i.e. the cosine baseline)."""
        key = self.select_key(question, hint=hint)
        if key is None:
            return self.shared
        return self.domains[key].head


class ConfidenceCalibration:
    """Evidence-conditioned confidence: a prototype bank plus the fitted ``noul`` map (P3).

    ``strength`` is the largest cosine between a state embedding and any training-state
    centroid — how like the training distribution this input is (spec, *P3 design*). The
    map turns that evidence into the confidence a ``noul`` answer reports: the *direction*
    still comes from the answer itself, so no input ever changes its mind, only how loudly
    it says it (P3's rule: Intelligence is independent of the Calibration axis).

    Built from ``confidence_calibration.json``, which embeds the bank and the map together
    so the pair can never drift apart. Unusable content is dropped with a warning naming
    the problem (B-8 discipline): the engine then falls back to the fitted temperature,
    exactly as an adapter that ships no asset would.
    """

    __slots__ = ("_dim", "centroids", "noul_knots", "prototypes")

    def __init__(
        self,
        centroids: list[list[float]],
        noul_knots: list[tuple[float, float]],
        *,
        prototypes: Mapping[str, Any] | None = None,
    ) -> None:
        self.centroids = centroids
        self.noul_knots = noul_knots
        self.prototypes = dict(prototypes or {})
        self._dim = len(centroids[0]) if centroids else 0

    @property
    def dim(self) -> int:
        """Embedding width the bank was built for; a mismatch is refused, never guessed."""
        return self._dim

    def strength(self, embedding: Sequence[float]) -> float | None:
        """Max cosine to the bank, or ``None`` when the bank is for a different space."""
        return prototype_strength(embedding, self.centroids)

    def noul_confidence(self, strength: float) -> float:
        """Confidence for this evidence, clamped to ``[0.5, 1]`` (a binary floor)."""
        value = interpolate_confidence(strength, self.noul_knots)
        return min(1.0, max(0.5, value))

    @classmethod
    def from_dict(cls, data: Any, *, warn: Callable[[str], None]) -> ConfidenceCalibration | None:
        """Validate the asset and build it; call ``warn`` and return ``None`` if unusable."""
        if not isinstance(data, Mapping):
            warn("asset is not a JSON object")
            return None
        if data.get("version") != CONFIDENCE_VERSION:
            warn(f"unsupported version {data.get('version')!r}")
            return None
        prototypes = data.get("prototypes")
        if not isinstance(prototypes, Mapping):
            warn("missing 'prototypes'")
            return None
        if prototypes.get("metric") != "cosine":
            warn(f"unsupported metric {prototypes.get('metric')!r}")
            return None
        raw_centroids = prototypes.get("centroids")
        dim = prototypes.get("dim")
        if not isinstance(raw_centroids, list) or not raw_centroids:
            warn("centroids must be a non-empty list")
            return None
        if not isinstance(dim, int) or dim < 1:
            warn(f"dim must be a positive int, got {dim!r}")
            return None
        if prototypes.get("k") != len(raw_centroids):
            warn(f"k={prototypes.get('k')!r} does not match {len(raw_centroids)} centroids")
            return None
        centroids: list[list[float]] = []
        for index, centroid in enumerate(raw_centroids):
            if not isinstance(centroid, list) or len(centroid) != dim:
                warn(f"centroid {index} is not a {dim}-vector")
                return None
            try:
                values = [float(value) for value in centroid]
            except (TypeError, ValueError):
                warn(f"centroid {index} contains a non-numeric value")
                return None
            if not all(math.isfinite(value) for value in values):
                warn(f"centroid {index} contains a non-finite value")
                return None
            centroids.append(values)
        noul = data.get("noul")
        raw_knots = noul.get("knots") if isinstance(noul, Mapping) else None
        knots: list[tuple[float, float]] = []
        if not isinstance(raw_knots, list) or not raw_knots:
            warn("missing 'noul' knots")
            return None
        previous = -math.inf
        for entry in raw_knots:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                warn("knots must be [strength, confidence] pairs")
                return None
            try:
                strength, level = float(entry[0]), float(entry[1])
            except (TypeError, ValueError):
                warn("knots must be numeric")
                return None
            if not (math.isfinite(strength) and 0.0 <= level <= 1.0):
                warn(f"knot out of range: {entry!r}")
                return None
            if strength <= previous:
                warn("knots must be strictly increasing in strength")
                return None
            previous = strength
            knots.append((strength, level))
        return cls(centroids, knots, prototypes=prototypes)


class EncoderModel:
    """Turns embeddings into typed answers with a similarity baseline."""

    def __init__(
        self,
        encode: EncodeFn,
        *,
        temperature: float = 1.0,
        temperatures: Mapping[str, float] | None = None,
        choice_bank: ChoiceHeadBank | None = None,
        confidence: ConfidenceCalibration | None = None,
    ) -> None:
        self._encode = encode
        self._temperature = temperature
        self._temperatures = dict(temperatures or {})
        self._choice_bank = choice_bank
        self._confidence = confidence

    @property
    def choice_bank(self) -> ChoiceHeadBank | None:
        """The loaded head bank (``None`` when the adapter ships no asset)."""
        return self._choice_bank

    @property
    def confidence(self) -> ConfidenceCalibration | None:
        """The loaded evidence-confidence asset (``None`` when the adapter ships no asset)."""
        return self._confidence

    def state_embedding(self, state: State) -> list[float]:
        """Embed ``state`` exactly as scoring does — same tokenization, same window.

        Exposed so the fitter can measure the evidence signal (``prototype_strength``) on
        the same vector the runtime will see; anything computed from a differently encoded
        state would be fitting a different input than the one deployed.
        """
        return self._encode([state_text(state)])[0]

    def _temp(self, kind: str, lang: str | None = None) -> float:
        """Temperature for ``kind``, preferring a per-language fit over the per-primitive one."""
        if lang:
            specific = self._temperatures.get(f"{kind}:{lang}")
            if specific is not None:
                return specific
        return self._temperatures.get(kind, self._temperature)

    def answer_state(
        self,
        state: State,
        questions: dict[str, Question],
        *,
        lang: str | None = None,
        choice_head: str | None = None,
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
        calibration = self._confidence
        strength = calibration.strength(state_embedding) if calibration is not None else None
        answers: dict[str, Answer] = {}
        for question_id, question, question_index, criterion_indices in plan:
            question_embedding = embeddings[question_index]
            if isinstance(question, NoulQuestion):
                raw = _cosine(question_embedding, state_embedding)
                if strength is not None and calibration is not None:
                    # P3: the answer keeps its own direction, the evidence sets the volume.
                    level = calibration.noul_confidence(strength)
                    answers[question_id] = NoulAnswer(noul=level if raw >= 0.0 else 1.0 - level)
                    continue
                score = raw / self._temp("noul", lang)
                answers[question_id] = NoulAnswer(noul=min(1.0, max(0.0, _sigmoid(score))))
                continue
            scores = [
                _cosine(state_embedding, embeddings[index])
                + _cosine(question_embedding, embeddings[index])
                for index in criterion_indices
            ]
            if isinstance(question, ChoiceQuestion):
                # One head per choice question: hint → lexical gate → shared (ADR-0016 §2).
                scorer = (
                    self._choice_bank.select(question, hint=choice_head)
                    if self._choice_bank is not None
                    else None
                )
                if scorer is not None:
                    residual = scorer.residual(
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
        choice_bank: ChoiceHeadBank | None = None,
        confidence: ConfidenceCalibration | None = None,
    ) -> None:
        self.info = info
        self.model = EncoderModel(
            encode,
            temperature=temperature,
            temperatures=temperatures,
            choice_bank=choice_bank,
            confidence=confidence,
        )

    def predict(
        self, states: list[State], questions: dict[str, Question], *, choice_head: str | None = None
    ) -> list[dict[str, Answer]]:
        return [
            self.model.answer_state(state, questions, choice_head=choice_head) for state in states
        ]


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
        from tachyone.fast import maybe_accelerate

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


#: Fitted temperatures shipped next to the adapter (per-primitive and per-language, B-1).
TEMPERATURE_ASSET = "temperature_calibration.json"
#: Trained ``choice`` scorer shipped next to the adapter (L-002).
CHOICE_HEAD_ASSET = "choice_head.json"
#: Evidence-conditioned confidence: prototype bank + fitted ``noul`` map (P3).
CONFIDENCE_ASSET = "confidence_calibration.json"
#: Schema version of that asset; the runtime refuses a version it does not know.
CONFIDENCE_VERSION = 1

_log = logging.getLogger(__name__)


def _asset_absent_from_repo(exc: Exception) -> bool:
    """True only when the Hub said *this repo does not ship the asset* — silence is correct.

    Anything else (an unpopulated cache under ``TACHYONE_OFFLINE=1``, a network error, a 500)
    used to be swallowed too, which is the failure B-8 is about: the engine then runs without
    temperature scaling and without the ``choice`` head, quietly reporting confident numbers
    that do not reflect the calibrated model. ``LocalEntryNotFoundError`` is never "absent":
    it means *not cached*, and a partially populated cache looks exactly like that.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "LocalEntryNotFoundError" in names:
        return False
    if "RepositoryNotFoundError" in names:  # a mis-typed adapter id, not a missing file
        return False
    return "EntryNotFoundError" in names


def _warn_asset(asset: str, source: str, problem: str) -> None:
    """Log (stderr, WARNING) that an adapter asset is unusable, naming both file and repo."""
    _log.warning(
        "tachyone: cannot load %s from %s (%s); continuing without it", asset, source, problem
    )


def _adapter_asset(adapter: str, asset: str, *, models_dir: str, offline: bool) -> Path | None:
    """Locate ``asset`` in a local adapter dir or download it, warning when that is not routine."""
    local = Path(os.path.expanduser(adapter))
    if local.is_dir():
        candidate = local / asset
        if candidate.exists():
            return candidate
        if offline:
            _warn_asset(
                asset,
                str(candidate),
                "absent from the local adapter directory and TACHYONE_OFFLINE=1 "
                "forbids fetching it",
            )
        return None  # a local adapter that does not ship the asset is a normal, documented case
    try:
        from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]

        return Path(
            hf_hub_download(
                repo_id=adapter,
                filename=asset,
                cache_dir=models_dir,
                local_files_only=offline,
            )
        )
    except Exception as exc:  # classified by _asset_absent_from_repo, never fatal
        if not _asset_absent_from_repo(exc):
            _warn_asset(asset, adapter, f"{type(exc).__name__}: {exc}")
        return None


def _read_asset_json(path: Path, asset: str) -> dict[str, Any] | None:
    """Read a JSON object from ``path``, warning when the file exists but cannot be used."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        _warn_asset(asset, str(path), f"unreadable, {type(exc).__name__}: {exc}")
        return None
    if not isinstance(payload, dict):
        _warn_asset(asset, str(path), f"expected a JSON object, found {type(payload).__name__}")
        return None
    return payload


def load_temperatures(
    info: CheckpointInfo, *, models_dir: str, offline: bool = False
) -> dict[str, float]:
    """Load fitted temperatures from the adapter repo/dir, if present.

    Keys are ``kind`` for the per-primitive fit and ``kind:lang`` for the per-language fit
    (B-1); :meth:`EncoderModel._temp` prefers the latter. A legacy per-primitive-only file still
    loads.

    An adapter that does not ship the file is normal and stays silent; a file that exists but
    cannot be read, parsed or interpreted — or that cannot be fetched offline — logs a WARNING
    naming the asset and its source before the engine degrades to the uncalibrated baseline
    (B-8).
    """
    if not info.adapter:
        return {}
    source = _adapter_asset(info.adapter, TEMPERATURE_ASSET, models_dir=models_dir, offline=offline)
    if source is None:
        return {}
    report = _read_asset_json(source, TEMPERATURE_ASSET)
    if report is None:
        return {}
    try:
        return parse_temperature_report(report)
    except (TypeError, ValueError) as exc:
        _warn_asset(TEMPERATURE_ASSET, str(source), f"invalid temperature value, {exc}")
        return {}


def load_choice_head(
    info: CheckpointInfo, *, models_dir: str, offline: bool = False
) -> ChoiceHeadBank | None:
    """Load the ``choice`` head bank from the adapter repo/dir, if present.

    Accepts both ADR-0016 shapes: the keyed ``{shared, domains}`` bank and the legacy
    single scorer (which loads as a shared-only bank, so every published adapter keeps
    answering exactly as before). Same contract as :func:`load_temperatures`: absent asset
    is silent, unusable asset warns by name and degrades — a corrupt entry drops only
    itself, never the whole engine (B-8).
    """
    if not info.adapter:
        return None
    source = _adapter_asset(info.adapter, CHOICE_HEAD_ASSET, models_dir=models_dir, offline=offline)
    if source is None:
        return None
    payload = _read_asset_json(source, CHOICE_HEAD_ASSET)
    if payload is None:
        return None
    return ChoiceHeadBank.from_dict(
        payload, warn=lambda message: _warn_asset(CHOICE_HEAD_ASSET, str(source), message)
    )


def load_confidence(
    info: CheckpointInfo, *, models_dir: str, offline: bool = False
) -> ConfidenceCalibration | None:
    """Load the evidence-confidence asset (prototype bank + ``noul`` map) if present.

    Same contract as :func:`load_temperatures`: an adapter that does not ship the file is
    normal and stays silent; a file that exists but cannot be read, parsed or interpreted —
    or cannot be fetched offline — logs a WARNING naming the asset and its source before the
    engine falls back to the fitted temperature (B-8). The bank and the map are one file on
    purpose: a map fitted against a different bank would be measuring a different signal.
    """
    if not info.adapter:
        return None
    source = _adapter_asset(info.adapter, CONFIDENCE_ASSET, models_dir=models_dir, offline=offline)
    if source is None:
        return None
    payload = _read_asset_json(source, CONFIDENCE_ASSET)
    if payload is None:
        return None
    return ConfidenceCalibration.from_dict(
        payload, warn=lambda message: _warn_asset(CONFIDENCE_ASSET, str(source), message)
    )


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
    """Load a full checkpoint: encoder (+ adapter), temperatures, choice head, confidence."""
    encode = load_encoder(info, models_dir=models_dir, device=device, offline=offline, fast=fast)
    temperatures = load_temperatures(info, models_dir=models_dir, offline=offline)
    choice_bank = load_choice_head(info, models_dir=models_dir, offline=offline)
    confidence = load_confidence(info, models_dir=models_dir, offline=offline)
    return EncoderCheckpoint(
        info,
        encode,
        temperatures=temperatures,
        choice_bank=choice_bank,
        confidence=confidence,
    )


def _usage(state: State, questions: dict[str, Question]) -> Usage:
    return Usage(
        input_tokens=max(1, len(state_text(state)) // 4),
        output_tokens=max(1, len(questions)),
    )


class EncoderBackend:
    """Offline :class:`~tachyone.backends.base.Backend` backed by local encoder checkpoints."""

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
        choice_head: str | None = None,
    ) -> PredictionResult:
        if not questions:
            return PredictionResult(answers={}, usage=Usage(input_tokens=0, output_tokens=0))
        checkpoint_id = (
            model if model in self._router.checkpoints else self._router.route(state).checkpoint_id
        )
        checkpoint = cast(EncoderCheckpointLike, self._router.get(checkpoint_id))

        def runner(states: list[State], shared: dict[str, Question]) -> list[dict[str, Answer]]:
            return checkpoint.predict(states, shared, choice_head=choice_head)

        agent = Agent(runner, max_batch_size=self._max_batch_size, hooks=self._hooks)
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
    "CHOICE_HEAD_ASSET",
    "CONFIDENCE_ASSET",
    "CONFIDENCE_VERSION",
    "MODEL_IDS",
    "TEMPERATURE_ASSET",
    "ChoiceHeadBank",
    "ChoiceScorer",
    "ConfidenceCalibration",
    "DomainChoiceHead",
    "EncodeFn",
    "EncoderBackend",
    "EncoderCheckpoint",
    "EncoderCheckpointLike",
    "EncoderModel",
    "apply_adapters",
    "load_checkpoint",
    "load_choice_head",
    "load_confidence",
    "load_encoder",
    "load_temperatures",
    "prototype_strength",
]
