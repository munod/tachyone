"""The S2 cross-encoder as an **opt-in** scoring override (B-21 W2 / S2-v V6).

One batched forward over ``[state ⊕ question ⊕ option]`` answers ``choice`` and ``noul``
questions: one logit per criterion, softmaxed over the question's own criteria. ``score``
questions keep the single-pass path untouched, and the fitted P3 knobs (temperatures,
knots, cells) are **bypassed on purpose** for the overridden questions — the teacher's
softmax *is* the calibration (the measured bench read). With
:data:`CROSS_ENCODER_ENV` unset the legacy path is byte-identical (additive extension).

    TACHYONE_CROSS_ENCODER=checkpoints/en_s2_teacher uv run tachyone --serve

The teacher is a **training-time instrument** that earned a serving variant only behind the
pre-registered latency gate (BACKLOG B-21 S2): raw p50 ≤ 150 ms at the measured arities.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

#: Opt-in knob: a trained teacher directory (``teacher_config.json`` + adapter + heads).
CROSS_ENCODER_ENV = "TACHYONE_CROSS_ENCODER"

_TRAIN_HINT = "the train extra is required for the cross-encoder: uv sync --extra train"


@runtime_checkable
class CrossScorerLike(Protocol):
    """Per-option logits for one question — the seam ``EncoderModel`` consumes."""

    def score(
        self, state_text: str, question_text: str, criterion_texts: Sequence[str]
    ) -> list[float]:
        """One logit per criterion text, in order (softmaxed by the caller)."""
        ...

    def temperature(self, kind: str, lang: str | None = None) -> float:
        """The fitted softmax temperature for ``kind`` (1.0 without a calibration)."""
        ...


def _load_calibration(directory: str | Path) -> dict[str, float]:
    """Flatten a ``training.fit_calibration`` report into ``{kind[, :lang] -> T}``.

    The W5 refit for the variant (BACKLOG B-21 S2-v3-f): probability-only, so the argmax
    is invariant by construction — the same contract as the published stack's fit.
    """
    path = Path(directory) / "temperature_calibration.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    flat: dict[str, float] = {}
    for kind, entry in (report.get("per_primitive") or {}).items():
        flat[str(kind)] = float(entry.get("temperature", 1.0))
        for lang, sub in (entry.get("by_language") or {}).items():
            flat[f"{kind}:{lang}"] = float(sub.get("temperature", 1.0))
    return flat


class CrossEncoderScorer:
    """The trained S2 teacher: joint ``[state ⊕ question ⊕ option]`` scorer (W2)..

    Loads the base trunk named by the teacher's own recipe, the LoRA adapter saved beside
    it and the scalar ``score_head.pt``. Torch/transformers/peft import lazily; without the
    ``train`` extra this raises a :class:`RuntimeError` naming the extra.
    """

    def __init__(
        self,
        teacher_dir: str | Path,
        *,
        models_dir: str | None = None,
        device: str = "auto",
        max_len: int = 512,
        temperatures: dict[str, float] | None = None,
    ) -> None:
        try:
            import torch  # pyright: ignore[reportMissingImports]
            from peft import PeftModel  # pyright: ignore[reportMissingImports]
            from transformers import (  # pyright: ignore[reportMissingImports]
                AutoModel,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(_TRAIN_HINT) from exc

        directory = Path(teacher_dir)
        recipe = json.loads((directory / "teacher_config.json").read_text(encoding="utf-8"))
        model_id = str(recipe["config"]["model_id"])
        cache: dict[str, Any] = {"cache_dir": models_dir} if models_dir else {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **cache)
        encoder = AutoModel.from_pretrained(model_id, **cache)
        model = PeftModel.from_pretrained(encoder, directory, is_trainable=False)
        resolved = device
        if device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(resolved)
        model.eval()
        head = torch.nn.Linear(int(encoder.config.hidden_size), 1)
        head.load_state_dict(torch.load(directory / "score_head.pt", map_location=resolved))
        head = head.to(resolved)
        head.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._head = head
        self._device = resolved
        self._max_len = max_len
        self._temperatures = dict(temperatures or {})

    def temperature(self, kind: str, lang: str | None = None) -> float:
        if lang and f"{kind}:{lang}" in self._temperatures:
            return self._temperatures[f"{kind}:{lang}"]
        return self._temperatures.get(kind, 1.0)

    def score(
        self, state_text: str, question_text: str, criterion_texts: Sequence[str]
    ) -> list[float]:
        torch = self._torch
        rights = [f"{question_text} {criterion}" for criterion in criterion_texts]
        batch = self._tokenizer(
            [state_text] * len(rights),
            rights,
            truncation=True,
            max_length=self._max_len,
            padding=True,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            hidden = self._model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            logits = self._head(pooled).squeeze(-1)
        return [float(value) for value in logits]


class CrossEncoderEnsemble:
    """Two or more teachers answering as one: the **mean of the per-option logits**.

    The S2-v2 strengthening (BACKLOG B-21): one teacher per GPU, and the average is itself
    a calibration lever. The cost is N batched forwards per question — the ≤ 150 ms gate
    still applies to the mean-latency read.
    """

    def __init__(
        self, scorers: Sequence[CrossScorerLike], *, temperatures: dict[str, float] | None = None
    ) -> None:
        if not scorers:
            raise ValueError("the ensemble needs at least one scorer")
        self._scorers = list(scorers)
        self._temperatures = dict(temperatures or {})

    def temperature(self, kind: str, lang: str | None = None) -> float:
        if lang and f"{kind}:{lang}" in self._temperatures:
            return self._temperatures[f"{kind}:{lang}"]
        return self._temperatures.get(kind, 1.0)

    def score(
        self, state_text: str, question_text: str, criterion_texts: Sequence[str]
    ) -> list[float]:
        votes = [
            scorer.score(state_text, question_text, criterion_texts) for scorer in self._scorers
        ]
        width = len(votes[0])
        return [sum(row[position] for row in votes) / len(votes) for position in range(width)]


def maybe_cross_encoder(
    *,
    checkpoint_id: str | None = None,
    models_dir: str | None = None,
    device: str = "auto",
) -> CrossScorerLike | None:
    """The opt-in scorer from :data:`CROSS_ENCODER_ENV`, or ``None`` when unset/unmatched.

    Two spec shapes (mirroring ``TACHYONE_ADAPTERS``): a bare directory applies to every
    checkpoint; ``checkpoint_id=dir[,checkpoint_id=dir...]`` scopes the override to the
    named checkpoints — the variant is an **en-arm** change and must never leak into the
    untouched multi arm (S2-v V4). A directory value may join several teachers with ``+``
    (the S2-v2 ensemble). The legacy path depends on this returning ``None``: nothing
    loads, nothing changes.
    """
    spec = os.environ.get(CROSS_ENCODER_ENV, "").strip()
    if not spec:
        return None
    if "=" not in spec:
        teacher_spec = spec
    else:
        teacher_spec = ""
        for part in spec.split(","):
            key, _, value = part.partition("=")
            if key.strip() == (checkpoint_id or ""):
                teacher_spec = value.strip()
                break
        if not teacher_spec:
            return None
    directories = [part.strip() for part in teacher_spec.split("+") if part.strip()]
    temperatures = _load_calibration(directories[0])
    scorers: list[CrossScorerLike] = [
        CrossEncoderScorer(
            directory, models_dir=models_dir, device=device, temperatures=temperatures
        )
        for directory in directories
    ]
    return (
        scorers[0]
        if len(scorers) == 1
        else CrossEncoderEnsemble(scorers, temperatures=temperatures)
    )
