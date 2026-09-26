"""LLM backend: answer primitives via structured outputs of an OpenAI-compatible endpoint.

Provider-agnostic by design (OD-1): any server exposing ``POST {base_url}/chat/completions``
works (OpenAI, Groq, Together, Ollama, vLLM, llama.cpp). The transport is injectable, so
tests run fully offline and no hosted dependency leaks into the core. HTTP uses the standard
library only; the base install needs no extra dependency (ADR-0004).
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from tachyone.backends.base import PredictionResult
from tachyone.config import Config
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
from tachyone.wire import BackendError, Overloaded, RateLimited, Usage

#: Sends an OpenAI chat-completions payload and returns the assistant message content.
type Transport = Callable[[dict[str, Any]], Awaitable[str]]

_SYSTEM_PROMPT = (
    "You answer typed decision questions about a STATE. "
    "Reply with a single JSON object and nothing else. "
    'The shape is {"answers": {"<question id>": {"noul": <number 0..1>} } } for noul, '
    '{"choice": "<option>", "probabilities": {"<option>": <number 0..1>}} for choice, and '
    '{"score": <number>, "probabilities": {"<index>": <number 0..1>}} for score. '
    "For noul return only noul. For choice include a probability for every option. "
    "For score, score is the probability-weighted position on the levels and may fall "
    "between levels; include a probability for every level index. "
    "Probabilities must be non-negative and sum to 1."
)


def _question_spec(question: Question) -> dict[str, Any]:
    if isinstance(question, NoulQuestion):
        return {
            "type": "noul",
            "instructions": question.instructions,
            "criteria": question.criteria.model_dump(mode="json") if question.criteria else None,
        }
    if isinstance(question, ChoiceQuestion):
        return {
            "type": "choice",
            "instructions": question.instructions,
            "criteria": question.criteria,
        }
    if isinstance(question, ScoreQuestion):
        return {
            "type": "score",
            "instructions": question.instructions,
            "criteria": question.criteria,
        }
    raise TypeError(f"unsupported question type: {type(question)!r}")


def build_messages(state: State, questions: dict[str, Question]) -> list[dict[str, str]]:
    """Assemble the chat messages: a fixed system prompt plus the state and questions."""
    spec = {qid: _question_spec(q) for qid, q in questions.items()}
    user = json.dumps({"state": state, "questions": spec}, ensure_ascii=False)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from ``text``, tolerating surrounding prose."""
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start == -1:
            raise ValueError("no JSON object found in model output") from None
        try:
            parsed = json.JSONDecoder().raw_decode(stripped[start:])[0]
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in model output: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("model output JSON root is not an object")
    return parsed


def _unit(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number in [0, 1], got {value!r}")
    return min(1.0, max(0.0, float(value)))


def _probabilities[K](raw: Any, keys: list[K]) -> dict[K, float]:
    values: dict[K, float] = {}
    for key in keys:
        candidate = raw.get(str(key)) if isinstance(raw, dict) else None
        if candidate is None and isinstance(raw, dict):
            candidate = raw.get(key)
        values[key] = _unit(candidate) if candidate is not None else 0.0
    total = sum(values.values())
    if total <= 0.0:
        uniform = 1.0 / len(values) if values else 0.0
        return dict.fromkeys(values, uniform)
    return {key: value / total for key, value in values.items()}


def _answer_from_raw(question: Question, raw: Any) -> Answer:
    if not isinstance(raw, dict):
        raise ValueError("answer entry must be an object")
    if isinstance(question, NoulQuestion):
        return NoulAnswer(noul=_unit(raw.get("noul")))
    if isinstance(question, ChoiceQuestion):
        probabilities = _probabilities(raw.get("probabilities"), list(question.criteria))
        choice = raw.get("choice")
        if not isinstance(choice, str) or choice not in probabilities:
            choice = max(probabilities, key=lambda key: probabilities[key])
        return ChoiceAnswer(
            choice=choice,
            probabilities=probabilities,
            confidence=max(probabilities.values()),
        )
    probabilities = _probabilities(raw.get("probabilities"), list(range(len(question.criteria))))
    expected = sum(index * probability for index, probability in probabilities.items())
    score = raw.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        score = expected
    score = min(float(len(question.criteria) - 1), max(0.0, float(score)))
    return ScoreAnswer(
        score=score,
        legend=dict(enumerate(question.criteria)),
        probabilities=probabilities,
        confidence=max(probabilities.values()),
    )


def build_answers(questions: dict[str, Question], payload: dict[str, Any]) -> dict[str, Answer]:
    """Map a model JSON payload to validated answers, one per question id."""
    raw_answers = payload.get("answers")
    if not isinstance(raw_answers, dict):
        raise ValueError("model output is missing an 'answers' object")
    answers: dict[str, Answer] = {}
    for question_id, question in questions.items():
        if question_id not in raw_answers:
            raise ValueError(f"model output is missing an answer for {question_id!r}")
        answers[question_id] = _answer_from_raw(question, raw_answers[question_id])
    return answers


def estimate_usage(messages: list[dict[str, str]], content: str) -> Usage:
    """Rough token estimate when the provider does not report usage."""
    input_chars = sum(len(message.get("content", "")) for message in messages)
    return Usage(
        input_tokens=max(1, input_chars // 4),
        output_tokens=max(1, len(content) // 4),
    )


def _post_chat_completions(
    url: str, api_key: str | None, payload: dict[str, Any], timeout: float
) -> str:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RateLimited("LLM provider rate limited the request") from exc
        if exc.code in (529, 503):
            raise Overloaded(f"LLM provider unavailable ({exc.code})") from exc
        raise BackendError(f"LLM provider returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise BackendError(f"LLM provider connection error: {exc.reason}") from exc
    try:
        parsed = json.loads(body)
        return str(parsed["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise BackendError("LLM provider returned an unexpected response body") from exc


def openai_transport(*, base_url: str, api_key: str | None, timeout: float) -> Transport:
    """Default transport: an OpenAI-compatible chat-completions call over stdlib HTTP."""
    url = f"{base_url.rstrip('/')}/chat/completions" if base_url else ""

    async def transport(payload: dict[str, Any]) -> str:
        if not url:
            raise BackendError("TACHYONE_LLM_BASE_URL is not configured")
        return await asyncio.to_thread(_post_chat_completions, url, api_key, payload, timeout)

    return transport


class LLMBackend:
    """Answers primitives by prompting an OpenAI-compatible model for structured JSON."""

    name = "llm"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        transport: Transport | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._max_retries = max(0, max_retries)
        self._transport: Transport = transport or openai_transport(
            base_url=base_url, api_key=api_key, timeout=timeout
        )

    @classmethod
    def from_config(cls, config: Config, *, transport: Transport | None = None) -> LLMBackend:
        return cls(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
            transport=transport,
            timeout=config.llm_timeout,
            max_retries=config.llm_retries,
        )

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

        messages = build_messages(state, questions)
        payload = {
            "model": self._model or model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }

        last_error: Exception | None = None
        for _ in range(self._max_retries + 1):
            content = await self._transport(payload)
            try:
                parsed = extract_json_object(content)
                answers = build_answers(questions, parsed)
            except (ValueError, ValidationError) as exc:
                last_error = exc
                continue
            return PredictionResult(answers=answers, usage=estimate_usage(messages, content))

        raise BackendError(
            f"LLM backend could not produce valid answers after "
            f"{self._max_retries + 1} attempt(s): {last_error}"
        )


__all__ = [
    "LLMBackend",
    "Transport",
    "build_answers",
    "build_messages",
    "estimate_usage",
    "extract_json_object",
    "openai_transport",
]
