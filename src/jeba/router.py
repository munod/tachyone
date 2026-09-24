"""Script/language routing: pick the English or multilingual checkpoint cheaply.

Detection is pure Python and model-free, so a routing decision costs microseconds (ROUTE-03)
and works offline. The rule is deliberately simple: non-Latin scripts go to the multilingual
checkpoint; Latin text goes to English unless a caller hint says otherwise. A caller-provided
``lang_guess`` (callable or code) overrides the heuristic (ROUTE-01, ROUTE-02).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from jeba.primitives import State

#: Built-in checkpoint ids.
ENGLISH = "jeba-en"
MULTILINGUAL = "jeba-multi"

#: ISO 639-1 codes served by the multilingual checkpoint (100+ languages).
SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "af",
    "am",
    "ar",
    "as",
    "az",
    "be",
    "bg",
    "bn",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fil",
    "fr",
    "ga",
    "gl",
    "gu",
    "ha",
    "he",
    "hi",
    "hr",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "lo",
    "lt",
    "lv",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "my",
    "ne",
    "nl",
    "no",
    "or",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "rw",
    "si",
    "sk",
    "sl",
    "so",
    "sq",
    "sr",
    "su",
    "sv",
    "sw",
    "ta",
    "te",
    "th",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "xh",
    "yi",
    "yo",
    "zh",
    "zu",
    "ceb",
    "co",
    "eo",
    "fy",
    "haw",
    "hmn",
    "ig",
    "ilo",
    "la",
    "lb",
    "mg",
    "mi",
    "ny",
    "sm",
    "sn",
    "st",
    "tg",
    "tk",
    "tt",
    "ug",
    "war",
)

#: Unicode block ranges per script, checked in order.
_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("han", ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF))),
    ("kana", ((0x3040, 0x30FF),)),
    ("hangul", ((0xAC00, 0xD7AF), (0x1100, 0x11FF))),
    ("cyrillic", ((0x0400, 0x04FF), (0x0500, 0x052F))),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFE70, 0xFEFF))),
    ("hebrew", ((0x0590, 0x05FF),)),
    ("devanagari", ((0x0900, 0x097F),)),
    ("bengali", ((0x0980, 0x09FF),)),
    ("thai", ((0x0E00, 0x0E7F),)),
    ("greek", ((0x0370, 0x03FF),)),
    ("latin", ((0x0041, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF))),
)

_ENGLISH_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "is",
        "are",
        "you",
        "your",
        "to",
        "of",
        "in",
        "it",
        "that",
        "this",
        "for",
        "we",
        "please",
        "not",
        "can",
        "with",
        "was",
        "have",
        "has",
        "as",
        "at",
        "on",
        "or",
    }
)
_NON_ENGLISH_STOPWORDS = frozenset(
    {
        "el",
        "la",
        "los",
        "las",
        "una",
        "que",
        "der",
        "die",
        "das",
        "und",
        "ist",
        "le",
        "les",
        "des",
        "une",
        "et",
        "não",
        "obrigado",
        "hola",
        "gracias",
        "bitte",
        "danke",
        "merci",
    }
)


class CheckpointInfo(BaseModel):
    """Static metadata for a loadable checkpoint."""

    model_config = ConfigDict(extra="ignore")

    id: str
    languages: list[str]
    context: int
    size_params: int


class RouteDecision(BaseModel):
    """The chosen checkpoint and why it was chosen."""

    model_config = ConfigDict(extra="ignore")

    checkpoint_id: str
    detected_script: str
    detected_language: str | None
    reason: str


#: Default checkpoints: English (ModernBERT-large class) and multilingual (mmBERT-base class).
DEFAULT_CHECKPOINTS: dict[str, CheckpointInfo] = {
    ENGLISH: CheckpointInfo(id=ENGLISH, languages=["en"], context=512, size_params=395_000_000),
    MULTILINGUAL: CheckpointInfo(
        id=MULTILINGUAL, languages=["*"], context=1024, size_params=315_000_000
    ),
}

LangGuess = Callable[[State], str | None]


def state_text(state: State) -> str:
    """Flatten any accepted state shape into text, using string leaves (not JSON keys)."""
    if isinstance(state, str):
        return state
    parts: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(state)
    return " ".join(parts)


def detect_script(text: str) -> str:
    """Return the dominant script of ``text`` (``"unknown"`` if no letters found)."""
    counts: dict[str, int] = {}
    for char in text:
        codepoint = ord(char)
        for name, ranges in _SCRIPT_RANGES:
            if any(low <= codepoint <= high for low, high in ranges):
                counts[name] = counts.get(name, 0) + 1
                break
    if not counts:
        return "unknown"
    return max(counts, key=lambda name: counts[name])


def detect_language(text: str, script: str) -> str | None:
    """Best-effort language code for Latin text; ``None`` when the script is not Latin."""
    if script != "latin":
        return None
    words = {word.strip(".,!?;:\"'()[]").lower() for word in text.split()}
    if words & _NON_ENGLISH_STOPWORDS:
        return "und"  # Latin script, likely non-English
    if words & _ENGLISH_STOPWORDS:
        return "en"
    return None


@dataclass
class Router:
    """Chooses a checkpoint id for a state without touching a model."""

    default: str = ENGLISH
    lang_guess: LangGuess | None = None
    checkpoints: dict[str, CheckpointInfo] = field(
        default_factory=lambda: dict(DEFAULT_CHECKPOINTS)
    )

    def route(self, state: State) -> RouteDecision:
        """Decide which checkpoint should answer ``state``."""
        if self.lang_guess is not None:
            guess = self.lang_guess(state)
            if guess:
                return self._from_language(guess, script="unknown", reason="lang_guess hint")
        text = state_text(state)
        script = detect_script(text)
        language = detect_language(text, script)
        if script == "latin" and language != "und":
            return RouteDecision(
                checkpoint_id=ENGLISH,
                detected_script=script,
                detected_language=language or "en",
                reason="Latin script, no non-English signal",
            )
        if script == "latin" and language == "und":
            return RouteDecision(
                checkpoint_id=MULTILINGUAL,
                detected_script=script,
                detected_language=None,
                reason="Latin script with non-English signal",
            )
        if script == "unknown":
            return RouteDecision(
                checkpoint_id=self.default,
                detected_script=script,
                detected_language=None,
                reason="no letters detected; using default checkpoint",
            )
        return RouteDecision(
            checkpoint_id=MULTILINGUAL,
            detected_script=script,
            detected_language=None,
            reason=f"non-Latin script ({script}) not readable by the English checkpoint",
        )

    def _from_language(self, code: str, *, script: str, reason: str) -> RouteDecision:
        primary = code.replace("_", "-").split("-")[0].lower()
        checkpoint_id = ENGLISH if primary in {"en", "eng", "english"} else MULTILINGUAL
        return RouteDecision(
            checkpoint_id=checkpoint_id,
            detected_script=script,
            detected_language=primary,
            reason=reason,
        )


__all__ = [
    "DEFAULT_CHECKPOINTS",
    "ENGLISH",
    "MULTILINGUAL",
    "SUPPORTED_LANGUAGES",
    "CheckpointInfo",
    "LangGuess",
    "RouteDecision",
    "Router",
    "detect_language",
    "detect_script",
    "state_text",
]
