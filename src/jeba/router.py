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

from jeba.hooks import ON_EVICT, ON_LOAD, ON_ROUTE, Hooks
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
#: Function words per non-English Latin language, used as a cheap language signal. Words that
#: also occur in English (e.g. "die") are deliberately omitted to avoid false non-English hits.
_LANGUAGE_STOPWORDS: dict[str, frozenset[str]] = {
    "pt": frozenset(
        {
            "quero",
            "queria",
            "preciso",
            "gostaria",
            "minha",
            "meu",
            "meus",
            "minhas",
            "para",
            "com",
            "uma",
            "um",
            "não",
            "nao",
            "sim",
            "obrigado",
            "obrigada",
            "favor",
            "por",
            "senha",
            "pagamento",
            "cobrança",
            "cobranca",
            "cancelar",
            "cancelamento",
            "ajuda",
            "ajudar",
            "agora",
            "hoje",
            "amanhã",
            "amanha",
            "fazer",
            "está",
            "esta",
            "estão",
            "muito",
            "bom",
            "dia",
            "tudo",
            "bem",
            "aqui",
            "isso",
            "esse",
            "essa",
            "você",
            "voce",
            "vocês",
            "assinatura",
            "conta",
            "problema",
            "erro",
            "falha",
            "falhou",
            "duplicado",
            "duplicada",
            "reembolso",
            "devolver",
            "devolva",
            "pedido",
            "empresa",
            "tempo",
            "quando",
            "como",
            "onde",
            "porque",
            "também",
            "tambem",
            "mais",
            "sobre",
            "depois",
            "antes",
            "sempre",
            "nunca",
            "ainda",
            "só",
            "pode",
            "consegue",
        }
    ),
    "es": frozenset(
        {
            "hola",
            "gracias",
            "por",
            "favor",
            "el",
            "la",
            "los",
            "las",
            "una",
            "que",
            "para",
            "con",
            "mi",
            "su",
            "está",
            "esta",
            "problema",
            "pago",
            "factura",
            "reembolso",
            "cancelar",
            "ayuda",
            "necesito",
            "quiero",
            "tengo",
            "cómo",
            "como",
            "dónde",
            "donde",
            "cuándo",
            "porque",
            "muy",
            "bien",
            "ahora",
            "hoy",
            "mañana",
            "devolver",
            "duplicado",
        }
    ),
    "fr": frozenset(
        {
            "bonjour",
            "merci",
            "le",
            "les",
            "des",
            "une",
            "un",
            "et",
            "pour",
            "avec",
            "mon",
            "ma",
            "mes",
            "est",
            "sont",
            "pas",
            "plus",
            "aide",
            "besoin",
            "remboursement",
            "annuler",
            "facture",
            "paiement",
            "problème",
            "maintenant",
            "comment",
            "pourquoi",
            "très",
            "bien",
            "vous",
            "aujourd'hui",
        }
    ),
    "de": frozenset(
        {
            "bitte",
            "danke",
            "der",
            "das",
            "und",
            "ist",
            "sind",
            "nicht",
            "mein",
            "meine",
            "mit",
            "für",
            "hilfe",
            "brauche",
            "rechnung",
            "zahlung",
            "rückerstattung",
            "stornieren",
            "problem",
            "jetzt",
            "heute",
            "warum",
            "sehr",
            "gut",
            "sie",
            "ich",
        }
    ),
    "it": frozenset(
        {
            "ciao",
            "grazie",
            "favore",
            "il",
            "lo",
            "gli",
            "una",
            "non",
            "mio",
            "mia",
            "aiuto",
            "bisogno",
            "fattura",
            "pagamento",
            "rimborso",
            "annullare",
            "problema",
            "adesso",
            "oggi",
            "perché",
            "molto",
            "bene",
            "sono",
        }
    ),
    "nl": frozenset(
        {
            "hallo",
            "bedankt",
            "alstublieft",
            "het",
            "een",
            "niet",
            "mijn",
            "met",
            "voor",
            "hulp",
            "nodig",
            "factuur",
            "betaling",
            "terugbetaling",
            "annuleren",
            "probleem",
            "vandaag",
            "waarom",
            "heel",
            "goed",
        }
    ),
}


class CheckpointInfo(BaseModel):
    """Static metadata for a loadable checkpoint."""

    model_config = ConfigDict(extra="ignore")

    id: str
    languages: list[str]
    context: int
    size_params: int
    base_model: str | None = None
    adapter: str | None = None


class RouteDecision(BaseModel):
    """The chosen checkpoint and why it was chosen."""

    model_config = ConfigDict(extra="ignore")

    checkpoint_id: str
    detected_script: str
    detected_language: str | None
    reason: str


#: Default checkpoints: English (ModernBERT-large) and multilingual (mmBERT-base), each with the
#: published jeba LoRA adapter loaded on top (override with ``JEBA_ADAPTERS``).
DEFAULT_CHECKPOINTS: dict[str, CheckpointInfo] = {
    ENGLISH: CheckpointInfo(
        id=ENGLISH,
        languages=["en"],
        context=512,
        size_params=395_000_000,
        base_model="answerdotai/ModernBERT-large",
        adapter="munod/jeba-en",
    ),
    MULTILINGUAL: CheckpointInfo(
        id=MULTILINGUAL,
        languages=["*"],
        context=1024,
        size_params=315_000_000,
        base_model="jhu-clsp/mmBERT-base",
        adapter="munod/jeba-multi",
    ),
}

LangGuess = Callable[[State], str | None]

#: Builds a loadable checkpoint object from its metadata.
Loader = Callable[[CheckpointInfo], object]


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
    """Best-effort language code for Latin text; ``None`` when the script is not Latin.

    Returns a specific code when function-word hits identify a language, ``"und"`` for
    accented Latin without a match, ``"en"`` for English signal, else ``None``.
    """
    if script != "latin":
        return None
    words = {word.strip(".,!?;:\"'()[]").lower() for word in text.split()}
    english_hits = len(words & _ENGLISH_STOPWORDS)
    best_code: str | None = None
    best_hits = 0
    for code, stopwords in _LANGUAGE_STOPWORDS.items():
        hits = len(words & stopwords)
        if hits > best_hits:
            best_code, best_hits = code, hits
    if best_code is not None and best_hits >= english_hits:
        return best_code
    if english_hits > 0:
        return "en"
    return "und" if any(0x00C0 <= ord(char) <= 0x024F for char in text) else None


@dataclass
class Router:
    """Chooses a checkpoint id for a state and manages checkpoint lifecycle.

    Routing is model-free; loading is delegated to an injected ``loader``. With
    ``max_loaded`` set, the least-recently-used checkpoint is evicted before a new one loads
    (``0`` means unlimited).
    """

    default: str = ENGLISH
    lang_guess: LangGuess | None = None
    checkpoints: dict[str, CheckpointInfo] = field(
        default_factory=lambda: dict(DEFAULT_CHECKPOINTS)
    )
    loader: Loader | None = None
    max_loaded: int = 2
    hooks: Hooks | None = None
    _loaded: dict[str, object] = field(default_factory=dict, repr=False)
    _order: list[str] = field(default_factory=list, repr=False)

    @property
    def loaded(self) -> tuple[str, ...]:
        """Ids of currently loaded checkpoints, least- to most-recently used."""
        return tuple(self._order)

    def preload(self, ids: list[str] | tuple[str, ...] | None = None) -> None:
        """Load the given checkpoints (all registered ones when ``ids`` is omitted)."""
        for checkpoint_id in ids if ids is not None else list(self.checkpoints):
            self.get(checkpoint_id)

    def get(self, checkpoint_id: str) -> object:
        """Return a loaded checkpoint, loading and (if needed) evicting on demand."""
        if checkpoint_id in self._loaded:
            self._touch(checkpoint_id)
            return self._loaded[checkpoint_id]
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"unknown checkpoint: {checkpoint_id!r}")
        if self.loader is None:
            raise RuntimeError(f"no loader configured to load {checkpoint_id!r}")
        if self.max_loaded > 0:
            while len(self._loaded) >= self.max_loaded:
                self._evict_lru(exclude=checkpoint_id)
        loaded = self.loader(self.checkpoints[checkpoint_id])
        self._loaded[checkpoint_id] = loaded
        self._touch(checkpoint_id)
        if self.hooks is not None:
            self.hooks.emit(ON_LOAD, checkpoint_id=checkpoint_id)
        return loaded

    def attach(self, checkpoint_id: str, checkpoint: object) -> None:
        """Register an externally built checkpoint without invoking the loader."""
        if checkpoint_id not in self._loaded and self.max_loaded > 0:
            while len(self._loaded) >= self.max_loaded:
                self._evict_lru(exclude=checkpoint_id)
        self._loaded[checkpoint_id] = checkpoint
        self._touch(checkpoint_id)

    def unload(self, checkpoint_id: str | None = None) -> None:
        """Drop one checkpoint, or all of them when ``checkpoint_id`` is omitted."""
        if checkpoint_id is None:
            self._loaded.clear()
            self._order.clear()
            return
        self._loaded.pop(checkpoint_id, None)
        if checkpoint_id in self._order:
            self._order.remove(checkpoint_id)

    def _touch(self, checkpoint_id: str) -> None:
        if checkpoint_id in self._order:
            self._order.remove(checkpoint_id)
        self._order.append(checkpoint_id)

    def _evict_lru(self, *, exclude: str) -> None:
        for candidate in self._order:
            if candidate != exclude and candidate in self._loaded:
                self._loaded.pop(candidate, None)
                self._order.remove(candidate)
                if self.hooks is not None:
                    self.hooks.emit(ON_EVICT, checkpoint_id=candidate)
                return

    def route(self, state: State) -> RouteDecision:
        """Decide which checkpoint should answer ``state`` and emit ``on_route``."""
        decision = self._decide(state)
        if self.hooks is not None:
            self.hooks.emit(
                ON_ROUTE, state=state, route=decision, checkpoint_id=decision.checkpoint_id
            )
        return decision

    def _decide(self, state: State) -> RouteDecision:
        """Decide which checkpoint should answer ``state``."""
        if self.lang_guess is not None:
            guess = self.lang_guess(state)
            if guess:
                return self._from_language(guess, script="unknown", reason="lang_guess hint")
        text = state_text(state)
        script = detect_script(text)
        language = detect_language(text, script)
        if script == "latin":
            if language in (None, "en"):
                return RouteDecision(
                    checkpoint_id=ENGLISH,
                    detected_script=script,
                    detected_language="en",
                    reason="Latin script, English or no non-English signal",
                )
            return RouteDecision(
                checkpoint_id=MULTILINGUAL,
                detected_script=script,
                detected_language=language,
                reason=f"Latin script detected as {language}",
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
    "Loader",
    "RouteDecision",
    "Router",
    "detect_language",
    "detect_script",
    "state_text",
]
