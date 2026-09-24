"""Deterministic synthetic supervision for the three primitives.

Same seed + config produces byte-identical JSONL. Records stream to disk one line at a time,
so generation never holds the dataset in memory (TRAIN-01, TRAIN-07). The output format is
documented in ``docs/training.md``; it is training data, not the wire contract.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Default language set; states are authored here, other tags fall back to English text.
DEFAULT_LANGUAGES: tuple[str, ...] = ("en", "pt", "es", "fr", "de")

_ENTITIES: tuple[str, ...] = (
    "refund",
    "invoice",
    "password reset",
    "outage",
    "upgrade",
    "duplicate charge",
    "login code",
    "cancellation",
)

#: Per-language phrasing loaded from committed data. ``{entity}`` is substituted.
_PHRASES: dict[str, dict[str, list[str]]] = json.loads(
    (Path(__file__).parent / "data" / "phrases.json").read_text(encoding="utf-8")
)

_AMBIGUOUS = "maybe"
_TEAMS: tuple[str, ...] = ("billing", "technical", "sales", "other")
_TEAM_CUES: dict[str, tuple[str, ...]] = {
    "billing": ("invoice", "refund", "duplicate charge", "payment"),
    "technical": ("outage", "password reset", "login code", "bug"),
    "sales": ("upgrade", "pricing", "plan"),
}
_LEVELS: tuple[str, ...] = ("none", "low", "medium", "high")
_LEVEL_BY_TONE: dict[str, int] = {"calm": 0, "neutral": 1, "request": 2, "urgent": 3}


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Inputs to a data-generation run."""

    seed: int = 42
    per_type: int = 200
    languages: tuple[str, ...] = DEFAULT_LANGUAGES
    source: str = "synthetic"


def _phrases(lang: str) -> dict[str, list[str]]:
    return _PHRASES.get(lang, _PHRASES["en"])


def _boundary_state(index: int, base: str) -> str:
    if index % 19 == 0:
        return ""  # empty input
    if index % 23 == 0:
        return base + " " + (" ".join([base] * 40))  # very long input
    return base


def _noul_record(index: int, lang: str, rng: random.Random) -> dict[str, Any]:
    entity = rng.choice(_ENTITIES)
    positive = (index % 2 == 0) if index else True
    tone = rng.choice(("request", "neutral"))
    state = _boundary_state(index, rng.choice(_phrases(lang)[tone]).format(entity=entity))
    if positive and tone == "neutral":
        positive = False
    target = 1 if positive else 0
    instructions = f"Does this message request a {entity}?"
    return {
        "id": f"noul-{index:06d}",
        "type": "noul",
        "state": state,
        "instructions": instructions,
        "criteria": {"true": f"asks for a {entity}", "false": "does not ask"},
        "target": target,
        "lang": lang,
    }


def _choice_record(index: int, lang: str, rng: random.Random) -> dict[str, Any]:
    cue = rng.choice(list(_TEAM_CUES))
    entity = rng.choice(_TEAM_CUES[cue])
    state = _boundary_state(index, rng.choice(_phrases(lang)["request"]).format(entity=entity))
    target = "other" if index % 11 == 0 else cue  # inject boundary: unmapped option
    return {
        "id": f"choice-{index:06d}",
        "type": "choice",
        "state": state,
        "instructions": "Which team should handle this request?",
        "criteria": dict.fromkeys(_TEAMS),
        "target": target,
        "lang": lang,
    }


def _score_record(index: int, lang: str, rng: random.Random) -> dict[str, Any]:
    entity = rng.choice(_ENTITIES)
    tone = rng.choice(("calm", "neutral", "request", "urgent"))
    state = _boundary_state(index, rng.choice(_phrases(lang)[tone]).format(entity=entity))
    target = _LEVEL_BY_TONE[tone]
    if index % 13 == 0:  # ambiguous near-tie label
        target = max(0, target - 1)
    return {
        "id": f"score-{index:06d}",
        "type": "score",
        "state": state,
        "instructions": "How urgent is this request?",
        "criteria": list(_LEVELS),
        "target": target,
        "lang": lang,
    }


_GENERATORS = {
    "noul": _noul_record,
    "choice": _choice_record,
    "score": _score_record,
}


def iter_records(config: DataConfig) -> Iterator[dict[str, Any]]:
    """Yield deterministic records: ``per_type`` of each primitive, languages interleaved."""
    rng = random.Random(config.seed)
    languages = config.languages or DEFAULT_LANGUAGES
    for kind in ("noul", "choice", "score"):
        generator = _GENERATORS[kind]
        for index in range(config.per_type):
            language = languages[index % len(languages)]
            record = generator(index, language, rng)
            record["source"] = config.source
            yield record


def _line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"


def generate(config: DataConfig, out_path: str | Path) -> int:
    """Stream records to ``out_path`` as JSONL and return the number written."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in iter_records(config):
            handle.write(_line(record))
            handle.flush()
            written += 1
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jeba-generate-data", description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-type", type=int, default=200, help="records per primitive")
    parser.add_argument(
        "--languages", default=",".join(DEFAULT_LANGUAGES), help="comma-separated language tags"
    )
    parser.add_argument("--source", default="synthetic")
    parser.add_argument("--out", required=True, help="output JSONL path")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    config = DataConfig(
        seed=args.seed,
        per_type=args.per_type,
        languages=tuple(item for item in args.languages.split(",") if item),
        source=args.source,
    )
    written = generate(config, args.out)
    print(f"wrote {written} records to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
