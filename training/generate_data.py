"""Deterministic synthetic supervision for the three primitives.

Same seed + config produces byte-identical JSONL. Records stream to disk one line at a time,
so generation never holds the dataset in memory (TRAIN-01, TRAIN-07). The output format is
documented in ``docs/training.md``; it is training data, not the wire contract.

Records are fully localized: the ``state``, the ``instructions``, the ``criteria`` and the
``noul``/``score`` entities all come from committed per-language data, so the multilingual
checkpoint learns language-specific cues instead of an English template (B-1).
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
DEFAULT_LANGUAGES: tuple[str, ...] = ("en", "pt", "es", "fr", "de", "it", "nl")

_DATA_DIR = Path(__file__).parent / "data"

#: Per-language phrasing loaded from committed data. ``{entity}``/``{distractor}`` substituted.
_PHRASES: dict[str, dict[str, list[str]]] = json.loads(
    (_DATA_DIR / "phrases.json").read_text(encoding="utf-8")
)

#: Per-language lexicon (entities, team cues, instructions, levels) from committed data.
_LEXICON: dict[str, dict[str, Any]] = json.loads(
    (_DATA_DIR / "lexicon.json").read_text(encoding="utf-8")
)

_TEAMS: tuple[str, ...] = ("billing", "technical", "sales", "other")

_LEVEL_BY_TONE: dict[str, int] = {"calm": 0, "neutral": 1, "request": 2, "urgent": 3}

#: One in this many ``choice`` records appends a hard-negative distractor from another team.
#: Kept low: appended distractors make the label ambiguous (the target is the *first* team), so a
#: high rate corrupts the term→team signal the head needs.
_HARD_NEGATIVE_RATE = 6

#: Rich option descriptions (esp. ``other``) so the option embedding c_k is informative; loaded
#: from committed data. ``other`` must be a genuinely learnable class, not a bare label.
_TEAM_DESCRIPTIONS: dict[str, dict[str, str]] = json.loads(
    (_DATA_DIR / "team_descriptions.json").read_text(encoding="utf-8")
)


def _lexicon(lang: str) -> dict[str, Any]:
    return _LEXICON.get(lang, _LEXICON["en"])


def _entities(lang: str) -> list[str]:
    return list(_lexicon(lang)["entities"])


def _instructions(lang: str, kind: str) -> str:
    table = _lexicon(lang)["instructions"]
    return table.get(kind, _LEXICON["en"]["instructions"][kind])


def _noul_criteria(lang: str) -> dict[str, str]:
    table = _lexicon(lang)["noul_criteria"]
    fallback = _LEXICON["en"]["noul_criteria"]
    return {key: table.get(key, fallback[key]) for key in ("true", "false")}


def _levels(lang: str) -> list[str]:
    return list(_lexicon(lang)["levels"])


def _team_description(lang: str, team: str) -> str:
    table = _TEAM_DESCRIPTIONS.get(lang, _TEAM_DESCRIPTIONS["en"])
    return table.get(team, _TEAM_DESCRIPTIONS["en"][team])


def _team_terms(lang: str, team: str) -> tuple[str, ...]:
    table = _lexicon(lang)["team_terms"]
    return tuple(table.get(team, _LEXICON["en"]["team_terms"][team]))


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
    entity = rng.choice(_entities(lang))
    positive = (index % 2 == 0) if index else True
    tone = rng.choice(("request", "neutral"))
    state = _boundary_state(index, rng.choice(_phrases(lang)[tone]).format(entity=entity))
    if positive and tone == "neutral":
        positive = False
    target = 1 if positive else 0
    instructions = _instructions(lang, "noul").format(entity=entity)
    criteria = {key: value.format(entity=entity) for key, value in _noul_criteria(lang).items()}
    return {
        "id": f"noul-{index:06d}",
        "type": "noul",
        "state": state,
        "instructions": instructions,
        "criteria": criteria,
        "target": target,
        "lang": lang,
    }


def _choice_record(index: int, lang: str, rng: random.Random) -> dict[str, Any]:
    # Cycle teams deterministically (so every team, including "other", is well represented)
    # and regularly pick a distractor clause from a different team as a hard negative.
    team = _TEAMS[index % len(_TEAMS)]
    term = rng.choice(_team_terms(lang, team))
    request_phrase = rng.choice(_phrases(lang)["request"])
    state = request_phrase.format(entity=term)
    if index % _HARD_NEGATIVE_RATE == 0:
        other_team = rng.choice([candidate for candidate in _TEAMS if candidate != team])
        distractor = rng.choice(_team_terms(lang, other_team))
        distractor_phrase = rng.choice(_phrases(lang)["distractor"])
        state = f"{state} {distractor_phrase.format(distractor=distractor)}"
    state = _boundary_state(index, state)
    return {
        "id": f"choice-{index:06d}",
        "type": "choice",
        "state": state,
        "instructions": _instructions(lang, "choice"),
        "criteria": {option: _team_description(lang, option) for option in _TEAMS},
        "target": team,
        "lang": lang,
    }


def _score_record(index: int, lang: str, rng: random.Random) -> dict[str, Any]:
    entity = rng.choice(_entities(lang))
    tone = rng.choice(("calm", "neutral", "request", "urgent"))
    state = _boundary_state(index, rng.choice(_phrases(lang)[tone]).format(entity=entity))
    target = _LEVEL_BY_TONE[tone]
    if index % 13 == 0:  # ambiguous near-tie label
        target = max(0, target - 1)
    return {
        "id": f"score-{index:06d}",
        "type": "score",
        "state": state,
        "instructions": _instructions(lang, "score"),
        "criteria": _levels(lang),
        "target": target,
        "lang": lang,
    }


_GENERATORS = {
    "noul": _noul_record,
    "choice": _choice_record,
    "score": _score_record,
}


def iter_records(config: DataConfig) -> Iterator[dict[str, Any]]:
    """Yield deterministic records: ``per_type`` of each primitive, languages interleaved.

    Each record draws from its own RNG seeded by ``(seed, kind, index, language)``. A single
    shared RNG makes feature choices correlate with the cyclic label (team/tone) across records,
    which the model then exploits as a shortcut that does not generalize; per-record seeding
    keeps choices independent while staying byte-for-byte deterministic.
    """
    languages = config.languages or DEFAULT_LANGUAGES
    for kind in ("noul", "choice", "score"):
        generator = _GENERATORS[kind]
        for index in range(config.per_type):
            language = languages[index % len(languages)]
            rng = random.Random(f"{config.seed}:{kind}:{index}:{language}")
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
