"""Deterministic synthetic supervision for the three primitives.

Same seed + config produces byte-identical JSONL. Records stream to disk one line at a time,
so generation never holds the dataset in memory (TRAIN-01, TRAIN-07). The output format is
documented in ``docs/training.md``; it is training data, not the wire contract.

Records are fully localized: the ``state``, the ``instructions``, the ``criteria`` and the
``noul``/``score`` entities all come from committed per-domain, per-language data, so the
checkpoint learns language- and domain-specific cues instead of one English template (B-1, B-5).

Domains (B-5)
-------------
``DataConfig.domains`` names the domains to emit and defaults to ``("support",)`` — the original
four-team triage every published number was measured on. Each domain is a committed file under
``training/data/domains/<domain>.json`` holding its option labels, option terms and descriptions,
entities, phrase banks, score levels, ``noul`` criteria and per-primitive instructions.

Two rules keep the guarantee in BACKLOG B-5 ("existing configs produce byte-identical output"):

1. ``support`` keeps the **legacy record seed** ``f"{seed}:{kind}:{index}:{language}"`` — it
   predates domains — while every other domain qualifies the seed with its name. Support records
   are therefore identical whether they are generated alone or inside a five-domain run, so a
   support measurement stays comparable across datasets.
2. The ``domain`` field is written only when the run is not the default single-domain one, so
   ``data.json``, ``data_multi.json`` and ``data_noisy.json`` regenerate byte-for-byte.

``tests/test_training_generate.py`` pins both rules with golden hashes.
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

#: Default language set; states are authored here, other tags fall back to English text.
DEFAULT_LANGUAGES: tuple[str, ...] = ("en", "pt", "es", "fr", "de", "it", "nl")

#: The pre-B-5 domain: customer-support triage. Everything published until now comes from it.
DEFAULT_DOMAIN = "support"
DEFAULT_DOMAINS: tuple[str, ...] = (DEFAULT_DOMAIN,)

_DATA_DIR = Path(__file__).parent / "data"
_DOMAIN_DIR = _DATA_DIR / "domains"


def _load_domains() -> dict[str, dict[str, Any]]:
    """Read every committed domain file, keyed by its ``domain`` field (file stem as fallback)."""
    domains: dict[str, dict[str, Any]] = {}
    for path in sorted(_DOMAIN_DIR.glob("*.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        domains[str(spec.get("domain", path.stem))] = spec
    if DEFAULT_DOMAIN not in domains:
        raise RuntimeError(f"default domain data missing: {_DOMAIN_DIR / f'{DEFAULT_DOMAIN}.json'}")
    return domains


#: Committed per-domain content (``training/data/domains/*.json``).
DOMAINS: dict[str, dict[str, Any]] = _load_domains()

_LEVEL_BY_TONE: dict[str, int] = {"calm": 0, "neutral": 1, "request": 2, "urgent": 3}

#: One in this many ``choice`` records appends a hard-negative distractor from another option.
#: Kept low: appended distractors make the label ambiguous (the target is the *first* option), so
#: a high rate corrupts the term→option signal the head needs.
_HARD_NEGATIVE_RATE = 6


@dataclass(frozen=True, slots=True)
class DomainData:
    """Committed content for one domain, with the same fallbacks the pre-B-5 loader had.

    A missing language falls back to the domain's English table (whole table for entities,
    phrases, levels and instructions; per key for criteria, option terms and descriptions).
    """

    name: str
    spec: dict[str, Any]

    @classmethod
    def load(cls, name: str) -> DomainData:
        if name not in DOMAINS:
            known = ", ".join(sorted(DOMAINS))
            raise SystemExit(f"unknown domain {name!r}; committed domains: {known}")
        return cls(name=name, spec=DOMAINS[name])

    @property
    def options(self) -> tuple[str, ...]:
        return tuple(self.spec["options"])

    @property
    def _languages(self) -> dict[str, Any]:
        return self.spec["languages"]

    def _lang(self, lang: str) -> dict[str, Any]:
        return self._languages.get(lang, self._languages["en"])

    def entities(self, lang: str) -> list[str]:
        return list(self._lang(lang)["entities"])

    def instructions(self, lang: str, kind: str) -> str:
        table = self._lang(lang)["instructions"]
        return table.get(kind, self._languages["en"]["instructions"][kind])

    def noul_criteria(self, lang: str) -> dict[str, str]:
        table = self._lang(lang)["noul_criteria"]
        fallback = self._languages["en"]["noul_criteria"]
        return {key: table.get(key, fallback[key]) for key in ("true", "false")}

    def levels(self, lang: str) -> list[str]:
        return list(self._lang(lang)["levels"])

    def phrases(self, lang: str) -> dict[str, list[str]]:
        return self._lang(lang)["phrases"]

    def option_terms(self, lang: str, option: str) -> tuple[str, ...]:
        table = self._lang(lang)["option_terms"]
        return tuple(table.get(option, self._languages["en"]["option_terms"][option]))

    def option_description(self, lang: str, option: str) -> str:
        table = self._lang(lang)["option_descriptions"]
        return table.get(option, self._languages["en"]["option_descriptions"][option])

    def criteria(self, lang: str) -> dict[str, str]:
        """Option labels → written descriptions (the ``choice`` criteria of this domain)."""
        return {option: self.option_description(lang, option) for option in self.options}


def _domain(name: str) -> DomainData:
    return DomainData.load(name)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Inputs to a data-generation run."""

    seed: int = 42
    per_type: int = 200
    languages: tuple[str, ...] = DEFAULT_LANGUAGES
    #: Domains to emit; the default keeps the pre-B-5 output byte-identical.
    domains: tuple[str, ...] = DEFAULT_DOMAINS
    #: Records per primitive for specific domains, overriding ``per_type``. Used to keep the
    #: incumbent domain's volume intact while new domains are added: training `support` on a third
    #: of its published records while four new domains compete for the same capacity cost 22
    #: points of `support` accuracy (B-5, measured 2026-09-27).
    per_domain: dict[str, int] = field(default_factory=dict)
    source: str = "synthetic"
    #: Fraction of records whose ``state`` gets one surface-noise edit (typos/accents/casing).
    #: ``0.0`` (default) reproduces the clean dataset byte-for-byte (B-4).
    noise_rate: float = 0.0


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _perturb_state(text: str, rng: random.Random) -> str:
    """Apply one deterministic surface-noise edit to ``text``.

    Realistic user input carries typos, missing accents, dropped punctuation and casing noise.
    Exactly one cheap edit is applied per call so the label stays recoverable; empty text passes
    through unchanged. The caller's ``rng`` guarantees the same seed → the same edit.
    """
    if not text:
        return text
    edit = rng.choice(("delete", "swap", "accent", "case", "punct"))
    if edit == "delete" and len(text) > 1:
        index = rng.randrange(len(text))
        return text[:index] + text[index + 1 :]
    if edit == "swap" and len(text) > 1:
        index = rng.randrange(len(text) - 1)
        return text[:index] + text[index + 1] + text[index] + text[index + 2 :]
    if edit == "accent":
        return _strip_accents(text)
    if edit == "case":
        index = rng.randrange(len(text))
        char = text[index]
        replacement = char.lower() if char.isupper() else char.upper()
        return text[:index] + replacement + text[index + 1 :]
    if text[-1] in ".,!?;:":
        return text[:-1]
    return text


def _boundary_state(index: int, base: str) -> str:
    if index % 19 == 0:
        return ""  # empty input
    if index % 23 == 0:
        return base + " " + (" ".join([base] * 40))  # very long input
    return base


def _noul_record(domain: DomainData, index: int, lang: str, rng: random.Random) -> dict[str, Any]:
    entity = rng.choice(domain.entities(lang))
    tone = rng.choice(("request", "neutral"))
    state = _boundary_state(index, rng.choice(domain.phrases(lang)[tone]).format(entity=entity))
    # B-11 (L-008): derive the label from the text just emitted, never from the loop index.
    # A request-toned state is a positive example and a neutral-toned one is negative; the empty
    # boundary state reads as neither, so it is negative too (deterministically, "" -> 0). The
    # old rule (`index % 2` overridden by tone) labelled half of every request-toned set 0,
    # capping `noul` accuracy at the label noise instead of the model (Bayes 0.746 on eval_en).
    # Draw order is unchanged, so only `noul.target` bytes move against the previous datasets.
    target = 1 if tone == "request" and state else 0
    instructions = domain.instructions(lang, "noul").format(entity=entity)
    criteria = {
        key: value.format(entity=entity) for key, value in domain.noul_criteria(lang).items()
    }
    return {
        "id": f"noul-{index:06d}",
        "type": "noul",
        "state": state,
        "instructions": instructions,
        "criteria": criteria,
        "target": target,
        "lang": lang,
    }


def _choice_record(domain: DomainData, index: int, lang: str, rng: random.Random) -> dict[str, Any]:
    # Cycle options deterministically (so every option, including the catch-all, is well
    # represented) and regularly pick a distractor term from a different option as a hard negative.
    options = domain.options
    option = options[index % len(options)]
    term = rng.choice(domain.option_terms(lang, option))
    request_phrase = rng.choice(domain.phrases(lang)["request"])
    state = request_phrase.format(entity=term)
    if index % _HARD_NEGATIVE_RATE == 0:
        other = rng.choice([candidate for candidate in options if candidate != option])
        distractor = rng.choice(domain.option_terms(lang, other))
        distractor_phrase = rng.choice(domain.phrases(lang)["distractor"])
        state = f"{state} {distractor_phrase.format(distractor=distractor)}"
    state = _boundary_state(index, state)
    return {
        "id": f"choice-{index:06d}",
        "type": "choice",
        "state": state,
        "instructions": domain.instructions(lang, "choice"),
        "criteria": domain.criteria(lang),
        "target": option,
        "lang": lang,
    }


def _score_record(domain: DomainData, index: int, lang: str, rng: random.Random) -> dict[str, Any]:
    entity = rng.choice(domain.entities(lang))
    tone = rng.choice(("calm", "neutral", "request", "urgent"))
    state = _boundary_state(index, rng.choice(domain.phrases(lang)[tone]).format(entity=entity))
    target = _LEVEL_BY_TONE[tone]
    if index % 13 == 0:  # ambiguous near-tie label
        target = max(0, target - 1)
    return {
        "id": f"score-{index:06d}",
        "type": "score",
        "state": state,
        "instructions": domain.instructions(lang, "score"),
        "criteria": domain.levels(lang),
        "target": target,
        "lang": lang,
    }


_GENERATORS = {
    "noul": _noul_record,
    "choice": _choice_record,
    "score": _score_record,
}


@cache
def _bank(domain: str, lang: str, tone: str) -> tuple[str, ...]:
    """Formatted sentences of one phrase bank, cached (the audit checks every ``noul`` record)."""
    data = DomainData.load(domain)
    return tuple(
        phrase.format(entity=entity)
        for entity in data.entities(lang)
        for phrase in data.phrases(lang)[tone]
    )


def noul_tone(domain: str, lang: str, state: str) -> str:
    """Recover which phrase bank produced a ``noul`` state (B-11): ``request`` / ``neutral`` /
    ``empty`` / ``unknown``.

    ``_boundary_state`` keeps the chosen phrase, repeats it (long input) or empties it, so the
    bank is recoverable as a prefix of the text. The audit uses it to judge a label against the
    text a reader would see (L-008): ``request`` should carry 1, ``neutral`` and ``empty`` 0.
    States no bank explains — surface-noise edits in the B-4 datasets — come back ``unknown``
    and are deliberately not judged.
    """
    if not state:
        return "empty"
    for tone in ("request", "neutral"):
        if any(state.startswith(phrase) for phrase in _bank(domain, lang, tone)):
            return tone
    return "unknown"


def _seed_base(seed: int, domain: str) -> str:
    """Record-seed prefix for a domain.

    ``support`` predates domains and keeps the legacy prefix, so its records are byte-identical
    to a single-domain run and identical inside a multi-domain run; every other domain is
    qualified with its name so two domains never share draws (L-003's lesson, per domain).
    """
    return str(seed) if domain == DEFAULT_DOMAIN else f"{seed}:{domain}"


def iter_records(config: DataConfig) -> Iterator[dict[str, Any]]:
    """Yield deterministic records: ``per_type`` (or ``per_domain``) of each primitive per domain.

    Each record draws from its own RNG seeded by ``(seed[, domain], kind, index, language)``. A
    single shared RNG makes feature choices correlate with the cyclic label (option/tone) across
    records, which the model then exploits as a shortcut that does not generalize; per-record
    seeding keeps choices independent while staying byte-for-byte deterministic.
    """
    languages = config.languages or DEFAULT_LANGUAGES
    domains = config.domains or DEFAULT_DOMAINS
    writes_domain = tuple(domains) != DEFAULT_DOMAINS
    for domain_name in domains:
        domain = DomainData.load(domain_name)
        base = _seed_base(config.seed, domain_name)
        count = config.per_domain.get(domain_name, config.per_type)
        for kind in ("noul", "choice", "score"):
            generator = _GENERATORS[kind]
            for index in range(count):
                language = languages[index % len(languages)]
                rng = random.Random(f"{base}:{kind}:{index}:{language}")
                record = generator(domain, index, language, rng)
                record["source"] = config.source
                if writes_domain:
                    record["domain"] = domain_name
                if config.noise_rate > 0.0:
                    # A dedicated RNG keeps the apply-or-not draw and the edit independent of the
                    # content draws (L-003), while remaining byte-for-byte deterministic.
                    noise_rng = random.Random(f"{base}:{kind}:{index}:{language}:noise")
                    if noise_rng.random() < config.noise_rate:
                        record["state"] = _perturb_state(record["state"], noise_rng)
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
    parser = argparse.ArgumentParser(prog="tachyone-generate-data", description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-type", type=int, default=200, help="records per primitive")
    parser.add_argument(
        "--languages", default=",".join(DEFAULT_LANGUAGES), help="comma-separated language tags"
    )
    parser.add_argument(
        "--domains",
        default=",".join(DEFAULT_DOMAINS),
        help="comma-separated domains from training/data/domains/ (default: support only)",
    )
    parser.add_argument(
        "--per-domain",
        default="",
        help="comma-separated name=per_type overrides, e.g. support=3000 (keeps the incumbent "
        "domain's volume while --per-type sets the new ones)",
    )
    parser.add_argument("--source", default="synthetic")
    parser.add_argument(
        "--noise-rate",
        type=float,
        default=0.0,
        help="fraction of records whose state gets one surface-noise edit (0.0 disables)",
    )
    parser.add_argument("--out", required=True, help="output JSONL path")
    return parser


def _parse_per_domain(raw: str) -> dict[str, int]:
    """Parse ``name=COUNT,name=COUNT`` into the :attr:`DataConfig.per_domain` override."""
    overrides: dict[str, int] = {}
    for item in (part for part in raw.split(",") if part.strip()):
        name, sep, value = item.partition("=")
        if not sep or not name.strip() or not value.strip().isdigit():
            raise SystemExit(f"--per-domain expects name=COUNT entries, got {item!r}")
        overrides[name.strip()] = int(value)
    return overrides


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if not 0.0 <= args.noise_rate <= 1.0:
        raise SystemExit("--noise-rate must be within [0, 1]")
    config = DataConfig(
        seed=args.seed,
        per_type=args.per_type,
        languages=tuple(item for item in args.languages.split(",") if item),
        domains=tuple(item for item in args.domains.split(",") if item),
        per_domain=_parse_per_domain(args.per_domain),
        source=args.source,
        noise_rate=args.noise_rate,
    )
    written = generate(config, args.out)
    print(f"wrote {written} records to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
