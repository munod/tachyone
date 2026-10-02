"""Build Tachyone records from the pinned public sources (B-13 P2, JB-2).

Three reviewed sources (``training/data/jev_sources.lock.json``) become wire-shaped
records in ``data/train_jev_sources.jsonl``:

- **MultiNLI → `choice`** over XNLI's three labels. The state is ``Premise: …\\nHypothesis: …``
  and the instruction is imported from ``benchmarks.probes`` so training asks the exact
  question the XNLI probe asks (the transfer target); the label index follows the pinned
  card (0 entailment, 1 neutral, 2 contradiction).
- **BoolQ → `noul`**: ``Passage: …\\nQuestion: …``, target taken from the emitted answer
  (never from an index).
- **Banking77 → `choice`** over 5 options — the true intent plus 4 distractors, mostly
  drawn from the same name-prefix group so the options are confusable siblings instead of
  obvious strangers. Descriptions are committed content
  (``training/data/jev_banking77_descriptions.json``), not fetched data.

Sampling is a seeded reservoir per source (L-003: one independent stream per source, so
nothing about the sample can correlate with a label). ``--public-dir`` asserts **zero**
normalized-text overlap with the 231 public JevBench items before anything is written —
those items stay evaluation-only forever (spec *Locked decisions*).

    uv run python -m training.build_jev_sources --public-dir /path/to/jevbench/datasets/public

pyarrow is imported lazily (the ``train`` extra); everything else is stdlib.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from benchmarks.probes import XNLI_INSTRUCTION, XNLI_LABELS
from training.fetch_jev_sources import load_lock

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = _ROOT / "training" / "configs" / "jev_sources.json"
DEFAULT_LOCK = _ROOT / "training" / "data" / "jev_sources.lock.json"
DEFAULT_DESCRIPTIONS = _ROOT / "training" / "data" / "jev_banking77_descriptions.json"
DEFAULT_DATA_DIR = _ROOT / "data" / "sources"
DEFAULT_OUT = _ROOT / "data" / "train_jev_sources.jsonl"

#: Criteria for the three-way NLI judgement, keyed in ``XNLI_LABELS`` order.
XNLI_CRITERIA: dict[str, str] = {
    "entailment": "The hypothesis is guaranteed to be true whenever the premise is true.",
    "neutral": "The premise neither guarantees nor rules out the hypothesis.",
    "contradiction": "The hypothesis is ruled out by the premise.",
}
BOOLQ_INSTRUCTION = "Given the passage, is the answer to the question yes?"
BOOLQ_CRITERIA: dict[str, str] = {
    "false": "The passage does not support answering yes.",
    "true": "The passage supports answering yes.",
}
BANKING_INSTRUCTION = "Which intent does this message express?"

_CONFIG_KEYS = {"seed", "counts", "distractors_per_choice"}


class BuildError(ValueError):
    """A source, config or label does not match what this builder expects."""


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the sampling recipe with strict keys (unknown keys are a mistake, L-011)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.pop("_comment", None)
    unknown = set(data) - _CONFIG_KEYS
    if unknown:
        raise BuildError(f"unknown config keys: {sorted(unknown)}")
    missing = _CONFIG_KEYS - set(data)
    if missing:
        raise BuildError(f"missing config keys: {sorted(missing)}")
    if not isinstance(data["seed"], int) or not isinstance(data["counts"], dict):
        raise BuildError("seed must be an int and counts a dict of source -> rows")
    if not isinstance(data["distractors_per_choice"], int) or data["distractors_per_choice"] < 1:
        raise BuildError("distractors_per_choice must be a positive int")
    return data


def load_descriptions(
    path: str | Path = DEFAULT_DESCRIPTIONS,
) -> tuple[list[str], dict[str, str]]:
    """The 77 intent names **in label-index order** plus their descriptions."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    intents = data.get("intents")
    if not isinstance(intents, dict) or not intents:
        raise BuildError("descriptions need a non-empty 'intents' object")
    names = list(intents)
    for name in names:
        if not isinstance(intents[name], str) or not intents[name].strip():
            raise BuildError(f"intent {name!r} needs a non-empty description")
    if len(set(names)) != len(names):
        raise BuildError("duplicate intent names")
    return names, intents


def reservoir(items: Iterable[Any], k: int, rng: random.Random) -> list[Any]:
    """A seeded reservoir sample: deterministic for a given seed *and* input sequence."""
    sample: list[Any] = []
    for index, item in enumerate(items):
        if index < k:
            sample.append(item)
            continue
        position = rng.randrange(index + 1)
        if position < k:
            sample[position] = item
    return sample


def intent_group(name: str) -> str:
    """The name-prefix sibling group (``card_arrival`` → ``card``) used for distractors."""
    return name.split("_")[0].lower()


def sample_distractors(target: str, names: Sequence[str], k: int, rng: random.Random) -> list[str]:
    """``k`` unique distractors, at least half from the target's sibling group when it can."""
    group = intent_group(target)
    siblings = [name for name in names if name != target and intent_group(name) == group]
    picks = rng.sample(siblings, min((k + 1) // 2, len(siblings)))
    rest = [name for name in names if name != target and name not in picks]
    picks.extend(rng.sample(rest, k - len(picks)))
    if len(set(picks)) != k or target in picks:
        raise BuildError(f"bad distractor draw for {target!r}: {picks}")
    return picks


def iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a parquet file as plain dicts (pyarrow is the ``train`` extra)."""
    try:
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError("pyarrow is required: uv sync --extra train") from exc
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(batch_size=4096):
        columns = batch.to_pydict()
        for index in range(batch.num_rows):
            yield {name: values[index] for name, values in columns.items()}


def mnli_pairs(row: dict[str, Any]) -> Iterator[tuple[str, str, int]]:
    """Yield ``(premise, hypothesis, label)`` pairs from one pinned-card row.

    The pinned card stores one pair per row (scalar columns); a list-valued layout (two
    pairs per row, as other revisions ship) is accepted too — the ragged check keeps the
    list layout honest.
    """
    premises, hypotheses, labels = row["premise"], row["hypothesis"], row["label"]
    if not isinstance(premises, list):
        premises, hypotheses, labels = [premises], [hypotheses], [labels]
    if not (len(premises) == len(hypotheses) == len(labels)):
        raise BuildError(f"ragged MultiNLI row: {len(premises)}/{len(hypotheses)}/{len(labels)}")
    for premise, hypothesis, label in zip(premises, hypotheses, labels, strict=True):
        yield str(premise), str(hypothesis), int(label)


def mnli_record(premise: str, hypothesis: str, label: int, ordinal: int) -> dict[str, Any]:
    """One XNLI-shaped `choice` record; the label is the row's own, never the ordinal."""
    if not 0 <= label < len(XNLI_LABELS):
        raise BuildError(f"MultiNLI label {label} outside 0..{len(XNLI_LABELS) - 1}")
    return {
        "id": f"src-mnli-{ordinal:05d}",
        "type": "choice",
        "state": f"Premise: {premise}\nHypothesis: {hypothesis}",
        "instructions": XNLI_INSTRUCTION,
        "criteria": dict(XNLI_CRITERIA),
        "target": XNLI_LABELS[label],
        "lang": "en",
    }


def boolq_record(question: str, passage: str, answer: Any, ordinal: int) -> dict[str, Any]:
    """One `noul` record; the target comes from the emitted answer (B-11 discipline)."""
    if not isinstance(answer, bool):
        raise BuildError(f"BoolQ answer must be a bool, got {answer!r}")
    return {
        "id": f"src-boolq-{ordinal:05d}",
        "type": "noul",
        "state": f"Passage: {passage}\nQuestion: {question}",
        "instructions": BOOLQ_INSTRUCTION,
        "criteria": dict(BOOLQ_CRITERIA),
        "target": 1 if answer else 0,
        "lang": "en",
    }


def banking_record(
    text: str,
    target: int,
    names: Sequence[str],
    descriptions: dict[str, str],
    k: int,
    rng: random.Random,
    ordinal: int,
) -> dict[str, Any]:
    """One 5-way `choice` record over the true intent and confusable siblings."""
    if not 0 <= target < len(names):
        raise BuildError(f"Banking77 label {target} outside 0..{len(names) - 1}")
    target_name = names[target]
    options = sorted({target_name, *sample_distractors(target_name, names, k, rng)})
    return {
        "id": f"src-bank77-{ordinal:05d}",
        "type": "choice",
        "state": str(text),
        "instructions": BANKING_INSTRUCTION,
        "criteria": {name: descriptions[name] for name in options},
        "target": target_name,
        "lang": "en",
    }


def normalized(text: str) -> str:
    """Lower-cased alphanumerics only — the overlap check compares like this."""
    return "".join(char for char in text.lower() if char.isalnum())


def public_texts(public_dir: str | Path) -> set[str]:
    """Normalized states **and** instructions of the public JevBench items."""
    texts: set[str] = set()
    directory = Path(public_dir)
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise BuildError(f"no *.jsonl under {directory}")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            texts.add(normalized(str(item["state"])))
            texts.add(normalized(str(item["question"]["instructions"])))
    return texts


def assert_no_public_overlap(records: Sequence[dict[str, Any]], public_dir: str | Path) -> None:
    """Fail loudly if any record collides with a public JevBench item (they stay eval-only)."""
    public = public_texts(public_dir)
    clashes = [
        record["id"]
        for record in records
        if normalized(record["state"]) in public
        or normalized(str(record["instructions"])) in public
    ]
    if clashes:
        raise BuildError(
            f"{len(clashes)} record(s) collide with public JevBench items: {clashes[:5]} — "
            "the public items are evaluation-only"
        )


def _require(path: Path, name: str) -> Path:
    if not path.exists():
        raise BuildError(
            f"missing source {name!r}: {path} — run python -m training.fetch_jev_sources"
        )
    return path


def build(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    lock_path: str | Path = DEFAULT_LOCK,
    descriptions_path: str | Path = DEFAULT_DESCRIPTIONS,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    out_path: str | Path = DEFAULT_OUT,
    public_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Sample the three pinned sources into one record file; return the run report."""
    config = load_config(config_path)
    lock = load_lock(lock_path)
    names, descriptions = load_descriptions(descriptions_path)
    counts: dict[str, int] = config["counts"]
    unknown = set(counts) - set(lock["sources"])
    if unknown:
        raise BuildError(f"config samples sources the lock does not pin: {sorted(unknown)}")
    directory = Path(data_dir)
    seed = config["seed"]
    k = config["distractors_per_choice"]
    records: list[dict[str, Any]] = []

    # MultiNLI: stream pairs, reservoir-sample, attach the pin's short revision.
    rng = random.Random(f"{seed}:multi_nli")
    rows = iter_parquet(_require(directory / "multi_nli.parquet", "multi_nli"))
    pairs = (pair for row in rows for pair in mnli_pairs(row))
    for ordinal, (premise, hypothesis, label) in enumerate(
        reservoir(pairs, counts["multi_nli"], rng)
    ):
        records.append(mnli_record(premise, hypothesis, label, ordinal))

    rng = random.Random(f"{seed}:boolq")
    rows = iter_parquet(_require(directory / "boolq.parquet", "boolq"))
    for ordinal, row in enumerate(reservoir(rows, counts["boolq"], rng)):
        records.append(boolq_record(row["question"], row["passage"], row["answer"], ordinal))

    rng = random.Random(f"{seed}:banking77")
    rows = iter_parquet(_require(directory / "banking77.parquet", "banking77"))
    for ordinal, row in enumerate(reservoir(rows, counts["banking77"], rng)):
        records.append(
            banking_record(row["text"], int(row["label"]), names, descriptions, k, rng, ordinal)
        )

    # Source attribution carries the pin (which exact revision produced these bytes).
    per_source = {
        "src-mnli-": f"multi_nli@{lock['sources']['multi_nli']['revision'][:12]}",
        "src-boolq-": f"boolq@{lock['sources']['boolq']['revision'][:12]}",
        "src-bank77-": f"banking77@{lock['sources']['banking77']['revision'][:12]}",
    }
    for record in records:
        record["source"] = next(
            tag for prefix, tag in per_source.items() if record["id"].startswith(prefix)
        )

    if public_dir is not None:
        assert_no_public_overlap(records, public_dir)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    per_type: dict[str, int] = {}
    for record in records:
        per_type[record["type"]] = per_type.get(record["type"], 0) + 1
    return {
        "out": str(out),
        "total": len(records),
        "per_source": {
            prefix.removeprefix("src-").removesuffix("-"): sum(
                1 for record in records if record["id"].startswith(prefix)
            )
            for prefix in per_source
        },
        "per_type": per_type,
        "seed": seed,
        "checked_against_public": public_dir is not None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-build-jev-sources", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--lock", default=str(DEFAULT_LOCK))
    parser.add_argument("--descriptions", default=str(DEFAULT_DESCRIPTIONS))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--public-dir",
        default=None,
        help="the jevbench datasets/public directory: refuse to write if any record collides",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = build(
        config_path=args.config,
        lock_path=args.lock,
        descriptions_path=args.descriptions,
        data_dir=args.data_dir,
        out_path=args.out,
        public_dir=args.public_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
