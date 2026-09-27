"""Public probes for **B-7**: typed-decisions, MASSIVE and XNLI — evaluation only.

Each loader turns one public dataset into the row shape ``benchmarks/compare.py`` already
consumes (``state / task / question / options / answer_index``, plus ``wire`` when the dataset
already speaks the ``/v1/systemone`` shape), so every engine, metric, temperature fit and the
renderer are reused untouched:

    uv run python -m benchmarks.compare run --engine tachyone --format probe \\
        --probe typed-decisions --data data/typed-decisions --per-task 500 \\
        --out benchmarks/results/probe_typed_decisions.json
    uv run python -m benchmarks.probes render --out benchmarks/probes.md \\
        --synthetic benchmarks/results/en.json benchmarks/results/probe_*.json

Two rules that keep the comparison comparable:

- **Probe rows never reach ``training/``.** These loaders are evaluation-only (B-7); using any of
  this data for training would turn the number into an in-sample one.
- **Nothing third-party is vendored.** The datasets are downloaded by the reader (commands are in
  the error messages and in ``benchmarks/README.md``), and only their *labels* are referenced
  here — the MASSIVE intent vocabulary is the dataset's own fixed 60-label set.

Licences are verified against the upstream sources, not against a card that may omit them:
``LocalLLaMA/typed-decisions`` declares ``license: apache-2.0`` on its card, MASSIVE ships
``LICENSE`` (CC-BY-4.0) in its repo and in its S3 tarball, and XNLI — whose HF card carries **no**
license tag — is **CC BY-NC 4.0** as published in ``facebookresearch/XNLI``'s LICENSE file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.compare import _cap, _header, _read_parquet, _row

#: Probes accepted by ``benchmarks.compare run --probe``.
PROBES: tuple[str, ...] = ("typed-decisions", "massive", "xnli")

#: ``state`` prompt used for XNLI. Chosen as ``choice`` over the three labels rather than the
#: alternative ``noul`` formulation: a 3-way distribution is what the harness scores directly
#: (B-7 records this as a design decision).
XNLI_INSTRUCTION = "Does the hypothesis follow from the premise?"
#: facebook/xnli label order (0 entailment, 1 neutral, 2 contradiction).
XNLI_LABELS: tuple[str, ...] = ("entailment", "neutral", "contradiction")

#: Fixed instruction MASSIVE gives with every utterance; the option set is its 60 intents.
MASSIVE_INSTRUCTION = "Which intent does this utterance express?"
#: Our language tag → MASSIVE locale (the dataset ships 51; we probe the trained ones).
MASSIVE_LOCALES: dict[str, str] = {
    "en": "en-US",
    "pt": "pt-PT",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "nl": "nl-NL",
}
#: Languages probed by default: English plus the six the multilingual checkpoint trains on.
MASSIVE_LANGUAGES: tuple[str, ...] = ("en", "pt", "es", "fr", "de", "it", "nl")
#: The dataset's own intent vocabulary (``massive.py::_INTENTS``, v1.1) — the option set every
#: language is scored against, so a row whose intent is not here is a label-set drift, not a
#: silently dropped option.
MASSIVE_INTENTS: tuple[str, ...] = (
    "datetime_query",
    "iot_hue_lightchange",
    "transport_ticket",
    "takeaway_query",
    "qa_stock",
    "general_greet",
    "recommendation_events",
    "music_dislikeness",
    "iot_wemo_off",
    "cooking_recipe",
    "qa_currency",
    "transport_traffic",
    "general_quirky",
    "weather_query",
    "audio_volume_up",
    "email_addcontact",
    "takeaway_order",
    "email_querycontact",
    "iot_hue_lightup",
    "recommendation_locations",
    "play_audiobook",
    "lists_createoradd",
    "news_query",
    "alarm_query",
    "iot_wemo_on",
    "general_joke",
    "qa_definition",
    "social_query",
    "music_settings",
    "audio_volume_other",
    "calendar_remove",
    "iot_hue_lightdim",
    "calendar_query",
    "email_sendemail",
    "iot_cleaning",
    "audio_volume_down",
    "play_radio",
    "cooking_query",
    "datetime_convert",
    "qa_maths",
    "iot_hue_lightoff",
    "iot_hue_lighton",
    "transport_query",
    "music_likeness",
    "email_query",
    "play_music",
    "audio_volume_mute",
    "social_post",
    "alarm_set",
    "qa_factoid",
    "calendar_set",
    "play_game",
    "alarm_remove",
    "lists_remove",
    "transport_taxi",
    "recommendation_movies",
    "iot_coffee",
    "music_query",
    "play_podcasts",
    "lists_query",
)

#: Where each probe comes from, how it is licensed and how it is cited (rendered by ``render``).
SOURCES: dict[str, dict[str, str]] = {
    "typed-decisions": {
        "repo": "LocalLLaMA/typed-decisions",
        "licence": "Apache-2.0",
        "licence_evidence": "dataset card front-matter `license: apache-2.0`",
        "size": "4 configs + `all`, n<1K rows each (3.15 MB)",
        "chance": "mixed — `noul` 0.50, `choice` 0.25-0.50, `score` 0.20-0.33",
        "citation": "LocalLLaMA, *Typed Decisions* (Hugging Face dataset), Apache-2.0.",
        "download": (
            'uv run python -c "from huggingface_hub import snapshot_download; '
            "snapshot_download('LocalLLaMA/typed-decisions', repo_type='dataset', "
            "local_dir='data/typed-decisions')\""
        ),
        "mapping": (
            "Already in our wire shape. Multi-question rows are flattened to **one row per "
            "question** — deliberately conservative, because it ignores the `choice` head's "
            "ability to score several options of one question in a single pass. `task` is the "
            "dataset's own config (workflow), and each question keeps its native `noul` / "
            "`choice` / `score` type. Two `noul` questions ship without `criteria` and are "
            "answered as bare true/false (`criteria: null`), which the wire allows."
        ),
    },
    "massive": {
        "repo": "AmazonScience/massive",
        "licence": "CC-BY-4.0",
        "licence_evidence": "`LICENSE` in the dataset repo and in the official S3 tarball",
        "size": "1M utterances, 60 intents, 51 locales (v1.1 tarball is 40 MB)",
        "chance": "0.017 (1 of 60 intents)",
        "citation": (
            "FitzGerald et al., *MASSIVE: A 1M-Example Multilingual Natural Language "
            "Understanding Dataset with 51 Typologically-Diverse Languages*, arXiv:2204.08582 "
            "(2022)."
        ),
        "download": (
            "curl -L -o data/massive/raw/amazon-massive-dataset-1.1.tar.gz "
            "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
        ),
        "mapping": (
            "`state` is the utterance, `options` the fixed 60-intent vocabulary, `task` the "
            "**language** — which is what makes the per-language accuracy/ECE table that `B-1` "
            "needs come out of the same run. Rows are routed to the English checkpoint for `en` "
            "and to the multilingual checkpoint for the other six."
        ),
    },
    "xnli": {
        "repo": "facebook/xnli",
        "licence": "CC BY-NC 4.0",
        "licence_evidence": (
            "`LICENSE` in `facebookresearch/XNLI` (the HF card carries **no** license tag "
            "and its Licensing Information section is a placeholder)"
        ),
        "size": "en: 5,010 test / 2,490 validation pairs",
        "chance": "0.333 (3 labels)",
        "citation": (
            "Conneau et al., *XNLI: Evaluating Cross-lingual Sentence Representations*, EMNLP "
            "2018, arXiv:1809.05053. CC BY-NC 4.0."
        ),
        "download": (
            'uv run python -c "from huggingface_hub import snapshot_download; '
            "snapshot_download('facebook/xnli', repo_type='dataset', allow_patterns=['en/*', "
            "'README.md'], local_dir='data/xnli')\""
        ),
        "mapping": (
            "`state` is `Premise: …\\nHypothesis: …`, the options are "
            "`{entailment, neutral, contradiction}` in the dataset's label order, and the "
            "question is a fixed instruction. Scored as `choice`, **not** as the alternative "
            "`noul` formulation (a design decision: the harness scores a full 3-way "
            "distribution directly)."
        ),
    },
}


# --------------------------------------------------------------------------- loaders
def _typed_decisions_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one typed-decisions record into one row per question, keeping the native wire."""
    state = str(record["state"])
    workflow = str(record["workflow"])
    questions = json.loads(str(record["questions"]))
    gold = json.loads(str(record["gold"]))
    rows: list[dict[str, Any]] = []
    for question_id, question in questions.items():
        answer = gold.get(question_id)
        if not isinstance(answer, dict):
            raise SystemExit(
                f"typed-decisions row {record.get('id')!r} has no gold answer for "
                f"{question_id!r}; the loader and the dataset have drifted apart"
            )
        kind = str(question["type"])
        criteria = question.get("criteria")
        label = str(answer["label"])
        wire: dict[str, Any]
        if kind == "noul":
            # 2 of the 5 noul questions (invoice/duplicate, security/credential_compromise)
            # ship without `criteria`: their option labels are the bare false/true pair, which
            # is exactly what the gold probabilities are keyed by.
            if isinstance(criteria, dict) and {"false", "true"} <= set(criteria):
                options = [str(criteria["false"]), str(criteria["true"])]
                wire_criteria: Any = criteria
            else:
                options = ["false", "true"]
                wire_criteria = None
            if label not in {"false", "true"}:
                raise SystemExit(f"typed-decisions noul label {label!r} is not false/true")
            answer_index = 1 if label == "true" else 0
            wire = {
                "type": kind,
                "instructions": str(question["instructions"]),
                "criteria": wire_criteria,
            }
        elif kind == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise SystemExit(
                    f"typed-decisions choice {question_id!r} has no criteria; the loader and "
                    "the dataset have drifted apart"
                )
            options = [str(option) for option in criteria]
            if label not in options:
                raise SystemExit(f"typed-decisions choice label {label!r} is not in its options")
            answer_index = options.index(label)
            wire = {
                "type": kind,
                "instructions": str(question["instructions"]),
                "criteria": criteria,
            }
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise SystemExit(
                    f"typed-decisions score {question_id!r} has no criteria; the loader and "
                    "the dataset have drifted apart"
                )
            options = [str(level) for level in criteria]
            answer_index = int(label)  # the gold label is the argmax level, as a string index
            if not 0 <= answer_index < len(options):
                raise SystemExit(
                    f"typed-decisions score label {label!r} is outside 0..{len(options) - 1}"
                )
            wire = {
                "type": kind,
                "instructions": str(question["instructions"]),
                "criteria": criteria,
            }
        else:
            raise SystemExit(f"unsupported typed-decisions question type {kind!r}")
        rows.append(
            {
                "state": state,
                "task": workflow,
                "question": str(question["instructions"]),
                "options": options,
                "answer_index": answer_index,
                "wire": wire,
            }
        )
    return rows


def load_typed_decisions(
    data: Path, *, split: str, per_task: int, limit: int = 0
) -> list[dict[str, Any]]:
    """``LocalLLaMA/typed-decisions``: one row per (record, question), capped per config."""
    files = sorted((data / "all").glob(f"{split}-*.parquet"))
    if not files:
        raise SystemExit(
            f"no {split!r} parquet under {data / 'all'}; download it first:\n"
            f"  {SOURCES['typed-decisions']['download']}"
        )
    rows: list[dict[str, Any]] = []
    for path in files:
        for record in _read_parquet(path):
            rows.extend(_typed_decisions_rows(record))
    return _cap(rows, per_task, limit)


def _massive_path(data: Path, language: str) -> Path | None:
    locale = MASSIVE_LOCALES[language]
    for candidate in (
        data / "data" / f"{locale}.jsonl",  # flat layout
        data / "1.1" / "data" / f"{locale}.jsonl",  # as the official tarball lays it out
        data / f"{locale}.jsonl",
    ):
        if candidate.exists():
            return candidate
    return None


def load_massive(
    data: Path,
    *,
    split: str,
    per_task: int,
    limit: int = 0,
    languages: tuple[str, ...] = MASSIVE_LANGUAGES,
) -> list[dict[str, Any]]:
    """``AmazonScience/massive``: intent classification, capped per **language**."""
    unknown = [language for language in languages if language not in MASSIVE_LOCALES]
    if unknown:
        raise SystemExit(
            f"MASSIVE has no locale for {', '.join(unknown)}; known tags: "
            f"{', '.join(MASSIVE_LOCALES)}"
        )
    tarball = data / "raw" / "amazon-massive-dataset-1.1.tar.gz"
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for language in languages:
        path = _massive_path(data, language)
        if path is None:
            missing.append(MASSIVE_LOCALES[language])
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if str(record["partition"]) != split:
                    continue
                intent = str(record["intent"])
                if intent not in MASSIVE_INTENTS:
                    raise SystemExit(
                        f"MASSIVE intent {intent!r} is not in the dataset's 60-label vocabulary; "
                        "the loader and the dataset have drifted apart"
                    )
                rows.append(
                    {
                        "state": str(record["utt"]),
                        "task": language,
                        "question": MASSIVE_INSTRUCTION,
                        "options": list(MASSIVE_INTENTS),
                        "answer_index": MASSIVE_INTENTS.index(intent),
                    }
                )
    if missing:
        locales = " ".join(f"1.1/data/{locale}.jsonl" for locale in missing)
        raise SystemExit(
            f"MASSIVE locales not extracted under {data}: {', '.join(missing)}\n"
            f"  download: {SOURCES['massive']['download']}\n"
            f"  then:     tar -xzf {tarball} -C {data} --wildcards '{locales}'"
        )
    return _cap(rows, per_task, limit)


def load_xnli(
    data: Path,
    *,
    split: str,
    per_task: int,
    limit: int = 0,
    languages: tuple[str, ...] = ("en",),
) -> list[dict[str, Any]]:
    """``facebook/xnli``: 3-way entailment, capped per language."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for language in languages:
        files = sorted((data / language).glob(f"{split}-*.parquet"))
        if not files:
            missing.append(language)
            continue
        for path in files:
            for record in _read_parquet(path):
                label = int(record["label"])
                if not 0 <= label < len(XNLI_LABELS):
                    raise SystemExit(f"XNLI label {label} is outside 0..{len(XNLI_LABELS) - 1}")
                rows.append(
                    {
                        "state": (
                            f"Premise: {record['premise']}\nHypothesis: {record['hypothesis']}"
                        ),
                        "task": language,
                        "question": XNLI_INSTRUCTION,
                        "options": list(XNLI_LABELS),
                        "answer_index": label,
                    }
                )
    if missing:
        raise SystemExit(
            f"no XNLI parquet for {', '.join(missing)} under {data}; download it first:\n"
            f"  {SOURCES['xnli']['download']}"
        )
    return _cap(rows, per_task, limit)


def load_probe(
    name: str,
    data: Path,
    *,
    split: str,
    per_task: int,
    limit: int = 0,
    languages: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Dispatch ``benchmarks.compare run --probe <name>`` to its loader."""
    if name == "typed-decisions":
        return load_typed_decisions(data, split=split, per_task=per_task, limit=limit)
    if name == "massive":
        return load_massive(
            data,
            split=split,
            per_task=per_task,
            limit=limit,
            languages=tuple(languages) if languages else MASSIVE_LANGUAGES,
        )
    if name == "xnli":
        return load_xnli(
            data,
            split=split,
            per_task=per_task,
            limit=limit,
            languages=tuple(languages) if languages else ("en",),
        )
    raise SystemExit(f"unknown probe {name!r}; choose one of: {', '.join(PROBES)}")


# -------------------------------------------------------------------------- render
def _method(artifact: dict[str, Any]) -> str:
    dataset = artifact["dataset"]
    temperature = artifact["temperature"]
    environment = artifact["environment"]
    fit = (
        f"fitted on {dataset.get('val_split')} (capped)"
        if temperature["fitted"]
        else "fixed at 1.0"
    )
    return (
        f"{dataset['rows']} rows · `{dataset['split']}` split · {dataset['per_task']} per "
        f"{'language' if dataset.get('probe') == 'massive' else 'task'} · temperature {fit} "
        f"(T={temperature['value']}) · {environment.get('gpu', 'unknown GPU')} · "
        f"Python {environment.get('python', '?')}"
    )


def _detail_table(artifacts: list[dict[str, Any]]) -> str:
    """Per-task accuracy **and** calibration: `B-1` gates on the worst language, not the average."""
    lines = [
        "| Task | Engine | n | Accuracy | ECE raw | ECE cal | Conf |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for artifact in sorted(artifacts, key=lambda item: str(item["engine"])):
        calibrated = artifact["per_task_calibrated"]
        raw = artifact["per_task_raw"]
        for task in sorted(calibrated):
            entry = calibrated[task]
            shipped = raw.get(task, entry)
            lines.append(
                f"| {task} | {artifact['engine']} | {entry['n']} | {entry['accuracy']:.3f} "
                f"| {float(shipped['ece']):.3f} | {float(entry['ece']):.3f} "
                f"| {entry['mean_confidence']:.3f} |"
            )
    return "\n".join(lines)


def _section(artifacts: list[dict[str, Any]], probe: str) -> list[str]:
    source = SOURCES[probe]
    lines = [
        f"## `{probe}` — {source['repo']} ({source['licence']})",
        "",
        f"- **Licence:** {source['licence']} — {source['licence_evidence']}.",
        f"- **Size:** {source['size']}.",
        f"- **Chance level:** {source['chance']}.",
        f"- **How it is mapped:** {source['mapping']}",
        "",
    ]
    for artifact in sorted(artifacts, key=lambda item: str(item["engine"])):
        lines += [f"### {artifact['engine']}", "", _method(artifact), ""]
    lines += ["#### Quality", "", _header("quality")]
    lines += [
        _row(artifact, "quality") for artifact in sorted(artifacts, key=lambda a: a["engine"])
    ]
    lines += ["", "#### Performance", "", _header("performance")]
    lines += [
        _row(artifact, "performance") for artifact in sorted(artifacts, key=lambda a: a["engine"])
    ]
    lines += ["", "#### Per task: accuracy and calibration", "", _detail_table(artifacts), ""]
    return lines


def _synthetic_row(path: Path) -> list[str]:
    report = json.loads(path.read_text(encoding="utf-8"))
    overall = report["overall"]
    return [
        "| in-sample synthetic eval (`benchmarks/report.md`) | synthetic, shares states with "
        f"training data | {overall['n']} | {overall['accuracy']:.3f} | {overall['ece']:.3f} |",
    ]


def render(paths: list[Path], out: Path, synthetic: Path | None) -> None:
    """Compose ``benchmarks/probes.md`` from probe artifacts (plus the synthetic row)."""
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_probe: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        probe = str(artifact["dataset"].get("probe") or "")
        if probe not in SOURCES:
            raise SystemExit(
                f"{artifact['engine']}: dataset.probe is {probe!r}; re-run it with --probe "
                "<one of the public probes> so it can be attributed to a dataset"
            )
        by_probe.setdefault(probe, []).append(artifact)

    ordered = [probe for probe in PROBES if probe in by_probe]
    body = [
        "# Public probes",
        "",
        "**Externally comparable numbers.** Everything here is measured on public datasets",
        "Tachyone did not train on, through the same `benchmarks/compare.py` metrics used in",
        "[`docs/compare.md`](../docs/compare.md). Evaluation only: no row on this page has ever",
        "reached `training/`, which is exactly what makes the numbers mean something.",
        "",
        "> **Do not read these next to `benchmarks/report.md` as if they were the same test.**",
        "> That report is the *in-sample synthetic* split (it shares states with the training",
        "> data — `B-9`, lesson L-005); these are public distributions. The table at the bottom",
        "> shows both, labelled, and makes no regression claim across them.",
        "",
        "### How to read this page",
        "",
        "- **Judge each number against that probe's chance level**, stated in its section — not",
        "  against the synthetic 0.859. Different tasks, different option counts, no shared",
        "  distribution with the training data.",
        "- **All three temperature fits landed on the grid ceiling (T=20.0).** That flattens the",
        "  distribution, so a low `ECE cal` is *bought* with confidence: read it next to `Conf`",
        "  and `Brier`, and treat `ECE raw` as what the adapter actually ships (same caveat as",
        "  [`docs/compare.md` §3](../docs/compare.md#3-why-not-another-open-system-one-scorer)).",
        "- **These are the released adapters, zero-shot**, trained on support tickets with four",
        "  team labels. Broadening what they know is `B-5`; these numbers are the evidence for it.",
        "",
        "## Method",
        "",
        "| | |",
        "| --- | --- |",
        "| Hardware | single RTX 3060 12GB · Python 3.12 · stock encoder forward (no `fast`) |",
        "| Engine | `encoder` backend through `tachyone.wire.answer`, released adapters "
        "(`munod/tachyone-en`, `munod/tachyone-multi`) selected per row by the language router |",
        "| Metrics | one implementation of accuracy / 10-bin ECE / Brier; a failed answer counts "
        "as **wrong** |",
        "| Temperature | fitted per probe on that probe's own development split, never on the "
        "split being reported |",
        "| Caps | rows are taken deterministically (dataset order) up to `--per-task` per task; "
        "the exact cap is in each section |",
        "",
    ]
    for probe in ordered:
        body += _section(by_probe[probe], probe)

    body += [
        "## Synthetic vs public",
        "",
        "| Evaluation set | Nature | n | Accuracy | ECE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    if synthetic:
        body += _synthetic_row(synthetic)
    for probe in ordered:
        artifact = max(by_probe[probe], key=lambda item: item["dataset"]["rows"])
        raw = artifact["raw"]
        source = SOURCES[probe]
        body.append(
            f"| {probe} `{artifact['dataset']['split']}` | public, {source['licence']} "
            f"| {raw['n']} | {raw['accuracy']:.3f} | {float(raw['ece']):.3f} |"
        )
    body += [
        "",
        "Raw (uncalibrated) ECE is what the engine shipped with; see",
        "[`docs/compare.md`](../docs/compare.md#3-why-not-another-open-system-one-scorer) for why "
        "ECE alone is not a quality metric.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "OUT=benchmarks/results",
        SOURCES["typed-decisions"]["download"],
        SOURCES["massive"]["download"],
        "tar -xzf data/massive/raw/amazon-massive-dataset-1.1.tar.gz -C data/massive "
        "--wildcards "
        + " ".join(f"'1.1/data/{MASSIVE_LOCALES[lang]}.jsonl'" for lang in MASSIVE_LANGUAGES),
        SOURCES["xnli"]["download"],
        "",
        "uv run python -m benchmarks.compare run --engine tachyone --format probe \\",
        "  --probe typed-decisions --data data/typed-decisions --per-task 500 \\",
        "  --out $OUT/probe_typed_decisions.json",
        "uv run python -m benchmarks.compare run --engine tachyone --format probe \\",
        "  --probe massive --data data/massive --per-task 512 \\",
        "  --languages en,pt,es,fr,de,it,nl --out $OUT/probe_massive.json",
        "uv run python -m benchmarks.compare run --engine tachyone --format probe \\",
        "  --probe xnli --data data/xnli --per-task 6000 \\",
        "  --out $OUT/probe_xnli.json",
        "uv run python -m benchmarks.probes render --out benchmarks/probes.md \\",
        "  --synthetic benchmarks/results/en.json $OUT/probe_*.json",
        "```",
        "",
        "Artifacts live under `benchmarks/results/` (gitignored); each records its own command,",
        "hardware, fitted temperature and per-task breakdown in `environment.command`.",
        "",
        "## Citations and licences",
        "",
    ]
    for probe in ordered:
        source = SOURCES[probe]
        body.append(f"- **{source['repo']}** — {source['licence']}. {source['citation']}")
    body += [
        "",
        "XNLI is **non-commercial**; the tables above are derived metrics (our model's scores),",
        "not a redistribution of the corpus — the data is never committed to this repository.",
        "",
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body), encoding="utf-8")
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.probes", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    render_parser = sub.add_parser("render", help="compose benchmarks/probes.md from artifacts")
    render_parser.add_argument("artifacts", nargs="+", type=Path)
    render_parser.add_argument("--out", type=Path, required=True)
    render_parser.add_argument(
        "--synthetic", type=Path, default=None, help="training/evaluate artifact for the first row"
    )
    args = parser.parse_args(argv)
    render(args.artifacts, args.out, args.synthetic)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
