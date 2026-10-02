"""Render a reproducible benchmark report from evaluation artifacts.

``training/evaluate.py`` writes JSON reports; this script aggregates them into Markdown with
the exact reproduction commands and the captured environment (OPS-06, NFR-D04). It is pure, so
the report can be regenerated without a GPU; real numbers are produced by the actual backends.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.config import environment


@dataclass(frozen=True, slots=True)
class ReportEntry:
    """One evaluated backend and its metrics report."""

    name: str
    report: dict[str, Any]

    @classmethod
    def from_files(cls, name: str, path: str | Path) -> ReportEntry:
        return cls(name=name, report=json.loads(Path(path).read_text(encoding="utf-8")))


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _worst(table: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The key with the lowest accuracy (tie-break: highest ECE), or ``None``.

    The project gates on the worst group, never the average (B-1, B-5).
    """
    if not table:
        return None
    return min(
        table.items(),
        key=lambda item: (item[1].get("accuracy", 0.0), -item[1].get("ece", 0.0)),
    )


def _worst_language(report: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The language with the lowest accuracy, or ``None``."""
    return _worst(report.get("per_language") or {})


def _noul_table(entry: ReportEntry) -> list[str]:
    """``noul`` accuracy per language beside the label audit of the same rows (B-11).

    Both columns come from one report, so a language can no longer look weak because its labels
    were contradictory (L-008): the contradiction count sits next to the accuracy it would
    otherwise be mistaken for. Absent when the report predates the audit.
    """
    metrics_by_lang = entry.report.get("noul_per_language") or {}
    audit_rows = (entry.report.get("noul_labels") or {}).get("per_language") or {}
    if not metrics_by_lang or not audit_rows:
        return []
    lines = [
        "#### `noul` per language — accuracy beside the label audit",
        "",
        "| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for lang in sorted(metrics_by_lang):
        metrics = metrics_by_lang[lang]
        row = audit_rows.get(lang, {})
        lines.append(
            f"| {lang} | {metrics.get('n', 0)} | {_fmt(metrics.get('accuracy', 0.0))} | "
            f"{_fmt(metrics.get('ece', 0.0))} | {row.get('request', 0)} | {row.get('neutral', 0)} "
            f"| {row.get('empty', 0)} | {row.get('unknown', 0)} | {row.get('contradictory', 0)} "
            f"({row.get('contradictory_rate', 0.0):.1%}) |"
        )
    lines.append("")
    return lines


def _gate_table(entry: ReportEntry) -> list[str]:
    """The `choice` gate beside the per-domain rows it routes (ADR-0016 consequences).

    A wrong gate can at worst fall back to the shared head, so the two harmless/ harmful
    outcomes are counted separately: ``fell to shared`` is today's behaviour, ``wrong domain``
    is a question answered with another domain's head. Absent when the report's asset ships
    no bank — which is every pre-ADR-0016 report.
    """
    gate = entry.report.get("choice_gate")
    if not isinstance(gate, Mapping) or not gate.get("n"):
        return []
    lines = [
        "#### `choice` gate",
        "",
        f"Answers routed by `{gate.get('answers_routed_by', 'gate')}` — strict accuracy "
        f"{_fmt(float(gate.get('strict_accuracy', 0.0)))}, fell to shared "
        f"{gate.get('fell_to_shared', 0)}, wrong domain {gate.get('wrong_domain', 0)} "
        f"of n={gate.get('n', 0)}.",
        "",
        "| Domain | n | Strict accuracy | Fell to shared | Wrong domain |",
        "| --- | --- | --- | --- | --- |",
    ]
    for domain, row in sorted((gate.get("per_domain") or {}).items()):
        lines.append(
            f"| {domain} | {row.get('n', 0)} | {_fmt(float(row.get('strict_accuracy', 0.0)))} | "
            f"{row.get('fell_to_shared', 0)} | {row.get('wrong_domain', 0)} |"
        )
    lines.append("")
    return lines


def _cross_table(entry: ReportEntry) -> list[str]:
    """Accuracy/ECE per ``domain/language`` cell, worst cell first (B-5b).

    ``per_domain`` hides the language spread inside a domain and ``per_language`` hides the
    domain spread inside a language; the project's worst-cell gate needs both at once. Absent
    when the report predates B5B-3 or the eval set carried a single domain.
    """
    cross = entry.report.get("per_domain_language")
    if not isinstance(cross, Mapping) or not cross:
        return []
    worst = _worst(cross)
    lines = [
        "#### domain x language",
        "",
        "| Cell | n | Accuracy | ECE |",
        "| --- | --- | --- | --- |",
    ]
    for cell, metrics in sorted(
        cross.items(), key=lambda item: (item[1].get("accuracy", 0.0), item[0])
    ):
        lines.append(
            f"| {cell} | {metrics.get('n', 0)} | {_fmt(metrics.get('accuracy', 0.0))} | "
            f"{_fmt(metrics.get('ece', 0.0))} |"
        )
    lines.append("")
    if worst is not None:
        cell, metrics = worst
        lines += [
            f"Worst cell (accuracy): `{cell}` — accuracy {_fmt(metrics.get('accuracy', 0.0))}, "
            f"ECE {_fmt(metrics.get('ece', 0.0))}.",
            "",
        ]
    return lines


def _text_seen_note(entry: ReportEntry) -> list[str]:
    """Accuracy split by whether the row's input text also occurs in training (B7).

    The shipped eval sets collide with their training sets on most rows, so this is the only
    line that says how much of the headline accuracy is generalization rather than recall.
    Absent when the run did not pass a training set — every report produced before B7.
    """
    split = entry.report.get("text_seen")
    if not isinstance(split, Mapping):
        return []
    seen = split.get("seen") or {}
    unseen = split.get("unseen") or {}
    if not isinstance(seen, Mapping) or not isinstance(unseen, Mapping):
        return []
    return [
        f"Input text seen in training: n={seen.get('n', 0)}, "
        f"accuracy {_fmt(float(seen.get('accuracy', 0.0)))} — never seen: "
        f"n={unseen.get('n', 0)}, accuracy {_fmt(float(unseen.get('accuracy', 0.0)))}.",
        "",
    ]


def _table(entry: ReportEntry) -> list[str]:
    lines = [
        f"### {entry.name}",
        "",
        "| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    scopes: list[tuple[str, dict[str, Any]]] = [("overall", entry.report["overall"])]
    scopes += sorted(entry.report.get("per_primitive", {}).items())
    scopes += [
        (f"lang:{lang}", metrics)
        for lang, metrics in sorted(entry.report.get("per_language", {}).items())
    ]
    domains = entry.report.get("per_domain") or {}
    # A single-domain report would just repeat `overall`, so the rows appear only for the
    # multi-domain datasets (B-5).
    if len(domains) > 1:
        scopes += [(f"domain:{domain}", metrics) for domain, metrics in sorted(domains.items())]
    for scope, metrics in scopes:
        latency = metrics.get("latency_ms", {})
        lines.append(
            f"| {scope} | {metrics.get('n', 0)} | {_fmt(metrics.get('accuracy', 0.0))} | "
            f"{_fmt(metrics.get('ece', 0.0))} | {_fmt(latency.get('p50', 0.0))} | "
            f"{_fmt(latency.get('p95', 0.0))} |"
        )
    worst = _worst_language(entry.report)
    if worst is not None:
        lang, metrics = worst
        lines += [
            "",
            f"Worst language (accuracy): `{lang}` — accuracy {_fmt(metrics.get('accuracy', 0.0))}, "
            f"ECE {_fmt(metrics.get('ece', 0.0))}.",
        ]
    worst_domain = _worst(domains) if len(domains) > 1 else None
    if worst_domain is not None:
        domain, metrics = worst_domain
        lines += [
            "",
            f"Worst domain (accuracy): `{domain}` — accuracy {_fmt(metrics.get('accuracy', 0.0))}, "
            f"ECE {_fmt(metrics.get('ece', 0.0))}.",
        ]
    noisy = entry.report.get("noisy")
    if isinstance(noisy, Mapping):
        noisy_overall = noisy.get("overall", {})
        latency = noisy_overall.get("latency_ms", {})
        lines += [
            "",
            f"Noisy view (noise_rate {entry.report.get('noise_rate', '?')}) — overall: "
            f"accuracy {_fmt(noisy_overall.get('accuracy', 0.0))}, "
            f"ECE {_fmt(noisy_overall.get('ece', 0.0))}, "
            f"p50 {_fmt(latency.get('p50', 0.0))} ms.",
        ]
    lines += _cross_table(entry)
    lines += _gate_table(entry)
    lines += _text_seen_note(entry)
    lines += _noul_table(entry)
    lines.append("")
    return lines


def render_report(
    entries: Sequence[ReportEntry],
    *,
    commands: Sequence[str] = (),
    title: str = "tachyone Benchmark Report",
    notes: str | None = None,
) -> str:
    """Render Markdown for the given entries, environment, and reproduction commands."""
    lines = [f"# {title}", ""]
    if notes:
        lines += [f"> {notes}", ""]
    lines += ["## Environment", "", "```json"]
    lines.append(json.dumps(environment(), indent=2, sort_keys=True, default=str))
    lines += ["```", ""]
    if commands:
        lines += ["## Reproduce", "", "```bash", *commands, "```", ""]
    lines += ["## Results", ""]
    for entry in entries:
        lines += _table(entry)
    lines += [
        "",
        "## See also",
        "",
        "The head-to-head against the open System One scorer and two local LLMs — accuracy, ECE, "
        "Brier, latency, throughput, memory and contract compliance on two evaluation sets — is "
        "published in [`docs/compare.md`]"
        "(../docs/compare.md#3-why-not-another-open-system-one-scorer) and reproduced from "
        "`benchmarks/README.md`.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_report(markdown: str, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-benchmark-report", description=__doc__)
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        metavar="NAME=REPORT.json",
        help="a labeled evaluation report (repeatable)",
    )
    parser.add_argument("--out", required=True, help="output Markdown path")
    parser.add_argument("--title", default="tachyone Benchmark Report")
    parser.add_argument("--note", default=None)
    return parser


def _parse_entries(specs: Iterable[str]) -> list[ReportEntry]:
    entries: list[ReportEntry] = []
    for spec in specs:
        # Split on the *last* ``=``: entry names legitimately contain one (``LoRA r=16``),
        # and the report's own section titles do.
        name, sep, path = spec.rpartition("=")
        if not sep or not name or not path:
            raise ValueError(f"--entry expects NAME=REPORT.json, got {spec!r}")
        entries.append(ReportEntry.from_files(name, path))
    return entries


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = render_report(
        _parse_entries(args.entry),
        commands=[
            "uv run python -m training.generate_data --seed 1 --per-type 3000 --languages en "
            "--out data/train_en.jsonl",
            "uv run python -m training.generate_data --seed 1 --per-type 6000 "
            "--languages pt,es,fr,de,it,nl --out data/train_multi.jsonl",
            "uv run python -m training.generate_data --seed 2 --per-type 500 --languages en "
            "--out data/eval_en.jsonl",
            "uv run python -m training.generate_data --seed 2 --per-type 500 "
            "--languages pt,es,fr,de,it,nl --out data/eval_multi.jsonl",
            "uv run python -m training.finetune_rlcd --config training/configs/finetune_en.json",
            "uv run python -m training.predict --data data/eval_en.jsonl --adapter checkpoints/en "
            "--out-predictions data/preds_en.jsonl",
            "uv run python -m training.fit_calibration --calibration data/preds_en.jsonl "
            "--out checkpoints/en/temperature_calibration.json",
            # B-11: both evaluates pin the local adapter (the Hub id would fetch a revision) and
            # carry --noise-rate, without which the noisy view the report prints is not produced.
            "TACHYONE_ADAPTERS=tachyone-en=checkpoints/en "
            "uv run python -m training.evaluate --data data/eval_en.jsonl "
            "--out benchmarks/results/en_split.json --backend encoder --noise-rate 0.15 "
            '--models-dir "$HOME/.cache/tachyone/models"',
            "TACHYONE_ADAPTERS=tachyone-multi=checkpoints/multi "
            "uv run python -m training.evaluate --data data/eval_multi.jsonl "
            "--out benchmarks/results/multi_split.json --backend encoder --noise-rate 0.15 "
            '--models-dir "$HOME/.cache/tachyone/models"',
            # B-5a: five domains, English, support keeping its published volume
            "uv run python -m training.generate_data --seed 1 --per-type 1000 --languages en "
            "--domains support,ecommerce,agent_tools,documents,voice --per-domain support=3000 "
            "--out data/train_en_domains.jsonl",
            "uv run python -m training.generate_data --seed 2 --per-type 500 --languages en "
            "--domains support,ecommerce,agent_tools,documents,voice "
            "--out data/eval_en_domains.jsonl",
            "uv run python -m training.finetune_rlcd "
            "--config training/configs/finetune_en_domains.json",
            "uv run python -m training.predict --data data/eval_en_domains.jsonl "
            "--adapter checkpoints/en_domains --out-predictions data/preds_en_domains.jsonl",
            "uv run python -m training.fit_calibration --calibration data/preds_en_domains.jsonl "
            "--out checkpoints/en_domains/temperature_calibration.json",
            "TACHYONE_ADAPTERS=tachyone-en=checkpoints/en_domains "
            "uv run python -m training.evaluate "
            "--data data/eval_en_domains.jsonl --out benchmarks/results/en_domains.json "
            "--backend encoder",
            # B-5: five-domain bank, fitted on a frozen trunk and evaluated through the gate
            "uv run python -m training.fit_choice_bank "
            "--config training/configs/fit_bank_en_domains.json",
            "uv run python -m training.predict --data data/eval_en_domains.jsonl "
            "--adapter checkpoints/en_domains_bank --train-data data/train_en_domains.jsonl "
            "--out-predictions data/preds_en_domains_bank_gate.jsonl "
            "--out-report benchmarks/results/en_domains_bank_gate.json",
            # B-5b: five domains, multilingual (support keeps its 18k), joint run + frozen-trunk
            # fit; the promoted checkpoints/multi is the fitted bank (fit_bank -> mv multi).
            "uv run python -m training.generate_data --seed 1 --per-type 1000 "
            "--languages pt,es,fr,de,it,nl "
            "--domains support,ecommerce,agent_tools,documents,voice --per-domain support=6000 "
            "--out data/train_multi_domains.jsonl",
            "uv run python -m training.generate_data --seed 2 --per-type 500 "
            "--languages pt,es,fr,de,it,nl "
            "--domains support,ecommerce,agent_tools,documents,voice "
            "--out data/eval_multi_domains.jsonl",
            "uv run python -m training.finetune_rlcd "
            "--config training/configs/finetune_multi_domains.json",
            "uv run python -m training.fit_choice_bank "
            "--config training/configs/fit_bank_multi_domains.json",
            "uv run python -m training.predict --data data/eval_multi_domains.jsonl "
            "--adapter checkpoints/multi_b5b_fit_bank --train-data data/train_multi_domains.jsonl "
            "--out-predictions data/preds_multi_domains_fit_bank.jsonl "
            "--out-report benchmarks/results/multi_domains_fit_bank.json",
            "uv run python -m training.fit_calibration "
            "--calibration data/preds_multi_domains_fit_bank.jsonl "
            "--out benchmarks/results/calibration_multi_domains_fit_bank.json",
            # B-13: JevBench-family data (pinned sources + rule-tree families + mixture),
            # two-arm training, frozen-trunk fit; the promoted checkpoints/en is the treatment
            # bank (fit_bank -> mv checkpoints/en_jev_bank checkpoints/en).
            "uv run python -m training.fetch_jev_sources",
            "uv run python -m training.build_jev_sources "
            "--public-dir <jevbench-clone>/datasets/public",
            "uv run python -m training.build_jev_families "
            "--public-dir <jevbench-clone>/datasets/public",
            "uv run python -m training.build_jev_mixture",
            "uv run python -m training.finetune_rlcd "
            "--config training/configs/finetune_en_jev.json",
            "uv run python -m training.fit_choice_bank "
            "--config training/configs/fit_bank_en_jev.json",
            "uv run python -m training.predict --data data/eval_en_domains.jsonl "
            "--adapter checkpoints/en_jev_bank --train-data data/train_en_domains.jsonl "
            "--out-predictions data/preds_en_jev_treat.jsonl "
            "--out-report benchmarks/results/en_jev_treat_raw.json",
            "uv run python -m training.fit_calibration "
            "--calibration data/preds_en_jev_treat.jsonl "
            "--out benchmarks/results/calibration_en_jev_treat.json",
            "uv run python -m training.predict --data data/eval_en_domains.jsonl "
            "--adapter checkpoints/en_jev_bank "
            "--temperature benchmarks/results/calibration_en_jev_treat.json "
            "--train-data data/train_en_domains.jsonl "
            "--out-predictions data/preds_en_jev_treat_cal.jsonl "
            "--out-report benchmarks/results/en_jev_treat.json",
            "TACHYONE_ADAPTERS=tachyone-en=checkpoints/en "
            "uv run python -m training.evaluate --data data/eval_en.jsonl "
            "--out benchmarks/results/en_split_jev.json --backend encoder --noise-rate 0.15 "
            '--models-dir "$HOME/.cache/tachyone/models"',
            "uv run python -m benchmarks.report "
            '--entry "english (ModernBERT-large + LoRA r=16 + choice head)'
            '=benchmarks/results/en_split_jev.json" '
            '--entry "english five-domain (B-13 JevBench-family fitted bank, frozen trunk)'
            '=benchmarks/results/en_jev_treat.json" '
            '--entry "multilingual (mmBERT-base + LoRA r=64 + choice head)'
            '=benchmarks/results/multi_split.json" '
            '--entry "multilingual five-domain (B-5b fitted choice-head bank, frozen trunk)'
            '=benchmarks/results/multi_domains_fit_bank.json" '
            "--out benchmarks/report.md",
        ],
        title=args.title,
        notes=args.note,
    )
    write_report(report, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["ReportEntry", "render_report", "write_report"]
