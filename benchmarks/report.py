"""Render a reproducible benchmark report from evaluation artifacts.

``training/evaluate.py`` writes JSON reports; this script aggregates them into Markdown with
the exact reproduction commands and the captured environment (OPS-06, NFR-D04). It is pure, so
the report can be regenerated without a GPU; real numbers are produced by the actual backends.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
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


def _table(entry: ReportEntry) -> list[str]:
    lines = [
        f"### {entry.name}",
        "",
        "| Scope | n | Accuracy | ECE | p50 (ms) | p95 (ms) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    scopes: list[tuple[str, dict[str, Any]]] = [("overall", entry.report["overall"])]
    scopes += sorted(entry.report.get("per_primitive", {}).items())
    for scope, metrics in scopes:
        latency = metrics.get("latency_ms", {})
        lines.append(
            f"| {scope} | {metrics.get('n', 0)} | {_fmt(metrics.get('accuracy', 0.0))} | "
            f"{_fmt(metrics.get('ece', 0.0))} | {_fmt(latency.get('p50', 0.0))} | "
            f"{_fmt(latency.get('p95', 0.0))} |"
        )
    lines.append("")
    return lines


def render_report(
    entries: Sequence[ReportEntry],
    *,
    commands: Sequence[str] = (),
    title: str = "jeba Benchmark Report",
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
    return "\n".join(lines).rstrip() + "\n"


def write_report(markdown: str, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jeba-benchmark-report", description=__doc__)
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        metavar="NAME=REPORT.json",
        help="a labeled evaluation report (repeatable)",
    )
    parser.add_argument("--out", required=True, help="output Markdown path")
    parser.add_argument("--title", default="jeba Benchmark Report")
    parser.add_argument("--note", default=None)
    return parser


def _parse_entries(specs: Iterable[str]) -> list[ReportEntry]:
    entries: list[ReportEntry] = []
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--entry expects NAME=REPORT.json, got {spec!r}")
        entries.append(ReportEntry.from_files(name, path))
    return entries


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = render_report(
        _parse_entries(args.entry),
        commands=[
            "uv run python -m training.generate_data --out data/eval.jsonl",
            "uv run python -m training.evaluate --data data/eval.jsonl "
            "--out benchmarks/results/eval.json --backend encoder",
            "uv run python -m benchmarks.report "
            "--entry encoder=benchmarks/results/eval.json --out benchmarks/report.md",
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
