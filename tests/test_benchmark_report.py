"""Tests for the benchmark report renderer (M6-T1, OPS-06)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.report import ReportEntry, _parse_entries, main, render_report, write_report
from jeba.primitives import (
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
from training.evaluate import evaluate, load_examples, save_report
from training.generate_data import DataConfig, generate


def _predictor(state: State, question: Question) -> Answer:
    del state
    if isinstance(question, NoulQuestion):
        return NoulAnswer(noul=0.6)
    keys = list(question.criteria)
    if isinstance(question, ChoiceQuestion):
        return ChoiceAnswer(
            choice=str(keys[0]),
            probabilities={key: (1.0 if key == keys[0] else 0.0) for key in keys},
            confidence=1.0,
        )
    assert isinstance(question, ScoreQuestion)
    return ScoreAnswer(
        score=0.0,
        legend=dict(enumerate(keys)),
        probabilities={index: (1.0 if index == 0 else 0.0) for index in range(len(keys))},
        confidence=1.0,
    )


def _evaluation_report(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "eval.jsonl"
    generate(DataConfig(seed=5, per_type=4, languages=("en",)), path)
    return evaluate(load_examples(path), _predictor)


def test_render_report_contains_tables_and_sections(tmp_path: Path) -> None:
    report = _evaluation_report(tmp_path)
    markdown = render_report(
        [ReportEntry(name="encoder", report=report)],
        commands=["uv run python -m training.evaluate --data data/eval.jsonl"],
        notes="pending run",
    )
    assert "# jeba Benchmark Report" in markdown
    assert "> pending run" in markdown
    assert "## Environment" in markdown
    assert "## Reproduce" in markdown
    assert "### encoder" in markdown
    assert "| overall |" in markdown
    assert "| noul |" in markdown
    assert "| lang:en |" in markdown
    assert "Worst language" in markdown


def test_render_report_renders_noisy_view(tmp_path: Path) -> None:
    report = _evaluation_report(tmp_path)
    report["noise_rate"] = 0.3
    report["noisy"] = evaluate(load_examples(tmp_path / "eval.jsonl"), _predictor)
    markdown = render_report([ReportEntry(name="encoder", report=report)])
    assert "Noisy view (noise_rate 0.3)" in markdown
    assert "accuracy" in markdown


def test_report_entry_from_files_and_write(tmp_path: Path) -> None:
    json_path = tmp_path / "encoder.json"
    save_report(_evaluation_report(tmp_path), json_path)
    entry = ReportEntry.from_files("encoder", json_path)
    assert entry.name == "encoder"
    out = tmp_path / "report.md"
    write_report(render_report([entry]), out)
    assert "| overall |" in out.read_text(encoding="utf-8")


def test_parse_entries_requires_name_and_path(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_evaluation_report(tmp_path)), encoding="utf-8")
    assert _parse_entries([f"encoder={path}"])[0].name == "encoder"
    with pytest.raises(ValueError, match=r"NAME=REPORT\.json"):
        _parse_entries(["just-a-path.json"])


def test_cli_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_evaluation_report(tmp_path)), encoding="utf-8")
    out = tmp_path / "report.md"
    assert main(["--entry", f"encoder={path}", "--out", str(out)]) == 0
    assert out.exists()
    assert "wrote" in capsys.readouterr().out
