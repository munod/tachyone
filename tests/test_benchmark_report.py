"""Tests for the benchmark report renderer (M6-T1, OPS-06)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.report import ReportEntry, _parse_entries, main, render_report, write_report
from tachyone.primitives import (
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
    assert "# tachyone Benchmark Report" in markdown
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


def test_parse_entries_splits_on_last_equals(tmp_path: Path) -> None:
    """Entry names contain ``=`` (``LoRA r=16``); only the final one separates the path."""
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_evaluation_report(tmp_path)), encoding="utf-8")
    entry = _parse_entries([f"english (ModernBERT-large + LoRA r=16 + choice head)={path}"])[0]
    assert entry.name == "english (ModernBERT-large + LoRA r=16 + choice head)"
    assert entry.report["overall"]["n"] > 0
    with pytest.raises(ValueError, match=r"NAME=REPORT\.json"):
        _parse_entries(["no-path-here"])


def test_cli_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_evaluation_report(tmp_path)), encoding="utf-8")
    out = tmp_path / "report.md"
    assert main(["--entry", f"encoder={path}", "--out", str(out)]) == 0
    assert out.exists()
    assert "wrote" in capsys.readouterr().out


def _multi_domain_report(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "eval_domains.jsonl"
    generate(DataConfig(seed=5, per_type=4, languages=("en",), domains=("support", "voice")), path)
    return evaluate(load_examples(path), _predictor)


def _multi_language_domain_report(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "eval_multi_lang_domains.jsonl"
    generate(
        DataConfig(seed=5, per_type=4, languages=("en", "pt"), domains=("support", "voice")),
        path,
    )
    return evaluate(load_examples(path), _predictor)


def test_render_report_publishes_the_domain_language_cross_table(tmp_path: Path) -> None:
    """B5B-03: worst cell first, so the worst-cell gate has one line to read (B-5b)."""
    markdown = render_report(
        [ReportEntry(name="encoder", report=_multi_language_domain_report(tmp_path))]
    )
    assert "#### domain x language" in markdown
    for cell in ("support/en", "support/pt", "voice/en", "voice/pt"):
        assert f"| {cell} |" in markdown
    assert "Worst cell (accuracy)" in markdown
    # Worst cell first: data rows (cells carry a "/", the header does not) sort by accuracy.
    table_body = markdown.split("#### domain x language", 1)[1].split("Worst cell", 1)[0]
    rows = [
        line
        for line in table_body.splitlines()
        if line.startswith("| ") and "/" in line.split("|")[1]
    ]
    assert len(rows) == 4
    accuracies = [float(line.split("|")[3]) for line in rows]
    assert accuracies == sorted(accuracies)


def test_render_report_omits_the_cross_table_for_single_domain_reports(tmp_path: Path) -> None:
    markdown = render_report([ReportEntry(name="encoder", report=_evaluation_report(tmp_path))])
    assert "domain x language" not in markdown
    assert "Worst cell" not in markdown


def test_render_report_adds_domain_rows_only_for_multi_domain_reports(tmp_path: Path) -> None:
    """Single-domain reports must not grow a row that just repeats `overall` (B-5)."""
    single = render_report([ReportEntry(name="encoder", report=_evaluation_report(tmp_path))])
    assert "| domain:support |" not in single
    assert "Worst domain" not in single

    multi = render_report([ReportEntry(name="encoder", report=_multi_domain_report(tmp_path))])
    assert "| domain:support |" in multi
    assert "| domain:voice |" in multi
    assert "Worst domain (accuracy)" in multi


def test_render_report_puts_noul_accuracy_beside_the_label_audit(tmp_path: Path) -> None:
    """B-11: per-language `noul` accuracy and the contradictory-label rate in one table."""
    markdown = render_report([ReportEntry(name="encoder", report=_evaluation_report(tmp_path))])
    assert "accuracy beside the label audit" in markdown
    assert (
        "| Lang | n | Accuracy | ECE | request | neutral | empty | unknown | Contradictory |"
        in markdown
    )
    assert "| en |" in markdown
    assert "(0.0%)" in markdown  # generated labels are not contradictory (B-11)


def test_render_report_omits_the_noul_table_when_the_audit_is_absent() -> None:
    """Reports produced before B-11 have no audit and must render unchanged."""
    markdown = render_report([ReportEntry(name="old", report={"overall": {"n": 10}})])
    assert "accuracy beside the label audit" not in markdown


# --- B6: the gate row beside the per-domain numbers it routes (ADR-0016) ------------------


def _report_with_choice_gate(tmp_path: Path) -> dict[str, Any]:
    report = _multi_domain_report(tmp_path)
    report["choice_gate"] = {
        "n": 8,
        "strict": 6,
        "fell_to_shared": 1,
        "wrong_domain": 1,
        "strict_accuracy": 0.75,
        "answers_routed_by": "gate",
        "per_domain": {
            "support": {
                "n": 4,
                "strict": 4,
                "fell_to_shared": 0,
                "wrong_domain": 0,
                "strict_accuracy": 1.0,
            },
            "voice": {
                "n": 4,
                "strict": 2,
                "fell_to_shared": 1,
                "wrong_domain": 1,
                "strict_accuracy": 0.5,
            },
        },
    }
    return report


def test_render_report_publishes_the_choice_gate(tmp_path: Path) -> None:
    markdown = render_report([ReportEntry(name="bank", report=_report_with_choice_gate(tmp_path))])
    assert "#### `choice` gate" in markdown
    assert "Answers routed by `gate`" in markdown
    assert "strict accuracy 0.750" in markdown
    assert "fell to shared 1" in markdown
    assert "wrong domain 1" in markdown
    assert "| support | 4 | 1.000 | 0 | 0 |" in markdown
    assert "| voice | 4 | 0.500 | 1 | 1 |" in markdown


def test_render_report_names_the_oracle_when_the_hint_routed_the_answers(tmp_path: Path) -> None:
    report = _report_with_choice_gate(tmp_path)
    report["choice_gate"]["answers_routed_by"] = "domain"
    markdown = render_report([ReportEntry(name="bank", report=report)])
    assert "Answers routed by `domain`" in markdown


def test_render_report_omits_the_gate_row_when_the_asset_ships_no_bank(tmp_path: Path) -> None:
    """Every pre-ADR-0016 report renders exactly as before."""
    markdown = render_report([ReportEntry(name="encoder", report=_evaluation_report(tmp_path))])
    assert "#### `choice` gate" not in markdown


def test_render_report_splits_accuracy_by_seen_input_text(tmp_path: Path) -> None:
    report = _evaluation_report(tmp_path)
    report["text_seen"] = {
        "seen": {"n": 10, "accuracy": 0.9, "ece": 0.0},
        "unseen": {"n": 5, "accuracy": 0.6, "ece": 0.0},
    }
    markdown = render_report([ReportEntry(name="encoder", report=report)])
    assert (
        "Input text seen in training: n=10, accuracy 0.900 — never seen: n=5, accuracy 0.600."
        in markdown
    )


def test_render_report_omits_the_seen_text_note_without_a_training_set(tmp_path: Path) -> None:
    markdown = render_report([ReportEntry(name="encoder", report=_evaluation_report(tmp_path))])
    assert "Input text seen in training" not in markdown
