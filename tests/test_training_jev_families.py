"""Tests for the JevBench-family generator (B-13 P2, JB-4).

The load-bearing property: **every generated target re-executes from the record's own
shipped facts + tree**. If that identity ever breaks, the labels came from somewhere
other than the text (B-11/B-12/L-008 territory). Also covered: label balance (L-010 —
audit the distribution before training on it), the 512-token state window (L-014),
determinism, the public-item overlap refusal, and wire validity of the shapes.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from tachyone.primitives import ChoiceQuestion, NoulQuestion
from training.build_jev_families import (
    DEFAULT_CONFIG,
    DEFAULT_CONTENT_DIR,
    FAMILIES,
    LONG_POLICY_MIN_CHARS,
    MAX_STATE_CHARS,
    BuildError,
    add_months_clamped,
    build,
    business_days,
    draw_facts,
    load_config,
    load_content,
    record_rng,
    render_fact,
    tree_fact_names,
)
from training.rule_trees import reexecute

JEV_PUBLIC = Path("/tmp/opencode/jevbench/datasets/public")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One full build, shared by the read-only assertions below."""
    out = tmp_path_factory.mktemp("families") / "train_jev_families.jsonl"
    report = build(out_path=out)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return report, records


def test_config_is_strict_and_names_every_family() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert isinstance(config["seed"], int)
    assert set(config["counts"]) == set(FAMILIES)
    assert all(count > 0 for count in config["counts"].values())


def test_every_committed_content_file_validates() -> None:
    content = load_content(DEFAULT_CONTENT_DIR)
    assert set(content) == set(FAMILIES)
    assert len(content["trap"]["cases"]) >= 6
    assert len(content["long_policy"]["routing"]["bands"]) == 5
    assert len(content["adequacy"]["requests"]) >= 5
    cases = content["counterfactual"]["cases"]
    assert len(cases) >= 12
    assert {case["shape"] for case in cases} == {
        "noul_policy",
        "noul_defined_term",
        "choice_band",
    }
    for case in cases:
        assert case["base"]["tree"] != case["counterfactual"]["tree"], case["key"]
        if case["shape"] == "choice_band":
            assert set(case["criteria"]) == {
                "approve",
                "decline",
                "escalate",
                "request_info",
            }, case["key"]
    trap_cases = content["policy_trap"]["cases"]
    assert len(trap_cases) >= 6
    assert len(content["policy_trap"]["outcomes"]) >= 8
    for case in trap_cases:
        assert len(case["pool"]) >= 3, case["key"]
        assert case["noise"], case["key"]
    assert len(content["tradeoff"]["themes"]) >= 12
    assert len(content["tradeoff"]["actions"]) >= 20


def test_add_months_clamped_handles_month_end_and_leap_years() -> None:
    assert add_months_clamped(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months_clamped(date(2028, 1, 31), 1) == date(2028, 2, 29)  # leap year
    assert add_months_clamped(date(2026, 5, 31), 1) == date(2026, 6, 30)
    assert add_months_clamped(date(2026, 1, 31), 12) == date(2027, 1, 31)
    assert add_months_clamped(date(2027, 8, 31), 6) == date(2028, 2, 29)
    assert add_months_clamped(date(2026, 12, 15), 1) == date(2027, 1, 15)


def test_business_days_subtracts_one_weekend_and_never_goes_negative() -> None:
    assert business_days(7, True) == 5
    assert business_days(7, False) == 7
    assert business_days(1, True) == 0


def test_draw_facts_needs_derived_deps_declared_first() -> None:
    spec = {
        "effective": {"kind": "derived", "formula": "business_days", "deps": ["elapsed"]},
        "elapsed": {"kind": "int", "min": 3, "max": 9, "template": "{elapsed} days"},
    }
    with pytest.raises(BuildError, match="drawn first"):
        draw_facts(spec, random_for_test())


def random_for_test() -> Any:
    import random

    return random.Random("deps")


def test_loader_rejects_unknown_formulas_and_bad_fact_kinds(tmp_path: Path) -> None:
    content = json.loads((DEFAULT_CONTENT_DIR / "trap.json").read_text(encoding="utf-8"))
    content["cases"][0]["facts"] = {
        "x": {"kind": "derived", "formula": "not_a_formula", "deps": ["y"]}
    }
    directory = tmp_path / "jev"
    directory.mkdir()
    for family in FAMILIES:
        source = DEFAULT_CONTENT_DIR / f"{family}.json"
        target = directory / source.name
        if family == "trap":
            target.write_text(json.dumps(content), encoding="utf-8")
        else:
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(BuildError, match="unknown formula"):
        load_content(directory)


def test_render_fact_quotes_the_fact_and_refuses_derived() -> None:
    node = {"kind": "int", "min": 1, "max": 9, "template": "on day {days}"}
    assert render_fact("days", node, 5) == "on day 5"
    assert (
        render_fact("ok", {"kind": "bool", "yes": "yes text", "no": "no text"}, False) == "no text"
    )
    with pytest.raises(BuildError, match="must not be rendered"):
        render_fact("z", {"kind": "derived"}, 1)
    with pytest.raises(BuildError, match="template must contain"):
        render_fact("days", {"kind": "int", "min": 1, "max": 2, "template": "{other}"}, 1)


def test_tree_fact_names_walks_every_op() -> None:
    tree = {
        "op": "select",
        "when": [
            [
                {
                    "op": "all",
                    "args": [
                        {"op": "fact", "name": "a"},
                        {"op": "not", "arg": {"op": "fact", "name": "b"}},
                    ],
                },
                {"op": "const", "value": "x"},
            ],
            [
                {
                    "op": "cmp",
                    "cmp": "<",
                    "left": {"op": "fact", "name": "c"},
                    "right": {"op": "const", "value": 1},
                },
                {"op": "fact", "name": "d"},
            ],
        ],
        "else": {"op": "fact", "name": "e"},
    }
    assert tree_fact_names(tree) == {"a", "b", "c", "d", "e"}


def test_counts_match_the_config(built: tuple[dict[str, Any], list[dict[str, Any]]]) -> None:
    report, records = built
    config = load_config(DEFAULT_CONFIG)
    assert report["total"] == sum(config["counts"].values()) == len(records)
    for family, count in config["counts"].items():
        assert report["per_family"][family] == count
        assert (
            sum(1 for record in records if record["source"] == f"synthetic:jev:{family}") == count
        )


def test_every_target_reexecutes_from_the_shipped_rule(
    built: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    """The whole point: target == evaluate(tree, facts), for all 3,540 records."""
    _, records = built
    for record in records:
        assert reexecute(record) == record["target"], record["id"]
        assert set(record["rule"]["facts"]) >= _needed(record["rule"]["tree"])
        if record["type"] == "choice":
            ChoiceQuestion(instructions=record["instructions"], criteria=record["criteria"])
            assert record["target"] in record["criteria"]
        else:
            NoulQuestion(instructions=record["instructions"], criteria=record["criteria"])
            assert record["target"] in (0, 1)


def _needed(tree: dict[str, Any]) -> set[str]:
    return tree_fact_names(tree)


def test_label_distribution_is_balanced_enough_to_train_on(
    built: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    """L-010: no label space may ship a class the model could win with a prior."""
    report, records = built
    for family in FAMILIES:
        counts: dict[str, int] = {}
        for record in records:
            if record["source"] == f"synthetic:jev:{family}":
                counts[str(record["target"])] = counts.get(str(record["target"]), 0) + 1
        for space in _label_spaces(family, counts):
            total = sum(space.values())
            assert len(space) >= 2, family
            # Small spaces keep the 10% prior-proof floor; drawn-arity spaces (W1.4's many
            # options) must hold at least 40% of the uniform share instead.
            floor = 0.10 if len(space) <= 6 else 0.4 / len(space)
            for label, count in space.items():
                assert count / total >= floor, f"{family}/{label} is only {count}/{total}"
    # the two exhaustive choice shapes are pinned exactly: five routing bands and the
    # three settlement outcomes.
    routing = {
        label: count
        for label, count in report["target_distribution"]["long_policy"].items()
        if label.startswith("level")
    }
    assert len(routing) == 5 and min(routing.values()) >= 50
    payout = {
        label: count
        for label, count in report["target_distribution"]["temporal_numeric"].items()
        if label.startswith(("decline", "pay_"))
    }
    assert len(payout) == 3 and min(payout.values()) >= 50


def _label_spaces(family: str, counts: dict[str, int]) -> list[dict[str, int]]:
    """Some families mix two independent label spaces; balance is judged inside each one."""
    if family == "long_policy":
        routing = {key: value for key, value in counts.items() if key.startswith("level")}
        coverage = {key: value for key, value in counts.items() if not key.startswith("level")}
        return [routing, coverage]
    if family == "temporal_numeric":
        binary = {key: value for key, value in counts.items() if key in ("0", "1")}
        settlement = {key: value for key, value in counts.items() if key not in ("0", "1")}
        return [binary, settlement]
    if family == "counterfactual":
        binary = {key: value for key, value in counts.items() if key in ("0", "1")}
        choice = {key: value for key, value in counts.items() if key not in ("0", "1")}
        return [binary, choice]
    return [counts]


def test_states_respect_the_truncation_window(
    built: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    """L-014: the decisive text must live inside the 512-token runtime window."""
    _, records = built
    longest = max(records, key=lambda record: len(record["state"]))
    assert len(longest["state"]) <= MAX_STATE_CHARS
    long_states = [
        len(record["state"])
        for record in records
        if record["source"] == "synthetic:jev:long_policy"
    ]
    assert min(long_states) >= LONG_POLICY_MIN_CHARS


def test_states_fit_the_real_tokenizer(built: tuple[dict[str, Any], list[dict[str, Any]]]) -> None:
    """The precise version of the char cap: tokenize every state when transformers is here."""
    try:
        from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]
    except ImportError:
        pytest.skip("transformers (train extra) not installed")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-large", local_files_only=True
        )
    except Exception:
        pytest.skip("ModernBERT tokenizer not cached on this machine")
    _, records = built
    counts = [len(tokenizer(record["state"], truncation=False)["input_ids"]) for record in records]
    assert max(counts) <= 512, f"longest state is {max(counts)} tokens"


def test_trap_records_ship_the_surface_answer_and_actually_trap(
    built: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    _, records = built
    traps = [record for record in records if record["source"] == "synthetic:jev:trap"]
    assert traps
    differing = 0
    for record in traps:
        surface = record["rule"]["surface_answer"]
        assert surface in (0, 1)
        assert "note" in record["state"]
        if int(record["target"]) != int(surface):
            differing += 1
    assert differing >= 100, "the agent note must disagree with the defined answer often enough"


def test_counterfactual_pairs_flip_and_share_their_facts(
    built: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    """The S1b lesson is the flip: same drawn facts, one rubric clause apart, other answer."""
    _, records = built
    pairs: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["source"] == "synthetic:jev:counterfactual":
            pairs.setdefault(record["pair"]["id"], []).append(record)
    assert len(pairs) >= 3000
    for pair_id, members in pairs.items():
        assert len(members) == 2, pair_id
        base = next(record for record in members if record["pair"]["role"] == "base")
        cf = next(record for record in members if record["pair"]["role"] == "counterfactual")
        assert base["pair"]["change"] == cf["pair"]["change"]
        assert base["pair"]["change"].strip(), pair_id
        assert base["rule"]["facts"] == cf["rule"]["facts"], pair_id
        assert base["target"] != cf["target"], pair_id
        assert base["id"] != cf["id"], pair_id
        assert (base["state"], base["instructions"]) != (cf["state"], cf["instructions"]), pair_id
        for record in (base, cf):
            if record["type"] == "choice":
                assert record["target"] in record["criteria"]
            else:
                assert int(record["target"]) in (0, 1)
                assert int(record.get("surface_answer", record["target"])) in (0, 1)


def test_policy_trap_draws_arity_permutes_and_ships_the_tempting_answer(
    built: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    """Provenance is the lesson: the noise tempts, the trusted facts decide (W1.3/W1.4)."""
    _, records = built
    traps = [record for record in records if record["source"] == "synthetic:jev:policy_trap"]
    assert traps
    content = load_content(DEFAULT_CONTENT_DIR)
    outcomes = set(content["policy_trap"]["outcomes"])
    sizes: set[int] = set()
    target_positions: set[int] = set()
    biting = 0
    for record in traps:
        assert record["type"] == "choice"
        assert record["target"] in record["criteria"]
        assert set(record["criteria"]) <= outcomes, record["id"]
        sizes.add(len(record["criteria"]))
        target_positions.add(list(record["criteria"]).index(record["target"]))
        surface = record["surface_answer"]
        assert surface in outcomes, record["id"]
        if surface != record["target"]:
            biting += 1
    assert min(sizes) >= 3 and max(sizes) >= 6, sizes
    assert len(target_positions) >= 3, "option order must be permuted (W1.4)"
    assert biting >= len(traps) // 3, "the tempting text must disagree with the target often"


def test_tradeoff_covers_the_many_option_regime_and_the_default(
    built: tuple[dict[str, Any], list[dict[str, Any]]],
) -> None:
    """Precedence with drawn arity 2..20; the no-rule default wins some records (L-010)."""
    _, records = built
    rows = [record for record in records if record["source"] == "synthetic:jev:tradeoff"]
    assert rows
    content = load_content(DEFAULT_CONTENT_DIR)
    default_key = str(content["tradeoff"]["no_rule_action"]["key"])
    pool = {str(action["key"]) for action in content["tradeoff"]["actions"]} | {default_key}
    sizes: set[int] = set()
    target_positions: set[int] = set()
    defaults = 0
    for record in rows:
        assert record["target"] in record["criteria"]
        assert set(record["criteria"]) <= pool, record["id"]
        sizes.add(len(record["criteria"]))
        target_positions.add(list(record["criteria"]).index(record["target"]))
        defaults += record["target"] == default_key
    assert min(sizes) >= 2 and max(sizes) >= 15, sizes
    assert len(target_positions) >= 3, "option order must be permuted (W1.4)"
    assert defaults >= 100, "the no-rule default must draw as a winner (L-010)"


def test_build_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    build(out_path=first)
    build(out_path=second)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731
    assert digest(first) == digest(second)


def test_build_refuses_a_collision_with_a_public_item(tmp_path: Path) -> None:
    content = load_content(DEFAULT_CONTENT_DIR)
    record = _single_trap_record(content)
    public = tmp_path / "public"
    public.mkdir()
    (public / "original.jsonl").write_text(
        json.dumps({"state": record["state"], "question": {"instructions": record["instructions"]}})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(BuildError, match="evaluation-only"):
        build(out_path=tmp_path / "out.jsonl", public_dir=public)


def _single_trap_record(content: dict[str, Any]) -> dict[str, Any]:
    from training.build_jev_families import _trap_record

    return _trap_record(0, record_rng(1, "trap", 0), content)


@pytest.mark.skipif(not JEV_PUBLIC.exists(), reason="jevbench public items not cloned here")
def test_real_public_items_produce_zero_collisions(tmp_path: Path) -> None:
    report = build(out_path=tmp_path / "out.jsonl", public_dir=JEV_PUBLIC)
    assert report["checked_against_public"] is True
