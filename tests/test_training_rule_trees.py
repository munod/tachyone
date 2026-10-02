"""Tests for the executable rule trees (B-13 P2, JB-3).

The trees are the label engine for the JevBench-shaped families: if the executor is
wrong, every synthetic label is wrong with it. Cases cover the ops, the boundary
comparisons the hard families live on, the trap shape, and the failure modes.
"""

from __future__ import annotations

import pytest

from training.rule_trees import COMPARISONS, TreeError, as_target, evaluate, reexecute


def test_const_and_fact_nodes() -> None:
    facts = {"days_ago": 12, "has_receipt": False}
    assert evaluate({"op": "const", "value": "deny"}, facts) == "deny"
    assert evaluate({"op": "fact", "name": "days_ago"}, facts) == 12
    with pytest.raises(TreeError, match="not bound"):
        evaluate({"op": "fact", "name": "absent"}, facts)


def test_all_any_not_short_circuit_semantics() -> None:
    facts = {"a": True, "b": False}
    all_node = {"op": "all", "args": [{"op": "fact", "name": "a"}, {"op": "fact", "name": "b"}]}
    any_node = {"op": "any", "args": [{"op": "fact", "name": "a"}, {"op": "fact", "name": "b"}]}
    assert evaluate(all_node, facts) is False
    assert evaluate(any_node, facts) is True
    assert evaluate({"op": "not", "arg": all_node}, facts) is True


def test_boolean_context_rejects_non_booleans() -> None:
    with pytest.raises(TreeError, match="boolean context"):
        evaluate({"op": "all", "args": [{"op": "fact", "name": "x"}]}, {"x": 3})
    with pytest.raises(TreeError, match="boolean context"):
        evaluate({"op": "not", "arg": {"op": "const", "value": "yes"}}, {})


@pytest.mark.parametrize("cmp", COMPARISONS)
def test_every_comparison_operator_works(cmp: str) -> None:
    node = {
        "op": "cmp",
        "cmp": cmp,
        "left": {"op": "fact", "name": "n"},
        "right": {"op": "const", "value": 10},
    }
    assert evaluate(node, {"n": 10}) == (cmp in {"<=", ">=", "=="})
    assert evaluate(node, {"n": 7}) == (cmp in {"<", "<=", "!="})
    assert evaluate(node, {"n": 13}) == (cmp in {">", ">=", "!="})


def test_date_boundary_compares_iso_strings_lexicographically() -> None:
    """30-day refund windows: the day after the window ends is the inclusive boundary."""
    within = {
        "op": "cmp",
        "cmp": "<=",
        "left": {"op": "fact", "name": "purchase_date"},
        "right": {"op": "fact", "name": "window_ends"},
    }
    assert evaluate(within, {"purchase_date": "2026-02-14", "window_ends": "2026-02-14"}) is True
    assert evaluate(within, {"purchase_date": "2026-02-15", "window_ends": "2026-02-14"}) is False


def test_mixed_kinds_and_unknown_ops_fail_loudly() -> None:
    with pytest.raises(TreeError, match="different kinds"):
        evaluate(
            {
                "op": "cmp",
                "cmp": "<",
                "left": {"op": "const", "value": 5},
                "right": {"op": "const", "value": "five"},
            },
            {},
        )
    with pytest.raises(TreeError, match="unknown comparison"):
        evaluate(
            {
                "op": "cmp",
                "cmp": "=~",
                "left": {"op": "const", "value": 1},
                "right": {"op": "const", "value": 1},
            },
            {},
        )
    with pytest.raises(TreeError, match="unknown op"):
        evaluate({"op": "magic"}, {})
    with pytest.raises(TreeError, match="must be an object"):
        evaluate("not a node", {})


def test_select_takes_the_first_matching_branch_in_order() -> None:
    facts = {"receipt": False, "days": 12}
    tree = {
        "op": "select",
        "when": [
            [
                {
                    "op": "all",
                    "args": [
                        {"op": "fact", "name": "receipt"},
                        {
                            "op": "cmp",
                            "cmp": "<=",
                            "left": {"op": "fact", "name": "days"},
                            "right": {"op": "const", "value": 30},
                        },
                    ],
                },
                {"op": "const", "value": "pay"},
            ],
            [
                {"op": "not", "arg": {"op": "fact", "name": "receipt"}},
                {"op": "const", "value": "deny_no_receipt"},
            ],
        ],
        "else": {"op": "const", "value": "deny_too_late"},
    }
    assert evaluate(tree, facts) == "deny_no_receipt"
    assert evaluate(tree, {"receipt": True, "days": 12}) == "pay"
    assert evaluate(tree, {"receipt": True, "days": 40}) == "deny_too_late"


def test_multi_hop_chain_all_of_the_emitted_edges() -> None:
    facts = {"edge_ann_bob": True, "edge_bob_cat": True, "edge_cat_desk": False}
    hops = [{"op": "fact", "name": name} for name in facts]
    assert evaluate({"op": "all", "args": hops}, facts) is False
    assert evaluate({"op": "all", "args": hops[:2]}, facts) is True


def test_trap_tree_ignores_the_distractor_fact() -> None:
    """The trap's surface clue is a fact in the text but never a node in the tree."""
    facts = {
        "distractor_says_vacant": True,
        "actually_unoccupied": True,
        "definitions_restore_coverage": True,
    }
    tree = {
        "op": "select",
        "when": [
            [
                {
                    "op": "all",
                    "args": [
                        {"op": "fact", "name": "actually_unoccupied"},
                        {"op": "fact", "name": "definitions_restore_coverage"},
                    ],
                },
                {"op": "const", "value": "pay_subject_to_sublimit"},
            ],
        ],
        "else": {"op": "const", "value": "deny_vacancy_exclusion"},
    }
    assert evaluate(tree, facts) == "pay_subject_to_sublimit"


def test_as_target_coerces_and_validates() -> None:
    assert as_target(True) == 1 and as_target(False) == 0
    assert as_target(3) == 3
    assert as_target("pay", options=["pay", "deny"]) == "pay"
    with pytest.raises(TreeError, match="not one of the options"):
        as_target("maybe", options=["pay", "deny"])
    with pytest.raises(TreeError, match="cannot be a target"):
        as_target(2.5)


def test_reexecute_reads_the_record_shipped_rule() -> None:
    record = {
        "id": "x",
        "criteria": {"a": "…", "b": "…"},
        "rule": {
            "facts": {"ok": True},
            "tree": {
                "op": "select",
                "when": [[{"op": "fact", "name": "ok"}, {"op": "const", "value": "a"}]],
                "else": {"op": "const", "value": "b"},
            },
        },
    }
    assert reexecute(record) == "a"
    with pytest.raises(TreeError, match="carries no rule"):
        reexecute({"id": "y"})
