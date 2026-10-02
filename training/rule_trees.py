"""Executable rule trees: a label is what the tree computes over the emitted facts.

The JevBench-shaped families (``long_policy``, ``trap``, ``multi_hop``,
``temporal_numeric``, ``adequacy``) are decided by *reasoning over stated facts*, so
their training records are built the other way around from template data: the generator
draws structured content, emits the **text** and the **facts** from that same content,
and the record's ``target`` is :func:`evaluate` of the record's tree over those facts —
never a loop index (B-11/B-12/L-008 discipline). The tree and the facts ship inside the
record (``rule``) so any label can be re-executed and audited later.

Trees are plain JSON dicts, so committed content can embed them directly::

    {"op": "select",
     "when": [[{"op": "all", "args": [ ... ]}, "pay_full"], [ ... , "deny_no_receipt"]],
     "else": "deny_other"}

Ops: ``const``, ``fact``, ``all``, ``any``, ``not``, ``cmp``, ``select``.
Pure Python — no torch, testable everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: Comparison operators accepted by the ``cmp`` op.
COMPARISONS: tuple[str, ...] = ("<", "<=", ">", ">=", "==", "!=")


class TreeError(ValueError):
    """A tree cannot be executed over the given facts (missing fact, bad type, bad op)."""


def evaluate(tree: Any, facts: Mapping[str, Any]) -> Any:
    """Run ``tree`` over ``facts`` and return the label value it computes."""
    if not isinstance(tree, Mapping) or "op" not in tree:
        raise TreeError(f"node must be an object with an 'op', got {tree!r}")
    op = tree["op"]
    if op == "const":
        return tree["value"]
    if op == "fact":
        name = tree["name"]
        if name not in facts:
            raise TreeError(f"fact {name!r} is not bound; available: {sorted(facts)}")
        return facts[name]
    if op in {"all", "any"}:
        values = [_as_bool(evaluate(arg, facts), node=arg) for arg in tree["args"]]
        return all(values) if op == "all" else any(values)
    if op == "not":
        return not _as_bool(evaluate(tree["arg"], facts), node=tree)
    if op == "cmp":
        return _compare(
            tree["cmp"],
            evaluate(tree["left"], facts),
            evaluate(tree["right"], facts),
        )
    if op == "select":
        for condition, value in tree["when"]:
            if _as_bool(evaluate(condition, facts), node=condition):
                return evaluate(value, facts)
        return evaluate(tree["else"], facts)
    raise TreeError(f"unknown op {op!r}")


def _as_bool(value: Any, *, node: Any) -> bool:
    if not isinstance(value, bool):
        raise TreeError(f"boolean context expected, got {value!r} from {node!r}")
    return value


def _compare(cmp: str, left: Any, right: Any) -> bool:
    if cmp not in COMPARISONS:
        raise TreeError(f"unknown comparison {cmp!r}; expected one of {COMPARISONS}")
    same_kind = isinstance(left, (int, float)) and isinstance(right, (int, float))
    same_kind = same_kind or (isinstance(left, str) and isinstance(right, str))
    same_kind = same_kind or (isinstance(left, bool) and isinstance(right, bool))
    if not same_kind:
        raise TreeError(f"cannot compare {left!r} {cmp} {right!r}: different kinds")
    if cmp == "<":
        return left < right  # type: ignore[operator]
    if cmp == "<=":
        return left <= right  # type: ignore[operator]
    if cmp == ">":
        return left > right  # type: ignore[operator]
    if cmp == ">=":
        return left >= right  # type: ignore[operator]
    if cmp == "==":
        return bool(left == right)
    return bool(left != right)


def as_target(value: Any, *, options: Sequence[str] | None = None) -> int | str:
    """Coerce a tree result into a record target, validating it against the option set."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if options is not None:
        if value not in options:
            raise TreeError(f"tree produced {value!r}, which is not one of the options")
        return str(value)
    if isinstance(value, str):
        return value
    raise TreeError(f"tree produced {value!r}, which cannot be a target")


def reexecute(record: Mapping[str, Any]) -> int | str:
    """Recompute a record's target from its own shipped facts + tree (the audit hook)."""
    rule = record.get("rule")
    if not isinstance(rule, Mapping):
        raise TreeError(f"record {record.get('id')!r} carries no rule to re-execute")
    criteria = record.get("criteria")
    options = list(criteria) if isinstance(criteria, Mapping) else None
    return as_target(evaluate(rule["tree"], rule["facts"]), options=options)


__all__ = ["COMPARISONS", "TreeError", "as_target", "evaluate", "reexecute"]
