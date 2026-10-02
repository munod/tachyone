"""Generate the JevBench-family records from committed content + rule trees (B-13, JB-4).

Five families shaped like the JevBench tiers whose synthetic coverage the P0 baseline
found missing (``.specs/features/jevbench/spec.md``):

- ``long_policy``   — approval-band routing and coverage decisions over a drawn policy
                      document (hard/standard long-document shape).
- ``trap``          — a defined term that contradicts the everyday reading, with an agent
                      note pushing the wrong answer (the ``trap`` shape, incl. its
                      ``surface_answer`` audit field).
- ``multi_hop``     — a stated reporting graph; follow the chain from the named owner
                      (``multi_hop`` shape, optional urgency override).
- ``temporal_numeric`` — period rules whose end the reader must compute (month-end clamp,
                      leap years) and deductible/cap settlement arithmetic.
- ``adequacy``      — request/response judged against explicit constraints (the judge-tier
                      proxy).

Every record is assembled **three ways from the same draw**: the text, the shipped
``rule.facts`` and the shipped ``rule.tree``; the ``target`` is
``as_target(evaluate(tree, facts))`` — never a loop index (B-11/B-12/L-008). States are
capped at ``MAX_STATE_CHARS`` so the decisive text survives the runtime's 512-token
truncation (L-014), and ``--public-dir`` refuses to write anything that collides with the
231 public JevBench items (they stay evaluation-only).

    uv run python -m training.build_jev_families --public-dir /path/to/jevbench/datasets/public
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Callable, Iterable, Mapping
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

from training.build_jev_sources import BuildError, assert_no_public_overlap
from training.rule_trees import as_target, evaluate

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = _ROOT / "training" / "configs" / "jev_families.json"
DEFAULT_CONTENT_DIR = _ROOT / "training" / "data" / "jev"
DEFAULT_OUT = _ROOT / "data" / "train_jev_families.jsonl"

#: Hard cap on state size: inference truncates English states at 512 tokens (L-014), so the
#: decisive text must fit. Measured density on the densest family shape (the routing policy
#: document) is 3.87 chars/token with the ModernBERT-large tokenizer, so 1800 chars is
#: ~465 tokens — under the limit with margin. tests/test_training_jev_families.py asserts
#: the real token count when transformers is available; this cap is the CI-safe guard.
MAX_STATE_CHARS = 1800
#: The long_policy family must actually be *long* (it exists to teach dense documents).
LONG_POLICY_MIN_CHARS = 700

FAMILIES: tuple[str, ...] = (
    "long_policy",
    "trap",
    "multi_hop",
    "temporal_numeric",
    "adequacy",
)

#: noul criteria shared by the boolean families, in the bench's semantic shape.
NOUL_CRITERIA: dict[str, str] = {
    "false": "The operative facts do not satisfy the stated condition.",
    "true": "The operative facts satisfy the stated condition under the definitions above.",
}

_CONFIG_KEYS = {"seed", "counts"}
_FACT_KINDS = {"bool", "int", "text", "derived"}


# --------------------------------------------------------------------------- math


def add_months_clamped(start: date, months: int) -> date:
    """``start`` shifted by ``months``, clamped to the last existing day of the month.

    The convention the temporal family's rule text states: an 18-month term from
    31 January ends on the last day of July's target month if the day number does not
    exist there (31 Feb → 28/29 Feb), leap years included.
    """
    total = start.month - 1 + months
    year = start.year + total // 12
    month = total % 12 + 1
    last = _month_length(year, month)
    return date(year, month, min(start.day, last))


def _month_length(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - date(year, month, 1)).days


def business_days(elapsed: int, weekend_in_between: bool) -> int:
    """Calendar days minus the two non-business days of one weekend, never below 0."""
    return max(0, elapsed - (2 if weekend_in_between else 0))


#: Derived-fact formulas: named, pure, unit-tested. Content references them by name.
FORMULAS: dict[str, Callable[..., Any]] = {
    "business_days": business_days,
    "months_from": lambda start_iso, months: add_months_clamped(
        date.fromisoformat(str(start_iso)), int(months)
    ).isoformat(),
    "days_from": lambda start_iso, days: (
        date.fromisoformat(str(start_iso)) + timedelta(days=int(days))
    ).isoformat(),
}


# ------------------------------------------------------------------------ loading


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Strict recipe loader (unknown keys are a mistake, L-011)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.pop("_comment", None)
    unknown = set(data) - _CONFIG_KEYS
    if unknown:
        raise BuildError(f"unknown config keys: {sorted(unknown)}")
    if _CONFIG_KEYS - set(data):
        raise BuildError(f"missing config keys: {sorted(_CONFIG_KEYS - set(data))}")
    if not isinstance(data["seed"], int) or not isinstance(data["counts"], dict):
        raise BuildError("seed must be an int and counts a dict")
    if set(data["counts"]) != set(FAMILIES):
        raise BuildError(f"counts must name exactly {FAMILIES}")
    return data


def _require_keys(data: Mapping[str, Any], keys: Iterable[str], what: str) -> None:
    missing = set(keys) - set(data)
    if missing:
        raise BuildError(f"{what} is missing keys: {sorted(missing)}")


def _validate_fact_specs(case_key: str, facts: Mapping[str, Any]) -> None:
    for name, spec in facts.items():
        if not isinstance(spec, Mapping) or spec.get("kind") not in _FACT_KINDS:
            raise BuildError(f"{case_key}: fact {name!r} needs a kind in {sorted(_FACT_KINDS)}")
        kind = spec["kind"]
        if kind == "bool" and not ({"yes", "no"} <= set(spec)):
            raise BuildError(f"{case_key}: bool fact {name!r} needs 'yes' and 'no' text")
        if kind == "int" and not ({"min", "max", "template"} <= set(spec)):
            raise BuildError(f"{case_key}: int fact {name!r} needs min/max/template")
        if kind == "text" and not ({"pool"} <= set(spec) and spec["pool"]):
            raise BuildError(f"{case_key}: text fact {name!r} needs a non-empty pool")
        if kind == "derived":
            if spec.get("formula") not in FORMULAS:
                raise BuildError(f"{case_key}: fact {name!r} uses unknown formula")
            if not spec.get("deps"):
                raise BuildError(f"{case_key}: derived fact {name!r} needs deps")


def load_content(directory: str | Path = DEFAULT_CONTENT_DIR) -> dict[str, dict[str, Any]]:
    """Load and structurally validate the five committed content files."""
    directory = Path(directory)
    content: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        path = directory / f"{family}.json"
        if not path.exists():
            raise BuildError(f"missing content file: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if family == "long_policy":
            _require_keys(data, ("orgs", "doc_titles", "claim_refs", "routing", "coverage"), family)
            _require_keys(
                data["routing"],
                (
                    "question",
                    "intro",
                    "clause_text",
                    "bands",
                    "component_templates",
                    "discount_notes",
                    "aggregation_notes",
                ),
                f"{family}.routing",
            )
            _require_keys(
                data["coverage"],
                (
                    "question",
                    "intro",
                    "clauses",
                    "incident_templates",
                    "causes",
                    "people",
                    "outcomes",
                ),
                f"{family}.coverage",
            )
        elif family == "trap":
            _require_keys(data, ("cases",), family)
            for case in data["cases"]:
                _require_keys(
                    case,
                    (
                        "key",
                        "rule",
                        "definition",
                        "question",
                        "facts",
                        "tree",
                        "note",
                        "surface_answer",
                    ),
                    f"{family}.case",
                )
                _validate_fact_specs(case["key"], case["facts"])
        elif family == "multi_hop":
            _require_keys(
                data,
                (
                    "org_header",
                    "orgs",
                    "chapters",
                    "people",
                    "units",
                    "edge_templates",
                    "seat_template",
                    "incident_template",
                    "urgency",
                    "requests",
                    "question",
                ),
                family,
            )
            for unit in data["units"]:
                _require_keys(unit, ("key", "name", "desc"), f"{family}.unit")
        elif family == "temporal_numeric":
            _require_keys(data, ("within_period", "payout_band"), family)
            _require_keys(
                data["within_period"],
                ("orgs", "cert_refs", "question", "doc", "rule_sentences", "products", "criteria"),
                f"{family}.within_period",
            )
            _require_keys(
                data["payout_band"],
                ("orgs", "file_refs", "question", "doc", "deductible", "caps", "outcomes"),
                f"{family}.payout_band",
            )
        else:  # adequacy
            _require_keys(data, ("question", "criteria", "requests", "quotes", "responses"), family)
            for request in data["requests"]:
                _require_keys(request, ("key", "kind", "op", "template"), f"{family}.request")
        content[family] = data
    return content


# -------------------------------------------------------------- facts and plumbing


def record_rng(seed: int, family: str, index: int) -> random.Random:
    """One independent RNG per record (L-003: no shared stream that could track a label)."""
    return random.Random(f"{seed}:{family}:{index}")


def draw_facts(spec: Mapping[str, Any], rng: random.Random) -> dict[str, Any]:
    """Draw every declared fact; derived facts run their named formula over the deps."""
    facts: dict[str, Any] = {}
    for name, node in spec.items():
        kind = node["kind"]
        if kind == "bool":
            facts[name] = rng.random() < 0.5
        elif kind == "int":
            facts[name] = rng.randint(int(node["min"]), int(node["max"]))
        elif kind == "text":
            facts[name] = rng.choice(list(node["pool"]))
        else:
            try:
                values = [facts[dep] for dep in node["deps"]]
            except KeyError as exc:
                raise BuildError(f"fact {name!r} needs {exc.args[0]!r} drawn first") from exc
            try:
                facts[name] = FORMULAS[str(node["formula"])](*values)
            except (TypeError, ValueError) as exc:
                raise BuildError(f"formula {node['formula']!r} failed for {name!r}: {exc}") from exc
    return facts


def render_fact(name: str, node: Mapping[str, Any], value: Any) -> str:
    """How one *stated* fact reads in the text (derived facts are never stated)."""
    kind = node["kind"]
    if kind == "bool":
        return str(node["yes"] if value else node["no"])
    if kind == "int":
        try:
            return str(node["template"]).format(**{name: value})
        except KeyError as exc:
            raise BuildError(f"int fact {name!r}: template must contain {{{name!r}}}") from exc
    if kind == "text":
        return str(value)
    raise BuildError(f"derived fact {name!r} must not be rendered into the text")


def tree_fact_names(tree: Any) -> set[str]:
    """Every fact name a tree references (the fact table must bind all of them)."""
    if not isinstance(tree, Mapping) or "op" not in tree:
        return set()
    names: set[str] = set()
    if tree["op"] == "fact":
        names.add(str(tree["name"]))
    for key in ("args", "when"):
        value = tree.get(key)
        if isinstance(value, list):
            for item in value:
                if key == "when":
                    names.update(tree_fact_names(item[0]))
                    names.update(tree_fact_names(item[1]))
                else:
                    names.update(tree_fact_names(item))
    for key in ("arg", "left", "right", "else"):
        if key in tree:
            names.update(tree_fact_names(tree[key]))
    return names


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _record(
    index: int,
    family: str,
    *,
    state: str,
    instructions: str,
    criteria: Any,
    target: int | str,
    facts: Mapping[str, Any],
    tree: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if len(state) > MAX_STATE_CHARS:
        raise BuildError(
            f"{family}[{index}]: state is {len(state)} chars, over the {MAX_STATE_CHARS} cap "
            "(the decisive text must survive the 512-token truncation, L-014)"
        )
    record: dict[str, Any] = {
        "id": f"jev-{family}-{index:05d}",
        "type": "noul" if isinstance(target, int) else "choice",
        "state": state,
        "instructions": instructions,
        "criteria": criteria,
        "target": target,
        "lang": "en",
        "source": f"synthetic:jev:{family}",
        "rule": {"facts": dict(facts), "tree": dict(tree)},
    }
    if extra:
        record.update(extra)
    return record


# ----------------------------------------------------------------------- families


def _long_policy_record(
    index: int, rng: random.Random, content: Mapping[str, Any]
) -> dict[str, Any]:
    data = content["long_policy"]
    if index % 2 == 0:
        return _routing_record(index, rng, data)
    return _coverage_record(index, rng, data)


def _routing_record(index: int, rng: random.Random, data: Mapping[str, Any]) -> dict[str, Any]:
    routing = data["routing"]
    org = rng.choice(list(data["orgs"]))
    title = rng.choice(list(data["doc_titles"]))
    clause = f"{rng.choice(['GP', 'PA', 'FM', 'AR'])}-{rng.randint(1, 20):02d}"
    version = rng.randint(2, 9)
    month = rng.choice(["January", "March", "June", "September", "November"])
    effective = f"{rng.randint(1, 28)} {month} {rng.choice([2024, 2025])}"
    ref = rng.choice(list(data["claim_refs"]))

    bands = list(routing["bands"])
    # Draw the *target band* first and decompose the TCV into real components that land in
    # it: natural draws put almost everything in level 4 (L-010 — audit the label
    # distribution before training on it), and level 1 would never appear at all.
    target_index = rng.randrange(len(bands))
    lower = 0 if target_index == 0 else int(bands[target_index - 1]["max"])
    upper = bands[target_index]["max"]
    tcv_target = (
        rng.randint(lower + 500, int(upper) - 500)
        if upper is not None
        else rng.randint(int(bands[-2]["max"]) + 1500, 720000)
    )
    remaining = tcv_target
    related = (
        rng.choice([0, rng.randint(5000, max(6000, remaining // 4))]) if target_index > 0 else 0
    )
    remaining -= related
    fee = rng.choice([0, 0, remaining // 6])
    remaining -= fee
    extra_fee = rng.choice([0, remaining // 8])
    remaining -= extra_fee
    years = rng.choice([1, 2, 3])
    annual = max(1000, remaining // years)
    tcv = annual * years + fee + extra_fee + related
    discount = rng.randint(2, 30) * 100  # excluded by clause 2.2, stated in 4.2

    components = rng.choice(list(routing["component_templates"])).format(
        annual=annual, years=years, fee=fee, extra=extra_fee
    )
    if related:
        aggregation_note = rng.choice(
            [note for note in routing["aggregation_notes"] if "{related}" in note]
        ).format(related=related)
    else:
        aggregation_note = next(
            note for note in routing["aggregation_notes"] if "{related}" not in note
        )
    discount_note = rng.choice(list(routing["discount_notes"])).format(discount=discount)

    bands = list(routing["bands"])
    band_table = "\n".join(f"{position}. {band['desc']}" for position, band in enumerate(bands, 1))
    purpose, definitions, levels, evidence = (str(text) for text in routing["clause_text"])
    state = routing["intro"].format(
        org=org, title=title, clause=clause, version=version, effective=effective, ref=ref
    )
    state += purpose
    state += definitions
    state += levels.format(bands=band_table)
    state += evidence.format(
        ref=ref,
        components=components,
        aggregation_note=aggregation_note,
        discount_note=discount_note,
    )

    when = [
        [
            {
                "op": "cmp",
                "cmp": "<=",
                "left": {"op": "fact", "name": "tcv"},
                "right": {"op": "const", "value": int(band["max"])},
            },
            {"op": "const", "value": str(band["key"])},
        ]
        for band in bands
        if band["max"] is not None
    ]
    tree = {"op": "select", "when": when, "else": {"op": "const", "value": str(bands[-1]["key"])}}
    facts = {"tcv": tcv}
    target = as_target(evaluate(tree, facts), options=[str(band["key"]) for band in bands])
    criteria = {str(band["key"]): str(band["desc"]) for band in bands}
    return _record(
        index,
        "long_policy",
        state=state,
        instructions=routing["question"].format(ref=ref, clause=clause),
        criteria=criteria,
        target=target,
        facts=facts,
        tree=tree,
    )


def _coverage_record(index: int, rng: random.Random, data: Mapping[str, Any]) -> dict[str, Any]:
    coverage = data["coverage"]
    org = rng.choice(list(data["orgs"]))
    title = rng.choice(list(data.get("doc_titles_insurance", data["doc_titles"])))
    clause = f"{rng.choice(['HM', 'SC', 'FM'])}-{rng.randint(1, 12):02d}"
    version = rng.randint(2, 7)
    ref = rng.choice(list(data["claim_refs"]))
    window = rng.choice([14, 30, 45])
    vacancy_days = rng.choice([45, 60, 90])
    sublimit = rng.choice([10000, 15000])
    deductible = rng.choice([500, 1000])

    # One denial condition at most, so the select order never picks between two true ones.
    scenario = rng.choice(["late", "vacancy", "covered"])
    if scenario == "late":
        report_delay = rng.randint(window + 1, window + 25)
        unoccupied = rng.randint(1, vacancy_days)
    elif scenario == "vacancy":
        report_delay = rng.randint(1, window)
        unoccupied = rng.randint(vacancy_days + 1, vacancy_days + 40)
    else:
        report_delay = rng.randint(1, window)
        unoccupied = rng.randint(1, vacancy_days)
    cause = rng.choice(list(coverage["causes"]))
    loss = rng.randint(3, 60) * 1000
    discovery = (date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))).isoformat()
    who = rng.choice(list(coverage["people"]))

    clauses = coverage["clauses"]
    state = coverage["intro"].format(org=org, title=title, clause=clause, version=version, ref=ref)
    state += clauses["window"].format(window=window) + "\n"
    state += clauses["vacancy"].format(vacancy_days=vacancy_days) + "\n"
    state += clauses["sublimit"].format(sublimit=sublimit) + "\n"
    state += clauses["deductible"].format(deductible=deductible) + "\n\n"
    state += rng.choice(list(coverage["incident_templates"])).format(
        ref=ref,
        who=who,
        discovery=discovery,
        report_delay=report_delay,
        cause_sentence=cause["text"],
        unoccupied=unoccupied,
        loss=loss,
    )

    facts = {
        "report_delay": report_delay,
        "window": window,
        "unoccupied_days": unoccupied,
        "vacancy_limit": vacancy_days,
        "concealed": bool(cause["concealed"]),
    }
    tree = {
        "op": "select",
        "when": [
            [
                {
                    "op": "cmp",
                    "cmp": ">",
                    "left": {"op": "fact", "name": "report_delay"},
                    "right": {"op": "fact", "name": "window"},
                },
                {"op": "const", "value": "deny_late_notice"},
            ],
            [
                {
                    "op": "cmp",
                    "cmp": ">",
                    "left": {"op": "fact", "name": "unoccupied_days"},
                    "right": {"op": "fact", "name": "vacancy_limit"},
                },
                {"op": "const", "value": "deny_vacancy_exclusion"},
            ],
            [
                {"op": "fact", "name": "concealed"},
                {"op": "const", "value": "pay_capped_at_sublimit"},
            ],
        ],
        "else": {"op": "const", "value": "pay_full_estimate_less_deductible"},
    }
    options = [str(outcome["key"]) for outcome in coverage["outcomes"]]
    target = as_target(evaluate(tree, facts), options=options)
    criteria = {
        str(outcome["key"]): str(outcome["desc"]).format(sublimit=sublimit, deductible=deductible)
        for outcome in coverage["outcomes"]
    }
    record = _record(
        index,
        "long_policy",
        state=state,
        instructions=coverage["question"].format(ref=ref),
        criteria=criteria,
        target=target,
        facts=facts,
        tree=tree,
    )
    if len(record["state"]) < LONG_POLICY_MIN_CHARS:
        raise BuildError(f"long_policy[{index}]: state too short to be the long family")
    return record


def _trap_record(index: int, rng: random.Random, content: Mapping[str, Any]) -> dict[str, Any]:
    cases = list(content["trap"]["cases"])
    case = cases[index % len(cases)]
    facts = draw_facts(case["facts"], rng)
    stated = [
        render_fact(name, case["facts"][name], value)
        for name, value in facts.items()
        if case["facts"][name]["kind"] != "derived"
    ]
    state = (
        f"{case['rule']}\n\n{case['definition']}\n\n"
        f"In this case: {'; '.join(stated)}.\n{case['note']}"
    )
    target = as_target(evaluate(case["tree"], facts))
    return _record(
        index,
        "trap",
        state=state,
        instructions=str(case["question"]),
        criteria=dict(NOUL_CRITERIA),
        target=target,
        facts=facts,
        tree=case["tree"],
        extra={
            "rule": {
                "facts": facts,
                "tree": case["tree"],
                "surface_answer": int(case["surface_answer"]),
            }
        },
    )


def _multi_hop_record(index: int, rng: random.Random, content: Mapping[str, Any]) -> dict[str, Any]:
    data = content["multi_hop"]
    units = list(data["units"])
    people = list(data["people"])
    org = rng.choice(list(data["orgs"]))
    chapter = rng.choice(list(data["chapters"]))
    rev = f"2026-{rng.randint(1, 9):02d}"
    request = rng.choice(list(data["requests"]))

    asked = rng.choice(people)
    mid_count = rng.randint(1, 3)
    pool = [person for person in people if person != asked]
    mids = rng.sample(pool, mid_count)
    chain = [asked, *mids]
    # The duty desk is reachable only through the urgency override (L-010: drawn as a
    # normal target too, it becomes 27% of all labels).
    normal_units = [unit for unit in units if unit["key"] != str(data["urgency"]["override_unit"])]
    target_unit = rng.choice(normal_units)

    stated_edges: list[tuple[str, str]] = []
    statements: list[str] = []
    for left, right in pairwise(chain):
        stated_edges.append((left, right))
        statements.append(rng.choice(list(data["edge_templates"])).format(a=left, b=right))
    stated_edges.append((chain[-1], target_unit["name"]))
    statements.append(data["seat_template"].format(a=chain[-1], unit=target_unit["name"]))

    # Distractor chains: other people seated elsewhere, never reachable from `asked`.
    distractor_units = rng.sample(
        [unit for unit in units if unit["key"] != target_unit["key"]], rng.randint(1, 2)
    )
    for unit in distractor_units:
        stranger = rng.choice([person for person in pool if person not in chain])
        hub = rng.choice([person for person in pool if person not in chain and person != stranger])
        stated_edges.append((stranger, hub))
        statements.append(rng.choice(list(data["edge_templates"])).format(a=stranger, b=hub))
        stated_edges.append((hub, unit["name"]))
        statements.append(data["seat_template"].format(a=hub, unit=unit["name"]))

    variant = index % 3 == 2
    urgency = data["urgency"]
    facts: dict[str, Any] = {}
    extra_clauses = ""
    if variant:
        urgent = rng.random() < 0.5
        facts["urgent"] = urgent
        extra_clauses = "\n" + str(urgency["manual_clause"])

    # Option set: the target (or the override unit), plus distractors.
    must_include = target_unit
    if variant and facts["urgent"]:
        must_include = next(unit for unit in units if unit["key"] == urgency["override_unit"])
    options = [must_include]
    for unit in rng.sample([unit for unit in units if unit["key"] != must_include["key"]], 4):
        options.append(unit)
    options = sorted(options, key=lambda unit: str(unit["key"]))

    true_edges = set(stated_edges)

    def path_branch(unit: Mapping[str, Any]) -> list[Any]:
        """All edges of the shortest stated path asked → unit, else one unstated edge."""
        path = _shortest_path(true_edges, asked, str(unit["name"]))
        if path is None:
            return [{"op": "fact", "name": f"edge_{slug(asked)}_{slug(str(unit['name']))}"}]
        return [{"op": "fact", "name": f"edge_{slug(left)}_{slug(right)}"} for left, right in path]

    when: list[list[Any]] = []
    if variant:
        override = str(urgency["override_unit"])
        when.append(
            [
                {"op": "fact", "name": "urgent"},
                {"op": "const", "value": override},
            ]
        )
    when.extend(
        [
            {"op": "all", "args": path_branch(unit)},
            {"op": "const", "value": str(unit["key"])},
        ]
        for unit in options
        if str(unit["key"]) != (str(urgency["override_unit"]) if variant else None)
    )
    fallback = str(target_unit["key"])
    tree = {"op": "select", "when": when, "else": {"op": "const", "value": fallback}}

    # Bind every fact the tree references: stated edges true, everything else false.
    for left, right in true_edges:
        facts[f"edge_{slug(left)}_{slug(right)}"] = True
    for name in tree_fact_names(tree):
        facts.setdefault(name, False)

    target = as_target(evaluate(tree, facts), options=[str(unit["key"]) for unit in options])
    incident = str(data["incident_template"]).format(request=request, owner=asked)
    if variant:
        incident = incident.rstrip("\n") + str(
            urgency["incident_flag_yes"] if facts["urgent"] else urgency["incident_flag_no"]
        )
    state = data["org_header"].format(org=org, chapter=chapter, rev=rev)
    state += "\n".join(statements)
    state += "\n" + incident
    state += extra_clauses
    criteria = {str(unit["key"]): str(unit["desc"]) for unit in options}
    return _record(
        index,
        "multi_hop",
        state=state,
        instructions=data["question"].format(request=request, owner=asked),
        criteria=criteria,
        target=target,
        facts=facts,
        tree=tree,
    )


def _shortest_path(
    edges: set[tuple[str, str]], start: str, goal: str
) -> list[tuple[str, str]] | None:
    """Breadth-first over the stated edges (undirected seats are stated one-way)."""
    frontier: list[tuple[str, list[tuple[str, str]]]] = [(start, [])]
    seen = {start}
    while frontier:
        node, path = frontier.pop(0)
        for left, right in edges:
            nxt = right if left == node else (left if right == node else None)
            if nxt is None or nxt in seen:
                continue
            step = (left, right)
            if nxt == goal:
                return [*path, step]
            seen.add(nxt)
            frontier.append((nxt, [*path, step]))
    return None


def _temporal_record(index: int, rng: random.Random, content: Mapping[str, Any]) -> dict[str, Any]:
    data = content["temporal_numeric"]
    if index % 5 < 3:
        return _within_period_record(index, rng, data["within_period"])
    return _payout_record(index, rng, data["payout_band"])


def _within_period_record(
    index: int, rng: random.Random, data: Mapping[str, Any]
) -> dict[str, Any]:
    org = rng.choice(list(data["orgs"]))
    cert = rng.choice(list(data["cert_refs"]))
    ref = f"CL-{rng.randint(2800000, 2899999)}"
    product = rng.choice(list(data["products"]))
    serial = f"SN-{rng.randint(100000, 999999)}"
    start = date(2025, 1, 1) + timedelta(days=rng.randint(0, 700))

    if rng.random() < 0.75:
        months = rng.choice([6, 12, 18])
        rule_sentence = rng.choice(list(data["rule_sentences"])[:2]).format(months=months)
        end = add_months_clamped(start, months)
    else:
        days = rng.choice([30, 45, 90, 180])
        rule_sentence = str(data["rule_sentences"][2]).format(days=days)
        end = start + timedelta(days=days)
    offset = rng.choice([-14, -7, -3, -1, 0, 1, 3, 7, 14])
    event = end + timedelta(days=offset)

    state = str(data["doc"]).format(
        org=org,
        cert=cert,
        product=product,
        serial=serial,
        start=start.isoformat(),
        rule_sentence=rule_sentence,
        ref=ref,
        event=event.isoformat(),
    )
    facts = {
        "start_date": start.isoformat(),
        "period_end": end.isoformat(),
        "event_date": event.isoformat(),
    }
    tree = {
        "op": "cmp",
        "cmp": "<=",
        "left": {"op": "fact", "name": "event_date"},
        "right": {"op": "fact", "name": "period_end"},
    }
    target = as_target(evaluate(tree, facts))
    return _record(
        index,
        "temporal_numeric",
        state=state,
        instructions=data["question"].format(ref=ref, cert=cert),
        criteria=dict(data["criteria"]),
        target=target,
        facts=facts,
        tree=tree,
    )


def _payout_record(index: int, rng: random.Random, data: Mapping[str, Any]) -> dict[str, Any]:
    payout = data
    org = rng.choice(list(payout["orgs"]))
    ref = rng.choice(list(payout["file_refs"]))
    deductible = int(payout["deductible"])
    caps = [int(cap) for cap in payout["caps"]]
    # Scenario first (L-010): natural draws never fall below the deductible, so the
    # decline branch would ship with zero examples.
    scenario = rng.choice(["below", "capped", "full"])
    if scenario == "below":
        cap = rng.choice(caps)
        loss = rng.randint(1, 2) * 500
    elif scenario == "capped":
        cap = rng.choice(caps)
        loss = cap + deductible + rng.randint(1, 24) * 500
    else:
        cap = rng.choice(caps)
        loss = deductible + rng.randint(1, max(2, (cap - deductible) // 500)) * 500
    settled = 0 if loss <= deductible else min(loss - deductible, cap)

    state = (
        str(payout["doc"])
        .format(org=org, file_ref=ref, deductible=deductible, cap=cap, loss=loss, ref=ref)
        .replace("{ref}", ref)
    )
    facts = {"loss": loss, "deductible": deductible, "cap": cap, "payout": settled}
    tree = {
        "op": "select",
        "when": [
            [
                {
                    "op": "cmp",
                    "cmp": "<=",
                    "left": {"op": "fact", "name": "loss"},
                    "right": {"op": "fact", "name": "deductible"},
                },
                {"op": "const", "value": "decline_below_deductible"},
            ],
            [
                {
                    "op": "cmp",
                    "cmp": "==",
                    "left": {"op": "fact", "name": "payout"},
                    "right": {"op": "fact", "name": "cap"},
                },
                {"op": "const", "value": "pay_capped_at_limit"},
            ],
        ],
        "else": {"op": "const", "value": "pay_estimate_less_deductible"},
    }
    options = [str(outcome["key"]) for outcome in payout["outcomes"]]
    target = as_target(evaluate(tree, facts), options=options)
    criteria = {
        str(outcome["key"]): str(outcome["desc"]).format(cap=cap) for outcome in payout["outcomes"]
    }
    return _record(
        index,
        "temporal_numeric",
        state=state,
        instructions=payout["question"].format(ref=ref),
        criteria=criteria,
        target=target,
        facts=facts,
        tree=tree,
    )


def _adequacy_record(index: int, rng: random.Random, content: Mapping[str, Any]) -> dict[str, Any]:
    data = content["adequacy"]
    request = list(data["requests"])[index % len(data["requests"])]
    key = request["key"]
    a, b = rng.randint(30, 99), rng.randint(5, 29)
    if key == "quote_words":
        quote = rng.choice(list(data["quotes"]))
        template_vars: dict[str, Any] = {"quote": quote}
        value = str(len(str(quote).split()))
        wrong = str(len(str(quote).split()) + rng.choice([1, -1]))
    elif key == "rounded":
        decimal = f"{rng.randint(10, 99)}.{rng.randint(10, 99)}"
        template_vars = {"decimal": decimal}
        value = f"{float(decimal):.1f}"
        wrong = f"{float(decimal) + rng.choice([0.2, -0.2, 0.3]):.1f}"
    elif key == "pair":
        template_vars = {"a": a, "b": b}
        total, difference = a + b, a - b
        value = f"{total}\n{difference}"
        wrong = f"{total}\n{difference + rng.choice([3, -4, 5])}"
    else:
        template_vars = {"a": a, "b": b}
        computed = {"sum": a + b, "diff": a - b, "product": a * b}[str(request["op"])]
        value = str(computed)
        wrong = str(computed + rng.choice([-7, -3, 4, 9, 11]))

    # Label first, then compose facts that realize it (L-010): a conjunction of two coin
    # flips would ship 75% negatives, and the bench's adequacy items are mostly positive.
    label_true = rng.random() < 0.5
    if label_true:
        value_correct, style = True, "clean"
    else:
        value_correct = rng.random() < 0.5
        style = rng.choice(["clean", "prose"]) if not value_correct else "prose"
    response = str(data["responses"][style]["true" if value_correct else "false"]).format(
        value=value, wrong=wrong
    )
    state = str(request["template"]).format(**template_vars) + f"\nResponse: {response}"
    facts = {"value_correct": value_correct, "constraints_respected": style == "clean"}
    tree = {
        "op": "all",
        "args": [
            {"op": "fact", "name": "value_correct"},
            {"op": "fact", "name": "constraints_respected"},
        ],
    }
    target = as_target(evaluate(tree, facts))
    return _record(
        index,
        "adequacy",
        state=state,
        instructions=str(data["question"]),
        criteria=dict(data["criteria"]),
        target=target,
        facts=facts,
        tree=tree,
    )


_BUILDERS: dict[str, Callable[[int, random.Random, Mapping[str, Any]], dict[str, Any]]] = {
    "long_policy": _long_policy_record,
    "trap": _trap_record,
    "multi_hop": _multi_hop_record,
    "temporal_numeric": _temporal_record,
    "adequacy": _adequacy_record,
}


# --------------------------------------------------------------------------- build


def build(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    content_dir: str | Path = DEFAULT_CONTENT_DIR,
    out_path: str | Path = DEFAULT_OUT,
    public_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate every family's records, audit-collide them, write the file, report."""
    config = load_config(config_path)
    content = load_content(content_dir)
    seed = int(config["seed"])
    records: list[dict[str, Any]] = []
    per_family: dict[str, int] = {}
    for family in FAMILIES:
        count = int(config["counts"][family])
        builder = _BUILDERS[family]
        for index in range(count):
            records.append(builder(index, record_rng(seed, family, index), content))
        per_family[family] = count

    if public_dir is not None:
        assert_no_public_overlap(records, public_dir)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    targets: dict[str, dict[str, int]] = {}
    for family in FAMILIES:
        counts: dict[str, int] = {}
        for record in records:
            if record["source"] == f"synthetic:jev:{family}":
                counts[str(record["target"])] = counts.get(str(record["target"]), 0) + 1
        targets[family] = dict(sorted(counts.items()))
    lengths = [len(record["state"]) for record in records]
    return {
        "out": str(out),
        "total": len(records),
        "per_family": per_family,
        "per_type": {
            kind: sum(1 for record in records if record["type"] == kind)
            for kind in ("noul", "choice")
        },
        "target_distribution": targets,
        "state_chars": {"min": min(lengths), "max": max(lengths)},
        "seed": seed,
        "checked_against_public": public_dir is not None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tachyone-build-jev-families", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--content-dir", default=str(DEFAULT_CONTENT_DIR))
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
        content_dir=args.content_dir,
        out_path=args.out,
        public_dir=args.public_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
