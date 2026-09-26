"""LangGraph flow: decide → branch on confidence → commit or hand off to System-2.

Embedded verbatim in ``docs/integrations/langgraph.md`` and executed by
``tests/test_docs_examples.py::test_langgraph_example_runs``.

Run it:

    TACHYONE_BACKEND=fake uv run python docs/assets/examples/langgraph_flow.py
    TACHYONE_BACKEND=encoder uv run python docs/assets/examples/langgraph_flow.py

Requires the ``langchain`` extra (``uv sync --extra langchain``).
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from tachyone.backends import build_backend
from tachyone.config import Config
from tachyone.handoff import assess_response
from tachyone.integrations.langchain import create_runnable
from tachyone.wire import SystemOneResponse

#: τ is a per-decision knob, not a default — see docs/cookbook-handoff.md.
TAU = float(os.environ.get("TACHYONE_TAU", "0.6"))

QUESTIONS = {
    "queue": {
        "type": "choice",
        "instructions": "Which queue should receive this ticket first?",
        "criteria": {
            "billing-queue": "invoices, payments, refunds",
            "technical-queue": "bugs, outages, system errors",
            "sales-queue": "pricing, new contracts",
            "general-queue": "everything else",
        },
    }
}


class Flow(TypedDict, total=False):
    state: str
    questions: dict[str, Any]
    decision: dict[str, Any]
    outcome: str
    routed_to: str


runnable = create_runnable(build_backend(Config.from_env()), model="tachyone-latest")


def decide(flow: Flow) -> Flow:
    """One forward pass through Tachyone; the canonical response lands in the state."""
    flow["decision"] = runnable.invoke(
        {"state": flow["state"], "questions": flow["questions"]}
    )
    return flow


def branch(flow: Flow) -> str:
    """`abstain` when any answer's confidence is below τ."""
    response = SystemOneResponse.model_validate(flow["decision"])
    report = assess_response(response, threshold=TAU)
    return "escalate" if report.abstain else "commit"


def commit(flow: Flow) -> Flow:
    flow["routed_to"] = flow["decision"]["answers"]["queue"]["choice"]
    flow["outcome"] = "committed"
    return flow


def escalate(flow: Flow) -> Flow:
    """System-2: hand the state to a human or a larger model — see the cookbook."""
    flow["routed_to"] = "human-review"
    flow["outcome"] = "handoff"
    return flow


def build_graph() -> Any:
    graph = StateGraph(Flow)
    graph.add_node("decide", decide)
    graph.add_node("commit", commit)
    graph.add_node("escalate", escalate)
    graph.set_entry_point("decide")
    graph.add_conditional_edges(
        "decide", branch, {"commit": "commit", "escalate": "escalate"}
    )
    graph.add_edge("commit", END)
    graph.add_edge("escalate", END)
    return graph.compile()


if __name__ == "__main__":
    result = build_graph().invoke(
        {"state": "We were charged twice for the March invoice.", "questions": QUESTIONS}
    )
    print(f"{result['outcome']} -> {result['routed_to']}")
