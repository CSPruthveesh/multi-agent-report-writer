from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

MAX_RESEARCH_LOOPS = 2
MAX_REVISIONS = 2

MAX_SEARCHES = 5

ROUTES = ("researcher", "analyst", "writer", "finalize")

CRITIC_TARGETS = tuple(r for r in ROUTES if r != "finalize")


class ReportState(TypedDict, total=False):
    topic: str

    findings: Annotated[list[dict[str, Any]], operator.add]
    searches_used: int
    research_loops: int
    unclosed_gaps: list[str]

    outline: str | None
    gaps: list[str]
    draft: str | None
    critique: dict[str, Any] | None

    revision_count: int
    route: str

    token_log: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]


def initial_state(topic: str) -> ReportState:
    return ReportState(
        topic=topic,
        findings=[],
        searches_used=0,
        research_loops=0,
        unclosed_gaps=[],
        outline=None,
        gaps=[],
        draft=None,
        critique=None,
        revision_count=0,
        route="",
        token_log=[],
        trace=[],
    )


def trace_event(node: str, action: str, **detail: Any) -> dict[str, Any]:
    return {"node": node, "action": action, **detail}


def log_entries(ledger: Any) -> list[dict[str, Any]]:
    return [r.as_dict() for r in ledger.records]
