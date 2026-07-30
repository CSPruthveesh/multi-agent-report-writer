from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

MAX_REVISIONS = 2

MAX_SEARCHES = 5

ROUTES = ("researcher", "analyst", "writer", "finalize")


class ReportState(TypedDict, total=False):
    topic: str

    findings: Annotated[list[dict[str, Any]], operator.add]
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
