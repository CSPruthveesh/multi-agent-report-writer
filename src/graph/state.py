from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

MAX_RESEARCH_LOOPS = 2
MAX_REVISIONS = 2

# Guard against a failing Writer spinning the graph. Distinct from MAX_REVISIONS:
# that bounds how many times the Critic may ask for changes, this bounds how many
# times the supervisor will send a run to a Writer that keeps coming back empty.
MAX_WRITE_ATTEMPTS = 2

MAX_SEARCHES = 5

ROUTES = ("researcher", "analyst", "writer", "finalize")

CRITIC_TARGETS = tuple(r for r in ROUTES if r != "finalize")


class ReportState(TypedDict, total=False):
    topic: str

    findings: Annotated[list[dict[str, Any]], operator.add]
    searches_used: int
    research_loops: int
    write_attempts: int
    # Off by default so Phase 9 can batch six topics unattended. An approval gate
    # that cannot be turned off is an approval gate that blocks the evaluation.
    hitl: bool
    # Gaps the Researcher could not afford to search. It cannot write unclosed_gaps —
    # that field belongs to the supervisor — and it cannot hand them back through gaps,
    # because the Analyst sits between them on every path and overwrites that field.
    unaddressed_gaps: list[str]
    unclosed_gaps: list[str]

    outline: str | None
    gaps: list[str]
    draft: str | None
    critique: dict[str, Any] | None
    # What the Critic said, carried to an upstream node the supervisor routed to.
    # It cannot travel on `critique`: routing upstream has to clear that field, or
    # the stale verdict routes upstream again on the return trip. Clearing it also
    # deleted the message before the recipient read it, so the criticism gets a field
    # of its own — written by the supervisor, read by the Analyst, wiped by nobody.
    revision_brief: dict[str, Any] | None

    revision_count: int
    route: str

    token_log: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]


def initial_state(topic: str, *, hitl: bool = False) -> ReportState:
    return ReportState(
        topic=topic,
        hitl=hitl,
        findings=[],
        searches_used=0,
        research_loops=0,
        write_attempts=0,
        unaddressed_gaps=[],
        unclosed_gaps=[],
        outline=None,
        gaps=[],
        draft=None,
        critique=None,
        revision_brief=None,
        revision_count=0,
        route="",
        token_log=[],
        trace=[],
    )


def trace_event(node: str, action: str, **detail: Any) -> dict[str, Any]:
    return {"node": node, "action": action, **detail}


def log_entries(ledger: Any) -> list[dict[str, Any]]:
    return [r.as_dict() for r in ledger.records]
