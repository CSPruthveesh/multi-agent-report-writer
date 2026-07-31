from __future__ import annotations

from typing import Any

from src.graph.state import (
    CRITIC_TARGETS,
    MAX_RESEARCH_LOOPS,
    MAX_REVISIONS,
    MAX_SEARCHES,
    ReportState,
    trace_event,
)

NODE = "supervisor"


def supervisor(state: ReportState) -> dict[str, Any]:
    gaps = list(state.get("gaps") or [])
    draft = state.get("draft")
    crit = state.get("critique")
    loops = state.get("research_loops", 0)
    revs = state.get("revision_count", 0)
    searches = state.get("searches_used", 0)

    updates: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []

    if gaps and loops < MAX_RESEARCH_LOOPS and searches < MAX_SEARCHES:
        return {
            "route": "researcher",
            "research_loops": loops + 1,
            "trace": [
                trace_event(NODE, "route", to="researcher", why="evidence gaps",
                            gaps=len(gaps), loop=f"{loops + 1}/{MAX_RESEARCH_LOOPS}",
                            searches=f"{searches}/{MAX_SEARCHES}")
            ],
        }

    if gaps:
        why = (
            "search budget spent"
            if searches >= MAX_SEARCHES
            else "research loop budget spent"
        )
        updates["gaps"] = []
        updates["unclosed_gaps"] = list(state.get("unclosed_gaps") or []) + gaps
        trace.append(
            trace_event(NODE, "retire_gaps", count=len(gaps), why=why,
                        loops=f"{loops}/{MAX_RESEARCH_LOOPS}",
                        searches=f"{searches}/{MAX_SEARCHES}")
        )

    def out(route: str, why: str, **detail: Any) -> dict[str, Any]:
        return {
            **updates,
            "route": route,
            "trace": trace + [trace_event(NODE, "route", to=route, why=why, **detail)],
        }

    if draft is None:
        return out("writer", "no draft")

    if crit is None:
        return out("finalize", "no critique (unexpected)")

    if crit.get("verdict") == "pass":
        scores = crit.get("scores") or {}
        return out("finalize", "critic passed",
                   min_score=min(scores.values()) if scores else None)

    if revs >= MAX_REVISIONS:
        return out("finalize", "revision budget exhausted", degraded=True)

    target = crit.get("target", "writer")
    if target not in CRITIC_TARGETS:
        trace.append(trace_event(NODE, "retarget", requested=target, to="writer",
                                 why="not a routable repair target"))
        target = "writer"
    res = out(target, "critic requested revision",
              revision=f"{revs + 1}/{MAX_REVISIONS}")
    res["revision_count"] = revs + 1
    return res


def finalize(state: ReportState) -> dict[str, Any]:
    draft = (state.get("draft") or "").rstrip()
    crit = state.get("critique") or {}
    unclosed = state.get("unclosed_gaps") or []
    degraded = crit.get("verdict") == "revise"

    if degraded or unclosed:
        lines = ["", "", "---", "", "## Known limitations", ""]
        for issue in crit.get("issues", []):
            problem = (issue.get("problem") or "").strip()
            name = (issue.get("criterion") or "").replace("_", " ")
            if problem:
                lines.append(f"- {problem} ({name})")
        for g in unclosed:
            lines.append(f"- Evidence gap not closed within the search budget: {g}")
        draft = draft + "\n".join(lines)

    return {
        "draft": draft,
        "route": "done",
        "trace": [
            trace_event("finalize", "shipped", degraded=degraded,
                        unclosed_gaps=len(unclosed), words=len(draft.split()))
        ],
    }
