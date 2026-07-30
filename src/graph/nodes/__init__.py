from __future__ import annotations

from typing import Any

from src.graph.state import MAX_REVISIONS, ROUTES, ReportState, trace_event

STUB = True


def researcher(state: ReportState) -> dict[str, Any]:
    gaps = state.get("gaps") or []
    n = len(state.get("findings") or [])
    new = [
        {
            "id": f"F{n + i:03d}",
            "claim": f"stub finding {n + i} for {'gap: ' + gaps[0][:30] if gaps else 'initial sweep'}",
            "source_url": "https://example.com/stub",
            "confidence": "medium",
        }
        for i in range(1, 4)
    ]
    return {
        "findings": new,
        "gaps": [],
        "trace": [trace_event("researcher", "searched", rounds=1, new_findings=len(new),
                              seeded_by_gaps=bool(gaps))],
    }


def analyst(state: ReportState) -> dict[str, Any]:
    findings = state.get("findings") or []
    rev = state.get("revision_count", 0)

    gaps = ["stub gap: no evidence on the second half of the question"] if rev == 0 and len(findings) <= 3 else []

    outline = (
        "## Section 1 — Framing [F001]\n"
        "## Section 2 — Evidence [F002, F003]\n"
        "## Section 3 — Implications [F004]"
    )
    return {
        "outline": outline,
        "gaps": gaps,
        "trace": [trace_event("analyst", "outlined", sections=3, gaps=len(gaps))],
    }


def writer(state: ReportState) -> dict[str, Any]:
    rev = state.get("revision_count", 0)
    revising = state.get("critique") is not None
    body = (
        "# Stub Report\n\n"
        "Section one asserts something and cites it [F001].\n\n"
        "Section two builds on it [F002, F003].\n\n"
        "Section three draws the implication [F004]."
    )
    if revising:
        body += f"\n\n(revision {rev})"
    return {
        "draft": body,
        "trace": [trace_event("writer", "revised" if revising else "drafted", revision=rev)],
    }


def critic(state: ReportState) -> dict[str, Any]:
    if state.get("critique") is None:
        crit = {
            "scores": {
                "factual_grounding": 4,
                "structural_coherence": 2,
                "depth_of_analysis": 3,
                "citation_integrity": 5,
                "absence_of_filler": 4,
            },
            "verdict": "revise",
            "target": "writer",
            "issues": [
                {
                    "span": "Section two builds on it",
                    "criterion": "structural_coherence",
                    "problem": "stub problem",
                    "fix": "stub fix",
                }
            ],
        }
    else:
        crit = {
            "scores": {
                "factual_grounding": 4,
                "structural_coherence": 4,
                "depth_of_analysis": 4,
                "citation_integrity": 5,
                "absence_of_filler": 4,
            },
            "verdict": "pass",
            "target": "writer",
            "issues": [],
        }
    return {
        "critique": crit,
        "trace": [trace_event("critic", "scored", verdict=crit["verdict"],
                              min_score=min(crit["scores"].values()))],
    }


def supervisor(state: ReportState) -> dict[str, Any]:
    gaps = state.get("gaps") or []
    draft = state.get("draft")
    crit = state.get("critique")
    rev = state.get("revision_count", 0)

    if gaps and rev < MAX_REVISIONS:
        return {
            "route": "researcher",
            "revision_count": rev + 1,
            "trace": [trace_event("supervisor", "route", to="researcher",
                                  why="evidence gaps", revision=rev + 1)],
        }

    if draft is None:
        return {
            "route": "writer",
            "trace": [trace_event("supervisor", "route", to="writer", why="no draft")],
        }

    if crit is None:
        return {
            "route": "finalize",
            "trace": [trace_event("supervisor", "route", to="finalize", why="no critique")],
        }

    if crit.get("verdict") == "pass":
        return {
            "route": "finalize",
            "trace": [trace_event("supervisor", "route", to="finalize", why="critic passed")],
        }

    if rev >= MAX_REVISIONS:
        return {
            "route": "finalize",
            "trace": [trace_event("supervisor", "route", to="finalize",
                                  why="revision budget exhausted", degraded=True)],
        }

    target = crit.get("target", "writer")
    if target not in ROUTES:
        return {
            "route": "finalize",
            "trace": [trace_event("supervisor", "route", to="finalize",
                                  why="unroutable critic target", target=target)],
        }

    return {
        "route": target,
        "revision_count": rev + 1,
        "trace": [trace_event("supervisor", "route", to=target,
                              why="critic requested revision", revision=rev + 1)],
    }


def finalize(state: ReportState) -> dict[str, Any]:
    draft = state.get("draft") or ""
    crit = state.get("critique") or {}
    gaps = state.get("gaps") or []
    degraded = crit.get("verdict") == "revise"

    if degraded or gaps:
        lines = ["", "---", "", "## Known limitations", ""]
        for issue in crit.get("issues", []):
            lines.append(f"- {issue.get('problem', '')} ({issue.get('criterion', '')})")
        for g in gaps:
            lines.append(f"- Unclosed evidence gap: {g}")
        draft = draft.rstrip() + "\n" + "\n".join(lines)

    return {
        "draft": draft,
        "route": "done",
        "trace": [trace_event("finalize", "shipped", degraded=degraded,
                              unclosed_gaps=len(gaps))],
    }
