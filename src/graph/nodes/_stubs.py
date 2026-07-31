from __future__ import annotations

from typing import Any

from src.graph.state import ReportState, trace_event


def analyst(state: ReportState) -> dict[str, Any]:
    findings = state.get("findings") or []
    loops = state.get("research_loops", 0)

    gaps = ["stub gap: no evidence on the second half of the question"] if loops == 0 else []

    ids = [f["id"] for f in findings[:6]]
    outline = "\n".join(
        [
            f"## Section 1 — Framing [{', '.join(ids[:2])}]",
            f"## Section 2 — Evidence [{', '.join(ids[2:4])}]",
            f"## Section 3 — Implications [{', '.join(ids[4:6])}]",
        ]
    )
    return {
        "outline": outline,
        "gaps": gaps,
        "trace": [trace_event("analyst", "outlined", stub=True, sections=3, gaps=len(gaps))],
    }


def writer(state: ReportState) -> dict[str, Any]:
    revising = state.get("critique") is not None
    rev = state.get("revision_count", 0)
    findings = state.get("findings") or []
    cites = ", ".join(f["id"] for f in findings[:4]) or "F001"
    body = "\n\n".join(
        [
            "# Stub Report",
            f"Section one asserts something and cites it [{cites}].",
            "Section two builds on it.",
            "Section three draws the implication.",
        ]
    )
    if revising:
        body += f"\n\n(revision {rev})"
    return {
        "draft": body,
        "trace": [trace_event("writer", "revised" if revising else "drafted",
                              stub=True, revision=rev)],
    }


def critic(state: ReportState) -> dict[str, Any]:
    first = state.get("critique") is None
    scores = {
        "factual_grounding": 4,
        "structural_coherence": 2 if first else 4,
        "depth_of_analysis": 3 if first else 4,
        "citation_integrity": 5,
        "absence_of_filler": 4,
    }
    crit: dict[str, Any] = {
        "scores": scores,
        "verdict": "revise" if first else "pass",
        "target": "writer",
        "issues": (
            [
                {
                    "span": "Section two builds on it",
                    "criterion": "structural_coherence",
                    "problem": "stub problem",
                    "fix": "stub fix",
                }
            ]
            if first
            else []
        ),
    }
    return {
        "critique": crit,
        "trace": [trace_event("critic", "scored", stub=True, verdict=crit["verdict"],
                              min_score=min(scores.values()))],
    }
