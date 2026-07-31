"""Remaining stubs. Each gets deleted from here when its phase lands.

    writer -> Phase 4    critic -> Phase 5

Keeping the unbuilt nodes as stubs means the graph always runs end to end, so you
can exercise real routing against a real Researcher without having built the other
four. Do not let these linger past their phase — a stub that silently survives into
Phase 9 would quietly fake half your results.
"""

from __future__ import annotations

from typing import Any

from src.graph.state import ReportState, trace_event


def writer(state: ReportState) -> dict[str, Any]:
    """STUB — Phase 4. Reads outline + findings, writes draft."""
    revising = state.get("critique") is not None
    rev = state.get("revision_count", 0)
    findings = state.get("findings") or []
    outline = state.get("outline") or ""
    cites = ", ".join(f["id"] for f in findings[:4]) or "F001"

    # Echo the real outline so you can eyeball what the Analyst produced without
    # having built the Writer yet.
    body = "# Stub Report\n\n(outline received from analyst)\n\n" + outline
    body += f"\n\nStub prose citing [{cites}]."
    if revising:
        body += f"\n\n(revision {rev})"
    return {
        "draft": body,
        "trace": [trace_event("writer", "revised" if revising else "drafted",
                              stub=True, revision=rev)],
    }


def critic(state: ReportState) -> dict[str, Any]:
    """STUB — Phase 5. Reads draft + findings, writes critique."""
    # Fail on the first critique, then pass. Keyed off `critique` rather than a
    # counter so the gap loop cannot accidentally skip this path.
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
