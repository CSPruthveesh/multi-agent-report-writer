"""Remaining stubs. Each gets deleted from here when its phase lands.

    critic -> Phase 5

Keeping the unbuilt nodes as stubs means the graph always runs end to end, so you
can exercise real routing against a real Researcher without having built the other
four. Do not let these linger past their phase — a stub that silently survives into
Phase 9 would quietly fake half your results.
"""

from __future__ import annotations

from typing import Any

from src.graph.state import ReportState, trace_event


def _span_from(draft: str) -> str:
    """A span quoted verbatim out of the draft.

    The Writer drops any issue whose span it cannot find, so a hardcoded span means
    the surgical revision path never runs and every forced revision costs a full
    rewrite. This stub shipped with a span from the old stub report — text no real
    draft contains — and the full-graph run showed the Writer silently redrafting
    instead of editing. A real Critic must quote too, for the same reason.
    """
    for para in draft.split("\n\n"):
        stripped = para.strip()
        if len(stripped.split()) > 25 and not stripped.startswith("#"):
            head = stripped[:90]
            return head.rsplit(" ", 1)[0] if " " in head else head
    return ""


def critic(state: ReportState) -> dict[str, Any]:
    """STUB — Phase 5. Reads draft + findings, writes critique."""
    # Fail on the first critique, then pass. Keyed off `critique` rather than a
    # counter so the gap loop cannot accidentally skip this path.
    first = state.get("critique") is None
    span = _span_from(state.get("draft") or "")
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
                    "span": span,
                    "criterion": "structural_coherence",
                    "problem": "This passage states its point but does not connect it to "
                               "what follows, so the section reads as a list rather than "
                               "an argument.",
                    "fix": "Add the causal link forward to the next section.",
                }
            ]
            if first and span
            else []
        ),
    }
    return {
        "critique": crit,
        "trace": [trace_event("critic", "scored", stub=True, verdict=crit["verdict"],
                              min_score=min(scores.values()))],
    }
