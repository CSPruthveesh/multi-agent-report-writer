"""Offline doubles for the nodes that cost money, used only by the routing check.

Not stubs. A stub stands in for a node nobody has written; these stand in for nodes
that exist, so that `python -m src.graph.build` can exercise real routing for free.
They must never appear in STUBBED — see section 8 of the Phase 2 write-up.

One rule: every double mirrors the state effects of the node it replaces, and stamps
fake=True into its trace so no output of it can be mistaken for a real run. Add a
double here whenever a node becomes real, or the routing check quietly starts
spending quota — that has now happened twice.
"""

from __future__ import annotations

from typing import Any

from src.graph.state import MAX_SEARCHES, ReportState, trace_event

NODE = "researcher"
ANALYST = "analyst"
WRITER = "writer"


def researcher(state: ReportState) -> dict[str, Any]:
    gaps = list(state.get("gaps") or [])
    existing = list(state.get("findings") or [])
    used = state.get("searches_used", 0)
    remaining = MAX_SEARCHES - used

    if remaining <= 0:
        return {
            "unaddressed_gaps": gaps,
            "trace": [
                trace_event(NODE, "skipped", fake=True, why="search budget exhausted",
                            used=used, left_for_supervisor=len(gaps))
            ],
        }

    if gaps:
        queries = gaps[:remaining]
        mode = "gap-driven"
    else:
        queries = ["fake query"] * min(3, remaining)
        mode = "cold-start"

    kept = [
        {
            "id": f"F{len(existing) + i:03d}",
            "claim": f"fake finding {len(existing) + i} ({mode})",
            "source_url": "https://example.com/fake",
            "confidence": "medium",
        }
        for i in range(1, 2 * len(queries) + 1)
    ]
    unaddressed = gaps[len(queries):]

    return {
        "findings": kept,
        "searches_used": used + len(queries),
        "gaps": [],
        "unaddressed_gaps": unaddressed,
        "token_log": [
            {
                "node": NODE,
                "call_type": "fake",
                "in_tokens": 0,
                "out_tokens": 0,
                "total_tokens": 0,
                "latency_ms": 0,
                "attempts": 1,
                "model": "fake",
            }
        ],
        "trace": [
            trace_event(NODE, "searched", fake=True, mode=mode, queries=len(queries),
                        found=len(kept), kept=len(kept), dropped_dupes=0,
                        unaddressed=len(unaddressed),
                        budget=f"{used + len(queries)}/{MAX_SEARCHES}", tokens=0)
        ],
    }


def analyst(state: ReportState) -> dict[str, Any]:
    """Offline double for the real Analyst. Raises one gap on the first pass so the
    routing check still exercises the research loop, then none, so it terminates."""
    findings = list(state.get("findings") or [])
    loops = state.get("research_loops", 0)

    if not findings:
        return {
            "outline": None,
            "gaps": [],
            "trace": [trace_event(ANALYST, "skipped", fake=True,
                                  why="no findings to analyse")],
        }

    ids = [f["id"] for f in findings[:6]]
    gaps = ["fake gap: evidence missing for the second section"] if loops == 0 else []
    outline = "\n".join(
        [
            "THESIS: a fake thesis the fake evidence supports",
            "",
            f"## Framing [{', '.join(ids[:2])}]",
            f"## Evidence [{', '.join(ids[2:4])}]",
            f"## Implications [{', '.join(ids[4:6])}]",
        ]
    )
    return {
        "outline": outline,
        "gaps": gaps,
        "token_log": [
            {
                "node": ANALYST,
                "call_type": "fake",
                "in_tokens": 0,
                "out_tokens": 0,
                "total_tokens": 0,
                "latency_ms": 0,
                "attempts": 1,
                "model": "fake",
            }
        ],
        "trace": [
            trace_event(ANALYST, "outlined", fake=True, sections=3, tensions=0,
                        gaps_raw=len(gaps), gaps_kept=len(gaps), dropped_unblocking=0,
                        dropped_repeat=0, revision=False, tokens=0)
        ],
    }


def writer(state: ReportState) -> dict[str, Any]:
    """Offline double for the real Writer. Echoes the outline it was handed, so the
    routing check still shows what the node upstream produced, and reports the same
    trace fields as the real one."""
    outline = state.get("outline") or ""
    findings = list(state.get("findings") or [])
    revising = state.get("critique") is not None
    rev = state.get("revision_count", 0)

    if not findings or not outline:
        return {
            "draft": "",
            "trace": [trace_event(WRITER, "skipped", fake=True,
                                  why="no findings or no outline")],
        }

    cites = ", ".join(f["id"] for f in findings[:4]) or "F001"
    draft = f"# Fake Report\n\n(outline received)\n\n{outline}\n\nFake prose citing [{cites}]."
    if revising:
        draft += f"\n\n(revision {rev})"

    words = len(draft.split())
    extra = {"edits_returned": 1, "edits_applied": 1, "fallback": False,
             "changed_pct": 2.0, "issues_addressed": 1} if revising else {}
    return {
        "draft": draft,
        "token_log": [
            {
                "node": WRITER,
                "call_type": "fake",
                "in_tokens": 0,
                "out_tokens": 0,
                "total_tokens": 0,
                "latency_ms": 0,
                "attempts": 1,
                "model": "fake",
            }
        ],
        "trace": [
            trace_event(WRITER, "revised" if revising else "drafted", fake=True,
                        revision=rev, words=words, in_range=False,
                        cited=len(findings[:4]), broken_cites=0, cite_repair=False,
                        tokens=0, **extra)
        ],
    }
