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

from src.graph.nodes.writer import LIMITS_HEADER
from src.graph.state import MAX_SEARCHES, ReportState, trace_event

NODE = "researcher"
ANALYST = "analyst"
WRITER = "writer"
CRITIC = "critic"


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
    """Offline double for the real Analyst. Raises one gap on every pass, so the routing
    check exercises the research loop and then the retirement at the end of it.

    It used to go quiet after the first loop, and termination rested on that. It now
    rests on the supervisor's loop cap instead, which is where it rests in the real
    graph: pass two raises the gap again, the cap is spent, and the supervisor retires
    it into unclosed_gaps rather than chasing it.

    That retirement is the point. unclosed_gaps is the field the Writer reads to decide
    whether the report needs a Known limitations section, and while this node went
    silent the field stayed empty for the entire free run — so the section, and the
    merge in finalize that deduplicates it, could only ever be exercised by paying for
    a live run. One did, on 2026-08-02, and shipped two sections with one name.
    """
    findings = list(state.get("findings") or [])

    if not findings:
        return {
            "outline": None,
            "gaps": [],
            "trace": [trace_event(ANALYST, "skipped", fake=True,
                                  why="no findings to analyse")],
        }

    ids = [f["id"] for f in findings[:6]]
    gaps = ["fake gap: evidence missing for the second section"]
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
    trace fields as the real one.

    unclosed_gaps, not gaps. By the time the Writer runs, gaps is always empty — the
    Researcher clears it on the way past and the supervisor clears it when retiring —
    so a double keyed on gaps would silently never emit the section. unclosed_gaps is
    what the real Writer reads (writer.py, _write_fresh's `unclosed`), and it is the
    field the supervisor writes when it gives up chasing one.
    """
    outline = state.get("outline") or ""
    findings = list(state.get("findings") or [])
    unclosed = list(state.get("unclosed_gaps") or [])
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

    # The real Writer is instructed to close with this block when the run has gaps it
    # could not close, and to use the header verbatim because downstream tooling matches
    # on it. Imported rather than restated: a second copy of the literal is how judge.py
    # ended up blinding on a string the Writer had never been told to produce.
    #
    # Last, after the revision marker, because the instruction is "at the very end,
    # AFTER the closing section" — and because finalize merges into this block by
    # scanning forward from the header to the next heading.
    #
    # \n throughout. renderMarkdown splits blocks on \n{2,} and a \r\n\r\n does not
    # match it, which collapses the whole report into one unparsed block.
    if unclosed:
        bullets = "\n".join(f"- {g}" for g in unclosed)
        draft += f"\n\n{LIMITS_HEADER}\n\n{bullets}"

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
                        has_limits=LIMITS_HEADER in draft, tokens=0, **extra)
        ],
    }


def _span_from(draft: str) -> str:
    """A span quoted verbatim out of the draft.

    The Writer discards any issue whose span it cannot find and falls back to a full
    rewrite. A double that invents a span would make the routing check exercise the
    fallback rather than the surgical path — which is precisely the defect the real
    stub Critic shipped with in Phase 4, silently, for a whole phase.
    """
    for para in draft.split("\n\n"):
        stripped = para.strip()
        if len(stripped.split()) > 6 and not stripped.startswith("#"):
            head = stripped[:60]
            return head.rsplit(" ", 1)[0] if " " in head else head
    return ""


def critic(state: ReportState) -> dict[str, Any]:
    """Offline double for the real Critic. Fails once, then passes, so the routing
    check still exercises one revision and then terminates."""
    draft = state.get("draft") or ""
    if not draft.strip():
        return {
            "critique": None,
            "trace": [trace_event(CRITIC, "skipped", fake=True, why="no draft to review")],
        }

    first = state.get("critique") is None
    span = _span_from(draft)
    scores = {
        "factual_grounding": 4,
        "structural_coherence": 2 if first else 4,
        "depth_of_analysis": 3 if first else 4,
        "citation_integrity": 5,
        "absence_of_filler": 4,
    }
    verdict = "revise" if first and span else "pass"
    critique: dict[str, Any] = {
        "scores": scores,
        "verdict": verdict,
        "target": "writer",
        "issues": (
            [
                {
                    "span": span,
                    "criterion": "structural_coherence",
                    "problem": "This passage does not connect to what follows.",
                    "fix": "Add the causal link forward to the next section.",
                }
            ]
            if verdict == "revise"
            else []
        ),
        "summary": "fake critique",
        "broken_citations": [],
        "citation_cap_applied": False,
    }
    return {
        "critique": critique,
        "token_log": [
            {
                "node": CRITIC,
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
            trace_event(CRITIC, "scored", fake=True, verdict=verdict,
                        target="writer" if verdict == "revise" else "-",
                        worst=f"structural_coherence={min(scores.values())}",
                        issues_raw=len(critique["issues"]),
                        issues_kept=len(critique["issues"]),
                        dropped_ungrounded=0, broken_cites=0, cite_cap=False, tokens=0)
        ],
    }
