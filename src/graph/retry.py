"""Node-level failure containment — Phase 6.

THREE FAILURE CLASSES, THREE DIFFERENT HANDLERS
-----------------------------------------------
Conflating these is the most common design smell in agent projects. Name them
separately in code and you can name them separately in the interview.

  1. Transport      429, 500, timeout. The request never really happened.
     Handler:       with_backoff() in common/llm.py. Exponential, 3 attempts.
     State impact:  none. Invisible to the graph.

  2. Parse          200 OK, but the text does not fit the schema. Backoff cannot
                    help — the model needs to see its own error.
     Handler:       one re-prompt inside generate(), temperature forced to 0.
     State impact:  none, but logged as `<call_type>:parse_retry` so Phase 8 can
                    price it separately.

  3. Semantic       The Critic says the output is not good enough. Nothing failed.
     Handler:       supervisor routing, capped by MAX_REVISIONS.
     State impact:  increments revision_count. Fully visible in the trace.

This module handles what is left over: a node that raises anyway. One retry, then
degrade to a known-safe update and let the run continue. A report with one weak
section beats a stack trace, and a graph that crashes on a transient fault is not a
system anyone would deploy.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from langgraph.errors import GraphBubbleUp

from src.graph.state import ReportState, trace_event

log = logging.getLogger(__name__)

# What each node hands back when it has failed twice. Every value here must let the
# supervisor make progress — a degradation that leaves state unchanged causes the
# supervisor to make the same decision again, which is a spin, not a degradation.
DEGRADED: dict[str, dict[str, Any]] = {
    # Clear gaps, and hand nothing back on the handover field either: an
    # unaddressed_gaps entry left behind by a half-finished Researcher would be
    # retired as a declared limitation, which is a claim about evidence rather than
    # about a node that crashed.
    "researcher": {"gaps": [], "unaddressed_gaps": []},
    "analyst": {"outline": None, "gaps": []},
    "writer": {},                        # keep whatever draft exists
    "critic": {"critique": {"scores": {}, "verdict": "pass", "target": "writer",
                            "issues": [], "node_failed": True}},
    "supervisor": {"route": "finalize"},
    "finalize": {"route": "done"},
}


def resilient(name: str, fn: Callable[[ReportState], dict[str, Any]]):
    """Wrap a node so an exception degrades the run instead of killing it."""

    def wrapped(state: ReportState) -> dict[str, Any]:
        try:
            return fn(state)
        except GraphBubbleUp:
            # Control flow, not failure. interrupt() and Command() work by raising,
            # and GraphInterrupt subclasses Exception — so without this the wrapper
            # would catch a legitimate pause, retry the node, catch the second
            # interrupt and degrade. The approval gate would never open and the
            # symptom would be a node that mysteriously fails twice.
            #
            # LangGraph names the base class GraphBubbleUp precisely because these
            # signals have to pass through user code untouched.
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] raised %s: %s — retrying once", name, type(e).__name__, e)
            time.sleep(1.0)
            try:
                out = fn(state)
                out.setdefault("trace", []).append(
                    trace_event(name, "recovered", after=type(e).__name__)
                )
                return out
            except Exception as e2:  # noqa: BLE001
                log.error("[%s] failed twice (%s) — degrading", name, type(e2).__name__)
                out = dict(DEGRADED.get(name, {}))
                out["trace"] = [
                    trace_event(name, "degraded", error=type(e2).__name__,
                                detail=str(e2)[:120])
                ]
                return out

    wrapped.__name__ = name
    return wrapped
