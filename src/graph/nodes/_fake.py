from __future__ import annotations

from typing import Any

from src.graph.state import MAX_SEARCHES, ReportState, trace_event

NODE = "researcher"


def researcher(state: ReportState) -> dict[str, Any]:
    gaps = list(state.get("gaps") or [])
    existing = list(state.get("findings") or [])
    used = state.get("searches_used", 0)
    remaining = MAX_SEARCHES - used

    if remaining <= 0:
        return {
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
        "gaps": unaddressed,
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
