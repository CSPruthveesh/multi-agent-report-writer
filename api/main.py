"""FastAPI backend — Phase 10.

    uv run uvicorn api.main:app --reload
    open http://localhost:8000

WHAT IS BEING DEMOED
--------------------
Not the report. The report is the least interesting output of this project — a
single agent writes a comparable one for a fraction of the cost, which is the whole
finding.

The demo is the TRACE: watching the gap loop fire, the critic route a revision, the
budgets hold, and the token counter climb past the baseline cost in real time. That
last one is the most persuasive thing here. Anyone can claim multi-agent is
expensive; watching the number cross 5x while the report is still being written
makes the argument without saying anything.

STREAMING
---------
`graph.astream(stream_mode="values")` yields the whole state after each super-step,
not a per-node delta. That is the right mode here even though it sends more: token
totals are recomputed from the full `token_log` on every frame rather than
accumulated, so a dropped or duplicated frame cannot drift the counter away from what
the ledger says. Trace events are sent by slicing off the ones already emitted.

Each frame carries the trace event plus a running cost total, so the client never
computes anything — it just renders what arrives.

NO DOLLAR FIGURES UNLESS SOMEONE SUPPLIED RATES
-----------------------------------------------
cost_usd returns None unless PRICE_IN_PER_M and PRICE_OUT_PER_M are set, because
8fd69a3 removed dollars from every report after finding the defaults were
placeholders nobody had checked. Its argument was that a placeholder which prints is
indistinguishable from a verified rate once it is in a table — and a demo is a bigger
table, not an exemption.

So usd is None here whenever it is unknown, `prices_supplied` says which case the
client is in, and the UI shows tokens and the multiple instead. Both are rate-free
and both are the numbers the README already quotes.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.analysis.cost import PRICES_SET, cost_usd
from src.common.io import RESULTS, load_topics
from src.graph.build import build
from src.graph.state import (
    MAX_RESEARCH_LOOPS,
    MAX_REVISIONS,
    MAX_SEARCHES,
    initial_state,
)

app = FastAPI(title="Multi-Agent Report Writer")
STATIC = Path(__file__).parent / "static"


class RunRequest(BaseModel):
    # Bounded because the public page lets anyone type anything and every run spends
    # real tokens. A one-word topic also gives the Researcher nothing to plan from,
    # so the floor is a quality guard as much as a cost one.
    topic: str = Field(min_length=15, max_length=300)
    topic_id: str | None = None


def _is_frozen(topic: str, topic_id: str | None) -> bool:
    """Whether this run is comparable to the recorded baseline.

    The baseline is the mean of six single-agent runs on six specific topics. Against
    anything else it is the cost of a different task, and putting a multiple next to a
    topic somebody just invented would be the same error as the placeholder prices —
    a number that is arithmetically real and means nothing.
    """
    return any(
        t["id"] == topic_id and t["topic"] == topic for t in load_topics()
    )


def _usd(in_tok: float, out_tok: float) -> float | None:
    """Rounded dollars, or None when nobody supplied rates.

    round(None, 5) raises, and cost_usd returns None by design. Every caller here went
    through round() directly, so the endpoint that loads the page 500'd before the
    client saw a single frame — and in _stream the crash happened above the try, so it
    could not even be reported as an error event.
    """
    v = cost_usd(int(in_tok), int(out_tok))
    return round(v, 5) if v is not None else None


def _baseline_reference() -> dict[str, Any]:
    """Mean baseline cost, so the UI can show the multiple as it climbs.

    Without this the token counter is a number with no meaning. With it, the
    counter is an argument.
    """
    runs = []
    for p in (RESULTS / "baseline").glob("*/run.json"):
        runs.append(json.loads(p.read_text(encoding="utf-8")))
    if not runs:
        return {"tokens": 0, "usd": 0.0, "n": 0}
    tk = [r["tokens"]["total"] for r in runs]
    ins = sum(r["tokens"]["total_in"] for r in runs) / len(runs)
    outs = sum(r["tokens"]["total_out"] for r in runs) / len(runs)
    return {
        "tokens": round(sum(tk) / len(tk)),
        "usd": _usd(ins, outs),
        "n": len(runs),
    }


@app.get("/api/topics")
def topics() -> dict[str, Any]:
    return {
        "topics": load_topics(),
        "baseline": _baseline_reference(),
        "budgets": {
            "searches": MAX_SEARCHES,
            "research_loops": MAX_RESEARCH_LOOPS,
            "revisions": MAX_REVISIONS,
        },
        # So the client can distinguish "no rates supplied" from "costs nothing".
        "prices_supplied": PRICES_SET,
    }


@app.get("/api/comparison")
def comparison() -> dict[str, Any]:
    p = RESULTS / "comparison.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream(topic: str, comparable: bool) -> AsyncIterator[str]:
    graph = build()
    state = initial_state(topic)

    spent_in = spent_out = spent = 0
    seen_trace = 0
    baseline = _baseline_reference()

    yield _sse("start", {"topic": topic, "baseline": baseline,
                         "comparable": comparable})

    try:
        async for chunk in graph.astream(state, config={"recursion_limit": 40},
                                         stream_mode="values"):
            # Token totals, recomputed from the authoritative log rather than
            # accumulated client-side.
            records = chunk.get("token_log") or []
            spent_in = sum(r.get("in_tokens", 0) for r in records)
            spent_out = sum(r.get("out_tokens", 0) for r in records)
            spent = sum(r.get("total_tokens", 0) for r in records)

            trace = chunk.get("trace") or []
            for ev in trace[seen_trace:]:
                yield _sse("node", {
                    "event": ev,
                    "cost": {
                        "tokens": spent,
                        "usd": _usd(spent_in, spent_out),
                        "multiple": round(spent / baseline["tokens"], 2)
                        if comparable and baseline["tokens"] else None,
                    },
                    "budgets": {
                        "searches": chunk.get("searches_used", 0),
                        "research_loops": chunk.get("research_loops", 0),
                        "revisions": chunk.get("revision_count", 0),
                    },
                    "counts": {
                        "findings": len(chunk.get("findings") or []),
                        "has_outline": bool(chunk.get("outline")),
                        "draft_words": len((chunk.get("draft") or "").split()),
                    },
                })
                # Let the event flush before the next node's work lands, so the UI
                # animates instead of jumping.
                await asyncio.sleep(0.05)
            seen_trace = len(trace)

            if chunk.get("route") == "done":
                crit = chunk.get("critique") or {}
                yield _sse("done", {
                    "report": chunk.get("draft") or "",
                    "findings": chunk.get("findings") or [],
                    "scores": crit.get("scores") or {},
                    "unclosed_gaps": chunk.get("unclosed_gaps") or [],
                    "cost": {
                        "tokens": spent,
                        "usd": _usd(spent_in, spent_out),
                        "multiple": round(spent / baseline["tokens"], 2)
                        if comparable and baseline["tokens"] else None,
                    },
                })
    except Exception as e:  # noqa: BLE001
        yield _sse("error", {"error": type(e).__name__, "detail": str(e)[:200]})


@app.post("/api/run")
async def run(req: RunRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream(req.topic, _is_frozen(req.topic, req.topic_id)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index() -> FileResponse:
    """The page for someone who wants a report on their own topic."""
    return FileResponse(STATIC / "app.html")


@app.get("/dev")
def dev() -> FileResponse:
    """The instrument view: full trace, frozen topics, cost against baseline.

    Kept separate rather than hidden behind a flag on one page. The two answer
    different questions — this one is for reading the machinery, / is for getting a
    report — and a single page trying to do both would do the second one badly.
    """
    return FileResponse(STATIC / "index.html")
