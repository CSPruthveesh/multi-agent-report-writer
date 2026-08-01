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
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.analysis.cost import PRICES_SET, cost_usd
from src.common.io import RESULTS, load_topics
from src.graph.build import build
from src.graph.checkpoint import async_sqlite_checkpointer, thread_config
from src.graph.state import (
    MAX_RESEARCH_LOOPS,
    MAX_REVISIONS,
    MAX_SEARCHES,
    initial_state,
)

app = FastAPI(title="Multi-Agent Report Writer")
STATIC = Path(__file__).parent / "static"

# Both pages share trace.css and trace.js, so the directory needs serving rather than
# two hand-written FileResponse routes. / and /dev stay explicit below: they map a
# reader to a page, which is a routing decision, not a file lookup.
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ResumeRequest(BaseModel):
    """A decision on a parked thread.

    `seen` is how many trace events the client already has. The resumed stream replays
    from a state whose trace holds everything so far, so without it the client would
    redraw the whole run under the rows it is already showing.
    """
    thread_id: str
    decision: dict[str, Any]
    seen: int = 0


class RunRequest(BaseModel):
    # Bounded because the public page lets anyone type anything and every run spends
    # real tokens. A one-word topic also gives the Researcher nothing to plan from,
    # so the floor is a quality guard as much as a cost one.
    topic: str = Field(min_length=15, max_length=300)
    topic_id: str | None = None
    # Inserts the approval node before the Writer. Needs a checkpointer: interrupt()
    # raises, LangGraph parks the thread on disk, and the answer arrives on a separate
    # request — there is nowhere to park without one.
    hitl: bool = False


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


@app.post("/api/resume")
async def resume(req: ResumeRequest) -> StreamingResponse:
    return StreamingResponse(
        _resume(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/comparison")
def comparison() -> dict[str, Any]:
    p = RESULTS / "comparison.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _pending_interrupt(graph, cfg) -> dict[str, Any] | None:
    """The payload an interrupt parked, or None if the thread simply finished.

    astream ends the same way whether the graph completed or stopped at interrupt(),
    so the difference is only visible in the checkpointed state afterwards.

    aget_state, not get_state. The saver behind a streaming graph is the async one, and
    it refuses sync calls from the event loop with a message saying exactly that — the
    graph ran to the gate and then fell over reading its own parked state.
    """
    snap = await graph.aget_state(cfg)
    for task in getattr(snap, "tasks", ()) or ():
        for itr in getattr(task, "interrupts", ()) or ():
            value = getattr(itr, "value", None)
            if value is not None:
                return value
    return None


async def _drive(graph, cfg, payload, *, comparable: bool, seen_trace: int,
                 thread_id: str | None) -> AsyncIterator[str]:
    """Stream one leg of a run: from the start, or from a resume.

    Both legs are the same loop. A gated run is two calls to this — the first ends on
    an approval event, the second is handed Command(resume=decision) as its payload and
    picks up the trace where the client left off.
    """
    spent_in = spent_out = spent = 0
    baseline = _baseline_reference()
    finished = False

    def cost() -> dict[str, Any]:
        return {
            "tokens": spent,
            "usd": _usd(spent_in, spent_out),
            "multiple": round(spent / baseline["tokens"], 2)
            if comparable and baseline["tokens"] else None,
        }

    try:
        async for chunk in graph.astream(payload, config=cfg, stream_mode="values"):
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
                    "cost": cost(),
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
                finished = True
                crit = chunk.get("critique") or {}
                yield _sse("done", {
                    "report": chunk.get("draft") or "",
                    "findings": chunk.get("findings") or [],
                    "scores": crit.get("scores") or {},
                    "unclosed_gaps": chunk.get("unclosed_gaps") or [],
                    "cost": cost(),
                })

        if not finished and thread_id:
            payload = await _pending_interrupt(graph, cfg)
            if payload:
                yield _sse("approval", {
                    "thread_id": thread_id,
                    "interrupt": payload,
                    # What the client has drawn, handed back on resume so the replayed
                    # state does not redraw the run under the rows already on screen.
                    "seen": seen_trace,
                    "cost": cost(),
                })
    except Exception as e:  # noqa: BLE001
        yield _sse("error", {"error": type(e).__name__, "detail": str(e)[:200]})


async def _close(saver) -> None:
    """Hand the checkpointer's connection back.

    aiosqlite runs each connection on its own thread, so a leaked one keeps a thread
    alive for the life of the process — one per gated request, none of them doing
    anything. The probe found it by refusing to exit after printing its results.
    """
    conn = getattr(saver, "conn", None)
    if conn is not None:
        try:
            await conn.close()
        except Exception:  # noqa: BLE001, S110
            pass  # A run that finished should not fail on tidying up after itself.


async def _stream(topic: str, comparable: bool, hitl: bool) -> AsyncIterator[str]:
    # A gate needs somewhere to park. Without a checkpointer interrupt() raises into a
    # graph that cannot save itself, so the gated build gets one and the ungated build
    # deliberately does not — a checkpointer on every request would write a thread file
    # for runs nobody will ever resume.
    thread_id = uuid.uuid4().hex[:12] if hitl else None
    saver = await async_sqlite_checkpointer() if hitl else None
    graph = build(checkpointer=saver, hitl=True) if hitl else build()
    cfg = thread_config(thread_id) if hitl else {"recursion_limit": 40}

    yield _sse("start", {"topic": topic, "baseline": _baseline_reference(),
                         "comparable": comparable, "hitl": hitl,
                         "thread_id": thread_id})

    try:
        async for frame in _drive(graph, cfg, initial_state(topic, hitl=hitl),
                                  comparable=comparable, seen_trace=0,
                                  thread_id=thread_id):
            yield frame
    finally:
        await _close(saver)


async def _resume(req: ResumeRequest) -> AsyncIterator[str]:
    """Answer a parked thread and stream the rest of the run."""
    saver = await async_sqlite_checkpointer()
    graph = build(checkpointer=saver, hitl=True)
    cfg = thread_config(req.thread_id)
    try:
        async for frame in _drive(graph, cfg, Command(resume=req.decision),
                                  comparable=False, seen_trace=req.seen,
                                  thread_id=req.thread_id):
            yield frame
    finally:
        await _close(saver)


@app.post("/api/run")
async def run(req: RunRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream(req.topic, _is_frozen(req.topic, req.topic_id), req.hitl),
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
