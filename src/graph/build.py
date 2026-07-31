from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from src.common.io import ROOT, save_run
from src.common.schemas import Finding
from src.graph import nodes
from src.graph.checkpoint import sqlite_checkpointer, thread_config
from src.graph.nodes import _fake
from src.graph.retry import resilient
from src.graph.state import (
    MAX_RESEARCH_LOOPS,
    MAX_REVISIONS,
    MAX_SEARCHES,
    ROUTES,
    ReportState,
    initial_state,
)


def build(checkpointer: Any = None, overrides: dict[str, Any] | None = None):
    fns: dict[str, Any] = {
        "researcher": nodes.researcher,
        "analyst": nodes.analyst,
        "supervisor": nodes.supervisor,
        "writer": nodes.writer,
        "critic": nodes.critic,
        "finalize": nodes.finalize,
    }
    unknown = set(overrides or ()) - set(fns)
    if unknown:
        raise ValueError(f"override for unknown node(s): {', '.join(sorted(unknown))}")
    fns.update(overrides or {})

    g = StateGraph(ReportState)

    # Every node wrapped: one retry, then degrade. See graph/retry.py for the three
    # failure classes and why they are handled in three different places. Overrides
    # are wrapped too — a fake that raises should degrade exactly like a real node,
    # or the chaos scenarios would be testing the wrapper rather than the graph.
    for name, fn in fns.items():
        g.add_node(name, resilient(name, fn))

    g.add_edge(START, "researcher")
    g.add_edge("researcher", "analyst")
    g.add_edge("analyst", "supervisor")
    g.add_edge("writer", "critic")
    g.add_edge("critic", "supervisor")
    g.add_edge("finalize", END)

    g.add_conditional_edges(
        "supervisor",
        lambda s: s["route"] if s.get("route") in ROUTES else "finalize",
        {r: r for r in ROUTES},
    )

    return g.compile(checkpointer=checkpointer)


GRAPH = build()


def _ledger_dict(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_node: dict[str, dict[str, int]] = {}
    for r in records:
        d = by_node.setdefault(r.get("node", "unknown"),
                               {"calls": 0, "in": 0, "out": 0, "total": 0, "ms": 0})
        d["calls"] += 1
        d["in"] += r.get("in_tokens", 0)
        d["out"] += r.get("out_tokens", 0)
        d["total"] += r.get("total_tokens", 0)
        d["ms"] += r.get("latency_ms", 0)
    return {
        "calls": len(records),
        "total_in": sum(r.get("in_tokens", 0) for r in records),
        "total_out": sum(r.get("out_tokens", 0) for r in records),
        "total": sum(r.get("total_tokens", 0) for r in records),
        "by_node": by_node,
        "records": records,
    }


def _print_trace(events: list[dict[str, Any]], seen: int = 0) -> int:
    for ev in events[seen:]:
        detail = " ".join(f"{k}={v}" for k, v in ev.items() if k not in ("node", "action"))
        print(f"  {ev['node']:<11}{ev['action']:<18}{detail}")
    return len(events)


def _persist(topic_id: str, final: ReportState, wall_ms: int) -> Path:
    findings = [Finding(**f) for f in (final.get("findings") or [])]
    return save_run(
        system="multiagent",
        topic_id=topic_id,
        report=final.get("draft") or "",
        findings=findings,
        ledger_dict=_ledger_dict(final.get("token_log") or []),
        wall_ms=wall_ms,
        extra={
            "revisions": final.get("revision_count", 0),
            "research_loops": final.get("research_loops", 0),
            "searches_used": final.get("searches_used", 0),
            "write_attempts": final.get("write_attempts", 0),
            "unclosed_gaps": final.get("unclosed_gaps") or [],
            "final_critique": final.get("critique"),
            "hitl": bool(final.get("hitl")),
            "trace": final.get("trace") or [],
        },
    )


def _guard_stubs() -> None:
    if nodes.STUBBED:
        raise RuntimeError(
            f"these nodes are still stubs: {', '.join(nodes.STUBBED)} — refusing to write "
            "results/multiagent/. results/ is the experimental record and a partly stubbed "
            "run is indistinguishable from a real one once it is on disk. Empty STUBBED in "
            "src/graph/nodes/__init__.py as each phase lands. Use `python -m src.graph.build` "
            "to exercise routing without writing anything."
        )


def run(
    topic_id: str,
    topic: str,
    *,
    verbose: bool = True,
    hitl: bool = False,
    thread_id: str | None = None,
    checkpoint: bool = True,
) -> dict[str, Any]:
    """Same signature as baseline.agent.run, plus persistence options.

    Returns early with status='awaiting_human' if an interrupt fires. Resume with
    resume_run(thread_id, decision).
    """
    _guard_stubs()

    tid = thread_id or f"{topic_id}-{uuid.uuid4().hex[:8]}"
    cp = sqlite_checkpointer() if checkpoint else None
    graph = build(checkpointer=cp) if cp else GRAPH
    cfg = thread_config(tid) if cp else {"recursion_limit": 40}

    t0 = time.perf_counter()
    final = cast(ReportState, graph.invoke(initial_state(topic, hitl=hitl), config=cfg))
    wall_ms = int((time.perf_counter() - t0) * 1000)

    seen = _print_trace(final.get("trace") or []) if verbose else 0

    # An interrupt leaves the thread parked rather than finished.
    snap = graph.get_state(cfg) if cp else None
    pending = []
    for task in (getattr(snap, "tasks", ()) or ()) if snap else ():
        for itr in getattr(task, "interrupts", ()) or ():
            pending.append(getattr(itr, "value", None))

    if pending:
        if verbose:
            print(f"\n  PAUSED — waiting for human input on thread {tid}")
        return {
            "status": "awaiting_human",
            "thread_id": tid,
            "interrupt": pending[0],
            "state": final,
            "_seen": seen,
        }

    out = _persist(topic_id, final, wall_ms)
    return {"status": "complete", "thread_id": tid, "dir": out, "state": final}


def resume_run(
    thread_id: str,
    decision: dict[str, Any],
    *,
    topic_id: str | None = None,
    verbose: bool = True,
    seen: int = 0,
) -> dict[str, Any]:
    """Resume a thread parked on an interrupt, or restarted after a crash.

    Pass decision={} to resume a crashed run — there is nothing to answer, the
    checkpointer just replays from the last completed node.
    """
    from langgraph.types import Command

    _guard_stubs()

    cp = sqlite_checkpointer()
    graph = build(checkpointer=cp)
    cfg = thread_config(thread_id)

    t0 = time.perf_counter()
    final = cast(
        ReportState,
        graph.invoke(Command(resume=decision) if decision else None, config=cfg),
    )
    wall_ms = int((time.perf_counter() - t0) * 1000)

    if verbose:
        _print_trace(final.get("trace") or [], seen)

    tid_topic = topic_id or thread_id.split("-")[0]
    out = _persist(tid_topic, final, wall_ms)
    return {"status": "complete", "thread_id": thread_id, "dir": out, "state": final}


def export_mermaid(path: str = "docs/graph.mmd") -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(GRAPH.get_graph().draw_mermaid(), encoding="utf-8")
    return p


if __name__ == "__main__":
    live = "--live" in sys.argv

    if live:
        print("--live: every node is real and will spend quota.")
        app = GRAPH
    else:
        print("routing check with fake nodes throughout — no API calls. "
              "Pass --live for the real ones.")
        app = build(overrides={"researcher": _fake.researcher, "analyst": _fake.analyst,
                               "writer": _fake.writer, "critic": _fake.critic})

    state = cast(ReportState, app.invoke(initial_state("stub topic"), config={"recursion_limit": 40}))
    print("\n--- trace ---")
    for ev in state["trace"]:
        detail = " ".join(f"{k}={v}" for k, v in ev.items() if k not in ("node", "action"))
        print(f"  {ev['node']:<11} {ev['action']:<9} {detail}")
    print("\n--- final ---")
    print("findings      ", len(state["findings"]))
    print("searches_used ", f"{state.get('searches_used', 0)}/{MAX_SEARCHES}")
    print("research_loops", f"{state.get('research_loops', 0)}/{MAX_RESEARCH_LOOPS}")
    print("revisions     ", f"{state['revision_count']}/{MAX_REVISIONS}")
    print("unclosed_gaps ", state.get("unclosed_gaps") or [])
    print("route         ", state["route"])
    print("verdict       ", (state.get("critique") or {}).get("verdict"))
    print("draft chars   ", len(state.get("draft") or ""))
    print("token rows    ", len(state.get("token_log") or []))
    print("still stubbed ", nodes.STUBBED)
    print("\n" + json.dumps(state["trace"], indent=2)[:400] + " ...")
