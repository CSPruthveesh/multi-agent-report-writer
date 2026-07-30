from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from src.common.io import ROOT, save_run
from src.common.schemas import Finding
from src.graph import nodes
from src.graph.state import ROUTES, ReportState, initial_state


def build(checkpointer: Any = None):
    g = StateGraph(ReportState)

    g.add_node("researcher", nodes.researcher)
    g.add_node("analyst", nodes.analyst)
    g.add_node("supervisor", nodes.supervisor)
    g.add_node("writer", nodes.writer)
    g.add_node("critic", nodes.critic)
    g.add_node("finalize", nodes.finalize)

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


def run(topic_id: str, topic: str, *, verbose: bool = True) -> dict[str, Any]:
    if nodes.STUB:
        raise RuntimeError(
            "graph nodes are still stubs — refusing to write results/multiagent/. "
            "results/ is the experimental record and a stub run is indistinguishable "
            "from a real one once it is on disk. Set STUB = False in src/graph/nodes "
            "when the real nodes land. Use `python -m src.graph.build` to exercise "
            "routing without writing anything."
        )

    t0 = time.perf_counter()

    final = cast(
        ReportState,
        GRAPH.invoke(initial_state(topic), config={"recursion_limit": 40}),
    )

    wall_ms = int((time.perf_counter() - t0) * 1000)

    if verbose:
        for ev in final.get("trace", []):
            detail = " ".join(f"{k}={v}" for k, v in ev.items() if k not in ("node", "action"))
            print(f"  {ev['node']:<11} {ev['action']:<9} {detail}")

    findings = [Finding(**f) for f in (final.get("findings") or [])]

    records = final.get("token_log") or []
    by_node: dict[str, dict[str, int]] = {}
    for r in records:
        d = by_node.setdefault(r.get("node", "unknown"), {"calls": 0, "in": 0, "out": 0, "total": 0, "ms": 0})
        d["calls"] += 1
        d["in"] += r.get("in_tokens", 0)
        d["out"] += r.get("out_tokens", 0)
        d["total"] += r.get("total_tokens", 0)
        d["ms"] += r.get("latency_ms", 0)

    ledger_dict = {
        "calls": len(records),
        "total_in": sum(r.get("in_tokens", 0) for r in records),
        "total_out": sum(r.get("out_tokens", 0) for r in records),
        "total": sum(r.get("total_tokens", 0) for r in records),
        "by_node": by_node,
        "records": records,
    }

    out = save_run(
        system="multiagent",
        topic_id=topic_id,
        report=final.get("draft") or "",
        findings=findings,
        ledger_dict=ledger_dict,
        wall_ms=wall_ms,
        extra={
            "revisions": final.get("revision_count", 0),
            "unclosed_gaps": final.get("gaps") or [],
            "final_critique": final.get("critique"),
            "trace": final.get("trace") or [],
        },
    )
    return {"dir": out, "state": final}


def export_mermaid(path: str = "docs/graph.mmd") -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(GRAPH.get_graph().draw_mermaid(), encoding="utf-8")
    return p


if __name__ == "__main__":
    state = GRAPH.invoke(initial_state("stub topic"), config={"recursion_limit": 40})
    print("\n--- trace ---")
    for ev in state["trace"]:
        detail = " ".join(f"{k}={v}" for k, v in ev.items() if k not in ("node", "action"))
        print(f"  {ev['node']:<11} {ev['action']:<9} {detail}")
    print("\n--- final ---")
    print("findings      ", len(state["findings"]))
    print("revisions     ", state["revision_count"])
    print("route         ", state["route"])
    print("verdict       ", (state.get("critique") or {}).get("verdict"))
    print("draft chars   ", len(state.get("draft") or ""))
    print("\n" + json.dumps(state["trace"], indent=2)[:400] + " ...")
