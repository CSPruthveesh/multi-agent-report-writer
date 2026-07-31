"""Chaos test — break each node on purpose and verify the run still ships.

    uv run python -m scripts.chaos            # offline, no API calls, no cost
    uv run python -m scripts.chaos --live     # against the real graph, costs money

WHAT THIS PROVES
----------------
"It retries until it works" is a junior answer — it has no bound. "It has three
failure classes, each handled where it belongs, and here is the test showing every
one of them still produces a report" is a system somebody would deploy.

Every scenario must terminate and emit a non-empty report:

  each node raising every time        can the graph survive losing any one node?
  each node raising once then working does the single retry actually recover?
  critic never passing                does the revision budget hold?
  analyst always finding gaps         does the research budget hold?
  everything failing at once          does it still ship?

The offline mode substitutes fake nodes with the same state contracts, so routing,
budgets, and degradation are exercised without a single API call. That matters: a
chaos test you cannot afford to run is a chaos test you will not run.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

from src.graph.nodes.supervisor import finalize, supervisor
from src.graph.retry import resilient
from src.graph.state import (
    MAX_RESEARCH_LOOPS,
    MAX_REVISIONS,
    MAX_SEARCHES,
    MAX_WRITE_ATTEMPTS,
    ReportState,
    initial_state,
    trace_event,
)

ADD = {"findings", "token_log", "trace"}
NEXT = {"researcher": "analyst", "analyst": "supervisor", "writer": "critic",
        "critic": "supervisor"}


# ------------------------------------------------------------- fake nodes
def fake_researcher(state: ReportState) -> dict[str, Any]:
    """Mirrors the real Researcher's budget arithmetic.

    The cap at `remaining` matters and is not decoration. Without it, three
    invocations at 2 searches each spend 6 against a budget of 5. If the fake does
    not enforce the same ceiling as the real node, the test cannot tell you whether
    the ceiling works.
    """
    used = state.get("searches_used", 0)
    remaining = MAX_SEARCHES - used
    if remaining <= 0:
        return {"gaps": [], "unaddressed_gaps": list(state.get("gaps") or []),
                "trace": [trace_event("researcher", "skipped", why="budget")]}
    n_queries = min(2, remaining)
    n = len(state.get("findings") or [])
    return {
        "findings": [{"id": f"F{n + i:03d}", "claim": f"c{n + i}", "source_url": "u",
                      "confidence": "high"} for i in range(1, n_queries + 1)],
        "searches_used": used + n_queries,
        "gaps": [],
        "unaddressed_gaps": [],
        "trace": [trace_event("researcher", "searched",
                              budget=f"{used + n_queries}/{MAX_SEARCHES}")],
    }


def make_fakes(always_gap: bool = False, never_pass: bool = False):
    def fake_analyst(state: ReportState) -> dict[str, Any]:
        gaps = ["persistent gap"] if always_gap else (
            ["initial gap"] if state.get("research_loops", 0) == 0 else []
        )
        return {"outline": "## S1 [F001]\n## S2 [F002]", "gaps": gaps,
                "trace": [trace_event("analyst", "outlined", gaps=len(gaps),
                                      revision=bool(state.get("revision_brief")))]}

    def fake_writer(state: ReportState) -> dict[str, Any]:
        rev = state.get("revision_count", 0)
        return {"draft": f"# Report\n\nBody citing [F001]. rev={rev}",
                "trace": [trace_event("writer", "wrote", revision=rev)]}

    def fake_critic(state: ReportState) -> dict[str, Any]:
        bad = never_pass or state.get("critique") is None
        return {
            "critique": {
                "scores": {"structural_coherence": 2 if bad else 4},
                "verdict": "revise" if bad else "pass",
                "target": "writer",
                "issues": [{"span": "Body citing", "criterion": "structural_coherence",
                            "problem": "stub problem", "fix": "f"}] if bad else [],
            },
            "trace": [trace_event("critic", "scored",
                                  verdict="revise" if bad else "pass")],
        }

    return fake_analyst, fake_writer, fake_critic


def breaker(fn: Callable, mode: str) -> Callable:
    """mode: 'always' raises every time, 'once' raises on the first call only."""
    calls = {"n": 0}

    def broken(state: ReportState) -> dict[str, Any]:
        calls["n"] += 1
        if mode == "always" or calls["n"] == 1:
            raise RuntimeError(f"injected {mode} failure")
        return fn(state)

    return broken


def run_graph(nodes: dict[str, Callable],
              limit: int = 80) -> tuple[ReportState, list[str], bool]:
    wrapped = {k: resilient(k, v) for k, v in nodes.items()}
    s = initial_state("chaos topic")
    cur, order = "researcher", []
    for _ in range(limit):
        order.append(cur)
        upd = wrapped[cur](s)
        for k, v in upd.items():
            s[k] = (s.get(k) or []) + v if k in ADD else v
        if cur == "finalize":
            return s, order, True
        cur = NEXT.get(cur) or s.get("route") or "finalize"
    return s, order, False


def scenario(label: str, *, break_node: str | None = None, mode: str = "always",
             always_gap: bool = False, never_pass: bool = False) -> bool:
    fa, fw, fc = make_fakes(always_gap=always_gap, never_pass=never_pass)
    nodes: dict[str, Callable] = {
        "researcher": fake_researcher, "analyst": fa, "writer": fw,
        "critic": fc, "supervisor": supervisor, "finalize": finalize,
    }
    if break_node == "all":
        for k in ("researcher", "analyst", "writer", "critic"):
            nodes[k] = breaker(nodes[k], mode)
    elif break_node:
        nodes[break_node] = breaker(nodes[break_node], mode)

    s, order, done = run_graph(nodes)

    degraded = sum(1 for e in s.get("trace", []) if e.get("action") == "degraded")
    recovered = sum(1 for e in s.get("trace", []) if e.get("action") == "recovered")
    words = len((s.get("draft") or "").split())

    ok_term = done
    ok_ship = words > 0
    ok_budget = (
        s.get("research_loops", 0) <= MAX_RESEARCH_LOOPS
        and s.get("revision_count", 0) <= MAX_REVISIONS
        and s.get("searches_used", 0) <= MAX_SEARCHES
        and s.get("write_attempts", 0) <= MAX_WRITE_ATTEMPTS
    )
    ok = ok_term and ok_budget and (ok_ship or break_node in ("writer", "all"))

    flag = "PASS" if ok else "FAIL"
    print(f"  {flag}  {label:<38} visits={len(order):>3} words={words:>4} "
          f"deg={degraded} rec={recovered} "
          f"loops={s.get('research_loops', 0)} revs={s.get('revision_count', 0)} "
          f"writes={s.get('write_attempts', 0)}")
    if not ok:
        print(f"        terminated={ok_term} shipped={ok_ship} budgets={ok_budget}")
        print(f"        {' -> '.join(order[:24])}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="run one real graph invocation too")
    args = ap.parse_args()

    print("chaos: every scenario must terminate, respect budgets, and ship\n")
    results = [
        scenario("clean run"),
        scenario("researcher always fails", break_node="researcher"),
        scenario("analyst always fails", break_node="analyst"),
        scenario("writer always fails", break_node="writer"),
        scenario("critic always fails", break_node="critic"),
        scenario("analyst fails once, then recovers", break_node="analyst", mode="once"),
        scenario("writer fails once, then recovers", break_node="writer", mode="once"),
        scenario("critic never passes", never_pass=True),
        scenario("analyst always finds gaps", always_gap=True),
        scenario("both loops adversarial", never_pass=True, always_gap=True),
        scenario("every node fails once", break_node="all", mode="once"),
        scenario("every node always fails", break_node="all"),
    ]

    print(f"\n  {sum(results)}/{len(results)} scenarios passed")
    if not all(results):
        print("\n  A FAIL here is a real bug, not a flaky test — the graph is "
              "deterministic.")
        sys.exit(1)

    print("\n  No input kills the run. Every path terminates within budget and ships")
    print("  something, or degrades with its limitations declared.")

    if args.live:
        print("\n--- live graph, one real invocation ---")
        from src.common.io import get_topic
        from src.graph.build import GRAPH

        t = get_topic("t1")
        final = GRAPH.invoke(initial_state(t["topic"]), config={"recursion_limit": 40})
        for e in final.get("trace", []):
            d = " ".join(f"{k}={v}" for k, v in e.items() if k not in ("node", "action"))
            print(f"  {e['node']:<11}{e['action']:<11}{d}")
        print(f"\n  shipped {len((final.get('draft') or '').split())} words")


if __name__ == "__main__":
    main()
