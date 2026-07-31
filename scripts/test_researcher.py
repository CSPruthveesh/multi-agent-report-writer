from __future__ import annotations

import argparse

from src.common.io import get_topic
from src.graph.nodes.researcher import researcher
from src.graph.state import MAX_SEARCHES, initial_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic-id", default="t1")
    ap.add_argument("--gap-mode", action="store_true", help="simulate arriving from the analyst")
    ap.add_argument("--gap", action="append", dest="gaps", metavar="TEXT",
                    help="gap statement to seed; repeatable. defaults are t1-specific")
    args = ap.parse_args()

    topic = get_topic(args.topic_id)
    state = initial_state(topic["topic"])

    if args.gap_mode:
        state["findings"] = [
            {"id": "F001", "claim": "placeholder from a prior sweep",
             "source_url": "https://example.com", "confidence": "medium"}
        ]
        state["searches_used"] = 3
        state["gaps"] = args.gaps or [
            "No evidence on costs after 2024",
            "Nothing on deployment volumes outside the largest market",
        ]

    gaps_in = len(state.get("gaps") or [])
    findings_in = len(state.get("findings") or [])

    print(f"topic   {topic['id']} ({topic['shape']})")
    print(f"mode    {'gap-driven' if args.gap_mode else 'cold-start'}")
    print(f"budget  {state.get('searches_used', 0)}/{MAX_SEARCHES} used going in\n")

    upd = researcher(state)

    for ev in upd.get("trace", []):
        detail = " ".join(f"{k}={v}" for k, v in ev.items() if k not in ("node", "action"))
        print(f"  {ev['node']:<11}{ev['action']:<10}{detail}")

    findings = upd.get("findings", [])
    print(f"\n{len(findings)} findings\n")
    for f in findings:
        url = f["source_url"][:52] or "(no source)"
        print(f"  [{f['id']}] ({f['confidence']:<6}) {f['claim'][:88]}")
        print(f"          {url}")

    print("\ntokens by call type:")
    agg: dict[str, dict[str, int]] = {}
    for r in upd.get("token_log", []):
        d = agg.setdefault(r["call_type"], {"calls": 0, "total": 0})
        d["calls"] += 1
        d["total"] += r["total_tokens"]
    for k, v in agg.items():
        print(f"  {k:<10} {v['calls']} calls  {v['total']:,} tokens")
    print(f"  {'TOTAL':<10} {sum(v['calls'] for v in agg.values())} calls  "
          f"{sum(v['total'] for v in agg.values()):,} tokens")

    ev = (upd.get("trace") or [{}])[0]
    queries = ev.get("queries", 0)
    gaps_back = len(upd.get("unaddressed_gaps") or [])
    ids = [f["id"] for f in findings]
    expect_ids = [f"F{i:03d}" for i in range(findings_in + 1, findings_in + 1 + len(findings))]
    want = 2 * queries
    no_source = sum(1 for f in findings if not f["source_url"])

    print("\nchecks:")
    print(f"  >= 2 findings per query   {'PASS' if len(findings) >= want else 'FAIL'} "
          f"({len(findings)} from {queries} queries, wanted {want})")
    print(f"  all have a source URL     {'PASS' if no_source == 0 else f'FAIL ({no_source} missing)'}")
    print(f"  IDs contiguous from F{findings_in + 1:03d}  {'PASS' if ids == expect_ids else 'FAIL'}")
    print(f"  gap handover consistent   {'PASS' if gaps_back == max(0, gaps_in - queries) else 'FAIL'} "
          f"({gaps_in} in, {queries} queried, {gaps_back} handed back)")
    print(f"  budget respected          "
          f"{'PASS' if upd.get('searches_used', 0) <= MAX_SEARCHES else 'FAIL'}")


if __name__ == "__main__":
    main()
