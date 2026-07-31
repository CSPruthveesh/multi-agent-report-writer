"""Exercise the outline approval gate.

    uv run python -m scripts.hitl_demo                    # interactive
    uv run python -m scripts.hitl_demo --auto approve     # scripted
    uv run python -m scripts.hitl_demo --auto reject

Shows the interrupt/resume cycle: the graph pauses before the Writer, hands the
outline back, and waits. Nothing downstream has been paid for yet, which is the whole
point of gating here rather than at the end — the tokens-spent figure printed at the
pause is the number the gate is protecting.
"""

from __future__ import annotations

import argparse
import time

from langgraph.types import Command

from src.common.io import get_topic
from src.graph.build import build
from src.graph.checkpoint import sqlite_checkpointer, thread_config
from src.graph.state import initial_state


def _tokens(state) -> int:
    return sum(r.get("total_tokens", 0) for r in (state.get("token_log") or []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic-id", default="t1")
    ap.add_argument("--auto", choices=["approve", "reject", "edit"])
    ap.add_argument("--thread-id", help="reuse a specific thread instead of a fresh one")
    args = ap.parse_args()

    topic = get_topic(args.topic_id)
    graph = build(checkpointer=sqlite_checkpointer(), hitl=True)
    # Fresh thread per invocation by default. A fixed id would mean the second run of
    # this script resumes the first one's finished thread rather than starting over,
    # and the demo would silently show a completed run instead of a pause.
    config = thread_config(args.thread_id or f"hitl-{args.topic_id}-{int(time.time())}")

    print(f"topic: {topic['id']} ({topic['shape']})\n")
    result = graph.invoke(initial_state(topic["topic"], hitl=True), config=config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        snap = graph.get_state(config)
        spent = _tokens(snap.values)

        print("=" * 62)
        print("PAUSED FOR APPROVAL")
        print("=" * 62)
        print(f"findings: {payload['finding_count']}   tokens spent so far: {spent:,}")
        if payload.get("unclosed_gaps"):
            print(f"unclosed gaps: {payload['unclosed_gaps']}")
        print(f"\n{payload['outline']}\n")
        print("Nothing has been written yet. Rejecting here costs one analyst call;")
        print("rejecting the finished report would cost the whole write.\n")

        if args.auto:
            action = args.auto
            print(f"[auto] {action}")
        else:
            action = input("approve / edit / reject > ").strip().lower() or "approve"

        payload_out: dict = {"action": action}
        if action == "edit":
            print("Paste the revised outline, blank line to finish:")
            lines = []
            while (ln := input()) != "":
                lines.append(ln)
            payload_out["outline"] = "\n".join(lines)
        elif action == "reject":
            payload_out["note"] = (
                "" if args.auto else input("What is wrong with it? > ").strip()
            ) or "Reviewer rejected the framing; rebuild the outline."

        result = graph.invoke(Command(resume=payload_out), config=config)

    print("\n--- trace ---")
    for ev in result.get("trace", []):
        detail = " ".join(f"{k}={v}" for k, v in ev.items() if k not in ("node", "action"))
        print(f"  {ev['node']:<11}{ev['action']:<11}{detail}")

    print(f"\nshipped {len((result.get('draft') or '').split())} words, "
          f"{_tokens(result):,} tokens")


if __name__ == "__main__":
    main()
