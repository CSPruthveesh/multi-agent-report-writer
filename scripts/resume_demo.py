"""Crash mid-run, resume from the checkpoint, and price the difference.

    uv run python -m scripts.resume_demo

WHY MEASURE INSTEAD OF ASSERT
-----------------------------
"It resumes after a crash" is a feature claim. "A crash after the research phase
costs 340 tokens to recover instead of 16,000, because the checkpoint means the
searches are not repeated" is a cost argument — and this project is about cost. The
number belongs in the README.

The contract suite already proves the mechanism against fake nodes and a temporary
database. What it cannot show is the size of the saving, because fake nodes spend
nothing. This spends real tokens once to produce the figure.

HOW THE CRASH IS SIMULATED
--------------------------
os._exit(1) from inside the streaming loop. Not sys.exit, not an exception — those
unwind the stack and give LangGraph a chance to finish writing. os._exit terminates
the process immediately with no cleanup, which is what a real crash, an OOM kill or
a lost spot instance actually looks like. And it is a stronger test than the suite's
KeyboardInterrupt, which still unwinds.

Three phases, orchestrated by --auto:
  1. child process runs, hard-exits once the research is banked
  2. parent confirms the checkpoint outlived the process
  3. a NEW child resumes on the same thread_id and finishes
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from src.common.io import ROOT, get_topic
from src.graph.build import build
from src.graph.checkpoint import sqlite_checkpointer, thread_config
from src.graph.state import initial_state

MARKER = ROOT / "checkpoints" / "demo_state.json"
DEMO_DB = ROOT / "checkpoints" / "demo.sqlite"


def _tokens(state) -> int:
    return sum(r.get("total_tokens", 0) for r in (state.get("token_log") or []))


def phase_crash(thread_id: str, topic_id: str) -> None:
    """Run until the analyst has finished, then die without cleanup."""
    topic = get_topic(topic_id)
    graph = build(checkpointer=sqlite_checkpointer(DEMO_DB))
    config = thread_config(thread_id)

    spent = 0
    for chunk in graph.stream(initial_state(topic["topic"]), config=config,
                              stream_mode="values"):
        spent = _tokens(chunk)
        done = {e["node"] for e in (chunk.get("trace") or [])}
        print(f"  ...{'+'.join(sorted(done)) or 'starting'}  {spent:,} tokens")

        # Kill once real research work is banked. That is the expensive part and
        # therefore the interesting thing to protect.
        if chunk.get("outline") and chunk.get("findings"):
            MARKER.write_text(json.dumps({
                "thread_id": thread_id,
                "tokens_before_crash": spent,
                "findings": len(chunk["findings"]),
            }), encoding="utf-8")
            print(f"\n  *** hard kill: os._exit(1), {spent:,} tokens already spent ***")
            sys.stdout.flush()
            os._exit(1)

    print("  (finished without crashing — the run took a short path)")


def phase_resume(thread_id: str) -> None:
    """Resume the same thread. Completed nodes are not re-executed."""
    graph = build(checkpointer=sqlite_checkpointer(DEMO_DB))
    config = thread_config(thread_id)

    snap = graph.get_state(config)
    before = _tokens(snap.values)
    print(f"  checkpoint found: {len(snap.values.get('findings') or [])} findings, "
          f"{before:,} tokens banked")
    print(f"  next node: {snap.next}")

    t0 = time.perf_counter()
    # None as input means "continue from the checkpoint" rather than "start over".
    final = graph.invoke(None, config=config)
    secs = time.perf_counter() - t0

    after = _tokens(final)
    recovery = after - before
    words = len((final.get("draft") or "").split())

    print(f"\n  resumed and finished in {secs:.1f}s")
    print(f"  tokens to complete : {recovery:,}")
    print(f"  tokens preserved   : {before:,}")
    print(f"  total for the run  : {after:,}")
    print(f"  report             : {words} words")

    if MARKER.exists():
        meta = json.loads(MARKER.read_text(encoding="utf-8"))
        saved = meta["tokens_before_crash"]
        pct = round(saved / after * 100, 1) if after else 0
        print("\n  " + "=" * 58)
        print(f"  Without checkpointing the crash would have cost {saved:,} tokens")
        print(f"  of repeated work — {pct}% of the run, re-paid for nothing.")
        print("  " + "=" * 58)


def phase_auto(topic_id: str) -> None:
    thread_id = f"resume-demo-{int(time.time())}"
    DEMO_DB.parent.mkdir(parents=True, exist_ok=True)
    for p in (DEMO_DB, MARKER):
        if p.exists():
            p.unlink()

    print(f"thread_id: {thread_id}\n")
    print("--- 1. running until crash ---")
    r = subprocess.run(
        [sys.executable, "-m", "scripts.resume_demo", "--phase", "crash",
         "--thread-id", thread_id, "--topic-id", topic_id],
        cwd=str(ROOT), check=False,  # a non-zero exit is the point
    )
    if r.returncode != 1:
        print(f"\n(child exited {r.returncode}; expected 1 from os._exit)")
        if not MARKER.exists():
            print("no checkpoint marker — nothing to resume")
            return

    print("\n--- 2. the checkpoint outlived the process ---")
    print(f"  {DEMO_DB.name}: {DEMO_DB.stat().st_size:,} bytes on disk")

    print("\n--- 3. resuming in a NEW process ---")
    subprocess.run(
        [sys.executable, "-m", "scripts.resume_demo", "--phase", "resume",
         "--thread-id", thread_id],
        cwd=str(ROOT), check=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["auto", "crash", "resume"], default="auto")
    ap.add_argument("--thread-id")
    ap.add_argument("--topic-id", default="t1")
    args = ap.parse_args()

    if args.phase == "auto":
        phase_auto(args.topic_id)
    elif args.phase == "crash":
        phase_crash(args.thread_id, args.topic_id)
    else:
        phase_resume(args.thread_id)


if __name__ == "__main__":
    main()
