"""CLI: run either system, resume a parked or crashed thread.

    uv run python -m src.run --system baseline   --topic-id t1
    uv run python -m src.run --system multiagent --all
    uv run python -m src.run --system multiagent --topic-id t1 --hitl
    uv run python -m src.run --resume <thread_id> --approve
    uv run python -m src.run --status <thread_id>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.common.io import RESULTS, get_topic, load_topics, print_summary
from src.graph.state import MAX_RESEARCH_LOOPS

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _handle_interrupt(res: dict) -> dict | None:
    """Print the approval request and read a decision from the terminal.

    Field names come from approval.py's interrupt payload. They are read here rather
    than guessed — an earlier draft of this function looked for findings_count while
    the node emitted finding_count, which would have printed None to a reviewer being
    asked to make a decision on the strength of it.
    """
    itr = res.get("interrupt") or {}
    print("\n" + "=" * 62)
    print("OUTLINE APPROVAL")
    print("=" * 62)
    print(f"findings  {itr.get('finding_count')}")
    if itr.get("unclosed_gaps"):
        print("unclosed gaps:")
        for g in itr["unclosed_gaps"]:
            print(f"  - {g}")
    print("\n" + (itr.get("outline") or ""))
    print("\n" + "-" * 62)
    print("  [a] approve   [e] edit outline   [r] reject with feedback   [q] quit")
    choice = input("> ").strip().lower()

    if choice.startswith("a"):
        return {"action": "approve"}
    if choice.startswith("e"):
        print("Paste the revised outline, then a blank line:")
        lines = []
        while (ln := input()) != "":
            lines.append(ln)
        return {"action": "edit", "outline": "\n".join(lines)}
    if choice.startswith("r"):
        # approval.py accepts note or feedback; sending the name it documents first.
        return {"action": "reject", "note": input("What is wrong with it? ").strip()}
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["baseline", "multiagent"])
    ap.add_argument("--topic-id")
    ap.add_argument("--all", action="store_true", help="run every topic in topics.json")
    ap.add_argument("--hitl", action="store_true", help="pause for outline approval")
    ap.add_argument("--no-checkpoint", action="store_true", help="disable persistence")
    # The gap loop is a third of the graph's run cost and two evaluations have failed to
    # show it buying quality. Setting this to 0 turns t2's accidental zero-loop run into
    # a matched pair on every topic — same first-pass budget, loop on and off — which
    # measures evidence coverage directly instead of inferring it from a judge.
    ap.add_argument("--max-research-loops", type=int, default=MAX_RESEARCH_LOOPS,
                    metavar="N", help=f"gap-loop cap (default {MAX_RESEARCH_LOOPS}; "
                                      f"0 disables the loop)")
    # A second arm has to land somewhere else or it overwrites the control it is being
    # compared against — which is the one file the experiment cannot afford to lose.
    ap.add_argument("--out-suffix", metavar="NAME", default="",
                    help="write to results/multiagent_NAME/ instead of "
                         "results/multiagent/, for a matched-pair arm")
    ap.add_argument("--resume", metavar="THREAD_ID")
    ap.add_argument("--approve", action="store_true", help="with --resume: auto-approve")
    ap.add_argument("--status", metavar="THREAD_ID")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.status:
        from src.graph.build import build
        from src.graph.checkpoint import describe, sqlite_checkpointer

        g = build(checkpointer=sqlite_checkpointer(), hitl=True)
        print(json.dumps(describe(g, args.status), indent=2))
        return 0

    if args.resume:
        from src.graph.build import resume_run

        decision = {"action": "approve"} if args.approve else {}
        res = resume_run(args.resume, decision, verbose=not args.quiet)
        print_summary(res["dir"])
        return 0

    if not args.system:
        ap.error("pass --system, --resume, or --status")
    if not args.topic_id and not args.all:
        ap.error("pass --topic-id or --all")

    topics = load_topics() if args.all else [get_topic(args.topic_id)]

    # Refused rather than ignored. The baseline has no gap loop, so there is no arm to
    # separate, and silently dropping the flag would write over results/baseline/.
    if args.out_suffix and args.system != "multiagent":
        ap.error("--out-suffix applies to --system multiagent")

    if args.system == "baseline":
        from src.baseline.agent import run as baseline_run

        def runner(t):
            return baseline_run(t["id"], t["topic"], verbose=not args.quiet)
    else:
        from src.graph.build import run as graph_run

        if args.hitl and args.all:
            ap.error("--hitl with --all would block on every topic; run them one at a time")

        def runner(t):
            return graph_run(
                t["id"], t["topic"], verbose=not args.quiet,
                hitl=args.hitl, checkpoint=not args.no_checkpoint,
                max_research_loops=args.max_research_loops,
                out_suffix=args.out_suffix,
            )

    for t in topics:
        print(f"\n=== {args.system} / {t['id']} ({t['shape']}) ===")
        print(f"    {t['topic'][:90]}...")
        try:
            res = runner(t)

            # The multi-agent path may park on an interrupt.
            while isinstance(res, dict) and res.get("status") == "awaiting_human":
                decision = _handle_interrupt(res)
                if decision is None:
                    print(f"\n  parked. resume with:\n"
                          f"    uv run python -m src.run --resume {res['thread_id']}")
                    break
                from src.graph.build import resume_run

                res = resume_run(res["thread_id"], decision, topic_id=t["id"],
                                 verbose=not args.quiet, seen=res.get("_seen", 0))

            if isinstance(res, dict) and res.get("status") == "complete":
                print_summary(res["dir"])
            elif args.system == "baseline":
                print_summary(RESULTS / "baseline" / t["id"])
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
