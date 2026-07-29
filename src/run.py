from __future__ import annotations

import argparse
import logging
import sys

from src.common.io import RESULTS, get_topic, load_topics, print_summary

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["baseline", "multiagent"], required=True)
    ap.add_argument("--topic-id")
    ap.add_argument("--all", action="store_true", help="run every topic in topics.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.topic_id and not args.all:
        ap.error("pass --topic-id or --all")

    topics = load_topics() if args.all else [get_topic(args.topic_id)]

    if args.system == "baseline":
        from src.baseline.agent import run
    else:
        try:
            from src.graph.build import run  # type: ignore[no-redef]
        except ImportError:
            print("multi-agent graph not built yet — that's Phase 1 onward.", file=sys.stderr)
            return 1

    for t in topics:
        print(f"\n=== {args.system} / {t['id']} ({t['shape']}) ===")
        print(f"    {t['topic'][:90]}...")
        try:
            run(t["id"], t["topic"], verbose=not args.quiet)
            print_summary(RESULTS / args.system / t["id"])
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
