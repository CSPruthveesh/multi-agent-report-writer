"""Backfill code_version into run records written before the field existed.

    uv run python -m scripts.backfill_code_version --dry-run
    uv run python -m scripts.backfill_code_version

The field is recorded at write time from here on. The twelve results already committed
predate it, and leaving them null would make every one of them permanently unverifiable
— the exact problem the field was added to solve, frozen into the record.

WHERE THESE SHAS COME FROM

They are inferred, not observed, so the reasoning is written down rather than left in a
commit message:

  baseline    ran 2026-07-31T08:30-08:38Z (14:00-14:08 IST). The nearest preceding
              commit is 9e6844c at 13:58:43 IST. The next, 23d2c06 at 14:24:53, is the
              commit that added these very results and touched nothing but results/ —
              so src/ at 9e6844c is what produced them.

  multiagent  ran 2026-08-01T06:47-06:53Z. HEAD was 626098d with a clean tree, and the
              two commits after it (b949796, 8fd69a3) are this run's own results and
              the fixes it provoked.

inferred=True is stamped on every backfilled record. A reader must be able to tell a
SHA that was observed at write time from one that was reconstructed afterwards, because
only the first is evidence. The dirty flag is left None for the same reason: the tree
state at those moments was not recorded and cannot be recovered.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from src.common.io import RESULTS, ROOT

BACKFILL = {
    "baseline": "9e6844c",
    "multiagent": "626098d",
}


def resolve(short: str) -> str:
    """Expand to a full SHA, and fail loudly if the commit is not in this repo."""
    out = subprocess.run(
        ["git", "rev-parse", short], cwd=ROOT, capture_output=True, text=True,
        check=False,
    )
    if out.returncode != 0:
        raise SystemExit(f"commit {short} not found in this repository")
    return out.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    written = skipped = 0
    for system, short in BACKFILL.items():
        full = resolve(short)
        for p in sorted((RESULTS / system).glob("*/run.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            existing = d.get("code_version")
            if isinstance(existing, dict) and existing.get("commit"):
                # Written by save_run() at generation time. Observed beats inferred, so
                # never overwrite one with the other.
                print(f"  skip  {system}/{p.parent.name}  has {existing['commit'][:7]}")
                skipped += 1
                continue
            d["code_version"] = {
                "commit": full,
                "commit_short": full[:7],
                "dirty": None,
                "inferred": True,
            }
            # Rebuild so code_version sits next to generated_at rather than at the end,
            # matching the order save_run() writes.
            order = ["system", "topic_id", "generated_at", "code_version"]
            rebuilt = {k: d[k] for k in order if k in d}
            rebuilt.update({k: v for k, v in d.items() if k not in rebuilt})
            print(f"  {'would set' if args.dry_run else 'set'}  "
                  f"{system}/{p.parent.name}  -> {full[:7]} (inferred)")
            if not args.dry_run:
                p.write_text(json.dumps(rebuilt, indent=2), encoding="utf-8")
            written += 1

    verb = "would write" if args.dry_run else "wrote"
    print(f"\n{verb} {written}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
