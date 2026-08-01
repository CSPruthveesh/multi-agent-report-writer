"""Backfill max_research_loops into run records written before the field existed.

    uv run python -m scripts.backfill_max_research_loops --dry-run
    uv run python -m scripts.backfill_max_research_loops

results/multiagent/ was produced at a cap of 2. The field that records the cap landed
two commits later, and the default has since been lowered to 1 — so the arm that defines
half the matched-pair experiment cannot say from its own record which arm it is, and a
fresh run today would produce a third configuration indistinguishable from it.

That is the Phase 9 provenance problem one level up. code_version answers "what code
produced this"; it says nothing about what the run was configured to do.

TWO SOURCES, AND THEY ARE NOT EQUALLY GOOD

The supervisor writes the cap into its own trace as "loop=1/2" and "loops=2/2", so most
records prove their cap without anything being inferred:

  t1 t3 t4 t5 t6   observed  — read out of the record's own trace
  t2               inferred  — never looped and never retired a gap, so no trace event
                              carries a cap. Taken from the constant at 2008af8, which
                              is the commit the record already names.

Every backfilled record is stamped in `backfilled`, and the two sources are kept apart
in `max_research_loops_source`. A cap recovered from the artifact is evidence; one
recovered from a constant at a commit is a reconstruction, and reporting them
identically would launder the difference — the same argument backfill_code_version.py
makes about SHAs, applied to configuration.

An observed value is never overwritten. The live writer records this field now, so this
script is a one-off that should report nothing left to do on every run after the first.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.common.io import RESULTS

SYSTEM = "multiagent"
# The value of MAX_RESEARCH_LOOPS at 2008af8, the commit every record in this directory
# already names in its code_version. Lowered to 1 at f553c09, after these runs.
CAP_AT_COMMIT = 2


def cap_from_trace(record: dict) -> int | None:
    """The cap the supervisor recorded while routing, or None if it never said.

    trace_event writes loop="1/2" on a route to the researcher and loops="2/2" when it
    retires gaps with the budget spent. A run that neither looped nor retired anything
    leaves no evidence of its own cap.
    """
    caps = set()
    for ev in record.get("trace") or []:
        for key in ("loop", "loops"):
            value = ev.get(key)
            if isinstance(value, str) and "/" in value:
                caps.add(value.rsplit("/", 1)[1])
    if len(caps) != 1:
        # Zero means the run never routed on the loop. More than one would mean the cap
        # changed mid-run, which cannot happen — and guessing which to trust is worse
        # than declining to answer.
        return None
    try:
        return int(caps.pop())
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    written = skipped = 0
    for p in sorted((RESULTS / SYSTEM).glob("*/run.json")):
        topic = p.parent.name
        d = json.loads(p.read_text(encoding="utf-8"))

        # Present is present, however it got there. Protecting only observed values would
        # leave the script rewriting its own output on every run — not idempotent, and it
        # re-touches six committed files to change nothing. Correcting a backfilled value
        # is a deliberate edit, not something a re-run should do silently.
        if isinstance(d.get("max_research_loops"), int):
            already = "max_research_loops" in (d.get("backfilled") or [])
            print(f"  skip  {topic}  cap {d['max_research_loops']} "
                  f"({'backfilled earlier' if already else 'observed at write time'})")
            skipped += 1
            continue

        observed = cap_from_trace(d)
        cap = observed if observed is not None else CAP_AT_COMMIT
        source = "trace" if observed is not None else "constant-at-commit"

        d["max_research_loops"] = cap
        d["max_research_loops_source"] = source
        d["backfilled"] = sorted({*(d.get("backfilled") or []), "max_research_loops"})

        print(f"  {'would set' if args.dry_run else 'set'}  {topic}  -> cap {cap} "
              f"({source})")
        if not args.dry_run:
            p.write_text(json.dumps(d, indent=2), encoding="utf-8")
        written += 1

    verb = "would write" if args.dry_run else "wrote"
    print(f"\n{verb} {written}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
