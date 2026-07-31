"""Exercise EXTRACT_SYSTEM alone, on fixed notes, with no search.

Extraction is the one prompt both systems share, so a change to it moves the baseline
and the graph together. Judging such a change on a live run is unreadable — the search
results move underneath the prompt — so this script holds the notes constant and lets
the prompt be the only variable.

The notes carry, on purpose, every failure mode observed so far: figures worth keeping,
two sources giving different values for one quantity, generic padding, and a figure no
source can be pinned to.

    uv run python -m scripts.test_extract
    uv run python -m scripts.test_extract --runs 3      # also check it reproduces
"""
from __future__ import annotations

import argparse
import re

from src.common.llm import TokenLedger
from src.common.search import extract

SOURCES = [
    {"title": "CATL — Naxtra sodium-ion launch disclosure",
     "url": "https://www.catl.com/en/news/6250.html"},
    {"title": "Wood Mackenzie — sodium-ion storage outlook 2025",
     "url": "https://www.woodmac.com/press-releases/sodium-ion-outlook-2025/"},
    {"title": "IRENA — electricity storage cost report",
     "url": "https://www.irena.org/publications/2024/electricity-storage-costs"},
    {"title": "Benchmark Mineral Intelligence — sodium-ion vs LFP",
     "url": "https://source.benchmarkminerals.com/article/sodium-ion-vs-lfp"},
]

NOTES = """\
IRENA reports lithium-ion pack prices averaged $115/kWh in 2024 \
(https://www.irena.org/publications/2024/electricity-storage-costs).

Benchmark Mineral Intelligence puts LFP cell costs at approximately $60/kWh \
(https://source.benchmarkminerals.com/article/sodium-ion-vs-lfp).

CATL states its Naxtra sodium-ion cell reaches 175 Wh/kg \
(https://www.catl.com/en/news/6250.html).

Wood Mackenzie reports grid-duty sodium-ion cycle life of 4,000 to 6,000 cycles \
(https://www.woodmac.com/press-releases/sodium-ion-outlook-2025/).

Benchmark Mineral Intelligence counts announced sodium-ion production capacity of \
220 GWh through 2027 \
(https://source.benchmarkminerals.com/article/sodium-ion-vs-lfp).

The sources disagree on sodium-ion cell cost. CATL's disclosure gives a production \
cost of $19/kWh (https://www.catl.com/en/news/6250.html), while Wood Mackenzie puts \
the 2025 industry average at $59/kWh \
(https://www.woodmac.com/press-releases/sodium-ion-outlook-2025/).

Sodium-ion batteries are considered a promising alternative for stationary storage \
(https://www.woodmac.com/press-releases/sodium-ion-outlook-2025/).

The technology is favoured for its safety characteristics \
(https://source.benchmarkminerals.com/article/sodium-ion-vs-lfp).

Grid-scale sodium-ion installations reached 2.3 GWh in 2025.
"""

# A claim that talks about the evidence rather than the world. One URL cannot support it.
META = re.compile(
    r"sources disagree|source disagree|one source|another source|other source|"
    r"estimates vary|reports vary|sources vary|figures vary|disagreement",
    re.IGNORECASE,
)

# The only figure in the notes with no URL, and four sources it could equally belong to.
UNATTRIBUTABLE = "2.3 GWh"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1, help="repeat to check it reproduces")
    args = ap.parse_args()

    runs: list[list[dict]] = []
    total_tokens = 0

    for i in range(args.runs):
        ledger = TokenLedger()
        out = extract(NOTES, node="fixture", ledger=ledger, sources=SOURCES)
        runs.append([f.model_dump() for f in out])
        total_tokens += ledger.total
        if args.runs > 1:
            print(f"run {i + 1}: {len(out)} findings, {ledger.total:,} tokens")

    findings = runs[0]
    if args.runs > 1:
        print()

    print(f"{len(findings)} findings from 10 assertions, {total_tokens // args.runs:,} tokens\n")
    for f in findings:
        print(f"  [{f['id']}] ({f['confidence']:<6}) {f['claim'][:88]}")
        print(f"          {f['source_url'][:72] or '(no source)'}")

    spread = {c: sum(1 for f in findings if f["confidence"] == c)
              for c in ("high", "medium", "low")}
    meta = [f for f in findings if META.search(f["claim"])]
    no_source = [f for f in findings if not f["source_url"]]
    ghost = [f for f in findings if UNATTRIBUTABLE in f["claim"]]
    cheap = [f for f in findings if "19" in f["claim"] and "kWh" in f["claim"]]
    dear = [f for f in findings if "59" in f["claim"] and "kWh" in f["claim"]]
    conflict = cheap + dear
    reproduced = all(
        [f["claim"] for f in r] == [f["claim"] for f in findings] for r in runs
    )

    print(f"\nconfidence  h{spread['high']} / m{spread['medium']} / l{spread['low']}")

    print("\nchecks:")
    print(f"  no claim about the evidence   {'PASS' if not meta else f'FAIL ({len(meta)})'}")
    for f in meta:
        print(f"      {f['id']} | {f['claim'][:76]}")
    print(f"  both conflicting values kept  {'PASS' if cheap and dear else 'FAIL'} "
          f"($19 {'yes' if cheap else 'no'}, $59 {'yes' if dear else 'no'})")
    print(f"  conflicting values are low    "
          f"{'PASS' if conflict and all(f['confidence'] == 'low' for f in conflict) else 'FAIL'}")
    print(f"  unattributable claim dropped  {'PASS' if not ghost else 'FAIL (kept)'}")
    print(f"  every finding has a source    "
          f"{'PASS' if not no_source else f'FAIL ({len(no_source)} missing)'}")
    print(f"  confidence did not collapse   "
          f"{'PASS' if sum(1 for v in spread.values() if v) > 1 else 'FAIL (all one level)'}")
    if args.runs > 1:
        print(f"  identical across {args.runs} runs        {'PASS' if reproduced else 'FAIL'}")


if __name__ == "__main__":
    main()
