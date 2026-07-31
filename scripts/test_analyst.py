"""Test the Analyst in isolation, and run the discrimination diagnostic.

    uv run python -m scripts.test_analyst --topic-id t1
    uv run python -m scripts.test_analyst --discriminate       # THE important one

WHY THE DISCRIMINATION TEST MATTERS
-----------------------------------
The Analyst's job is to notice when evidence is insufficient. The failure mode is an
Analyst that notices that always — asked "what is missing?", a model will always find
something. An Analyst like that is not detecting gaps, it is generating them, and it
will fire the research loop on every topic including the ones that never needed it.

That failure is invisible in a single run. It looks like the system working. You only
see it by comparing across topics of different evidence density:

    t1  abundant sources   -> expect 0 gaps
    t3  thin evidence      -> expect 1-2 gaps

Similar gap counts on both means the Analyst is not discriminating. Fix the prompt
before you build anything else, because every downstream number — token multiple,
latency, the whole Phase 9 comparison — is contaminated by a gap loop that fires
indiscriminately.

The diagnostic reuses findings already in results/baseline/, so it costs one Analyst
call per topic and no searches at all.
"""

from __future__ import annotations

import argparse
import json

from src.common.io import RESULTS, get_topic
from src.graph.nodes.analyst import analyst
from src.graph.state import initial_state


def _load_findings(topic_id: str) -> list[dict]:
    """Reuse the baseline's findings so the diagnostic costs no searches."""
    p = RESULTS / "baseline" / topic_id / "run.json"
    if not p.exists():
        raise SystemExit(
            f"No baseline findings at {p}.\n"
            f"Run: uv run python -m src.run --system baseline --topic-id {topic_id}"
        )
    return json.loads(p.read_text(encoding="utf-8"))["findings"]


def _run_one(topic_id: str, *, show: bool = True) -> dict:
    topic = get_topic(topic_id)
    findings = _load_findings(topic_id)

    state = initial_state(topic["topic"])
    state["findings"] = findings

    upd = analyst(state)
    ev = (upd.get("trace") or [{}])[0]

    if show:
        print(f"\ntopic     {topic_id} ({topic['shape']})")
        print(f"findings  {len(findings)} in")
        detail = " ".join(f"{k}={v}" for k, v in ev.items() if k not in ("node", "action"))
        print(f"  analyst  {ev.get('action')}  {detail}")
        print("\n--- outline ---")
        print(upd.get("outline") or "(none)")
        gaps = upd.get("gaps") or []
        print(f"\n--- gaps ({len(gaps)}) ---")
        for g in gaps:
            print(f"  - {g}")
        if not gaps:
            print("  (none — correct if the evidence covers the topic)")

    return {
        "topic_id": topic_id,
        "shape": topic["shape"],
        "findings": len(findings),
        "gaps_raw": ev.get("gaps_raw", 0),
        "gaps_kept": ev.get("gaps_kept", 0),
        "dropped": ev.get("dropped_unblocking", 0) + ev.get("dropped_repeat", 0),
        "sections": ev.get("sections", 0),
        "tensions": ev.get("tensions", 0),
        "tokens": ev.get("tokens", 0),
    }


def discriminate() -> None:
    """Run the Analyst across all topics with baseline findings and compare gap counts."""
    available = [p.parent.name for p in (RESULTS / "baseline").glob("*/run.json")]
    if not available:
        raise SystemExit("No baseline results found. Run Phase 0 first.")

    rows = [_run_one(t, show=False) for t in sorted(available)]

    print(f"\n{'topic':<7}{'shape':<20}{'findings':>9}{'raw':>6}{'kept':>6}"
          f"{'dropped':>9}{'sections':>10}")
    print("-" * 67)
    for r in rows:
        print(f"{r['topic_id']:<7}{r['shape']:<20}{r['findings']:>9}{r['gaps_raw']:>6}"
              f"{r['gaps_kept']:>6}{r['dropped']:>9}{r['sections']:>10}")

    thin = [r for r in rows if r["shape"] in ("thin-evidence", "contested")]
    rich = [r for r in rows if r["shape"] == "abundant-sources"]

    print("\n--- discrimination ---")
    if not thin or not rich:
        print("  need at least one abundant-sources and one thin-evidence topic")
        return

    m_thin = sum(r["gaps_kept"] for r in thin) / len(thin)
    m_rich = sum(r["gaps_kept"] for r in rich) / len(rich)
    print(f"  mean gaps, abundant-sources : {m_rich:.2f}")
    print(f"  mean gaps, thin/contested   : {m_thin:.2f}")

    if m_thin > m_rich + 0.4:
        print("\n  PASS — the Analyst reports more gaps where evidence is genuinely thinner.")
        print("  The research loop will fire selectively, which is the behaviour the")
        print("  architecture is betting on.")
    elif m_rich > m_thin:
        print("\n  FAIL, inverted — more gaps on well-covered topics than thin ones.")
        print("  Something is wrong with the gap prompt. Investigate before continuing.")
    elif sum(r["gaps_raw"] for r in rows) == 0:
        # Under-detection. The model proposes nothing, so the post-filter is not the
        # cause and every remedy that removes gaps is the wrong direction. Split on
        # gaps_raw rather than gaps_kept: same symptom in the kept column, opposite
        # causes, opposite fixes.
        print("\n  FAIL, under-detecting — the Analyst proposed NO gaps on any topic.")
        print("  The research loop cannot fire, so the gap-driven Researcher mode is")
        print("  dead code. Note raw is 0: nothing is being filtered out, so tightening")
        print("  the rules will not help. Things to try, in order:")
        print("    1. Give it a test for when a gap is REQUIRED, not just permitted")
        print("    2. Stop `tensions` absorbing conflicts it should be reporting as gaps")
        print("    3. Raise temperature — 0.3 may be too conservative for a judgement")
        print("    4. Only if gaps then appear everywhere, start tightening")
    elif sum(r["gaps_kept"] for r in rows) < len(rows):
        # Gaps exist but do not track density. Not over-generation — the volume is too
        # low for that — so tightening the rules would just return it to zero.
        print("\n  FAIL, not discriminating — gaps appear, but not where evidence is")
        print("  thinner. Check WHICH test is firing before changing anything: a topic")
        print("  can be thin in count, in confidence, or in substance, and a test for")
        print("  one will not catch the others. Read the gap text on a failing topic.")
    else:
        print("\n  FAIL, over-generating — gaps on every topic regardless of density.")
        print("  The Analyst is generating gaps rather than detecting them, so the loop")
        print("  fires on topics that never needed it. Things to try, in order:")
        print("    1. Strengthen the 'zero gaps is correct' line in GAP_DISCIPLINE")
        print("    2. Lower MAX_GAPS to 1 — forcing one choice sharpens the ranking")
        print("    3. Require the blocked section be quoted verbatim from `sections`")
        print("    4. Drop temperature to 0.1")

    total = sum(r["tokens"] for r in rows)
    print(f"\n  diagnostic cost: {total:,} tokens across {len(rows)} topics")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic-id", default="t1")
    ap.add_argument("--discriminate", action="store_true",
                    help="run every topic and compare gap counts by evidence density")
    args = ap.parse_args()

    if args.discriminate:
        discriminate()
    else:
        _run_one(args.topic_id)


if __name__ == "__main__":
    main()
