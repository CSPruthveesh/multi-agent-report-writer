"""Build the comparison — the single artifact this project exists to produce.

    uv run python -m evals.compare
    uv run python -m evals.compare --markdown > docs/comparison.md

Joins quality (judge + hand scores) to cost (results/cost.json) and runs four
integrity checks on its own output before reporting anything.

THE CHECKS MATTER AS MUCH AS THE TABLE
--------------------------------------
A comparison table is easy to produce and easy to produce wrongly. These four run
automatically and print warnings into the output, so a reader sees the caveats at
the same time as the numbers:

  1. Length-bias sweep   Did the judge prefer the longer report on every criterion
                         of every topic? If so the scores are measuring length.
  2. Baseline strength   Is the baseline scoring so low it looks sandbagged? A weak
                         control is the fastest way to lose credibility.
  3. Judge/human agreement  Mean absolute difference per criterion. Divergence above
                         1.0 means the judge is not measuring what you are.
  4. Loop value          Did the revision loop cost more than it returned? Answering
                         this honestly is the strongest thing in the writeup.

None of these are decorative. Each one has a plausible failure mode attached to it
that would otherwise make the headline number wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evals.handscore import is_topic_dir, stale_labels
from src.common.io import RESULTS, get_topic

CRITERIA = [
    "factual_grounding",
    "structural_coherence",
    "depth_of_analysis",
    "citation_integrity",
    "absence_of_filler",
]
LABEL = {
    "factual_grounding": "Factual grounding",
    "structural_coherence": "Structural coherence",
    "depth_of_analysis": "Depth of analysis",
    "citation_integrity": "Citation integrity",
    "absence_of_filler": "Absence of filler",
}


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def collect() -> dict[str, Any]:
    judge = _load(RESULTS / "judge.json")
    cost = _load(RESULTS / "cost.json")
    if not judge:
        raise SystemExit("No results/judge.json — run: uv run python -m evals.judge")
    if "topics" not in judge:
        # The older scripts/judge.py wrote {judge_model, tokens, scores} to this path.
        # A bare KeyError here would look like a bug in this file rather than a file
        # written by a different tool, so name both.
        raise SystemExit(
            f"results/judge.json has the older per-system schema "
            f"(keys: {', '.join(judge)}), which this file cannot read.\n"
            f"It was written by scripts/judge.py. Run: uv run python -m evals.judge"
        )

    rows = judge["topics"]
    means = {
        sysname: {
            c: sum(r[sysname][c] for r in rows) / len(rows) for c in CRITERIA
        }
        for sysname in ("baseline", "multiagent")
    }

    # Per-shape breakdown. If multi-agent's grounding advantage appears only on
    # thin-evidence topics, that is a sharper and more useful finding than an
    # average across six — it tells you WHEN to use the architecture.
    by_shape: dict[str, dict[str, Any]] = {}
    for r in rows:
        shape = get_topic(r["topic_id"])["shape"]
        b = by_shape.setdefault(shape, {"n": 0, "baseline": {c: 0.0 for c in CRITERIA},
                                        "multiagent": {c: 0.0 for c in CRITERIA}})
        b["n"] += 1
        for c in CRITERIA:
            b["baseline"][c] += r["baseline"][c]
            b["multiagent"][c] += r["multiagent"][c]
    for b in by_shape.values():
        for s in ("baseline", "multiagent"):
            for c in CRITERIA:
                b[s][c] /= b["n"]

    # Broken citations — mechanical, no model involved. The cleanest number here.
    broken = {}
    for sysname in ("baseline", "multiagent"):
        n = 0
        for p in (RESULTS / sysname).glob("*/run.json"):
            n += json.loads(p.read_text(encoding="utf-8"))["citations"]["broken_count"]
        broken[sysname] = n

    hand = collect_hand()
    return {"judge": judge, "rows": rows, "means": means, "by_shape": by_shape,
            "broken": broken, "cost": cost, "hand": hand}


def collect_hand() -> dict[str, Any] | None:
    """Hand scores, with any that describe a superseded run left out.

    This function is where a stale hand score does its damage. It joins whatever is on
    disk to the current judge numbers, so a directory left over from an earlier run
    puts a human column and a judge column in the same table describing different
    documents. stale_labels() is imported rather than restated so the two files cannot
    drift on what "stale" means — the same argument evals/judge.py makes for importing
    _body rather than reimplementing the blinding.
    """
    hs = RESULTS / "handscore"
    if not hs.exists():
        return None
    per_sys: dict[str, list[dict[str, int]]] = {"baseline": [], "multiagent": []}
    topics: list[str] = []
    stale: list[str] = []
    for d in sorted(hs.glob("*/")):
        mp, sc = d / "mapping.json", d / "handscores.json"
        if not (mp.exists() and sc.exists()):
            continue
        # An archived directory keeps its mapping and scores and passes the staleness
        # check for the wrong reason, so the name is validated before either is read.
        if not is_topic_dir(d.name):
            continue
        if stale_labels(d.name):
            stale.append(d.name)
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        s = json.loads(sc.read_text(encoding="utf-8"))["scores"]
        for label, sysname in m.items():
            per_sys[sysname].append(s[label])
        topics.append(d.name)
    if not topics:
        return {"topics": [], "means": {}, "stale": stale} if stale else None
    return {
        "topics": topics,
        "stale": stale,
        "means": {
            sysname: {c: sum(x[c] for x in lst) / len(lst) for c in CRITERIA}
            for sysname, lst in per_sys.items() if lst
        },
    }


# ------------------------------------------------------------------ checks
def check_length_bias(rows: list[dict]) -> str | None:
    sweeps = 0
    for r in rows:
        longer = "multiagent" if r["words"]["multiagent"] > r["words"]["baseline"] else "baseline"
        other = "baseline" if longer == "multiagent" else "multiagent"
        if all(r[longer][c] >= r[other][c] for c in CRITERIA):
            sweeps += 1
    if sweeps == len(rows) and len(rows) >= 3:
        return (f"The longer report won or tied every criterion on all {len(rows)} topics. "
                f"That is the signature of length bias, not quality. Weight the hand "
                f"scores heavily and say so in the README.")
    return None


def check_baseline_strength(means: dict) -> str | None:
    m = sum(means["baseline"].values()) / len(CRITERIA)
    if m < 2.8:
        return (f"Baseline mean is {m:.2f}/5. A control that weak invites the question of "
                f"whether it was given a fair prompt. Check src/baseline/agent.py before "
                f"publishing this comparison.")
    return None


def check_agreement(means: dict, hand: dict | None) -> tuple[str | None, dict]:
    if not hand or "multiagent" not in hand["means"]:
        return None, {}
    diffs = {}
    for c in CRITERIA:
        jd = means["multiagent"][c] - means["baseline"][c]
        hd = hand["means"]["multiagent"][c] - hand["means"]["baseline"][c]
        diffs[c] = round(abs(jd - hd), 2)
    worst = max(diffs.values())
    if worst > 1.0:
        bad = [c for c, v in diffs.items() if v > 1.0]
        return (f"Judge and hand scores diverge by more than 1.0 on: {', '.join(bad)}. "
                f"Trust your own read on those and explain the divergence."), diffs
    return None, diffs


def check_stale_hand(hand: dict | None) -> str | None:
    """Say what was dropped. A silent exclusion reads as full coverage."""
    dropped = (hand or {}).get("stale") or []
    if not dropped:
        return None
    return (f"Hand scores for {', '.join(dropped)} were left out: their copied reports no "
            f"longer match the reports on disk, so they describe a superseded run. "
            f"Re-score them, or say in the write-up that the hand column covers "
            f"{len((hand or {}).get('topics') or [])} topics and not those.")


def check_loop_value(cost: dict | None, means: dict) -> str | None:
    if not cost:
        return None
    ph = cost.get("multiagent", {}).get("by_phase", {})
    tot = sum(v["total"] for v in ph.values()) or 1
    rev = ph.get("revision_loop", {}).get("total", 0)
    pct = rev / tot * 100
    delta = sum(means["multiagent"][c] - means["baseline"][c] for c in CRITERIA) / len(CRITERIA)
    if pct > 20 and delta < 0.3:
        return (f"The revision loop is {pct:.0f}% of run cost while the overall quality delta "
                f"is {delta:+.2f} points. On this evidence it is not paying for itself — cap "
                f"revisions at 1 and report that as a finding.")
    return None


# ------------------------------------------------------------------ output
def render(d: dict, md: bool) -> None:
    means, cost, hand = d["means"], d["cost"], d["hand"]
    p = print

    p("# Single agent vs multi-agent\n" if md else "=" * 72)
    if not md:
        p("SINGLE AGENT vs MULTI-AGENT")
        p("=" * 72)

    p(f"\nJudge: {d['judge']['judge_model']}, {d['judge']['repeats']} round(s), "
      f"{len(d['rows'])} topics, blind and order-randomised.\n")

    # quality
    if md:
        p("| Criterion | Single agent | Multi-agent | Delta |")
        p("|---|---:|---:|---:|")
    else:
        p(f"  {'criterion':<24}{'single':>9}{'multi':>9}{'delta':>9}")
        p("  " + "-" * 51)
    for c in CRITERIA:
        b, m = means["baseline"][c], means["multiagent"][c]
        if md:
            p(f"| {LABEL[c]} | {b:.2f} | {m:.2f} | {m - b:+.2f} |")
        else:
            p(f"  {LABEL[c]:<24}{b:>9.2f}{m:>9.2f}{m - b:>+9.2f}")

    bm = sum(means["baseline"].values()) / 5
    mm = sum(means["multiagent"].values()) / 5
    if md:
        p(f"| **Mean** | **{bm:.2f}** | **{mm:.2f}** | **{mm - bm:+.2f}** |")
        p(f"| Broken citations (count) | {d['broken']['baseline']} | "
          f"{d['broken']['multiagent']} | — |")
    else:
        p("  " + "-" * 51)
        p(f"  {'MEAN':<24}{bm:>9.2f}{mm:>9.2f}{mm - bm:>+9.2f}")
        p(f"  {'broken citations':<24}{d['broken']['baseline']:>9}"
          f"{d['broken']['multiagent']:>9}")

    # cost
    if cost and "multiples" in cost:
        mult = cost["multiples"]
        bt = cost["baseline"]["totals"]
        mt = cost["multiagent"]["totals"]
        p("\n### Cost\n" if md else "\nCost")
        if md:
            p("| Metric | Single agent | Multi-agent | Multiple |")
            p("|---|---:|---:|---:|")
            p(f"| Total tokens | {bt['total']:,} | {mt['total']:,} | {mult['tokens']}x |")
            p(f"| Model calls | {bt['calls']} | {mt['calls']} | {mult['calls']}x |")
            p(f"| Wall clock (s) | {bt['ms'] / 1000:.0f} | {mt['ms'] / 1000:.0f} | "
              f"{mult['latency']}x |")
            # usd_per_report is None unless PRICE_IN_PER_M and PRICE_OUT_PER_M were
            # supplied from the environment — cost.py withholds dollars rather than
            # print placeholder rates. Formatting None raises, so the row is dropped.
            # Not merely cosmetic: the cost multiple moves with the price sheet,
            # because the two systems have different input/output mixes. Only the
            # token multiples are rate-free.
            if bt.get("usd_per_report") and mt.get("usd_per_report"):
                p(f"| Cost per report | ${bt['usd_per_report']:.4f} | "
                  f"${mt['usd_per_report']:.4f} | {mult.get('cost')}x |")
            else:
                p("| Cost per report | — | — | no rates supplied |")
        else:
            tok_x = f"{mult['tokens']}x"
            call_x = f"{mult['calls']}x"
            lat_x = f"{mult['latency']}x"
            p(f"  {'total tokens':<24}{bt['total']:>9,}{mt['total']:>9,}{tok_x:>9}")
            p(f"  {'model calls':<24}{bt['calls']:>9}{mt['calls']:>9}{call_x:>9}")
            p(f"  {'wall clock (s)':<24}{bt['ms'] / 1000:>9.0f}"
              f"{mt['ms'] / 1000:>9.0f}{lat_x:>9}")

        ph = cost["multiagent"].get("by_phase", {})
        tot = sum(v["total"] for v in ph.values()) or 1
        p("\nWhere the multiple goes:" if not md else "\n**Where the multiple goes:**\n")
        for k, name in [("first_pass", "first pass"),
                        ("gap_loop", "gap loop (analyst->researcher)"),
                        ("revision_loop", "revision loop (critic->writer)")]:
            if k in ph:
                p(f"  {name:<34}{ph[k]['total'] / tot * 100:>5.1f}% of run cost")

    # by shape
    p("\n### By topic shape\n" if md else "\nBy topic shape (where the architecture pays)")
    if md:
        p("| Shape | n | Grounding delta | Coherence delta | Mean delta |")
        p("|---|---:|---:|---:|---:|")
    for shape, b in sorted(d["by_shape"].items()):
        g = b["multiagent"]["factual_grounding"] - b["baseline"]["factual_grounding"]
        s = b["multiagent"]["structural_coherence"] - b["baseline"]["structural_coherence"]
        mn = (sum(b["multiagent"].values()) - sum(b["baseline"].values())) / 5
        if md:
            p(f"| {shape} | {b['n']} | {g:+.2f} | {s:+.2f} | {mn:+.2f} |")
        else:
            p(f"  {shape:<20} n={b['n']}  grounding {g:+.2f}  coherence {s:+.2f}  "
              f"mean {mn:+.2f}")

    # hand scores. `hand` can be truthy with no usable topics when every scored
    # directory was dropped as stale, so this branches on the topics, not the dict.
    if hand and hand.get("topics"):
        p("\n### Hand scores\n" if md else "\nHand scores (blind, "
          f"{len(hand['topics'])} topics: {', '.join(hand['topics'])})")
        _, diffs = check_agreement(means, hand)
        if md:
            p("| Criterion | Single | Multi | Delta | Judge delta | Gap |")
            p("|---|---:|---:|---:|---:|---:|")
        for c in CRITERIA:
            hb = hand["means"]["baseline"][c]
            hm = hand["means"]["multiagent"][c]
            jd = means["multiagent"][c] - means["baseline"][c]
            if md:
                p(f"| {LABEL[c]} | {hb:.2f} | {hm:.2f} | {hm - hb:+.2f} | {jd:+.2f} | "
                  f"{diffs.get(c, 0):.2f} |")
            else:
                p(f"  {LABEL[c]:<24}{hb:>7.2f}{hm:>7.2f}  delta {hm - hb:+.2f}  "
                  f"judge {jd:+.2f}  gap {diffs.get(c, 0):.2f}")
    else:
        p("\n(no hand scores yet — run: uv run python -m evals.handscore)")

    # checks
    warnings = [
        check_length_bias(d["rows"]),
        check_baseline_strength(means),
        check_agreement(means, hand)[0],
        check_loop_value(cost, means),
        check_stale_hand(hand),
    ]
    warnings = [w for w in warnings if w]
    p("\n### Integrity checks\n" if md else "\n" + "=" * 72)
    if not md:
        p("INTEGRITY CHECKS")
        p("=" * 72)
    if warnings:
        for w in warnings:
            p(f"\n  ! {w}" if not md else f"\n> **Warning.** {w}")
    else:
        p("\n  No warnings. Length bias, baseline strength, judge/human agreement, and"
          if not md else "\nNo warnings raised.")
        if not md:
            p("  revision-loop value all within expected bounds.")

    out = RESULTS / "comparison.json"
    out.write_text(json.dumps({
        "quality": means, "by_shape": d["by_shape"], "broken_citations": d["broken"],
        "hand": hand, "cost_multiples": (cost or {}).get("multiples"),
        "warnings": warnings,
    }, indent=2), encoding="utf-8")
    # stderr, not stdout: --markdown is redirected into docs/comparison.md, and a status
    # line carrying an absolute local path does not belong in a committed document. The
    # operator still sees it. Same defect 8fd69a3 fixed in cost_report.py; this file was
    # written before that fix and inherited the shape.
    print(f"\nwrote {out}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    render(collect(), args.markdown)


if __name__ == "__main__":
    main()
