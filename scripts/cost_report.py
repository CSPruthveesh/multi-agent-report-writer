"""Print the cost breakdown tables.

    uv run python -m scripts.cost_report
    uv run python -m scripts.cost_report --markdown > docs/cost.md

Reads results/*/run.json. No API calls, so run it as often as you like.

The per-phase table is the one to look at hardest. If the revision loop is a large
share of the run, the comparison has to justify it with a quality delta — and if it
cannot, the honest move is to cap revisions at one and say so. Finding that out is a
better outcome than a clean win.
"""

from __future__ import annotations

import argparse
import sys

from src.analysis.cost import (
    PRICES_SET,
    by_call_type,
    by_node,
    by_phase,
    code_versions,
    load_runs,
    retry_overhead,
    tagged_share,
    totals,
    write_json,
)

NODE_ORDER = ["researcher", "analyst", "writer", "critic", "supervisor",
              "finalize", "approval", "agent"]


def _bar(pct: float, width: int = 18) -> str:
    n = round(pct / 100 * width)
    return "#" * n + "." * (width - n)


def table_nodes(runs, title: str, md: bool) -> None:
    agg = by_node(runs)
    tot = sum(v["total"] for v in agg.values()) or 1
    rows = sorted(agg.items(),
                  key=lambda kv: NODE_ORDER.index(kv[0]) if kv[0] in NODE_ORDER else 99)

    print(f"\n### {title}" if md else f"\n{title}")
    if md:
        print("\n| node | calls | in | out | total | % |")
        print("|---|---:|---:|---:|---:|---:|")
        for n, v in rows:
            print(f"| {n} | {v['calls']:.0f} | {v['in']:,.0f} | {v['out']:,.0f} | "
                  f"{v['total']:,.0f} | {v['total'] / tot * 100:.1f}% |")
        print(f"| **TOTAL** | **{sum(v['calls'] for v in agg.values()):.0f}** | | | "
              f"**{tot:,.0f}** | **100%** |")
        return

    print(f"  {'node':<12}{'calls':>6}{'in':>10}{'out':>9}{'total':>10}{'%':>7}  share")
    print("  " + "-" * 70)
    for n, v in rows:
        pct = v["total"] / tot * 100
        print(f"  {n:<12}{v['calls']:>6.0f}{v['in']:>10,.0f}{v['out']:>9,.0f}"
              f"{v['total']:>10,.0f}{pct:>6.1f}%  {_bar(pct)}")
    print("  " + "-" * 70)
    print(f"  {'TOTAL':<12}{sum(v['calls'] for v in agg.values()):>6.0f}"
          f"{'':>19}{tot:>10,.0f}{100:>6.1f}%")


def table_phases(runs, md: bool) -> None:
    agg = by_phase(runs)
    tot = sum(v["total"] for v in agg.values()) or 1
    share = tagged_share(runs)
    order = ["first_pass", "gap_loop", "revision_loop"]
    label = {
        "first_pass": "first pass",
        "gap_loop": "gap loop (analyst -> researcher)",
        "revision_loop": "revision loop (critic -> writer)",
    }
    print("\n### Cost by phase" if md else "\nCost by phase")
    if md:
        print("\n| phase | calls | tokens | % of run |")
        print("|---|---:|---:|---:|")
        for k in order:
            if k in agg:
                v = agg[k]
                print(f"| {label[k]} | {v['calls']:.0f} | {v['total']:,.0f} | "
                      f"{v['total'] / tot * 100:.1f}% |")
    else:
        print(f"  {'phase':<34}{'calls':>6}{'tokens':>10}{'%':>7}  share")
        print("  " + "-" * 78)
        for k in order:
            if k in agg:
                v = agg[k]
                pct = v["total"] / tot * 100
                print(f"  {label[k]:<34}{v['calls']:>6.0f}{v['total']:>10,.0f}"
                      f"{pct:>6.1f}%  {_bar(pct)}")

    if share < 1.0:
        note = (f"only {share:.0%} of records carry a phase tag — runs recorded before "
                f"tagging existed all fall into first_pass, so the loop rows are a "
                f"floor, not a measurement")
        print(f"\n  !! {note}" if not md else f"\n> **Caveat:** {note}")


def provenance(sets: dict[str, list], md: bool) -> None:
    """Which code produced each set, and whether the comparison is safe to read.

    Printed above the tables rather than below them. A reader who scrolls to the
    numbers first should have already passed the reason not to trust them.
    """
    rows, warnings = [], []
    for name, runs in sets.items():
        if not runs:
            continue
        cv = code_versions(runs)
        commits = ", ".join(cv["commits"]) or "unknown"

        # Each condition is reported on its own. An earlier version branched on the
        # single `consistent` flag, which folds "more than one commit" together with
        # "some records have none" — a set that was dirty and partly unlabelled printed
        # "MIXED, 6 runs span 1 commits" and dropped the dirty warning entirely. A
        # validity check that garbles which invalidation occurred is worse than none,
        # because it is still read as authoritative.
        if len(cv["commits"]) > 1:
            state = "MIXED — not comparable"
            warnings.append(
                f"{name}: {cv['runs']} runs span {len(cv['commits'])} commits "
                f"({commits}). A set built from more than one code version cannot be "
                f"read as one measurement — re-run it."
            )
        elif cv["missing"] == cv["runs"]:
            state = "predates this check"
        elif cv["dirty"]:
            state = "dirty tree"
        elif cv["inferred"] == cv["runs"]:
            state = "inferred, not observed"
        elif cv["inferred"]:
            state = f"{cv['inferred']}/{cv['runs']} inferred"
        else:
            state = "clean"

        if cv["dirty"]:
            warnings.append(
                f"{name}: produced from a modified working tree, so it is not "
                f"reproducible from commit {commits}."
            )
        if cv["missing"] and cv["missing"] != cv["runs"]:
            warnings.append(
                f"{name}: {cv['missing']} of {cv['runs']} runs carry no code version, "
                f"so the set is only partly verifiable."
            )
        rows.append((name, str(cv["runs"]), commits, state))

    if not rows:
        return
    if md:
        print("\n### Provenance\n")
        print("| system | runs | commit | state |")
        print("|---|---:|---|---|")
        for r in rows:
            print(f"| {r[0]} | {r[1]} | `{r[2]}` | {r[3]} |")
        for w in warnings:
            print(f"\n> **Warning:** {w}")
    else:
        print("\nProvenance")
        print(f"  {'system':<12}{'runs':>5}  {'commit':<22}state")
        print("  " + "-" * 62)
        for r in rows:
            print(f"  {r[0]:<12}{r[1]:>5}  {r[2]:<22}{r[3]}")
        for w in warnings:
            print(f"\n  !! {w}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()
    md = args.markdown

    bl, ma = load_runs("baseline"), load_runs("multiagent")
    if not bl and not ma:
        raise SystemExit("No results found. Run at least one system first.")

    if md:
        print("# Cost breakdown")
    else:
        print("=" * 74)
        print("COST BREAKDOWN")
        print("=" * 74)

    provenance({"baseline": bl, "multiagent": ma}, md)

    if bl:
        table_nodes(bl, f"Baseline ({len(bl)} reports)", md)
    if ma:
        table_nodes(ma, f"Multi-agent ({len(ma)} reports)", md)
        table_phases(ma, md)

        ct = by_call_type(ma)
        tot = sum(v["total"] for v in ct.values()) or 1
        print("\n### Cost by call type" if md else "\nCost by call type")
        if md:
            print("\n| call type | calls | tokens | % |")
            print("|---|---:|---:|---:|")
        else:
            print(f"  {'call type':<22}{'calls':>6}{'tokens':>10}{'%':>7}")
            print("  " + "-" * 46)
        for k, v in sorted(ct.items(), key=lambda kv: -kv[1]["total"]):
            pct = v["total"] / tot * 100
            if md:
                print(f"| {k} | {v['calls']:.0f} | {v['total']:,.0f} | {pct:.1f}% |")
            else:
                print(f"  {k:<22}{v['calls']:>6.0f}{v['total']:>10,.0f}{pct:>6.1f}%")

        r = retry_overhead(ma)
        if md:
            print("\n### Failure-class overhead\n")
            print("| failure class | count | tokens |")
            print("|---|---:|---:|")
            print(f"| transport retries (429/500/timeout) | {r['transport_retries']} | — |")
            print(f"| parse retries (schema mismatch) | {r['parse_retries']} | "
                  f"{r['parse_retry_tokens']:,} ({r['parse_retry_pct']}% of run) |")
            print("\n> Semantic retries are the revision loop, counted in the phase table"
                  "\n> above — they are not failures. A run where the Critic never asked for"
                  "\n> a revision has no revision-loop row at all.")
        else:
            print("\nFailure-class overhead")
            print(f"  transport retries (429/500/timeout) : {r['transport_retries']}")
            print(f"  parse retries (schema mismatch)     : {r['parse_retries']}"
                  f"  ({r['parse_retry_tokens']:,} tokens, {r['parse_retry_pct']}% of run)")
            print("  semantic retries are the revision loop, counted in the phase table"
                  " above — not failures")

    if bl and ma:
        bt, mt = totals(bl), totals(ma)
        if md:
            print("\n### Head to head")
        else:
            print("\n" + "=" * 74)
            print("HEAD TO HEAD")
            print("=" * 74)
        rows = [
            ("reports", bt["runs"], mt["runs"], None),
            ("model calls", bt["calls"], mt["calls"],
             mt["calls"] / bt["calls"] if bt["calls"] else 0),
            ("input tokens", bt["in"], mt["in"], mt["in"] / bt["in"] if bt["in"] else 0),
            ("output tokens", bt["out"], mt["out"],
             mt["out"] / bt["out"] if bt["out"] else 0),
            ("total tokens", bt["total"], mt["total"],
             mt["total"] / bt["total"] if bt["total"] else 0),
            ("wall clock (s)", bt["ms"] / 1000, mt["ms"] / 1000,
             mt["ms"] / bt["ms"] if bt["ms"] else 0),
            ("words written", bt["words"], mt["words"], None),
        ]
        if md:
            print("\n| metric | baseline | multi-agent | multiple |")
            print("|---|---:|---:|---:|")
            for n, b, m, x in rows:
                print(f"| {n} | {b:,.0f} | {m:,.0f} | {f'{x:.2f}x' if x else '—'} |")
            if bt.get("usd_per_report") and mt.get("usd_per_report"):
                print(f"| cost per report (USD) | ${bt['usd_per_report']:.4f} | "
                      f"${mt['usd_per_report']:.4f} | "
                      f"{mt['usd_per_report'] / bt['usd_per_report']:.2f}x |")
        else:
            print(f"  {'metric':<18}{'baseline':>13}{'multi-agent':>14}{'multiple':>11}")
            print("  " + "-" * 58)
            for n, b, m, x in rows:
                print(f"  {n:<18}{b:>13,.0f}{m:>14,.0f}"
                      f"{(f'{x:.2f}x' if x else '—'):>11}")
            if bt.get("usd_per_report") and mt.get("usd_per_report"):
                mult = mt["usd_per_report"] / bt["usd_per_report"]
                print(f"  {'USD per report':<18}{bt['usd_per_report']:>13.4f}"
                      f"{mt['usd_per_report']:>14.4f}{f'{mult:.2f}x':>11}")
        if PRICES_SET and md:
            print("\n> **Dollar figures use the rates supplied in the environment** —"
                  "\n> they are only as good as those numbers. Token counts are exact.")
        elif PRICES_SET:
            print("\n  (dollar figures use the rates supplied in the environment —"
                  "\n   they are only as good as those numbers; token counts are exact)")
        elif md:
            print("\n> **Tokens and ratios only.** No pricing rates were supplied, so no"
                  "\n> dollar figures are reported. Check current rates for the model in"
                  "\n> `GEMINI_MODEL`, then set `PRICE_IN_PER_M` and `PRICE_OUT_PER_M`.")
        else:
            print("\n  tokens and ratios only — no pricing rates supplied, so no dollar")
            print("  figures are reported. Check current rates for the model in")
            print("  GEMINI_MODEL, then set PRICE_IN_PER_M and PRICE_OUT_PER_M.")

    p = write_json()
    # stderr, not stdout: --markdown is redirected into docs/cost.md, and a status
    # line on stdout lands in the committed file along with an absolute local path.
    print(f"\nwrote {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
