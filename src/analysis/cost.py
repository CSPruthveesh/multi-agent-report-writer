"""Cost analysis — Phase 8.

Turns raw token logs into the three tables the comparison needs:

  1. Per-node       which node buys the multiple
  2. Per-phase      what the gap loop and the revision loop each cost
  3. Per-call-type  what schema enforcement and retries cost

"Multi-agent is expensive" is an observation anyone can make. "The gap loop is 31%
of the run and the revision loop is 0%, because it never fires" is an answer, and it
is the difference between having built the thing and having understood it.

PRICING
-------
Token counts are measured and exact. Dollars are not reported at all unless both
rates are supplied from the environment, per million tokens:

    $env:PRICE_IN_PER_M="0.30"; $env:PRICE_OUT_PER_M="2.50"

Default behaviour is tokens and ratios only. A wrong dollar figure is worse than no
dollar figure, because the ratio is the real finding and a bad absolute number
invites doubt about it — so the placeholder constants below are never printed. They
exist to give cost_usd() a shape, not to be quoted.

Check current rates for the model in GEMINI_MODEL before supplying them.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.common.io import RESULTS

PRICE_IN_PER_M = float(os.getenv("PRICE_IN_PER_M", "0.30"))
PRICE_OUT_PER_M = float(os.getenv("PRICE_OUT_PER_M", "2.50"))

# Dollars are reported only when BOTH rates are supplied from the environment. The
# defaults above are placeholders, and a placeholder that prints looks exactly like a
# verified rate once it is in a table — so the absence of a dollar column is the
# signal that nobody has checked. Setting the env vars to the same numbers as the
# defaults still counts as supplying them: the point is that a person chose them.
PRICES_SET = "PRICE_IN_PER_M" in os.environ and "PRICE_OUT_PER_M" in os.environ


def cost_usd(in_tok: int, out_tok: int) -> float | None:
    """None when rates were never supplied — see PRICES_SET."""
    if not PRICES_SET:
        return None
    return (in_tok / 1e6) * PRICE_IN_PER_M + (out_tok / 1e6) * PRICE_OUT_PER_M


def load_runs(system: str) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((RESULTS / system).glob("*/run.json"))
    ]


def _records(run: dict[str, Any]) -> list[dict[str, Any]]:
    return run.get("tokens", {}).get("records", []) or []


def by_node(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    agg: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "in": 0, "out": 0, "total": 0, "ms": 0}
    )
    for r in runs:
        for rec in _records(r):
            d = agg[rec.get("node", "?")]
            d["calls"] += 1
            d["in"] += rec.get("in_tokens", 0)
            d["out"] += rec.get("out_tokens", 0)
            d["total"] += rec.get("total_tokens", 0)
            d["ms"] += rec.get("latency_ms", 0)
    return dict(agg)


def by_phase(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """first_pass / gap_loop / revision_loop.

    The two loops are different products. The gap loop buys evidence coverage; the
    revision loop buys prose quality. Averaging them into one "overhead" number
    destroys the only interesting finding in the project.

    Records written before the tagging existed have no phase key and land in
    first_pass, which understates both loops rather than inventing a number for
    them. The `tagged` count in the summary says how much of the data can answer
    this question at all.
    """
    agg: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "in": 0, "out": 0, "total": 0}
    )
    for r in runs:
        for rec in _records(r):
            d = agg[rec.get("phase", "first_pass")]
            d["calls"] += 1
            d["in"] += rec.get("in_tokens", 0)
            d["out"] += rec.get("out_tokens", 0)
            d["total"] += rec.get("total_tokens", 0)
    return dict(agg)


def tagged_share(runs: list[dict[str, Any]]) -> float:
    """Proportion of records carrying a phase tag. Below 1.0, by_phase is a floor."""
    recs = [rec for r in runs for rec in _records(r)]
    if not recs:
        return 0.0
    return round(sum(1 for rec in recs if "phase" in rec) / len(recs), 3)


def by_call_type(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    agg: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "total": 0})
    for r in runs:
        for rec in _records(r):
            d = agg[rec.get("call_type", "?")]
            d["calls"] += 1
            d["total"] += rec.get("total_tokens", 0)
    return dict(agg)


def retry_overhead(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """What the three failure classes actually cost.

    Transport retries are counted from attempts > 1; parse retries from the
    :parse_retry call-type suffix. Semantic retries are the revision loop and appear
    in by_phase, not here — they are not failures.
    """
    transport = parse = parse_tokens = 0
    for r in runs:
        for rec in _records(r):
            if rec.get("attempts", 1) > 1:
                transport += rec["attempts"] - 1
            if str(rec.get("call_type", "")).endswith(":parse_retry"):
                parse += 1
                parse_tokens += rec.get("total_tokens", 0)
    total = sum(r.get("tokens", {}).get("total", 0) for r in runs)
    return {
        "transport_retries": transport,
        "parse_retries": parse,
        "parse_retry_tokens": parse_tokens,
        "parse_retry_pct": round(parse_tokens / total * 100, 1) if total else 0.0,
    }


def totals(runs: list[dict[str, Any]]) -> dict[str, Any]:
    t = {"runs": len(runs), "in": 0, "out": 0, "total": 0, "calls": 0, "ms": 0,
         "words": 0}
    for r in runs:
        tk = r.get("tokens", {})
        t["in"] += tk.get("total_in", 0)
        t["out"] += tk.get("total_out", 0)
        t["total"] += tk.get("total", 0)
        t["calls"] += tk.get("calls", 0)
        t["ms"] += r.get("wall_ms", 0)
        t["words"] += r.get("word_count", 0)
    usd = cost_usd(t["in"], t["out"])
    t["usd"] = round(usd, 4) if usd is not None else None
    t["usd_per_report"] = (
        round(t["usd"] / len(runs), 4) if usd is not None and runs else None
    )
    return t


def write_json(path: Path | None = None) -> Path:
    """Persist the cost side of the comparison. Quality is joined to this elsewhere."""
    ma = load_runs("multiagent")
    bl = load_runs("baseline")
    payload: dict[str, Any] = {
        "pricing_note": (
            "Verified rates supplied via env; dollar figures are only as good as them."
            if PRICES_SET
            else "No rates supplied — tokens and ratios only. Set PRICE_IN_PER_M and "
            "PRICE_OUT_PER_M in the environment to report dollars."
        ),
        "prices_supplied": PRICES_SET,
        "price_in_per_m": PRICE_IN_PER_M if PRICES_SET else None,
        "price_out_per_m": PRICE_OUT_PER_M if PRICES_SET else None,
        "baseline": {"totals": totals(bl), "by_node": by_node(bl)},
        "multiagent": {
            "totals": totals(ma),
            "by_node": by_node(ma),
            "by_phase": by_phase(ma),
            "phase_tagged_share": tagged_share(ma),
            "by_call_type": by_call_type(ma),
            "retries": retry_overhead(ma),
        },
    }
    bt, mt = payload["baseline"]["totals"], payload["multiagent"]["totals"]
    if bt["total"] and mt["total"]:
        payload["multiples"] = {
            "tokens": round(mt["total"] / bt["total"], 2),
            "calls": round(mt["calls"] / bt["calls"], 2) if bt["calls"] else None,
            "latency": round(mt["ms"] / bt["ms"], 2) if bt["ms"] else None,
            "cost": round(mt["usd"] / bt["usd"], 2) if bt["usd"] and mt["usd"] else None,
        }
    p = path or (RESULTS / "cost.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
