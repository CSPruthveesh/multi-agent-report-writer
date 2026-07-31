"""Forced pairwise choice between the two systems' reports on the same topic.

    uv run python -m scripts.pairwise            # 12 calls, both orders per topic
    uv run python -m scripts.pairwise --topic t4 # one pair

WHY ABSOLUTE SCORING WAS NOT ENOUGH
-----------------------------------
scripts/judge.py scored all twelve reports 1-5 against the frozen rubric and returned
straight 5s on ten of them. The judge is calibrated — it drops citation_integrity from
5 to 1 on a report with its citations stripped — so the 5s mean "no damage found"
rather than "not looking. But an instrument that detects vandalism does not thereby
rank two competent reports, and three of five criteria returned identical scores for
every report in the set.

The one topic that did separate, t4, separated for the wrong reason. The baseline
covered monopsony and made a single unsupported claim inside that section; the graph
never raised monopsony at all and so could not be caught. It scored higher for
covering less. The rubric penalises an unsupported claim and does not reward ground
covered, so a narrower report is structurally safer under absolute scoring.

A forced choice between two reports asks the question absolute scores cannot: not "is
this report good" but "which of these two would you rather have read". Ties are not
allowed, so the judge cannot retreat to 5/5.

THE CONTROL THAT MAKES IT MEAN ANYTHING
---------------------------------------
Every pair is judged twice, with the reports swapped between position A and B. A
preference that survives the swap is a preference. A preference that follows position
A regardless of content is position bias, and would otherwise look exactly like a
result. Order is the only thing that changes between the two calls.

Reports are stripped of their Known-limitations section first, for the same reason as
in judge.py: four of six graph reports carry one and no baseline report does, which
identifies the system before a word of the body is read.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from pydantic import BaseModel, Field

from scripts.judge import DEFAULT_JUDGE_MODEL, TOPICS, _body
from src.common.io import RESULTS
from src.common.llm import TokenLedger, generate
from src.graph.nodes.critic import RUBRIC

RESULT_FILE = RESULTS / "pairwise.json"

PERSONA = """You are an impartial reviewer comparing two research reports on the same
question, written from the same body of evidence. You did not write either and you do not
know how either was produced.

One of them is better. Say which, and say what decided it. Do not hedge, do not call them
equivalent, and do not reward length or confident tone."""


class Choice(BaseModel):
    winner: str = Field(description="Exactly 'A' or 'B'. No other value is acceptable.")
    criterion: str = Field(
        description="Which rubric criterion decided it: factual_grounding, "
        "structural_coherence, depth_of_analysis, citation_integrity or absence_of_filler."
    )
    reason: str = Field(description="One or two sentences. Specific to these two reports.")
    margin: str = Field(description="'clear' or 'narrow'.")


def _load(system: str, topic: str) -> tuple[str, str]:
    d = RESULTS / system / topic
    report = _body((d / "report.md").read_text(encoding="utf-8"))
    findings = json.loads((d / "run.json").read_text(encoding="utf-8"))["findings"]
    evidence = "\n".join(f"[{f['id']}] ({f['confidence']}) {f['claim']}" for f in findings)
    return report, evidence


def _compare(topic: str, a_system: str, b_system: str,
             ledger: TokenLedger, model: str) -> dict[str, Any] | None:
    a_report, a_ev = _load(a_system, topic)
    b_report, b_ev = _load(b_system, topic)

    resp = generate(
        f"Both reports answer the same question and each was written from its own "
        f"evidence set. Judge the reports, not the evidence they were given.\n\n"
        f"=== REPORT A ===\n{a_report}\n\n"
        f"Evidence available to A:\n{a_ev}\n\n"
        f"=== REPORT B ===\n{b_report}\n\n"
        f"Evidence available to B:\n{b_ev}\n\n"
        f"Which report is better? Use the rubric below as your standard, then name the "
        f"single criterion that decided it.{RUBRIC}",
        node="pairwise",
        call_type="compare",
        ledger=ledger,
        system=PERSONA,
        schema=Choice,
        temperature=0.0,
        model=model,
    )
    parsed = getattr(resp, "parsed", None)
    if not parsed:
        return None
    c = parsed.model_dump()
    c["winner_system"] = a_system if c["winner"].strip().upper() == "A" else b_system
    c["a_system"], c["b_system"] = a_system, b_system
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", help="one topic id; default is all six")
    ap.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    args = ap.parse_args()

    topics = [args.topic] if args.topic else list(TOPICS)
    ledger = TokenLedger()

    # Keyed by model, not overwritten per run. Two judges reading the same pairs is
    # a stronger result than one judge reading them twice — agreement between models
    # is evidence the preference is about the reports, and disagreement is evidence
    # it is about the judge. Either is worth more than a file with one of them in it.
    #
    # This is not hypothetical. The first judge, gemini-2.5-flash, became unavailable
    # to new users partway through the comparison — its three completed pairs cannot
    # be reproduced and would have been silently replaced by a plain overwrite.
    saved: dict[str, Any] = {}
    if RESULT_FILE.exists():
        prev = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
        # migrate the original single-model shape
        saved = prev.get("by_model") or (
            {prev.get("judge_model", "unknown"): prev.get("results") or {}}
        )
    out: dict[str, list[dict[str, Any]]] = saved.setdefault(args.model, {})

    print(f"judge model: {args.model}\n")
    for t in topics:
        out[t] = []
        # Same pair, both orders. Only position changes.
        for a, b in (("baseline", "multiagent"), ("multiagent", "baseline")):
            c = _compare(t, a, b, ledger, args.model)
            if c:
                out[t].append(c)
                RESULT_FILE.write_text(
                    json.dumps({"by_model": saved}, indent=2), encoding="utf-8")
                print(f"  {t}  A={a[:9]:<10} B={b[:9]:<10} -> {c['winner_system']:<11}"
                      f"({c['margin']}, {c['criterion']})")
            else:
                print(f"  {t}  A={a:<11} PARSE FAILED")

    print("\n" + "=" * 72)
    consistent = graph = base = 0
    for t in topics:
        rs = out.get(t) or []
        if len(rs) != 2:
            print(f"  {t}  incomplete")
            continue
        w1, w2 = rs[0]["winner_system"], rs[1]["winner_system"]
        if w1 == w2:
            consistent += 1
            graph += w1 == "multiagent"
            base += w1 == "baseline"
            print(f"  {t}  {w1:<11} wins in both orders   <- a real preference")
        else:
            # The judge picked whichever report sat in the same slot both times.
            slot = "A" if rs[0]["winner"].upper() == "A" and rs[1]["winner"].upper() == "A" else "B"
            print(f"  {t}  order-dependent ({w1} then {w2}) — the judge followed slot {slot}")

    print("-" * 72)
    print(f"  consistent preferences : {consistent}/{len(topics)}")
    print(f"  of those, graph wins   : {graph}")
    print(f"  of those, baseline wins: {base}")
    if consistent == 0:
        print("\n  Every result flipped with position. The judge has no preference between")
        print("  these reports — it has a preference for whichever it reads first.")
    elif consistent < len(topics):
        print("\n  Only the consistent rows are evidence. The rest measured position.")

    print("\nreasons given, first order only:")
    for t in topics:
        if out.get(t):
            r = out[t][0]
            print(f"  {t}  {r['winner_system']:<11} {r['reason'][:110]}")

    print(f"\ncost: {ledger.total:,} tokens across {len(ledger.records)} calls")


if __name__ == "__main__":
    main()
