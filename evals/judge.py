"""Blind LLM judge — Phase 9.

    uv run python -m evals.judge
    uv run python -m evals.judge --repeats 3     # measure judge stability

Scores both systems' reports against the frozen rubric, blind.

FOUR BIASES THIS FIGHTS, AND HOW
---------------------------------
An LLM judge is a measuring instrument with known defects. Naming them and
controlling for each is what separates a real evaluation from a rubber stamp.

  1. Position bias — judges favour whichever report comes first.
     Control: the starting order is fixed per topic by a stable hash and recorded
     outside the prompt, then ALTERNATES with each round. --repeats 2 therefore
     shows every topic in both orders exactly once, which cancels the effect
     instead of hoping a random draw happens to balance.

  2. Length bias — judges reliably prefer longer outputs. Multi-agent is expected
     to produce longer reports, so this bias points directly at the result we are
     testing, which is the worst kind.
     Control: explicit instruction that length is not quality, word counts shown
     for both so the judge cannot be fooled about which is longer, plus an
     automatic sanity check in compare.py that flags a clean sweep for the longer
     report.

  3. Self-preference — a model scoring text produced by the same model family
     rates it higher.
     Control: cannot be eliminated here. Set GEMINI_JUDGE_MODEL to something other
     than the writer model if you can, and disclose the limitation either way.

  4. Halo — one strong criterion inflates the rest.
     Control: criteria scored with explicit independence instructions and anchored
     descriptions. Imperfect. This is why 3 of 6 topics get hand-scored.

The output of this file is EVIDENCE, not truth. Where the judge and your hand
scores disagree by 2 or more points, the writeup should say so and explain which
you trust. Those sentences are the most interesting paragraph in the README.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from scripts.judge import _body
from src.common.io import RESULTS, load_topics
from src.common.llm import TokenLedger, generate

# The judge must not be the writer. An earlier version read
#
#     os.getenv("GEMINI_JUDGE_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
#
# which looks like a safe chain and is not: GEMINI_MODEL is set in .env to the
# pipeline model, so with GEMINI_JUDGE_MODEL unset this silently resolved to the
# model that wrote the reports — bias 3 above, with no control at all and no
# indication in the output. The "gemini-2.5-flash" default was unreachable.
#
# That is not hypothetical. results/pairwise.json contains a run judged by
# gemini-3.1-flash-lite which returned an exact 6-6 tie, and a self-judging model
# is the first thing to suspect in a null result.
#
# gemini-3.5-flash is the default because it is the only judge this project has
# calibrated: 4/4 degradations caught on the right criterion, against the same
# damage suite the Critic is held to.
DEFAULT_JUDGE_MODEL = "gemini-3.5-flash"
PIPELINE_MODEL = os.getenv("GEMINI_MODEL", "")
JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
OUT = RESULTS / "judge.json"

CRITERIA = [
    "factual_grounding",
    "structural_coherence",
    "depth_of_analysis",
    "citation_integrity",
    "absence_of_filler",
]

PERSONA = """You are an exacting editor comparing two research reports on the same topic. You
did not write either one and you have no stake in either. Your job is to score them honestly
against a fixed rubric.

Two reports being different does not mean one is better. If they are genuinely close on a
criterion, give them the same score — you are not required to separate them."""

RUBRIC = """
Score each report on each criterion, 1-5, integers only.

1. factual_grounding — do substantive claims trace to the evidence provided?
   1: claims float free of the evidence.  5: every substantive claim traces to a finding,
   and where evidence is absent the report says so instead of asserting anyway.
   Do NOT reward confident tone. Fluent unsourced assertion is the failure this catches.
   A report that admits "the evidence here is thin" is scoring HIGHER, not lower.

2. structural_coherence — does it build an argument, or is it a pile of sections?
   1: sections repeat, contradict, or could be shuffled without loss.  5: later sections
   use what earlier ones established; the conclusion follows from the body.
   Look specifically for two sections making the same point in different words.

3. depth_of_analysis — does it synthesise or summarise?
   1: restates sources one after another.  5: connects findings, surfaces tensions between
   sources rather than averaging them away, draws conclusions the sources do not state
   individually. Smoothing a real disagreement into consensus FAILS this criterion.

4. citation_integrity — do citations resolve and support what they are attached to?
   1: IDs missing, invented, or pointing at unrelated findings.  5: every citation supports
   the exact sentence it sits on, and every claim needing one has one.
   Any ID not in the evidence list caps this at 1.

5. absence_of_filler — does every paragraph carry information?
   1: throat-clearing, restated headers, generic hedging, a conclusion that only summarises.
   5: nothing could be cut without losing content.
   A hedge tied to specific missing evidence is NOT filler — that is criterion 1 working.

Rules that override your instincts:
- LENGTH IS NOT QUALITY. The word counts are given below. A shorter report that says
  everything it needs scores HIGHER than a longer one that says the same thing with padding.
  Do not let thoroughness of appearance substitute for thoroughness of content.
- Score criteria INDEPENDENTLY. A beautifully structured report can be factually
  ungrounded. Do not let a strong score on one criterion pull the others up.
- When a report sits between two anchors, round DOWN.
- Report A and Report B are in random order. Neither position indicates anything."""


class ReportScores(BaseModel):
    factual_grounding: int = Field(ge=1, le=5)
    structural_coherence: int = Field(ge=1, le=5)
    depth_of_analysis: int = Field(ge=1, le=5)
    citation_integrity: int = Field(ge=1, le=5)
    absence_of_filler: int = Field(ge=1, le=5)


class Judgement(BaseModel):
    report_a: ReportScores
    report_b: ReportScores
    reasoning: str = Field(
        description="2-3 sentences on the main difference between them. Name specifics."
    )
    closest_criterion: str = Field(
        description="Which criterion they were most similar on."
    )


def _load(system: str, topic_id: str) -> dict[str, Any] | None:
    p = RESULTS / system / topic_id / "run.json"
    r = RESULTS / system / topic_id / "report.md"
    if not (p.exists() and r.exists()):
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    # _body strips the "Known limitations" section. Some graph reports carry one and
    # no baseline report ever does, which is a format tell strong enough to defeat the
    # blinding on its own — the judge would be identifying the system, not scoring the
    # writing. Imported from scripts.judge rather than restated so the two judges
    # cannot drift on what blinding means.
    #
    # The count is a property of the run, not a constant: it was 4 of 6 when
    # scripts/judge.py was written and is 3 of 6 on the run at b949796. Only the
    # asymmetry matters — the baseline has no mechanism to declare a gap, so any
    # number above zero is a tell.
    #
    # Stripping is not free: criterion 1 rewards a report that says where evidence is
    # absent, so removing the declaration removes something the graph genuinely earns.
    # The honest handling is to score the bodies and report the limitations feature
    # separately as a categorical fact rather than burying it inside a number that
    # cannot be attributed.
    d["report"] = _body(r.read_text(encoding="utf-8"))
    return d


def _evidence(run: dict[str, Any]) -> str:
    return "\n".join(f"[{f['id']}] {f['claim']}" for f in run.get("findings", []))


def _order_bit(topic_id: str) -> int:
    """Stable per-topic starting order, identical in every process.

    Python's hash() of a str is salted by PYTHONHASHSEED and differs between runs, so
    it cannot decide a presentation order that has to be reproducible — the same topic
    would be shown in a different order on every invocation and the recorded ordering
    could never be replayed. sha256 is stable by construction.
    """
    return hashlib.sha256(topic_id.encode()).digest()[0] & 1


def judge_topic(
    topic_id: str, topic: str, *, rep: int, ledger: TokenLedger
) -> dict[str, Any] | None:
    bl, ma = _load("baseline", topic_id), _load("multiagent", topic_id)
    if not bl or not ma:
        print(f"  {topic_id}: missing results for one system, skipping")
        return None

    # Deterministic alternation, not a fresh random draw. The previous version drew
    # again from a re-seeded RNG each round, which does not guarantee the orders
    # differ — two rounds could present the same order twice and double the position
    # bias while the docstring claimed to be cancelling it. Adding rep to a stable
    # per-topic bit makes round 2 the exact mirror of round 1, which is what actually
    # cancels the effect.
    a_is_baseline = (_order_bit(topic_id) + rep) % 2 == 0
    first, second = (bl, ma) if a_is_baseline else (ma, bl)

    # One evidence block per report, NOT a union keyed by ID.
    #
    # Both systems number their findings independently from F001, so the IDs collide
    # completely — 67 of 67 on t1, with different claims behind the same ID. A merged
    # dict lets one system's claims overwrite the other's, and the judge then checks
    # Report A's citations against Report B's evidence. Criterion 4 caps citation
    # integrity at 1 for an ID that does not resolve, so the corruption would show up
    # as a confident, systematic penalty against whichever system lost the merge,
    # inside a table that looks entirely normal.
    #
    # Each report cites its own IDs, so each is shown its own block and the judge is
    # told they are separate namespaces.
    resp = generate(
        f"Topic both reports address:\n{topic}\n\n"
        f"Each report cites its OWN evidence set. The two sets use the same ID format "
        f"and the same ID means DIFFERENT things in each — check every citation "
        f"against that report's own block below, never the other one.\n\n"
        f"--- Evidence for REPORT A ---\n{_evidence(first)}\n\n"
        f"--- Evidence for REPORT B ---\n{_evidence(second)}\n\n"
        f"=== REPORT A ({len(first['report'].split())} words) ===\n{first['report']}\n\n"
        f"=== REPORT B ({len(second['report'].split())} words) ===\n{second['report']}\n\n"
        f"Score both.{RUBRIC}",
        node="judge",
        call_type="judge",
        ledger=ledger,
        system=PERSONA,
        schema=Judgement,
        temperature=0.1,
        model=JUDGE_MODEL,
    )

    parsed: Judgement | None = getattr(resp, "parsed", None)
    if parsed is None:
        print(f"  {topic_id}: judgement did not parse, skipping")
        return None

    a, b = parsed.report_a.model_dump(), parsed.report_b.model_dump()
    return {
        "topic_id": topic_id,
        "presented_a_as": "baseline" if a_is_baseline else "multiagent",
        "baseline": a if a_is_baseline else b,
        "multiagent": b if a_is_baseline else a,
        "reasoning": parsed.reasoning,
        "closest_criterion": parsed.closest_criterion,
        "words": {"baseline": len(bl["report"].split()),
                  "multiagent": len(ma["report"].split())},
    }


def _preserve_legacy(path: Path) -> None:
    """Move an incompatible older judge file aside instead of overwriting it.

    scripts/judge.py wrote absolute per-system scores to this same path under a
    different schema — {judge_model, tokens, scores} against this file's
    {judge_model, repeats, topics, judge_tokens}. Overwriting silently would destroy
    the record that produced the current write-ups, and compare.py reads judge["topics"]
    so it cannot consume the old shape anyway.
    """
    if not path.exists():
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if "topics" in existing:
        return  # same schema — a normal re-run
    legacy = path.with_name("judge_legacy.json")
    legacy.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\n  !! {path.name} held the older per-system schema from scripts/judge.py")
    print(f"     (judge_model={existing.get('judge_model')}). Copied to {legacy.name}")
    print("     before overwriting. It is also in git history.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=1,
                    help="judge each topic N times with different orderings and average")
    ap.add_argument("--topic-id")
    ap.add_argument("--allow-self-judge", action="store_true",
                    help="permit the judge model to equal the pipeline model")
    args = ap.parse_args()

    # A judge that is also the writer scores its own family higher. This is a stop
    # rather than a warning because the failure is invisible in the output: the run
    # completes, the numbers look ordinary, and nothing in judge.json says the
    # instrument was measuring itself.
    if PIPELINE_MODEL and JUDGE_MODEL == PIPELINE_MODEL and not args.allow_self_judge:
        raise SystemExit(
            f"judge model ({JUDGE_MODEL}) is the pipeline model — it would be scoring "
            f"its own output.\nSet GEMINI_JUDGE_MODEL to a different model, or pass "
            f"--allow-self-judge to record the limitation deliberately."
        )

    topics = load_topics()
    if args.topic_id:
        topics = [t for t in topics if t["id"] == args.topic_id]

    ledger = TokenLedger()
    all_rounds: list[list[dict[str, Any]]] = []

    for rep in range(args.repeats):
        print(f"\nround {rep + 1}/{args.repeats}  (judge: {JUDGE_MODEL})")
        rows = []
        for t in topics:
            # rep, not a seed. The order alternates deterministically per round, so
            # --repeats 2 guarantees each topic is seen in both orders exactly once.
            r = judge_topic(t["id"], t["topic"], rep=rep, ledger=ledger)
            if r:
                rows.append(r)
                d = sum(r["multiagent"][c] - r["baseline"][c] for c in CRITERIA)
                print(f"  {t['id']}  presented_a={r['presented_a_as']:<10} "
                      f"net delta {d:+d}")
        all_rounds.append(rows)

    # Average across rounds per topic.
    merged: list[dict[str, Any]] = []
    by_topic: dict[str, list[dict[str, Any]]] = {}
    for rows in all_rounds:
        for r in rows:
            by_topic.setdefault(r["topic_id"], []).append(r)

    for tid, rs in by_topic.items():
        merged.append({
            "topic_id": tid,
            "rounds": len(rs),
            "baseline": {c: sum(r["baseline"][c] for r in rs) / len(rs) for c in CRITERIA},
            "multiagent": {c: sum(r["multiagent"][c] for r in rs) / len(rs) for c in CRITERIA},
            "orderings": [r["presented_a_as"] for r in rs],
            "reasoning": rs[0]["reasoning"],
            "words": rs[0]["words"],
            # Spread across rounds. High spread means the judge is unstable and the
            # scores deserve less weight than the hand scores.
            "spread": max(
                abs(rs[i]["multiagent"][c] - rs[j]["multiagent"][c])
                for c in CRITERIA for i in range(len(rs)) for j in range(len(rs))
            ) if len(rs) > 1 else 0,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    _preserve_legacy(OUT)
    OUT.write_text(json.dumps({
        "judge_model": JUDGE_MODEL,
        "pipeline_model": PIPELINE_MODEL,
        "self_judged": bool(PIPELINE_MODEL) and JUDGE_MODEL == PIPELINE_MODEL,
        "repeats": args.repeats,
        "topics": merged,
        "judge_tokens": ledger.total,
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {OUT}  ({ledger.total:,} judge tokens)")
    if args.repeats > 1:
        worst = max((m["spread"] for m in merged), default=0)
        print(f"max score spread across rounds: {worst}")
        if worst >= 2:
            print("  Judge is unstable — the same report scored 2+ points apart across")
            print("  orderings. Weight the hand scores more heavily and say so.")


if __name__ == "__main__":
    main()
