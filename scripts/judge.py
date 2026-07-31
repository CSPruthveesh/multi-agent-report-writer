"""Score both systems' reports against the frozen rubric, blind to which is which.

    uv run python -m scripts.judge              # one pass, 12 calls
    uv run python -m scripts.judge --runs 2     # two passes, 24 calls

WHY THIS EXISTS
---------------
Everything measured so far is cost and mechanism. The six-topic comparison can say
the graph costs 26% more and that its gap loop fires where evidence is thinner. It
cannot say whether the reports are better, which is the question the project exists
to answer.

Two things stand in the way of the numbers already collected:

  - the Critic scored the graph's reports as part of the graph, so it graded its own
    pipeline's output. Those scores are not comparable to anything.
  - nobody has ever scored a baseline report. evals/rubric.md was frozen in Phase 0
    and has only ever been applied to one of the two systems.

So this runs the same rubric over all twelve reports as an outside judge, with no
knowledge of which system produced each.

WHAT BLIND MEANS HERE, AND WHAT IT DOES NOT
-------------------------------------------
The prompt never names the system. But four of six graph reports carry a "Known
limitations" section and no baseline report does, which is a format tell strong
enough to defeat the blinding on its own. That section is stripped before scoring.

Stripping it is not free. Criterion 1 rewards a report that says where evidence is
absent rather than asserting anyway, so removing the declaration removes something
the graph genuinely earns. The honest handling is to score the bodies, and report the
limitations feature separately as a categorical fact — the graph declares unclosed
evidence gaps on 4 of 6 topics, the baseline has no mechanism to declare any — rather
than burying it inside a number that cannot be attributed.

The rubric and the score schema are imported from the Critic rather than restated, so
the judge cannot drift from the standard the system was built against.
"""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Any

from src.common.io import RESULTS
from src.common.llm import TokenLedger, generate
from src.graph.nodes.critic import RUBRIC, Scores

# A model outside the pipeline, for two reasons that happen to coincide.
#
# Methodological: the pipeline writes with gemini-3.1-flash-lite, so judging with the
# same model is a family grading its own work. An outside judge is the stronger
# comparison, which is what GEMINI_CRITIC_MODEL was put in common/llm.py to allow.
#
# Practical: the free tier caps requests per day PER MODEL, and a day of six-topic
# runs exhausts the pipeline model's 500. A different model has its own allowance.
#
# Passed explicitly rather than read from the environment, because llm.py calls
# load_dotenv(override=True) — .env beats a shell variable, so setting
# GEMINI_CRITIC_MODEL in the shell would silently have no effect.
DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"

# Scores are written here as each one lands. The first attempt at this script held
# everything in memory and printed at the end, so when the quota ran out on call nine
# the eight that had succeeded were thrown away — the same argument that justified
# checkpointing the graph, not applied to the script measuring it.
SCORES_FILE = RESULTS / "judge.json"

SYSTEMS = ("baseline", "multiagent")
TOPICS = ("t1", "t2", "t3", "t4", "t5", "t6")
CRITERIA = ("factual_grounding", "structural_coherence", "depth_of_analysis",
            "citation_integrity", "absence_of_filler")

JUDGE_PERSONA = """You are an impartial reviewer scoring research reports against a fixed
rubric. You did not write these reports and you do not know how they were produced. You
have the evidence each one was written from, so you can check whether its claims are
actually supported.

Score honestly and independently. A report that reads well but asserts things the evidence
does not establish is a bad report."""


def _body(markdown: str) -> str:
    """The report without its declared-limitations section.

    That section is a graph feature the baseline cannot produce, so leaving it in
    would tell the judge which system it is looking at.
    """
    marker = "## Known limitations"
    cut = markdown.find(marker)
    if cut == -1:
        return markdown.strip()
    # also drop the horizontal rule that precedes it
    head = markdown[:cut].rstrip()
    if head.endswith("---"):
        head = head[:-3].rstrip()
    return head


def _load(system: str, topic: str) -> tuple[str, list[dict[str, Any]]]:
    d = RESULTS / system / topic
    report = _body((d / "report.md").read_text(encoding="utf-8"))
    findings = json.loads((d / "run.json").read_text(encoding="utf-8"))["findings"]
    return report, findings


def _load_scores() -> dict[str, dict[str, list[dict[str, int]]]]:
    if SCORES_FILE.exists():
        saved = json.loads(SCORES_FILE.read_text(encoding="utf-8"))
        return saved.get("scores") or {s: {t: [] for t in TOPICS} for s in SYSTEMS}
    return {s: {t: [] for t in TOPICS} for s in SYSTEMS}


def _save_scores(scores: dict, model: str, tokens: int) -> None:
    SCORES_FILE.write_text(
        json.dumps({"judge_model": model, "tokens": tokens, "scores": scores}, indent=2),
        encoding="utf-8",
    )


def _score(report: str, findings: list[dict[str, Any]],
           ledger: TokenLedger, model: str) -> dict[str, int] | None:
    evidence = "\n".join(
        f"[{f['id']}] ({f['confidence']}) {f['claim']}" for f in findings
    )
    resp = generate(
        f"Evidence the report was written from:\n{evidence}\n\n"
        f"Report:\n---\n{report}\n---\n\n"
        f"Score it.{RUBRIC}",
        node="judge",
        call_type="judge",
        ledger=ledger,
        system=JUDGE_PERSONA,
        schema=Scores,
        temperature=0.0,
        model=model,
    )
    parsed = getattr(resp, "parsed", None)
    return parsed.model_dump() if parsed else None


def calibrate(model: str) -> None:
    """Damage a report in four known ways and check the judge notices.

    A judge that awards 5/5/5/5/5 to almost everything is not measuring quality, it
    is agreeing — and a comparison decided by an instrument with no spread is not a
    comparison. Ten of twelve reports scored straight 5s on the first pass, which is
    the "always passes" failure scripts/test_critic.py was built to catch and which
    was not applied to the judge before its numbers were read.

    The degradations are imported from that script rather than restated, so the judge
    is held to the same standard as the Critic and cannot drift from it.
    """
    from scripts.test_critic import DEGRADATIONS

    ledger = TokenLedger()
    report, findings = _load("baseline", "t1")
    base = _score(report, findings, ledger, model)
    if not base:
        raise SystemExit("the judge could not score the clean report")

    print(f"judge model: {model}\n")
    print("clean       " + "/".join(str(base[c]) for c in CRITERIA))

    rows = []
    for name, (fn, expect) in DEGRADATIONS.items():
        got = _score(fn(report), findings, ledger, model)
        if not got:
            print(f"{name:<12}PARSE FAILED")
            continue
        rows.append((name, expect, base[expect], got[expect]))
        print(f"{name:<12}" + "/".join(str(got[c]) for c in CRITERIA)
              + f"   ({expect} {base[expect]} -> {got[expect]})")

    passes = sum(1 for _, _, was, now in rows if now < was)
    print(f"\n  {passes}/{len(rows)} degradations detected on the right criterion")
    if passes >= 3:
        print("  CALIBRATED — the judge responds to real damage, so its scores carry.")
    elif passes == 0:
        print("  FAIL — the judge scores damaged reports the same as clean ones. Its")
        print("  numbers measure nothing and the comparison has to be redone with a")
        print("  stronger model or a forced pairwise choice instead of absolute scores.")
    else:
        print("  PARTIAL — trust only the criteria that moved. A criterion the judge")
        print("  cannot see is a column of noise in the comparison table.")
    print(f"\n  cost: {ledger.total:,} tokens across {len(ledger.records)} calls")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true",
                    help="check the judge notices deliberate damage before trusting it")
    ap.add_argument("--runs", type=int, default=1,
                    help="passes per report; >1 shows whether the scores reproduce")
    ap.add_argument("--model", default=DEFAULT_JUDGE_MODEL,
                    help="judge model; must differ from the pipeline's, see the note above")
    ap.add_argument("--fresh", action="store_true", help="discard saved scores and rescore")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.model)
        return

    ledger = TokenLedger()
    scores = {s: {t: [] for t in TOPICS} for s in SYSTEMS} if args.fresh else _load_scores()

    print(f"judge model: {args.model}\n")
    for run in range(args.runs):
        for system in SYSTEMS:
            for topic in TOPICS:
                if len(scores[system][topic]) > run:
                    print(f"  pass {run + 1}  {system:<11}{topic}  (already scored)")
                    continue
                report, findings = _load(system, topic)
                got = _score(report, findings, ledger, args.model)
                if got:
                    scores[system][topic].append(got)
                    # Written after every call, so a quota failure costs one call.
                    _save_scores(scores, args.model, ledger.total)
                print(f"  pass {run + 1}  {system:<11}{topic}  "
                      f"{'/'.join(str(got[c]) for c in CRITERIA) if got else 'PARSE FAILED'}")

    def mean_of(system: str, criterion: str) -> float:
        vals = [r[criterion] for t in TOPICS for r in scores[system][t]]
        return statistics.mean(vals) if vals else 0.0

    print("\n" + "=" * 72)
    print(f"{'criterion':<24}{'baseline':>10}{'graph':>10}{'delta':>10}")
    print("-" * 72)
    for c in CRITERIA:
        b, m = mean_of("baseline", c), mean_of("multiagent", c)
        print(f"{c:<24}{b:>10.2f}{m:>10.2f}{m - b:>+10.2f}")
    ball = [r[c] for t in TOPICS for r in scores["baseline"][t] for c in CRITERIA]
    mall = [r[c] for t in TOPICS for r in scores["multiagent"][t] for c in CRITERIA]
    print("-" * 72)
    print(f"{'OVERALL':<24}{statistics.mean(ball):>10.2f}"
          f"{statistics.mean(mall):>10.2f}{statistics.mean(mall) - statistics.mean(ball):>+10.2f}")

    print("\nper topic, overall mean:")
    print(f"  {'':4}{'baseline':>10}{'graph':>10}{'winner':>10}")
    for t in TOPICS:
        b = statistics.mean([r[c] for r in scores["baseline"][t] for c in CRITERIA])
        m = statistics.mean([r[c] for r in scores["multiagent"][t] for c in CRITERIA])
        win = "graph" if m > b else ("baseline" if b > m else "tie")
        print(f"  {t:4}{b:>10.2f}{m:>10.2f}{win:>10}")

    if args.runs > 1:
        stable = all(
            len({tuple(sorted(r.items())) for r in scores[s][t]}) == 1
            for s in SYSTEMS for t in TOPICS
        )
        print(f"\nidentical across {args.runs} passes: {'yes' if stable else 'NO'}")
        if not stable:
            print("  scores move between passes — treat small deltas above as noise")

    print("\nnot scored, because it cannot be blinded:")
    for s in SYSTEMS:
        n = sum(1 for t in TOPICS
                if "Known limitations" in (RESULTS / s / t / "report.md")
                .read_text(encoding="utf-8"))
        print(f"  {s:<11} declares unclosed evidence gaps on {n}/6 topics")
    print("  The baseline has no mechanism to declare any. That is a real difference")
    print("  between the systems and it is deliberately not folded into a score.")

    print(f"\njudge cost: {ledger.total:,} tokens across {len(ledger.records)} calls")


if __name__ == "__main__":
    main()
