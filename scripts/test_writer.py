"""Test the Writer in isolation — cold draft and the revision path.

    uv run python -m scripts.test_writer --topic-id t1
    uv run python -m scripts.test_writer --topic-id t1 --revise    # THE important one

Reuses baseline findings, so it costs one Analyst call plus one or two Writer calls
and no searches.

WHY THE REVISION TEST MATTERS
-----------------------------
The revision path is where a multi-agent writer quietly wastes its budget. The naive
implementation regenerates all 1000 words on every loop, which costs a full write,
regresses passages that were fine, and makes it impossible to say what changed.

This Writer returns {find, replace} edits applied in code. `--revise` feeds it a
synthetic criticism and reports:

    changed_pct    how much of the document actually moved
    edits_applied  how many edits matched verbatim
    fallback       whether it gave up and rewrote wholesale

Healthy surgical revision: changed_pct under about 25, all edits applied, no fallback.
If changed_pct is near 100 or fallback keeps firing, the model is not copying spans
verbatim — tighten the `find` instruction before Phase 5, because from Phase 5 onward
the Critic generates these spans and you will not be able to tell the two failures apart.
"""

from __future__ import annotations

import argparse
import json

from src.common.io import RESULTS, check_citations, get_topic
from src.common.schemas import Finding
from src.graph.nodes.analyst import analyst
from src.graph.nodes.writer import writer
from src.graph.state import initial_state


def _load_findings(topic_id: str) -> list[dict]:
    p = RESULTS / "baseline" / topic_id / "run.json"
    if not p.exists():
        raise SystemExit(f"No baseline findings at {p}. Run Phase 0 for {topic_id} first.")
    return json.loads(p.read_text(encoding="utf-8"))["findings"]


def _show(upd: dict) -> dict:
    ev = (upd.get("trace") or [{}])[0]
    detail = " ".join(f"{k}={v}" for k, v in ev.items() if k not in ("node", "action"))
    print(f"  writer   {ev.get('action')}  {detail}")
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic-id", default="t1")
    ap.add_argument("--revise", action="store_true", help="also exercise the revision path")
    ap.add_argument("--issues", type=int, default=2, metavar="N",
                    help="how many synthetic criticisms to seed (default 2). One edit "
                         "matching proves little — a real Critic returns several at once "
                         "and they all have to match verbatim")
    ap.add_argument("--show-draft", action="store_true")
    args = ap.parse_args()

    topic = get_topic(args.topic_id)
    findings = _load_findings(args.topic_id)

    state = initial_state(topic["topic"])
    state["findings"] = findings

    print(f"topic     {topic['id']} ({topic['shape']})")
    print(f"findings  {len(findings)}\n")

    print("planning...")
    state.update({k: v for k, v in analyst(state).items() if k != "trace"})
    if not state.get("outline"):
        raise SystemExit("analyst produced no outline — fix Phase 3 first")

    print("writing...")
    upd = writer(state)
    _show(upd)
    draft = upd["draft"]
    state["draft"] = draft

    fobjs = [Finding(**f) for f in findings]
    c = check_citations(draft, fobjs)
    print(f"\n  words          {len(draft.split())}")
    print(f"  cited / broken {c['cited_count']} / {c['broken_count']}")
    print(f"  coverage       {c['coverage']:.0%} of findings cited")
    if c["broken"]:
        print(f"  !! broken      {', '.join(c['broken'])}")

    if args.show_draft:
        print("\n--- draft ---\n")
        print(draft)

    if args.revise:
        # Synthetic criticisms with spans taken verbatim from the draft, which is what
        # the real Critic is supposed to produce in Phase 5. Several at once, from
        # paragraphs spread through the document: one edit matching proves the mechanism
        # works, and a real revision has to land every edit in a batch.
        paras = [p for p in draft.split("\n\n")
                 if len(p.split()) > 25 and not p.lstrip().startswith("#")]
        if not paras:
            print("\n(draft too short to build a synthetic criticism)")
            return

        step = max(1, len(paras) // max(1, args.issues))
        chosen = paras[::step][:args.issues]

        template = [
            ("structural_coherence",
             ("This passage states its point but does not connect it to what follows, so "
              "the section reads as a list rather than an argument."),
             "Add the causal link forward to the next section."),
            ("depth_of_analysis",
             "This passage reports the evidence without saying what follows from it.",
             "State the implication the cited findings support."),
            ("absence_of_filler",
             "This passage restates what an earlier section already established.",
             "Cut the restatement and keep only what is new here."),
        ]

        issues = []
        for i, para in enumerate(chosen):
            criterion, problem, fix = template[i % len(template)]
            issues.append({
                "span": " ".join(para.split()[:14]),
                "criterion": criterion,
                "problem": problem,
                "fix": fix,
            })

        state["critique"] = {
            "scores": {"structural_coherence": 2},
            "verdict": "revise",
            "target": "writer",
            "issues": issues,
        }
        state["revision_count"] = 1

        print(f"\nrevising against {len(issues)} criticism(s):")
        for i in issues:
            print(f"  [{i['criterion']}] {i['span'][:64]}...")
        upd2 = writer(state)
        ev = _show(upd2)
        after = upd2["draft"]

        c2 = check_citations(after, fobjs)
        print(f"\n  words          {len(draft.split())} -> {len(after.split())}")
        print(f"  broken cites   {c['broken_count']} -> {c2['broken_count']}")

        print("\nchecks:")
        ok_applied = ev.get("edits_applied", 0) == ev.get("edits_returned", 0)
        ok_surgical = ev.get("changed_pct", 100) < 25
        ok_nofall = not ev.get("fallback", True)
        print(f"  all edits applied     {'PASS' if ok_applied else 'FAIL'} "
              f"({ev.get('edits_applied')}/{ev.get('edits_returned')})")
        print(f"  surgical (<25% moved) {'PASS' if ok_surgical else 'FAIL'} "
              f"({ev.get('changed_pct')}%)")
        print(f"  no rewrite fallback   {'PASS' if ok_nofall else 'FAIL'}")
        print(f"  no new broken cites   "
              f"{'PASS' if c2['broken_count'] <= c['broken_count'] else 'FAIL'}")

        if not (ok_applied and ok_nofall):
            print("\n  The model is not copying `find` spans verbatim. Before Phase 5:")
            print("    1. Strengthen 'copied EXACTLY, character for character' in _revise")
            print("    2. Ask for longer `find` spans — whole sentences, not phrases")
            print("    3. Drop revision temperature to 0.2")


if __name__ == "__main__":
    main()

