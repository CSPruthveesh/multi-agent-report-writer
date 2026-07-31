"""Critic node — Phase 5.

    Reads:  draft, findings, outline
    Writes: critique
    Tools:  none

THE DIVISION OF LABOUR HERE IS THE POINT
----------------------------------------
The model does what needs judgement: reading the prose and scoring it against the
rubric. The code does what is a rule: deciding the verdict, routing the revision,
and checking citations. Letting the model decide the verdict produces the classic
inconsistency where it scores three criteria at 2 and then returns "pass" anyway.

Deterministic in code:
  - verdict     revise if any criterion <= 3, else pass  (rubric.md)
  - target      derived from the lowest-scoring criterion
  - criterion 4 capped at 1 if any cited ID does not exist  (rubric.md hard rule)
  - issues      any issue whose span is not in the draft is DISCARDED

THE REFLEXION LESSON, REAPPLIED
-------------------------------
An ungrounded critic writes plausible-sounding criticism, the Writer dutifully acts
on it, and the draft gets worse while the scores go up. Requiring a verbatim span
for every issue is the fix, and it has to be enforced in code — a prompt asking for
grounding is a request, a substring check is a guarantee.

Phase 4 learned this the expensive way from the other side: the stub Critic carried
a hardcoded span no real draft contained, the Writer discarded every issue, and each
forced revision silently became a full rewrite. A Critic that paraphrases instead of
quoting breaks the surgical revision path and nothing warns you.

TWO OPPOSITE FAILURE MODES, BOTH INVISIBLE IN ONE RUN
-----------------------------------------------------
  Always passes   The model is agreeable. The revision loop never fires and the
                  architecture is researcher -> analyst -> writer with extra steps.
  Always revises   Asked to find problems, it finds problems. Every run burns the
                  full revision budget for nothing.

Neither is visible from a single trace — both look like the system working. See
scripts/test_critic.py, which scores a good draft against deliberately degraded
copies and checks the scores move on the RIGHT criterion.

SELF-SCORING BIAS
-----------------
By default this is the same model that wrote the draft, and it will be lenient about
its own work. If the diagnostic shows it always passing, set GEMINI_CRITIC_MODEL to
something stronger — the hook is already in common/llm.py. A weak critic is worse
than no critic, because it manufactures the appearance of quality control.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from src.common.llm import CRITIC_MODEL, TokenLedger, generate
from src.graph.state import ReportState, log_entries, trace_event

log = logging.getLogger(__name__)

NODE = "critic"

CITE_RE = re.compile(r"\bF\d{3}\b")

# Which node can actually fix a failure of each criterion. Sending a structural
# complaint to the Writer produces cosmetic edits; the Analyst owns structure.
CRITERION_OWNER = {
    "factual_grounding": "researcher",
    "structural_coherence": "analyst",
    "depth_of_analysis": "analyst",
    "citation_integrity": "writer",
    "absence_of_filler": "writer",
}

# Revise if any criterion is at or below this. Both values have now been measured
# across all six topics and neither is right.
#
#   3   the loop never fires. Thirty criterion scores came back as twenty-two 5s
#       and eight 4s, so nothing ever tripped it. Cost +26% against the baseline
#       for a Writer/Critic loop that is idle in production.
#   4   the loop never stops. A 4 is a good score and there is always one, so
#       revise is permanent: every topic spent both revisions and none reached
#       pass. Cost +70%, t1 alone going 25,048 -> 66,139, for zero resolved topics.
#
# Back to 3, because paying 26% for an idle loop beats paying 70% for one that
# settles nothing. This is not the fix — it is the cheaper of two known-wrong
# settings, recorded as such.
#
# The real problem is that a single threshold on the minimum score cannot express
# "this draft is good but improvable". Options, none tried: revise only when a
# criterion is at 4 AND the Critic raised a grounded issue for that specific
# criterion; or allow one revision at 4 and require 3 for a second; or drop the
# threshold idea and let the Critic ask for a revision explicitly, with the code
# still deciding the target and the cap.
#
# ISSUE_RULES below has to agree with this number. See the note there.
PASS_THRESHOLD = 3

PERSONA = """You are a demanding editor reviewing a research report before publication. You
are not the author and you owe them nothing. You have the evidence the report was written
from, so you can check whether its claims are actually supported.

Score honestly. A report that reads well but asserts things the evidence does not establish
is a bad report, and fluent confident prose is exactly what you are here to catch."""

RUBRIC = """
Score each criterion 1-5. Integers only. Score them INDEPENDENTLY — a report can be
beautifully structured and factually ungrounded.

1. factual_grounding — do substantive claims trace to the evidence provided?
   1: claims float free of the evidence.  5: every substantive claim traces to a finding,
   and where evidence is absent the report says so instead of asserting anyway.
   Do not reward confident tone. Fluent unsourced assertion is the failure this catches.

2. structural_coherence — does it build an argument, or is it a pile of sections?
   1: sections repeat, contradict, or could be shuffled without loss.  5: later sections
   use what earlier ones established; the conclusion follows from the body.

3. depth_of_analysis — does it synthesise or summarise?
   1: restates sources one after another.  5: connects findings, surfaces tensions between
   sources rather than averaging them away, draws conclusions the sources do not state
   individually. A report that smooths a real disagreement into consensus is FAILING this.

4. citation_integrity — do citations resolve and support what they are attached to?
   1: IDs missing, invented, or pointing at unrelated findings.  5: every citation supports
   the exact sentence it sits on, and every claim needing one has one.
   Absence is not perfection. A report that cites nothing has not avoided citation
   errors, it has failed to support any claim at all — that scores 1, not 5. Judge how
   much of what needs support has it, not whether the citations present are correct.

5. absence_of_filler — does every paragraph carry information?
   1: throat-clearing, restated headers, generic hedging, a conclusion that only summarises.
   5: nothing could be cut without losing content.
   A hedge tied to specific missing evidence is NOT filler — that is criterion 1 working.

When a report sits between two anchors, round DOWN. Optimistic scoring hides the
differences this review exists to find."""

# The "4 or below" line below must match PASS_THRESHOLD. When they disagree the
# failure is silent: the verdict computes to revise, no issues were raised because the
# prompt forbade it, and the rule further down converts the verdict back to pass. The
# loop stays dead and nothing in the trace says why.
#
# It was already disagreeing — the threshold was 3 and this said "3 or below", and the
# loop only looked one change away from working because the model was raising issues
# for 4s in defiance of the instruction. Relying on an instruction not being followed
# is not a mechanism.
ISSUE_RULES = """
Issue rules — this is the part that matters:

- Every issue MUST quote `span` VERBATIM from the draft, copied character for character.
  Issues whose span does not appear in the draft are discarded automatically and your
  criticism is lost.
- Make the span long enough to be unique — a full sentence, not a phrase.
- One issue per real problem. Do not pad the list.
- Only raise issues for criteria you scored 3 or below. If everything scored 4+, return
  an empty list — that is a valid and complete response.
- `fix` must be actionable and achievable with the evidence available. Do not ask for
  claims the evidence cannot support."""


class Scores(BaseModel):
    factual_grounding: int = Field(ge=1, le=5)
    structural_coherence: int = Field(ge=1, le=5)
    depth_of_analysis: int = Field(ge=1, le=5)
    citation_integrity: int = Field(ge=1, le=5)
    absence_of_filler: int = Field(ge=1, le=5)


class RawIssue(BaseModel):
    span: str = Field(description="Verbatim text from the draft. A full sentence.")
    criterion: str = Field(description="Which criterion this violates.")
    problem: str = Field(description="What is wrong. Specific.")
    fix: str = Field(description="What would fix it. Actionable.")


class RawCritique(BaseModel):
    scores: Scores
    issues: list[RawIssue] = Field(default_factory=list)
    summary: str = Field(description="One sentence on the report's main weakness.")


def _format_findings(findings: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{f['id']}] ({f['confidence']}) {f['claim']}" for f in findings)


def _cite_span(draft: str, broken: list[str]) -> str:
    """A span quoted verbatim from the draft, for a citation issue built in code.

    With broken IDs there is an obvious place to point: the sentence carrying the
    first bad one. With no citations at all there is nowhere obvious, so it points at
    the first substantive sentence — enough for the Writer to locate the problem,
    since the fix applies to the whole document anyway.
    """
    m = CITE_RE.search(draft) if broken else None
    if m:
        start = max(0, draft.rfind(".", 0, m.start()) + 1)
        end = draft.find(".", m.end())
        span = draft[start : end + 1 if end != -1 else len(draft)].strip()
        return span if span and span in draft else ""

    for para in draft.split("\n\n"):
        stripped = para.strip()
        if len(stripped.split()) > 20 and not stripped.startswith("#"):
            end = stripped.find(".")
            span = stripped[: end + 1] if end != -1 else stripped[:120]
            return span if span in draft else ""
    return ""


def critic(state: ReportState) -> dict[str, Any]:
    ledger = TokenLedger()
    draft = state.get("draft") or ""
    findings = list(state.get("findings") or [])
    outline = state.get("outline") or ""

    if not draft.strip():
        return {
            "critique": None,
            "trace": [trace_event(NODE, "skipped", why="no draft to review")],
        }

    valid_ids = {f["id"] for f in findings}
    cited = set(CITE_RE.findall(draft))
    broken = sorted(cited - valid_ids)

    resp = generate(
        f"Evidence the report was written from:\n{_format_findings(findings)}\n\n"
        f"Outline it was supposed to follow:\n{outline}\n\n"
        f"Report:\n---\n{draft}\n---\n\n"
        f"Review it.{RUBRIC}{ISSUE_RULES}",
        node=NODE,
        call_type="critique",
        ledger=ledger,
        system=PERSONA,
        schema=RawCritique,
        temperature=0.2,
        model=CRITIC_MODEL,
    )

    parsed: RawCritique | None = getattr(resp, "parsed", None)
    if parsed is None:
        # Do not block the run on a parse failure. Pass it through and let the
        # deterministic citation check stand as the only quality gate — recorded
        # honestly in the trace rather than silently.
        log.warning("[%s] critique did not parse; passing draft through", NODE)
        return {
            "critique": {
                "scores": {},
                "verdict": "pass",
                "target": "writer",
                "issues": [],
                "parse_failed": True,
            },
            "token_log": log_entries(ledger, state),
            "trace": [trace_event(NODE, "parse_failed", why="structured output unusable")],
        }

    scores = parsed.scores.model_dump()

    # Hard rule from rubric.md: an invented citation caps criterion 4 at 1,
    # regardless of how good the rest of the citation work is. Fabricated citations
    # are what make a research tool unusable. Enforced here, not asked for in the
    # prompt, because a model reviewing its own citations is not a reliable check.
    capped = False
    cap_reason = ""
    if broken and scores.get("citation_integrity", 5) > 1:
        scores["citation_integrity"] = 1
        capped, cap_reason = True, f"invented ids: {', '.join(broken)}"
    elif findings and not cited and scores.get("citation_integrity", 5) > 1:
        # Measured: a draft with every citation stripped out scored 5 on this
        # criterion. The model reads "do citations resolve and support what they are
        # attached to" as vacuously true when there are none — nothing is wrong
        # because nothing is claimed. That is the worst possible direction for this
        # project, because an uncited report is the exact failure the whole design
        # exists to catch, and the deterministic checker cannot see it either: it
        # counts broken IDs and finds zero.
        scores["citation_integrity"] = 1
        capped, cap_reason = True, "no citations at all"

    # Drop ungrounded issues. This is the Reflexion lesson enforced in code.
    grounded = []
    dropped = 0
    for i in parsed.issues:
        if i.span and i.span in draft:
            grounded.append(
                {
                    "span": i.span,
                    "criterion": i.criterion,
                    "problem": i.problem,
                    "fix": i.fix,
                }
            )
        else:
            dropped += 1
            log.debug("[%s] dropped ungrounded issue: %r", NODE, (i.span or "")[:60])

    # If the cap fired but the model raised no issue about citations, add one. The
    # defect is mechanically verified, so the Writer must not depend on the model
    # having remembered to mention it — and without a grounded issue the revise
    # verdict below is converted back to a pass and the defect ships.
    if capped and not any(i["criterion"] == "citation_integrity" for i in grounded):
        span = _cite_span(draft, broken)
        if span:
            grounded.append(
                {
                    "span": span,
                    "criterion": "citation_integrity",
                    "problem": (
                        f"Cites IDs that do not exist: {', '.join(broken)}"
                        if broken
                        else "The report cites no evidence at all, so nothing it asserts "
                             "is supported by the findings it was given."
                    ),
                    "fix": (
                        "Replace with a valid ID that supports the claim, or remove the "
                        "citation and soften the claim."
                        if broken
                        else "Cite the finding IDs that support each substantive claim, "
                             "and soften any claim no finding supports."
                    ),
                }
            )

    # Verdict is computed, never taken from the model. A model that scores three
    # criteria at 2 and then says "pass" is the standard inconsistency here.
    worst = min(scores.values()) if scores else 5
    verdict = "revise" if worst <= PASS_THRESHOLD else "pass"

    # Route to whoever can actually fix the lowest-scoring criterion. Ties break
    # toward the earlier node in the pipeline — an evidence problem is upstream of a
    # prose problem, and fixing prose over a bad evidence base is wasted effort.
    order = ["factual_grounding", "structural_coherence", "depth_of_analysis",
             "citation_integrity", "absence_of_filler"]
    lowest = min(order, key=lambda c: (scores.get(c, 5), order.index(c)))
    target = CRITERION_OWNER.get(lowest, "writer")

    # Nothing to act on means nothing to route. Prevents a revision loop that
    # sends the Writer an empty issue list and gets an identical draft back.
    if verdict == "revise" and not grounded:
        verdict = "pass"

    critique = {
        "scores": scores,
        "verdict": verdict,
        "target": target,
        "issues": grounded,
        "summary": parsed.summary,
        "broken_citations": broken,
        "citation_cap_applied": capped,
        "citation_cap_reason": cap_reason,
    }

    return {
        "critique": critique,
        "token_log": log_entries(ledger, state),
        "trace": [
            trace_event(
                NODE,
                "scored",
                verdict=verdict,
                target=target if verdict == "revise" else "-",
                worst=f"{lowest}={worst}",
                mean=round(sum(scores.values()) / len(scores), 2) if scores else None,
                issues_raw=len(parsed.issues),
                issues_kept=len(grounded),
                dropped_ungrounded=dropped,
                broken_cites=len(broken),
                cites=len(cited),
                cite_cap=cap_reason or False,
                tokens=ledger.total,
            )
        ],
    }
