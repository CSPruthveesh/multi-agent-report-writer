"""Analyst node — Phase 3.

    Reads:  findings, critique (on a structural revision), unclosed_gaps
    Writes: outline, gaps
    Tools:  none, ever

No tools is a structural decision, not an oversight. An Analyst with search access
turns into a second Researcher, and then you have two nodes gathering evidence and
nobody deciding whether it is sufficient. The whole value here comes from the
separation: one node gathers, a different node judges whether what was gathered can
carry a report.

THE FAILURE MODE THIS FILE IS BUILT AROUND
------------------------------------------
Ask a language model "what is missing?" and it will always find something. Always.
An Analyst that reports gaps on every run is not detecting gaps — it is generating
them, and the research loop fires every time, burns the search budget on topics
that never needed it, and inflates your token multiple for no quality gain.

Three defences, all of them in the prompt and the post-filter below:

  1. A gap must be LOAD-BEARING — it has to block a section of the outline the
     report actually needs. "More recent data would be nice" is not a gap.
  2. A hard cap of 2. Forcing a ranking makes the model choose rather than list.
  3. An explicit instruction that reporting zero gaps is the expected outcome on
     well-covered topics, plus a required `blocks` field naming the section each
     gap blocks. Making the model name the blocked section is what stops it
     inventing gaps — it has to point at real damage.

The diagnostic in scripts/test_analyst.py checks discrimination directly: run it on
t1 (abundant sources) and t3 (thin evidence). If the gap counts come out similar,
the Analyst is not discriminating and the prompt needs work — not the architecture.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.common.llm import TokenLedger, generate
from src.graph.state import ReportState, log_entries, trace_event

log = logging.getLogger(__name__)

NODE = "analyst"

MAX_GAPS = 2

PERSONA = """You are a research analyst. You are given evidence someone else gathered, and
you decide what report it can support. You do not search, you do not write prose, and you
do not pad. You are the person who says "the evidence does not support that claim" before
it reaches a reader."""


class Gap(BaseModel):
    missing: str = Field(
        description="The specific evidence that is absent. Name what would close it."
    )
    blocks: str = Field(
        description="Which outline section cannot be written without it. Must be a section "
        "you actually listed."
    )


class Analysis(BaseModel):
    sections: list[str] = Field(
        description="3-5 section headers, each followed by the finding IDs that support it, "
        "e.g. 'Cost trajectory [F002, F007, F011]'. Every section needs at least one ID."
    )
    thesis: str = Field(
        description="One sentence the report argues. Not a topic restatement — a claim."
    )
    tensions: list[str] = Field(
        default_factory=list,
        description="Places the findings disagree with each other. Empty if they do not. "
        "Do not manufacture disagreement.",
    )
    gaps: list[Gap] = Field(
        default_factory=list,
        description=f"At most {MAX_GAPS}. Only evidence absences that BLOCK a listed section. "
        "Empty is the correct answer when the evidence covers the topic.",
    )


GAP_DISCIPLINE = f"""
Gap rules — read these carefully, they are the part people get wrong.

Two tests that REQUIRE a gap. Apply both to every section you listed:

- Thin support. The section rests on fewer than three findings, or on nothing above
  medium confidence. Evidence that thin cannot carry a section. Say which measurement,
  figure or source would make it stand up.
- Nothing measured. The section's findings say what exists but never say how much, how
  many, how fast or how well — no figures, dates, rates or named outcomes. A long list
  of confident claims that a thing exists is still no evidence of how it performs, and
  a section cannot analyse what nobody has measured. Say what number is missing.
- Unresolved conflict. Two findings give different values for the same quantity and
  nothing in the evidence settles which is right. That is missing evidence, not a
  tension. A tension is a real disagreement in the world worth showing the reader; a
  conflict you cannot resolve is a hole in what was gathered. Say what would settle it.

If either test fires on a section, you must report it, up to the cap.

Then the limits:

- A gap is evidence you NEED and do not HAVE. It is not evidence you would like more of.
- Every gap must name the outline section it blocks. If you cannot name a blocked
  section, it is not a gap — drop it.
- Maximum {MAX_GAPS}. If you have more candidates, keep only the ones that do the most
  damage to the report.
- Reporting ZERO gaps is correct when neither test above fires anywhere. An empty list
  is a valid, complete response. Do not invent a gap to seem thorough — but do not
  report zero merely because the finding list is long. A long list of weakly supported
  claims is exactly what the first test is for.
- Never report a gap you already reported and that came back unclosed. It has been
  searched for and not found; saying so again wastes the budget."""


def _format_findings(findings: list[dict[str, Any]]) -> str:
    lines = []
    for f in findings:
        src = f.get("source_url") or "no source"
        lines.append(f"[{f['id']}] ({f['confidence']}) {f['claim']}  — {src}")
    return "\n".join(lines)


def _render_outline(a: Analysis) -> str:
    parts = [f"THESIS: {a.thesis}", ""]
    for s in a.sections:
        parts.append(f"## {s}")
    if a.tensions:
        parts.append("")
        parts.append("TENSIONS TO PRESERVE (do not resolve these into false consensus):")
        for t in a.tensions:
            parts.append(f"- {t}")
    return "\n".join(parts)


def analyst(state: ReportState) -> dict[str, Any]:
    ledger = TokenLedger()
    findings = list(state.get("findings") or [])
    # Not `critique`: the supervisor must clear that when it routes here, or the stale
    # verdict routes upstream again on the return trip. The criticism arrives on its
    # own field instead. See the comment in supervisor.py rule 7.
    brief = state.get("revision_brief")
    already_unclosed = list(state.get("unclosed_gaps") or [])
    loops = state.get("research_loops", 0)

    if not findings:
        # No evidence at all — the Researcher failed or its budget was spent before
        # it found anything. Do not invent an outline over nothing.
        return {
            "outline": None,
            "gaps": [],
            "trace": [trace_event(NODE, "skipped", why="no findings to analyse")],
        }

    evidence = _format_findings(findings)

    # Structural revision: the Critic routed here rather than to the Writer, meaning
    # the problem is the plan, not the prose. Feed it the criticism, not the draft —
    # the Analyst does not need to see prose to fix a structure.
    revision_note = ""
    if brief and brief.get("target") == "analyst":
        issues = brief.get("issues") or []
        listed = "\n".join(
            f"- {i.get('problem', '')} → {i.get('fix', '')}"
            for i in issues
            if i.get("problem")
        )
        if listed:
            revision_note = (
                f"\n\nA critic rejected the structure built from this evidence:\n{listed}\n"
                f"Rebuild the outline to address this. Do not simply reorder the same sections."
            )

    unclosed_note = ""
    if already_unclosed:
        listed = "\n".join(f"- {g}" for g in already_unclosed)
        unclosed_note = (
            f"\n\nThese gaps were already searched for and NOT found. The search budget for "
            f"them is spent. Do not report them again — plan a report that works without "
            f"them and lets the reader know what is unestablished:\n{listed}"
        )

    budget_note = ""
    if loops > 0:
        budget_note = (
            f"\n\nNote: {loops} follow-up research round(s) have already run. Report a new gap "
            f"only if it genuinely blocks a section."
        )

    resp = generate(
        f"Evidence gathered:\n{evidence}{revision_note}{unclosed_note}{budget_note}\n\n"
        f"Plan the report this evidence can support.{GAP_DISCIPLINE}",
        node=NODE,
        call_type="analyze",
        ledger=ledger,
        system=PERSONA,
        schema=Analysis,
        temperature=0.3,
    )

    parsed: Analysis | None = getattr(resp, "parsed", None)
    if parsed is None:
        log.warning("[%s] analysis did not parse; continuing with no outline", NODE)
        return {
            "outline": None,
            "gaps": [],
            "token_log": log_entries(ledger, state),
            "trace": [trace_event(NODE, "parse_failed", why="structured output unusable")],
        }

    # Post-filter the gaps. The prompt asks for discipline; this enforces it, because
    # a prompt instruction is a request and a filter is a guarantee.
    section_text = " ".join(parsed.sections).lower()
    kept: list[str] = []
    dropped_unblocking = 0
    dropped_repeat = 0
    prior = {g.lower()[:60] for g in already_unclosed}

    for g in parsed.gaps:
        # Must reference a section that exists. Cheap heuristic: some content word
        # from the named section has to appear in the outline.
        words = [w for w in g.blocks.lower().split() if len(w) > 4]
        if words and not any(w in section_text for w in words):
            dropped_unblocking += 1
            continue
        if g.missing.lower()[:60] in prior:
            dropped_repeat += 1
            continue
        kept.append(g.missing)

    kept = kept[:MAX_GAPS]

    return {
        "outline": _render_outline(parsed),
        "gaps": kept,
        "token_log": log_entries(ledger, state),
        "trace": [
            trace_event(
                NODE,
                "outlined",
                sections=len(parsed.sections),
                tensions=len(parsed.tensions),
                gaps_raw=len(parsed.gaps),
                gaps_kept=len(kept),
                dropped_unblocking=dropped_unblocking,
                dropped_repeat=dropped_repeat,
                revision=bool(revision_note),
                tokens=ledger.total,
            )
        ],
    }
