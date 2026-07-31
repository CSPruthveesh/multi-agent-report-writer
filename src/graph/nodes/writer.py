"""Writer node — Phase 4.

    Reads:  outline, findings, draft, critique
    Writes: draft
    Tools:  none
    Calls:  exactly one per invocation

ONE NODE, ONE CALL, THE WHOLE REPORT
------------------------------------
No parallel section writers. This is the single most tempting "optimisation" in a
multi-agent writing system and it is where the architecture loses. Parallel writers
produce sections that repeat each other and do not transition, because no call ever
sees the whole document. The fix — a stitching pass over the full text — costs more
than the parallelism saved on a document this size. You will be asked why you did
not parallelise; the answer is coherence, and it is measurable in criterion 2.

SURGICAL REVISION, NOT REWRITING
--------------------------------
The revision path is where a naive implementation burns its budget. Ask a model to
"revise this report addressing the following issues" and it regenerates all 1000
words, which:

  - costs a full write every loop
  - regresses passages that were already fine
  - makes it impossible to say what the revision actually changed

Instead the Writer returns EDITS — {find, replace} pairs — applied in code. That
gives three things a rewrite cannot:

  1. Cost. Output tokens scale with the size of the problem, not the report.
  2. Containment. Untouched text is byte-identical, so a revision cannot regress
     what it was not asked to fix.
  3. Measurement. `changed_pct` is computed exactly, which is what lets Phase 9
     say "the second revision loop cost 22% of the run and changed 4% of the text."
     That sentence is the strongest thing in the whole writeup, and you can only
     write it if revisions are surgical.

Fallback: if fewer than half the edits apply cleanly, fall back to a full rewrite
and record it in the trace. Do not fail the node — a degraded revision beats none.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from src.common.llm import TokenLedger, generate
from src.graph.state import ReportState, log_entries, trace_event

log = logging.getLogger(__name__)

NODE = "writer"

TARGET_WORDS = (800, 1200)
CITE_RE = re.compile(r"\bF\d{3}\b")

PERSONA = """You are a writer producing a briefing for a smart reader who does not know this
topic. Someone else gathered the evidence and someone else planned the structure. Your job is
the prose, and you are accountable for the report reading as one coherent piece rather than a
set of assembled parts."""

RULES = f"""
Requirements:
- {TARGET_WORDS[0]}-{TARGET_WORDS[1]} words.
- Cite finding IDs inline on the sentence they support: [F003] or [F003, F012].
- ONLY cite IDs that appear in the evidence list. Never invent one. A fabricated
  citation is worse than an uncited claim.
- Follow the outline's sections and thesis. If the outline lists tensions, preserve them —
  do not resolve a real disagreement into a smooth consensus.
- Build an argument. Later sections should use what earlier ones established. Do not
  restate the introduction as a conclusion.
- Where evidence is thin, say so plainly rather than writing around it with confident
  prose. Fluent unsourced assertion is the specific failure being graded here.
- No filler. No "in today's rapidly evolving landscape". No throat-clearing. Every
  paragraph carries information.
- Markdown. One H1 title, then the sections. No preamble before the title."""


class Edit(BaseModel):
    find: str = Field(
        description="Verbatim text from the draft to replace. Long enough to be unique — "
        "include surrounding words if a short phrase would match in several places."
    )
    replace: str = Field(description="Replacement text. May be longer or shorter.")
    why: str = Field(description="Which criticism this addresses. One short phrase.")


class Revision(BaseModel):
    edits: list[Edit] = Field(
        description="Targeted edits. Change only what the criticism identified — leave "
        "everything else untouched."
    )


def _format_findings(findings: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{f['id']}] ({f['confidence']}) {f['claim']}" for f in findings
    )


def _valid_ids(findings: list[dict[str, Any]]) -> set[str]:
    return {f["id"] for f in findings}


def _broken_cites(text: str, valid: set[str]) -> list[str]:
    return sorted(set(CITE_RE.findall(text)) - valid)


def _changed_pct(before: str, after: str) -> float:
    """Rough proportion of the document that changed, by word-level difference."""
    a, b = before.split(), after.split()
    if not a:
        return 100.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return round((1 - sm.ratio()) * 100, 1)


# ---------------------------------------------------------------- cold draft
def _write_fresh(
    topic: str,
    outline: str,
    findings: list[dict[str, Any]],
    unclosed: list[str],
    ledger: TokenLedger,
) -> str:
    gap_note = ""
    if unclosed:
        listed = "\n".join(f"- {g}" for g in unclosed)
        gap_note = (
            f"\n\nEvidence that was searched for and not found. Acknowledge these in the "
            f"report where relevant rather than writing around them — a reader is better "
            f"served knowing what is unestablished:\n{listed}"
        )

    resp = generate(
        f"Topic:\n{topic}\n\nOutline to follow:\n{outline}\n\n"
        f"Evidence you may cite:\n{_format_findings(findings)}{gap_note}\n\n"
        f"Write the report.{RULES}",
        node=NODE,
        call_type="write",
        ledger=ledger,
        system=PERSONA,
        temperature=0.6,
    )
    return (resp.text or "").strip()


# ------------------------------------------------------------ surgical revise
def _revise(
    draft: str,
    issues: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    ledger: TokenLedger,
) -> tuple[str, dict[str, Any]]:
    listed = "\n\n".join(
        f"PROBLEM: {i.get('problem', '')}\n"
        f"IN THIS TEXT: {i.get('span', '')}\n"
        f"SUGGESTED FIX: {i.get('fix', '')}"
        for i in issues
    )

    resp = generate(
        f"Current draft:\n---\n{draft}\n---\n\n"
        f"A critic raised these problems:\n\n{listed}\n\n"
        f"Evidence you may cite (never invent an ID):\n{_format_findings(findings)}\n\n"
        f"Return targeted edits that fix these problems. Rules:\n"
        f"- `find` must be copied EXACTLY from the draft, character for character.\n"
        f"- Make it long enough to appear only once. If a phrase repeats, include the\n"
        f"  sentence around it.\n"
        f"- Change only what the criticism identified. Do not rewrite passages that were\n"
        f"  not criticised, and do not make stylistic edits nobody asked for.\n"
        f"- If a criticism cannot be fixed with the evidence available, skip it rather than\n"
        f"  inventing support.",
        node=NODE,
        call_type="revise",
        ledger=ledger,
        system=PERSONA,
        schema=Revision,
        temperature=0.4,
    )

    parsed: Revision | None = getattr(resp, "parsed", None)
    if parsed is None or not parsed.edits:
        return draft, {"edits_returned": 0, "edits_applied": 0, "fallback": False}

    out = draft
    applied = 0
    for e in parsed.edits:
        if e.find and e.find in out:
            out = out.replace(e.find, e.replace, 1)
            applied += 1
        else:
            log.debug("[%s] edit did not match: %r", NODE, (e.find or "")[:60])

    stats = {
        "edits_returned": len(parsed.edits),
        "edits_applied": applied,
        "fallback": False,
    }

    # If most edits missed, the model was not copying verbatim. Fall back to a full
    # rewrite rather than shipping a draft that ignored the criticism.
    if applied < len(parsed.edits) / 2:
        stats["fallback"] = True
        rewrite = generate(
            f"Current draft:\n---\n{draft}\n---\n\n"
            f"Problems to fix:\n\n{listed}\n\n"
            f"Evidence:\n{_format_findings(findings)}\n\n"
            f"Rewrite the report addressing these problems. Preserve everything that was "
            f"not criticised.{RULES}",
            node=NODE,
            call_type="rewrite",
            ledger=ledger,
            system=PERSONA,
            temperature=0.5,
        )
        out = (rewrite.text or "").strip() or draft

    return out, stats


# ------------------------------------------------------------------- the node
def writer(state: ReportState) -> dict[str, Any]:
    ledger = TokenLedger()
    topic = state["topic"]
    outline = state.get("outline")
    findings = list(state.get("findings") or [])
    draft = state.get("draft")
    crit = state.get("critique") or {}
    unclosed = list(state.get("unclosed_gaps") or [])
    rev = state.get("revision_count", 0)

    if not findings:
        return {
            "draft": "",
            "trace": [trace_event(NODE, "skipped", why="no findings to write from")],
        }

    if not outline:
        return {
            "draft": "",
            "trace": [trace_event(NODE, "skipped", why="no outline to follow")],
        }

    valid = _valid_ids(findings)

    # Only spans that actually exist in the draft. The Critic filters these too, but
    # the Writer must not trust an upstream node to have done it — an issue with a
    # hallucinated span produces an edit that cannot apply, and burns a loop.
    issues = [
        i for i in (crit.get("issues") or [])
        if i.get("span") and draft and i["span"] in draft
    ]

    revising = bool(draft and issues)

    if revising:
        before = draft
        new_draft, stats = _revise(draft, issues, findings, ledger)
        changed = _changed_pct(before, new_draft)
        action = "revised"
        extra = {**stats, "changed_pct": changed, "issues_addressed": len(issues)}
    else:
        new_draft = _write_fresh(topic, outline, findings, unclosed, ledger)
        action = "drafted"
        extra = {}

    # Citation check. One repair attempt, then let it through — the Critic and the
    # deterministic checker will both catch what remains, and the broken count is a
    # number worth reporting honestly rather than papering over.
    broken = _broken_cites(new_draft, valid)
    repaired = False
    if broken:
        listed = ", ".join(broken)
        fix = generate(
            f"This report cites IDs that do not exist: {listed}\n\n"
            f"Valid IDs and what they say:\n{_format_findings(findings)}\n\n"
            f"Report:\n---\n{new_draft}\n---\n\n"
            f"Return the corrected report. For each invalid citation, either replace it with "
            f"a valid ID that genuinely supports the sentence, or remove the citation and "
            f"soften the claim. Do not change anything else.",
            node=NODE,
            call_type="cite_fix",
            ledger=ledger,
            system=PERSONA,
            temperature=0.2,
        )
        candidate = (fix.text or "").strip()
        if candidate:
            still = _broken_cites(candidate, valid)
            if len(still) < len(broken):
                new_draft = candidate
                repaired = True
                broken = still

    words = len(new_draft.split())
    return {
        "draft": new_draft,
        "token_log": log_entries(ledger, state),
        "trace": [
            trace_event(
                NODE,
                action,
                revision=rev,
                words=words,
                in_range=TARGET_WORDS[0] <= words <= TARGET_WORDS[1],
                cited=len(set(CITE_RE.findall(new_draft))),
                broken_cites=len(broken),
                cite_repair=repaired,
                tokens=ledger.total,
                **extra,
            )
        ],
    }
