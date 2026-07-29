from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from src.common.io import save_run
from src.common.llm import TokenLedger, generate
from src.common.schemas import Finding
from src.common.search import research

log = logging.getLogger(__name__)

MAX_SEARCH_ROUNDS = 5

PERSONA = """You are an experienced research analyst writing a briefing for a smart reader
who does not know this topic. You gather evidence, judge whether it is sufficient, and
write the report yourself. You are accountable for the whole thing end to end."""


class QueryPlan(BaseModel):
    queries: list[str] = Field(description="2-3 search queries, specific and non-overlapping")
    reasoning: str = Field(description="One sentence: why these queries")


class Sufficiency(BaseModel):
    sufficient: bool = Field(description="True if the evidence can support a full report")
    missing: list[str] = Field(
        default_factory=list, description="What is still missing. Empty if sufficient."
    )
    next_queries: list[str] = Field(
        default_factory=list, description="1-2 queries targeting the gaps. Empty if sufficient."
    )


def _plan(topic: str, ledger: TokenLedger) -> list[str]:
    resp = generate(
        f"Topic:\n{topic}\n\nPlan your opening searches. Cover different angles — do not "
        f"paraphrase the topic three times.",
        node="agent",
        call_type="plan",
        ledger=ledger,
        system=PERSONA,
        schema=QueryPlan,
        temperature=0.4,
    )
    plan = resp.parsed
    return list(plan.queries)[:3] if plan else [topic]


def _assess(topic: str, findings: list[Finding], ledger: TokenLedger) -> Sufficiency:
    evidence = "\n".join(f"[{f.id}] ({f.confidence}) {f.claim}" for f in findings)
    resp = generate(
        f"Topic:\n{topic}\n\nEvidence gathered so far:\n{evidence}\n\n"
        f"Can you write a well-supported 800-1200 word report from this? Be honest. "
        f"If key parts of the topic have no evidence behind them, say what is missing.",
        node="agent",
        call_type="assess",
        ledger=ledger,
        system=PERSONA,
        schema=Sufficiency,
        temperature=0.2,
    )
    return resp.parsed or Sufficiency(sufficient=True)


WRITE_SYSTEM = PERSONA + """

Write the report now.

Requirements:
- 800-1200 words.
- Cite finding IDs inline, like [F003] or [F003, F012], on the sentence they support.
- Only cite IDs that appear in the evidence list. Never invent one.
- Build an argument. Later sections should use what earlier sections established.
- Where the evidence is thin or the sources disagree, say so plainly. Do not smooth a real
  disagreement into a false consensus, and do not assert what you cannot support.
- No filler. No "in today's rapidly evolving landscape". No conclusion that only restates
  the introduction. Every paragraph carries information.
- Markdown. Start with a single H1 title. No preamble before it."""


def _write(topic: str, findings: list[Finding], gaps: list[str], ledger: TokenLedger) -> str:
    evidence = "\n".join(f"[{f.id}] ({f.confidence}) {f.claim} — {f.source_url}" for f in findings)
    gap_note = ""
    if gaps:
        listed = "\n".join(f"- {g}" for g in gaps)
        gap_note = (
            f"\n\nKnown evidence gaps you could not close — acknowledge these in the report "
            f"rather than writing around them:\n{listed}"
        )
    resp = generate(
        f"Topic:\n{topic}\n\nEvidence:\n{evidence}{gap_note}",
        node="agent",
        call_type="write",
        ledger=ledger,
        system=WRITE_SYSTEM,
        temperature=0.6,
    )
    return (resp.text or "").strip()


def run(topic_id: str, topic: str, *, verbose: bool = True) -> dict[str, Any]:
    ledger = TokenLedger()
    t0 = time.perf_counter()

    findings: list[Finding] = []
    queries = _plan(topic, ledger)
    gaps: list[str] = []
    rounds = 0

    while rounds < MAX_SEARCH_ROUNDS and queries:
        q = queries.pop(0)
        rounds += 1
        if verbose:
            print(f"  [{rounds}/{MAX_SEARCH_ROUNDS}] search: {q[:70]}")
        new = research(
            q, node="agent", ledger=ledger, context=f"Overall topic: {topic}",
            start_index=len(findings) + 1,
        )
        findings.extend(new)
        if verbose:
            print(f"        +{len(new)} findings (total {len(findings)})")

        if not queries and rounds < MAX_SEARCH_ROUNDS:
            s = _assess(topic, findings, ledger)
            if s.sufficient:
                if verbose:
                    print("        evidence judged sufficient")
                break
            gaps = s.missing
            queries = list(s.next_queries)[: MAX_SEARCH_ROUNDS - rounds]
            if verbose and gaps:
                print(f"        gaps: {'; '.join(gaps)[:100]}")

    report = _write(topic, findings, gaps, ledger)
    wall_ms = int((time.perf_counter() - t0) * 1000)

    out = save_run(
        system="baseline",
        topic_id=topic_id,
        report=report,
        findings=findings,
        ledger_dict=ledger.as_dict(),
        wall_ms=wall_ms,
        extra={"search_rounds": rounds, "unclosed_gaps": gaps},
    )
    return {"dir": out, "findings": findings, "report": report, "ledger": ledger}
