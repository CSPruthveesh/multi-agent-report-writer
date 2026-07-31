from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.common.llm import TokenLedger, generate
from src.common.search import research
from src.graph.state import MAX_SEARCHES, ReportState, log_entries, trace_event

log = logging.getLogger(__name__)

NODE = "researcher"

PERSONA = """You are a research specialist. Your only job is to find evidence. You do not
outline, you do not write prose, and you do not draw conclusions — other people do that."""


class QueryPlan(BaseModel):
    queries: list[str] = Field(
        description="2-3 search queries covering DIFFERENT angles. Never paraphrase the topic."
    )


class GapQueries(BaseModel):
    queries: list[str] = Field(
        description="One targeted search query per gap. Specific enough to find the missing evidence."
    )


def _plan_queries(topic: str, ledger: TokenLedger, limit: int) -> list[str]:
    resp = generate(
        f"Topic:\n{topic}\n\nPlan your opening searches. Each query must attack a different "
        f"angle — if two of them would return the same results, you have wasted one.",
        node=NODE,
        call_type="plan",
        ledger=ledger,
        system=PERSONA,
        schema=QueryPlan,
        temperature=0.0,
    )
    plan = getattr(resp, "parsed", None)
    return (list(plan.queries)[:limit] if plan and plan.queries else [topic])[:limit]


def _gap_queries(
    topic: str,
    gaps: list[str],
    existing: list[dict[str, Any]],
    ledger: TokenLedger,
    limit: int,
) -> list[str]:
    listed = "\n".join(f"- {g}" for g in gaps)
    known = "\n".join(f"- {f['claim'][:110]}" for f in existing[:12]) or "- (nothing yet)"
    resp = generate(
        f"Overall topic:\n{topic}\n\n"
        f"Evidence we already have:\n{known}\n\n"
        f"Evidence gaps another analyst identified:\n{listed}\n\n"
        f"Write one search query per gap, in the same order as the gaps.\n\n"
        f"Every gap names a subject and a constraint — a date bound, a place, an "
        f"exclusion, or a kind of measurement. The constraint is the whole reason the gap "
        f"exists, so the query must carry it:\n"
        f"- Resolve indirect references into explicit terms. If a gap says 'the largest "
        f"market', 'the excluded countries' or 'the leading vendor', name them in the "
        f"query. A query that keeps the phrase instead of resolving it will not match "
        f"anything.\n"
        f"- Keep date bounds as years. 'After 2024' means the query names 2025, 2026 or "
        f"later, or the word forecast or projection.\n"
        f"- Keep the measurement. If the gap asks for adoption rates, transaction volumes "
        f"or costs, those words belong in the query. Searching the subject alone returns "
        f"descriptions of it, not measurements of it.\n\n"
        f"Then check each query against the evidence listed above. If it would mostly "
        f"return what we already have, it is the wrong query — rewrite it.",
        node=NODE,
        call_type="gap_plan",
        ledger=ledger,
        system=PERSONA,
        schema=GapQueries,
        temperature=0.0,
    )
    plan = getattr(resp, "parsed", None)
    return (list(plan.queries)[:limit] if plan and plan.queries else gaps[:limit])[:limit]


def _dedupe(new: list[dict[str, Any]], existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {f["claim"].strip().lower()[:120] for f in existing}
    out = []
    for f in new:
        k = f["claim"].strip().lower()[:120]
        if k and k not in seen:
            seen.add(k)
            out.append(f)
    return out


def researcher(state: ReportState) -> dict[str, Any]:
    ledger = TokenLedger()
    topic = state["topic"]
    gaps = list(state.get("gaps") or [])
    existing = list(state.get("findings") or [])
    used = state.get("searches_used", 0)
    remaining = MAX_SEARCHES - used

    if remaining <= 0:
        return {
            "trace": [
                trace_event(NODE, "skipped", why="search budget exhausted", used=used,
                            left_for_supervisor=len(gaps))
            ],
        }

    if gaps:
        queries = _gap_queries(topic, gaps, existing, ledger, remaining)
        mode = "gap-driven"
    else:
        queries = _plan_queries(topic, ledger, min(3, remaining))
        mode = "cold-start"

    new: list[dict[str, Any]] = []
    for q in queries:
        found = research(
            q,
            node=NODE,
            ledger=ledger,
            context=f"Overall topic: {topic}",
            start_index=len(existing) + len(new) + 1,
        )
        new.extend(f.model_dump() for f in found)

    kept = _dedupe(new, existing)

    for i, f in enumerate(kept, start=len(existing) + 1):
        f["id"] = f"F{i:03d}"

    unaddressed = gaps[len(queries):]

    spread = {c: sum(1 for f in kept if f["confidence"] == c) for c in ("high", "medium", "low")}

    return {
        "findings": kept,
        "searches_used": used + len(queries),
        "gaps": unaddressed,
        "token_log": log_entries(ledger),
        "trace": [
            trace_event(
                NODE,
                "searched",
                mode=mode,
                queries=len(queries),
                found=len(new),
                kept=len(kept),
                dropped_dupes=len(new) - len(kept),
                unaddressed=len(unaddressed),
                confidence="/".join(f"{k[0]}{v}" for k, v in spread.items()),
                budget=f"{used + len(queries)}/{MAX_SEARCHES}",
                tokens=ledger.total,
                sent=[q[:70] for q in queries],
            )
        ],
    }
