from __future__ import annotations

import logging

from src.common.llm import TokenLedger, generate, sources_from
from src.common.schemas import Finding, FindingList

log = logging.getLogger(__name__)

GROUND_SYSTEM = """You are a research assistant. Search for current, credible sources and
report what they say.

Rules:
- Report facts, figures, dates, and named findings. Not opinions, not your own analysis.
- If the sources disagree, report the disagreement rather than picking a side.
- If you cannot find evidence for part of the question, say so explicitly. Do not fill
  the gap with general knowledge — an admitted gap is more useful than a confident guess.
- Prefer primary sources: papers, filings, government data, company disclosures.
- Be dense. No preamble, no summary of what you are about to do."""

EXTRACT_SYSTEM = """Convert research notes into discrete findings.

Rules:
- One finding = one factual assertion. Split compound sentences.
- Do not add anything not present in the notes. No inference, no background.
- Attach the source URL that supports each claim. If the notes do not attribute a
  claim to a specific URL, use the empty string and set confidence to "low".
- confidence: "high" if a credible source states it directly, "medium" if implied or
  from a weaker source, "low" if inferred, contested, or single-source.
- Drop anything that is opinion, hedging, or connective prose."""


def ground(
    query: str,
    *,
    node: str,
    ledger: TokenLedger,
    context: str = "",
) -> tuple[str, list[dict[str, str]]]:
    prompt = query if not context else f"{context}\n\nResearch question:\n{query}"
    resp = generate(
        prompt,
        node=node,
        call_type="search",
        ledger=ledger,
        system=GROUND_SYSTEM,
        search=True,
        temperature=0.2,
    )
    return (resp.text or ""), sources_from(resp)


def extract(
    notes: str,
    *,
    node: str,
    ledger: TokenLedger,
    sources: list[dict[str, str]] | None = None,
    start_index: int = 1,
) -> list[Finding]:
    if not notes.strip():
        return []

    src_block = ""
    if sources:
        listed = "\n".join(f"- {s['title']}: {s['url']}" for s in sources if s.get("url"))
        if listed:
            src_block = f"\n\nSources consulted:\n{listed}"

    resp = generate(
        f"Research notes:\n{notes}{src_block}",
        node=node,
        call_type="extract",
        ledger=ledger,
        system=EXTRACT_SYSTEM,
        schema=FindingList,
        temperature=0.0,
    )

    parsed = getattr(resp, "parsed", None)
    if parsed is None:
        log.warning("[%s] structured extraction returned nothing parseable", node)
        return []

    raw = parsed.findings if isinstance(parsed, FindingList) else list(parsed)
    out: list[Finding] = []
    for i, f in enumerate(raw, start=start_index):
        out.append(
            Finding(
                id=f"F{i:03d}",
                claim=f.claim.strip(),
                source_url=f.source_url or "",
                confidence=f.confidence,
            )
        )
    return out


def research(
    query: str,
    *,
    node: str,
    ledger: TokenLedger,
    context: str = "",
    start_index: int = 1,
) -> list[Finding]:
    notes, sources = ground(query, node=node, ledger=ledger, context=context)
    return extract(notes, node=node, ledger=ledger, sources=sources, start_index=start_index)
