from __future__ import annotations

import logging
import os
from typing import Any

from src.common.llm import TokenLedger, generate, sources_from
from src.common.schemas import Finding, FindingList

log = logging.getLogger(__name__)

MAX_RESULTS = 6

_gemini_grounding_ok = True

NATIVE_SYSTEM = """You are a research assistant. Search for current, credible sources and
report what they say.

Rules:
- Report facts, figures, dates, and named findings. Not opinions, not your own analysis.
- After every fact, put the URL it came from in parentheses.
- If the sources disagree, report the disagreement rather than picking a side.
- If you cannot find evidence for part of the question, say so explicitly. Do not fill
  the gap with general knowledge — an admitted gap is more useful than a confident guess.
- Prefer primary sources: papers, filings, government data, company disclosures.
- Be dense. No preamble, no summary of what you are about to do."""

GROUND_SYSTEM = """You are a research assistant. You are given raw web search results and a
research question. Report what the results say.

Rules:
- Report facts, figures, dates, and named findings. Not opinions, not your own analysis.
- After every fact, put the URL it came from in parentheses. A fact with no URL is useless
  to the next step.
- Use only what is in the search results. Do not add anything from your own knowledge, even
  if you are confident it is true.
- If the results disagree, report the disagreement rather than picking a side.
- If the results do not answer part of the question, say so explicitly. Do not fill the gap
  with general knowledge — an admitted gap is more useful than a confident guess.
- Weight primary sources higher: papers, filings, government data, company disclosures.
  Say when a fact rests only on a blog or an aggregator.
- Be dense. No preamble, no summary of what you are about to do."""

EXTRACT_SYSTEM = """Convert research notes into discrete findings.

Rules:
- One finding = one factual assertion. Split compound sentences.
- Do not add anything not present in the notes. No inference, no background.
- Attach the source URL that supports each claim. If the notes do not attribute the
  claim, take the URL from the sources list where the attribution is unambiguous; if it
  is genuinely unattributable, drop the claim rather than recording it without a source.
- Drop anything that is opinion, hedging, or connective prose.

Prefer specific claims, but do not discard evidence to achieve it:
- A claim carrying a figure, a date or a named measured outcome is always worth recording.
- A claim with none of those is still worth recording if it is specific to this subject.
  Mark it low. Drop it only if it would be true of almost any subject in the field.
- Record a restatement of an earlier claim once, keeping the most specific version.

confidence measures how much weight the claim can bear, not whether some source said
it. Nearly everything in the notes was stated by a source, so "a source says so" is not
sufficient for high.
- high   - a specific quantified or dated fact from a primary or institutional source:
           a regulator, statistics agency, standards body, company filing or disclosure,
           peer-reviewed work, or a named report that describes its methodology.
- medium - a specific fact from trade press or a secondary aggregator, or a figure whose
           date, scope or definition is unclear.
- low    - unquantified or generic, or from a marketing page, SEO aggregator, blog,
           forum or unattributed source, or contradicted elsewhere in the notes.

If two sources give different values for the same quantity, record each value as its own
finding attributed to its own source, set both to low, and say inside each claim that
sources disagree. Do not emit a separate finding whose subject is the disagreement: a
finding is one assertion about the world, attributable to one URL, and a claim about the
state of the evidence cannot be cited by anything.

Expect a spread of confidence values. If every finding comes out "high", the rule above
was not applied."""


def _gemini(
    query: str, *, node: str, ledger: TokenLedger, context: str
) -> tuple[str, list[dict[str, str]]]:
    prompt = query if not context else f"{context}\n\nResearch question:\n{query}"
    resp = generate(
        prompt,
        node=node,
        call_type="search",
        ledger=ledger,
        system=NATIVE_SYSTEM,
        search=True,
        temperature=0.2,
    )
    return (resp.text or "").strip(), sources_from(resp)


def _tavily(query: str, max_results: int) -> list[dict[str, str]]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    from tavily import TavilyClient

    payload: dict[str, Any] = TavilyClient(api_key=key).search(
        query=query, max_results=max_results, search_depth="basic"
    )
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in payload.get("results", [])
        if r.get("url")
    ]


def _ddgs(query: str, max_results: int) -> list[dict[str, str]]:
    from ddgs import DDGS

    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")}
        for r in DDGS().text(query, max_results=max_results)
        if r.get("href")
    ]


def web_search(query: str, *, max_results: int = MAX_RESULTS) -> tuple[list[dict[str, str]], str]:
    for name, fn in (("tavily", _tavily), ("ddgs", _ddgs)):
        try:
            hits = fn(query, max_results)
        except Exception as e:  # noqa: BLE001
            log.warning("%s search failed for %r: %s", name, query[:60], e)
            continue
        if hits:
            return _dedupe(hits), name
    log.warning("no search backend returned results for %r", query[:60])
    return [], "none"


def _dedupe(hits: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for h in hits:
        if h["url"] not in seen:
            seen.add(h["url"])
            out.append(h)
    return out


def ground(
    query: str,
    *,
    node: str,
    ledger: TokenLedger,
    context: str = "",
) -> tuple[str, list[dict[str, str]]]:
    global _gemini_grounding_ok

    if _gemini_grounding_ok:
        try:
            notes, sources = _gemini(query, node=node, ledger=ledger, context=context)
            if notes:
                return notes, sources
            log.warning("gemini grounding returned no text; falling back to web search")
        except Exception as e:  # noqa: BLE001
            log.warning("gemini grounding unavailable (%s); using web search for the rest "
                        "of this run", type(e).__name__)
        _gemini_grounding_ok = False

    hits, backend = web_search(query)
    if not hits:
        return "", []

    results = "\n\n".join(
        f"[{i}] {h['title']}\n{h['url']}\n{h['content']}" for i, h in enumerate(hits, start=1)
    )
    prompt = f"Research question:\n{query}\n\nSearch results ({backend}):\n\n{results}"
    if context:
        prompt = f"{context}\n\n{prompt}"

    resp = generate(
        prompt,
        node=node,
        call_type="search",
        ledger=ledger,
        system=GROUND_SYSTEM,
        temperature=0.2,
    )
    sources = [{"url": h["url"], "title": h["title"]} for h in hits]
    return (resp.text or ""), sources


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
                claim=" ".join(f.claim.split()),
                source_url=" ".join(f.source_url.split()) if f.source_url else "",
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
