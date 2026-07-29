from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

log = logging.getLogger(__name__)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CRITIC_MODEL = os.getenv("GEMINI_CRITIC_MODEL", MODEL)

_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set. Copy .env.example to .env.")
        _client = genai.Client(api_key=key)
    return _client


@dataclass
class CallRecord:
    node: str
    call_type: str
    in_tokens: int
    out_tokens: int
    total_tokens: int
    latency_ms: int
    attempts: int
    model: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TokenLedger:
    records: list[CallRecord] = field(default_factory=list)

    def add(self, r: CallRecord) -> None:
        self.records.append(r)

    @property
    def total_in(self) -> int:
        return sum(r.in_tokens for r in self.records)

    @property
    def total_out(self) -> int:
        return sum(r.out_tokens for r in self.records)

    @property
    def total(self) -> int:
        return sum(r.total_tokens for r in self.records)

    def by_node(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for r in self.records:
            d = out.setdefault(r.node, {"calls": 0, "in": 0, "out": 0, "total": 0, "ms": 0})
            d["calls"] += 1
            d["in"] += r.in_tokens
            d["out"] += r.out_tokens
            d["total"] += r.total_tokens
            d["ms"] += r.latency_ms
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": len(self.records),
            "total_in": self.total_in,
            "total_out": self.total_out,
            "total": self.total,
            "by_node": self.by_node(),
            "records": [r.as_dict() for r in self.records],
        }


_RETRYABLE = ("429", "500", "502", "503", "504", "deadline", "timeout", "unavailable")


def _is_retryable(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(t in s for t in _RETRYABLE)


def with_backoff[T](fn: Callable[[], T], *, attempts: int = 3, base: float = 1.5) -> tuple[T, int]:
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            return fn(), i
        except Exception as e:
            last = e
            if not _is_retryable(e) or i == attempts:
                raise
            wait = base**i + random.uniform(0, 0.4)
            log.warning("retryable failure (attempt %d/%d): %s — sleeping %.1fs", i, attempts, e, wait)
            time.sleep(wait)
    assert last is not None
    raise last


def generate(
    prompt: str,
    *,
    node: str,
    call_type: str,
    ledger: TokenLedger,
    system: str | None = None,
    search: bool = False,
    schema: Any = None,
    temperature: float = 0.3,
    model: str | None = None,
    max_output_tokens: int | None = None,
):
    if search and schema is not None:
        raise ValueError(
            "search + schema on one call is not supported by the API. "
            "Ground first, then extract structure in a second call."
        )

    mdl = model or MODEL
    cfg: dict[str, Any] = {"temperature": temperature}
    if system:
        cfg["system_instruction"] = system
    if max_output_tokens:
        cfg["max_output_tokens"] = max_output_tokens
    if search:
        cfg["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    if schema is not None:
        cfg["response_mime_type"] = "application/json"
        cfg["response_schema"] = schema

    t0 = time.perf_counter()
    resp, attempts = with_backoff(
        lambda: client().models.generate_content(
            model=mdl,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg),
        )
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    um = getattr(resp, "usage_metadata", None)
    in_tok = getattr(um, "prompt_token_count", 0) or 0
    out_tok = getattr(um, "candidates_token_count", 0) or 0
    tool_tok = getattr(um, "tool_use_prompt_token_count", 0) or 0
    total_tok = getattr(um, "total_token_count", 0) or (in_tok + out_tok + tool_tok)

    ledger.add(
        CallRecord(
            node=node,
            call_type=call_type,
            in_tokens=in_tok + tool_tok,
            out_tokens=out_tok,
            total_tokens=total_tok,
            latency_ms=latency_ms,
            attempts=attempts,
            model=mdl,
        )
    )
    return resp


def sources_from(resp) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    try:
        gm = resp.candidates[0].grounding_metadata
        for chunk in getattr(gm, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web and getattr(web, "uri", None):
                out.append({"url": web.uri, "title": getattr(web, "title", "") or ""})
    except (AttributeError, IndexError, TypeError):
        pass

    seen: set[str] = set()
    deduped = []
    for s in out:
        if s["url"] not in seen:
            seen.add(s["url"])
            deduped.append(s)
    return deduped
