from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.common.schemas import Finding

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

CITE_RE = re.compile(r"\bF\d{3}\b")


def load_topics() -> list[dict[str, Any]]:
    payload = json.loads((DATA / "topics.json").read_text(encoding="utf-8"))
    return payload["topics"]


def get_topic(topic_id: str) -> dict[str, Any]:
    for t in load_topics():
        if t["id"] == topic_id:
            return t
    raise KeyError(f"unknown topic id: {topic_id}")


def check_citations(draft: str, findings: list[Finding]) -> dict[str, Any]:
    valid = {f.id for f in findings}
    cited = set(CITE_RE.findall(draft))
    broken = sorted(cited - valid)
    return {
        "cited_count": len(cited),
        "valid_count": len(valid),
        "broken": broken,
        "broken_count": len(broken),
        "uncited_findings": sorted(valid - cited),
        "coverage": round(len(cited & valid) / len(valid), 3) if valid else 0.0,
    }


def word_count(text: str) -> int:
    return len(text.split())


def save_run(
    *,
    system: str,
    topic_id: str,
    report: str,
    findings: list[Finding],
    ledger_dict: dict[str, Any],
    wall_ms: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir = RESULTS / system / topic_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "report.md").write_text(report, encoding="utf-8")

    payload: dict[str, Any] = {
        "system": system,
        "topic_id": topic_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "word_count": word_count(report),
        "findings_count": len(findings),
        "citations": check_citations(report, findings),
        "wall_ms": wall_ms,
        "tokens": ledger_dict,
        "findings": [f.model_dump() for f in findings],
    }
    if extra:
        payload.update(extra)

    (out_dir / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_dir


def print_summary(payload_dir: Path) -> None:
    d = json.loads((payload_dir / "run.json").read_text(encoding="utf-8"))
    c = d["citations"]
    tk = d["tokens"]
    print(f"\n  {d['system']} / {d['topic_id']}")
    print(f"  words          {d['word_count']}")
    print(f"  findings       {d['findings_count']}")
    print(f"  cited / broken {c['cited_count']} / {c['broken_count']}")
    print(f"  coverage       {c['coverage']:.0%} of findings cited")
    print(f"  tokens         {tk['total']:,}  (in {tk['total_in']:,} / out {tk['total_out']:,})")
    print(f"  model calls    {tk['calls']}")
    print(f"  wall clock     {d['wall_ms'] / 1000:.1f}s")
    if c["broken"]:
        print(f"  !! broken IDs  {', '.join(c['broken'])}")
    print(f"  -> {payload_dir}")
