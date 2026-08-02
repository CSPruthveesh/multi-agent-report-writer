from __future__ import annotations

import json
import re
import subprocess
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.common.schemas import Finding

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

CITE_RE = re.compile(r"\bF\d{3}\b")


# Tracked paths a run writes or regenerates, excluded from the dirty check as git
# pathspecs. A run editing its own output must not report itself unreproducible.
#
# analysis/ joined the list when the phase write-ups stopped being ignored. Until then
# they never appeared in git status at all, which is why this did not name them — and
# the moment they became tracked, regenerating one made every later run record
# dirty=True with no source file touched. Reproduced at 6c5c38e before fixing.
#
# The same defect this function was rewritten to remove, arriving through a change
# nowhere near it. That is the standing cost of an exclude list: it has to be updated
# when something starts being tracked. Still the right direction — an allowlist of
# source paths would have missed a directory added later and reported clean when it was
# not, which is the dangerous way to be wrong.
GENERATED = (
    ":(exclude)results",
    ":(exclude)docs",
    ":(exclude)analysis",
)


@lru_cache(maxsize=1)
def code_version() -> dict[str, Any]:
    """The commit a run was produced at, and whether the tree was dirty.

    Two result sets are only comparable if the code that produced them was the same.
    Without this the check costs a git archaeology session — read the commits between
    two runs, diff the shared modules, and argue about whether an added branch could
    have executed. With it the check is one comparison, and it stays true forever.

    dirty matters as much as the commit: a run from a modified tree is not reproducible
    from any commit, so the SHA alone would be a more confident claim than the facts
    support.

    DIRTY MEANS THE CODE, NOT THE TREE
    ----------------------------------
    The first version asked git status about the whole working tree, which meant it
    could never answer False. results/ is tracked, save_run() writes report.md before
    it stamps run.json, and every topic after the first also sees the previous topic's
    output — so a run was dirty by the time it recorded its own provenance. Six runs
    of six recorded dirty=True with no source file modified, and the report told the
    reader they were not reproducible from a commit that had in fact produced them.

    A field added so that the next run would be the first observed clean one could not
    report clean. Same shape as check_loop_value in compare.py: an unreachable branch
    printing a conclusion about the run when the problem was the check.

    Generated output is excluded rather than source enumerated. An allowlist of source
    paths silently misses a directory added later, and missing one makes this report
    clean when it is not — the dangerous direction. An excluded output added later
    over-reports dirty instead, which is merely annoying. Ignored paths need no entry:
    they never appear in git status at all.

    Records written before this change used whole-tree scope. Nothing has to be
    corrected, because none of them claimed False — they say True or unknown, and both
    are still true under the narrower meaning.

    Never raises. Outside a repo, or with no git, every field is None and the report
    says the version is unknown — which is the honest reading, and is distinct from a
    record written before this field existed.
    """
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5,
                check=False,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--", ".", *GENERATED)
    return {
        "commit": commit,
        "commit_short": commit[:7] if commit else None,
        "dirty": bool(status) if status is not None else None,
    }


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
        "code_version": code_version(),
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
