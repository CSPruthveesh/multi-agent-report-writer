"""Hand-scoring harness — enforces blindness so you cannot cheat yourself.

    uv run python -m evals.handscore            # score the next unscored topic
    uv run python -m evals.handscore --reveal   # show the mapping AFTER scoring

WHY THIS EXISTS
---------------
You know which system you spent a week building. Reading a report knowing it is the
multi-agent one is not a blind evaluation, it is a confirmation exercise — and you
will find what you expect to find without ever noticing you did.

This copies both reports to results/handscore/<topic>/report_1.md and report_2.md in
random order, writes the mapping to a file it tells you not to open, and takes your
scores. Reveal only after all scoring is done.

Three topics is enough: one abundant-sources, one thin/contested, one cross-domain.
The point is not statistical power — six topics has none either. The point is a human
check on whether the judge is measuring anything real.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from src.common.io import RESULTS, get_topic

HS = RESULTS / "handscore"
CRITERIA = [
    ("factual_grounding", "claims trace to evidence; admitted gaps score HIGH"),
    ("structural_coherence", "builds an argument; watch for repeated points"),
    ("depth_of_analysis", "synthesises; preserves real disagreement"),
    ("citation_integrity", "IDs resolve and support their sentence"),
    ("absence_of_filler", "every paragraph carries information"),
]
SUGGESTED = ["t1", "t3", "t5"]  # one per shape


def _write_evidence(d: Path, topic_id: str, systems: list[str]) -> None:
    """Each report's own findings, under the same shuffled label as the report.

    Criterion 4 asks whether every ID resolves and supports the sentence it sits on.
    Without this the only way to check was to open results/<system>/<topic>/run.json,
    which names the system and destroys the blinding for the rest of the session — so
    the criterion was either unscorable or scored at the cost of the whole exercise.

    The two sets are separate namespaces. Both systems number from F001 independently
    and the claims behind a shared ID differ — 67 of 67 collide on t1 — so the header
    says so. Checking report_1's citations against evidence_2 would produce confident
    nonsense.
    """
    for i, sysname in enumerate(systems, start=1):
        run = RESULTS / sysname / topic_id / "run.json"
        if not run.exists():
            raise SystemExit(f"missing {run}")
        findings = json.loads(run.read_text(encoding="utf-8"))["findings"]
        lines = [
            f"# Evidence for report_{i}",
            "",
            (f"{len(findings)} findings. Check every citation in `report_{i}.md` "
             f"against THIS file."),
            "",
            ("IDs are per-report: F001 here is a different claim from F001 in the "
             "other evidence file. Do not cross-check."),
            "",
        ]
        for f in findings:
            conf = f.get("confidence")
            tag = f"({conf}) " if conf else ""
            lines.append(f"- **[{f['id']}]** {tag}{f['claim']}")
        (d / f"evidence_{i}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(topic_id: str) -> Path:
    d = HS / topic_id
    if (d / "mapping.json").exists():
        # Backfill evidence for a topic prepared before this existed, WITHOUT
        # re-shuffling. The mapping is read here but never printed — re-rolling it
        # would invalidate any scores already recorded against these labels.
        m = json.loads((d / "mapping.json").read_text(encoding="utf-8"))
        if not (d / "evidence_1.md").exists():
            _write_evidence(d, topic_id, [m["report_1"], m["report_2"]])
            print(f"added evidence files to existing: {d}")
        else:
            print(f"already prepared: {d}")
        return d
    d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(topic_id)
    systems = ["baseline", "multiagent"]
    rng.shuffle(systems)

    for i, sysname in enumerate(systems, start=1):
        src = RESULTS / sysname / topic_id / "report.md"
        if not src.exists():
            raise SystemExit(f"missing {src}")
        shutil.copy(src, d / f"report_{i}.md")

    _write_evidence(d, topic_id, systems)
    (d / "mapping.json").write_text(
        json.dumps({"report_1": systems[0], "report_2": systems[1]}), encoding="utf-8"
    )
    return d


def score(topic_id: str) -> None:
    d = prepare(topic_id)
    topic = get_topic(topic_id)

    print("=" * 68)
    print(f"HAND SCORING — {topic_id} ({topic['shape']})")
    print("=" * 68)
    print(f"\n{topic['topic']}\n")
    print(f"Watch for: {topic['watch_for']}\n")
    print(f"Read BOTH before scoring EITHER:\n  {d / 'report_1.md'}\n  {d / 'report_2.md'}")
    print("\nEvidence for citation_integrity — each report against its OWN file:")
    print(f"  {d / 'evidence_1.md'}\n  {d / 'evidence_2.md'}")
    print("  (same IDs, different claims — do not cross-check)")
    print("\nDo not open mapping.json until you have finished all three topics.\n")
    input("Press Enter when you have read both... ")

    out = {}
    for label in ("report_1", "report_2"):
        print(f"\n--- {label} ---")
        scores = {}
        for name, hint in CRITERIA:
            while True:
                raw = input(f"  {name:<22} ({hint})\n  1-5 > ").strip()
                if raw.isdigit() and 1 <= int(raw) <= 5:
                    scores[name] = int(raw)
                    break
                print("  integer 1-5 please")
        out[label] = scores

    note = input("\nOne sentence on the main difference between them:\n> ").strip()
    (d / "handscores.json").write_text(
        json.dumps({"scores": out, "note": note}, indent=2), encoding="utf-8"
    )
    print(f"\nsaved {d / 'handscores.json'}")


def reveal() -> None:
    print("\nMapping and hand scores by system:\n")
    for d in sorted(HS.glob("*/")):
        mp = d / "mapping.json"
        hs = d / "handscores.json"
        if not (mp.exists() and hs.exists()):
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        s = json.loads(hs.read_text(encoding="utf-8"))["scores"]
        print(f"  {d.name}")
        for label, sysname in m.items():
            tot = sum(s[label].values())
            print(f"    {label} = {sysname:<11} total {tot}/25  {s[label]}")
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic-id")
    ap.add_argument("--reveal", action="store_true")
    args = ap.parse_args()

    if args.reveal:
        reveal()
        return

    if args.topic_id:
        score(args.topic_id)
        return

    for t in SUGGESTED:
        if not (HS / t / "handscores.json").exists():
            score(t)
            return
    print("All suggested topics scored. Run --reveal to see the mapping.")


if __name__ == "__main__":
    main()
