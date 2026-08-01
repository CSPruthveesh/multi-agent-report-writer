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
import hashlib
import json
import random
from pathlib import Path

from scripts.judge import _body
from src.common.io import RESULTS, get_topic, load_topics

HS = RESULTS / "handscore"
CRITERIA = [
    ("factual_grounding", "claims trace to evidence; admitted gaps score HIGH"),
    ("structural_coherence", "builds an argument; watch for repeated points"),
    ("depth_of_analysis", "synthesises; preserves real disagreement"),
    ("citation_integrity", "IDs resolve and support their sentence"),
    ("absence_of_filler", "every paragraph carries information"),
]
SUGGESTED = ["t1", "t3", "t5"]  # one per shape

STALE_REFUSAL = """\
Hand scores exist for reports that are no longer on disk:

{listing}

prepare() returns early for an already-prepared topic, so a fresh run leaves the old
copies in place and the old scores beside them. Those scores describe text that no
longer exists.

Scoring on top of them — or letting compare.py join them to the current judge numbers
— produces a table whose human column and judge column describe different documents,
and nothing in the output would say so.

Move each directory OUT of results/handscore/, then run this again:

{commands}

Out of the directory, not renamed inside it. Everything that reads results/handscore/
globs it, and an archived copy left in place still holds a mapping and a set of scores.
The staleness check would clear it — it looks for the source report under the archived
name, finds nothing to compare, and correctly reports no difference — so the old scores
would rejoin the current numbers beside the new ones. The previous set is committed at
2e33d58 either way.
"""


def _free_archive_path(topic_id: str) -> Path:
    """A destination that does not already exist.

    Superseding happens more than once — a topic can be re-scored, superseded again,
    and archived again — so a fixed destination collides on the second round and the
    printed command fails. Third time in this phase that a correct check shipped
    remediation advice which did not work; the advice is part of the fix.
    """
    base = HS.parent / "handscore_superseded"
    p = base / topic_id
    n = 2
    while p.exists():
        p = base / f"{topic_id}-{n}"
        n += 1
    return p


def _stale_refusal(topics: list[str]) -> str:
    listing = "\n".join(
        f"  {HS / t}  ({len(stale_labels(t))} of 2 reports changed)" for t in topics
    )
    commands = "\n".join(
        [f"  New-Item -ItemType Directory -Force {HS.parent / 'handscore_superseded'}"]
        + [f"  Move-Item {HS / t} {_free_archive_path(t)}" for t in topics]
    )
    return STALE_REFUSAL.format(listing=listing, commands=commands)


def _norm(s: str) -> str:
    return s.strip() + "\n"


def _digest(s: str) -> str:
    return hashlib.sha256(_norm(s).encode("utf-8")).hexdigest()[:12]


def blind_text(system: str, topic_id: str) -> str:
    """A report as the scorer should see it: without its declared-limitations section.

    The judge has stripped this since scripts/judge.py was written, because the
    baseline has no mechanism to declare an evidence gap and the graph declares one on
    most topics — a format tell strong enough to identify the system on sight.

    Hand scoring did not strip it, and that was a hole for as long as this file has
    existed. It cost the first two topics of the second scoring round: the section was
    recognised, the system was identified, and the remaining four criteria stopped
    being blind. A leak that only affects a human is still a leak, and the human is
    the instrument that matters most here — the judge scored a set of disconnected
    topic buckets 4.83 out of 5 on structural coherence and could not see the problem
    at all.

    _body is imported from the judge rather than restated so the two instruments
    cannot drift on what blinding means. The declared-gaps asymmetry is not lost by
    this — it is reported by compare.py as a categorical fact, which is what
    scripts/judge.py's own docstring argued for and what was never done.
    """
    src = RESULTS / system / topic_id / "report.md"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    return _norm(_body(src.read_text(encoding="utf-8")))


def declares_gaps(system: str, topic_id: str) -> bool:
    """Whether this report carries a declared-limitations section, before stripping."""
    src = RESULTS / system / topic_id / "report.md"
    return src.exists() and "## Known limitations" in src.read_text(encoding="utf-8")


def is_topic_dir(name: str) -> bool:
    """Whether a directory under results/handscore/ names a real topic.

    Everything that reads this directory globs it, and a glob believes whatever it
    finds. An archived t1.old still holds mapping.json and handscores.json, so it
    reads as a valid topic — and stale_labels() clears it, because it looks for
    results/<system>/t1.old/report.md, finds nothing to compare, and honestly reports
    no difference. The old scores would then join the current judge numbers next to
    the new ones, which is the failure the staleness check exists to prevent, arrived
    at by archiving the evidence exactly as instructed.

    So the name has to be checked against topics.json rather than trusted.
    """
    return name in {t["id"] for t in load_topics()}


def stale_labels(topic_id: str) -> list[str]:
    """Labels whose copied report no longer matches the report it was copied from.

    The copy is its own record. No stored hash to keep in sync, and no way for the
    check to drift from what is actually on disk — the two files either match or they
    do not.

    Reads mapping.json to locate each label's source and never returns or prints a
    system name, so calling this does not cost the blinding. Returning labels rather
    than systems is deliberate for the same reason.
    """
    d = HS / topic_id
    mp = d / "mapping.json"
    if not mp.exists():
        return []
    m = json.loads(mp.read_text(encoding="utf-8"))
    out = []
    for i, label in enumerate(("report_1", "report_2"), start=1):
        src = RESULTS / m[label] / topic_id / "report.md"
        copy = d / f"report_{i}.md"
        if not (src.exists() and copy.exists()):
            continue
        # Compare against the blinded form, not the raw file — the copy is stripped, so
        # a raw comparison would report every topic carrying a limitations section as
        # permanently stale and there would be no way to prepare one that satisfied it.
        if _digest(blind_text(m[label], topic_id)) != _digest(
            copy.read_text(encoding="utf-8")
        ):
            out.append(label)
    return out


def scored_topics() -> list[str]:
    return sorted(
        p.parent.name for p in HS.glob("*/handscores.json") if is_topic_dir(p.parent.name)
    )


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


def _materialise(d: Path, topic_id: str, systems: list[str]) -> None:
    """Write both reports, blinded, and both evidence files, under the label order."""
    for i, sysname in enumerate(systems, start=1):
        (d / f"report_{i}.md").write_text(
            blind_text(sysname, topic_id), encoding="utf-8"
        )
    _write_evidence(d, topic_id, systems)


def prepare(topic_id: str) -> Path:
    d = HS / topic_id
    if (d / "mapping.json").exists():
        # The mapping is read on every branch here and printed on none of them.
        m = json.loads((d / "mapping.json").read_text(encoding="utf-8"))
        systems = [m["report_1"], m["report_2"]]
        stale = stale_labels(topic_id)

        if stale and (d / "handscores.json").exists():
            raise SystemExit(_stale_refusal([topic_id]))

        if stale:
            # A newer run exists and nothing has been scored against these copies, so
            # replacing them costs nothing. The label order is taken from the existing
            # mapping rather than re-rolled: the shuffle is seeded by topic_id and would
            # come out the same anyway, and depending on that is a worse habit than not.
            _materialise(d, topic_id, systems)
            print(f"re-prepared from the current run ({len(stale)} of 2 changed): {d}")
            return d

        # Backfill evidence for a topic prepared before those files existed, WITHOUT
        # re-shuffling — re-rolling would invalidate scores already recorded against
        # these labels.
        if not (d / "evidence_1.md").exists():
            _write_evidence(d, topic_id, systems)
            print(f"added evidence files to existing: {d}")
        else:
            print(f"already prepared: {d}")
        return d

    d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(topic_id)
    systems = ["baseline", "multiagent"]
    rng.shuffle(systems)

    _materialise(d, topic_id, systems)
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
    print("\nAny 'Known limitations' section has been removed from both copies. Only one")
    print("system can produce one, so it identifies the system on sight. It is counted")
    print("and reported separately by compare.py. Do not mark a report down for lacking")
    print("one — you are not being shown it.")
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


def refuse_stale_scored() -> None:
    """Guard every entry point, not just the one that was being thought about.

    prepare() catches this for the topic it is preparing, but --reveal never calls
    prepare() and the all-scored branch returns before it. A check that covers only
    the path you had in mind is the exact shape of defect this project keeps finding.
    """
    bad = [t for t in scored_topics() if stale_labels(t)]
    if bad:
        raise SystemExit(_stale_refusal(bad))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic-id")
    ap.add_argument("--reveal", action="store_true")
    args = ap.parse_args()

    refuse_stale_scored()

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
