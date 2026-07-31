"""Test the Critic, and check its calibration by degrading a draft on purpose.

    uv run python -m scripts.test_critic --topic-id t1
    uv run python -m scripts.test_critic --calibrate        # THE important one

WHY CALIBRATION MATTERS
-----------------------
The Critic has two opposite failure modes and both look like the system working:

    always passes    Model is agreeable about work it just produced. The revision
                     loop never fires. The architecture is researcher -> analyst ->
                     writer with extra steps and extra cost.
    always revises    Asked to find problems, it finds problems. Every run burns
                     the full revision budget and Phase 9 shows cost with no gain.

Neither shows up in a single trace. The only way to see it is to feed the Critic
drafts you have broken in known ways and check that the scores move on the RIGHT
criterion:

    shuffled paragraphs  -> structural_coherence should drop
    stripped citations   -> citation_integrity and factual_grounding should drop
    injected filler      -> absence_of_filler should drop
    invented citations   -> citation_integrity must be capped at 1 (code, not model)

A Critic that scores a shuffled draft the same as a clean one is not reading; it is
pattern-matching on surface fluency. Fix that before Phase 9 — an uncalibrated critic
makes the revision loop pure cost, and worse, it makes the multi-agent system look
like it has quality control when it does not.

Costs ~5 Critic calls. Cheap for what it tells you.
"""

from __future__ import annotations

import argparse
import json
import random
import re

from src.common.io import RESULTS, get_topic
from src.graph.nodes.analyst import analyst
from src.graph.nodes.critic import critic
from src.graph.nodes.writer import writer
from src.graph.state import initial_state

CITE_RE = re.compile(r"\s*\[F\d{3}(?:,\s*F\d{3})*\]")

FILLER = [
    "It is important to note that this is a complex and multifaceted issue.",
    "In today's rapidly evolving landscape, various factors may play a role.",
    "That said, the situation continues to develop in interesting ways.",
]


def _load_findings(topic_id: str) -> list[dict]:
    p = RESULTS / "baseline" / topic_id / "run.json"
    if not p.exists():
        raise SystemExit(f"No baseline findings at {p}. Run Phase 0 for {topic_id} first.")
    return json.loads(p.read_text(encoding="utf-8"))["findings"]


# ------------------------------------------------------------- degradations
def degrade_shuffle(draft: str) -> str:
    """Break structural coherence, leaving every sentence individually intact."""
    parts = draft.split("\n\n")
    head, body = parts[0], parts[1:]
    rng = random.Random(7)
    rng.shuffle(body)
    return "\n\n".join([head] + body)


def degrade_strip_cites(draft: str) -> str:
    """Remove every citation. Claims stay, support vanishes."""
    return CITE_RE.sub("", draft)


def degrade_filler(draft: str) -> str:
    """Pad every other paragraph with content-free sentences."""
    parts = draft.split("\n\n")
    out = []
    for i, p in enumerate(parts):
        out.append(p)
        if i % 2 == 1 and not p.startswith("#"):
            out.append(FILLER[i % len(FILLER)])
    return "\n\n".join(out)


def degrade_fake_cites(draft: str) -> str:
    """Inject IDs that do not exist. Must trigger the hard cap in code."""
    parts = draft.split("\n\n")
    for i, p in enumerate(parts):
        if not p.startswith("#") and len(p.split()) > 20:
            parts[i] = p.rstrip(".") + " [F901, F902]."
            break
    return "\n\n".join(parts)


DEGRADATIONS = {
    "shuffled": (degrade_shuffle, "structural_coherence"),
    "no-citations": (degrade_strip_cites, "citation_integrity"),
    "filler": (degrade_filler, "absence_of_filler"),
    "fake-citations": (degrade_fake_cites, "citation_integrity"),
}


def _score(state: dict, draft: str) -> dict:
    s = dict(state)
    s["draft"] = draft
    upd = critic(s)
    crit = upd.get("critique") or {}
    ev = (upd.get("trace") or [{}])[0]
    return {"critique": crit, "trace": ev}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic-id", default="t1")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    topic = get_topic(args.topic_id)
    findings = _load_findings(args.topic_id)
    state = initial_state(topic["topic"])
    state["findings"] = findings

    print(f"topic     {topic['id']} ({topic['shape']})")
    print("planning + writing a clean draft...")
    state.update({k: v for k, v in analyst(state).items() if k != "trace"})
    w = writer(state)
    draft = w["draft"]
    state["draft"] = draft
    print(f"  {len(draft.split())} words\n")

    base = _score(state, draft)
    bc = base["critique"]
    print("--- clean draft ---")
    for k, v in bc["scores"].items():
        print(f"  {k:<22} {v}")
    print(f"  {'VERDICT':<22} {bc['verdict']}"
          + (f" -> {bc['target']}" if bc["verdict"] == "revise" else ""))
    print(f"  summary: {bc.get('summary', '')[:90]}")
    ev = base["trace"]
    print(f"  issues kept/raw: {ev.get('issues_kept')}/{ev.get('issues_raw')}"
          f"  dropped ungrounded: {ev.get('dropped_ungrounded')}")

    if not args.calibrate:
        return

    print("\n--- degraded copies ---")
    rows = []
    for name, (fn, expect) in DEGRADATIONS.items():
        bad = fn(draft)
        r = _score(state, bad)
        rows.append((name, expect, r["critique"], r["trace"]))
        sc = r["critique"]["scores"]
        print(f"\n  {name}  (expect {expect} to drop)")
        # Every criterion, not just the expected one. Collateral movement is
        # information: a degradation that drags unrelated scores down means the
        # Critic is reacting to damage in general rather than reading criteria
        # independently, and a criterion that never moves anywhere is a blind spot.
        for k in bc["scores"]:
            was, got = bc["scores"].get(k, 5), sc.get(k, 5)
            mark = "DROP" if got < was else ("same" if got == was else "ROSE")
            star = " <-" if k == expect else ""
            print(f"    {k:<22} {was} -> {got}  [{mark}]{star}")
        print(f"    verdict: {r['critique']['verdict']}"
              + (f" -> {r['critique']['target']}"
                 if r["critique"]["verdict"] == "revise" else ""))
        if r["critique"].get("citation_cap_applied"):
            print(f"    citation cap applied in code: "
                  f"{r['critique'].get('citation_cap_reason', '')}")

    print("\n" + "=" * 60)
    print("CALIBRATION")
    print("=" * 60)

    passes = 0
    for name, expect, crit, _ in rows:
        got = crit["scores"].get(expect, 5)
        was = bc["scores"].get(expect, 5)
        ok = got < was
        passes += ok
        print(f"  {name:<16} {expect:<22} {was} -> {got}  {'PASS' if ok else 'FAIL'}")

    cap_row = next((r for r in rows if r[0] == "fake-citations"), None)
    cap_ok = bool(cap_row and cap_row[2].get("citation_cap_applied"))
    print(f"  {'hard cap':<16} {'invented IDs -> 1':<22} {'':>6}  "
          f"{'PASS' if cap_ok else 'FAIL'}")

    clean_pass = bc["verdict"] == "pass"
    print(f"  {'clean draft':<16} {'should mostly pass':<22} {bc['verdict']:>6}  "
          f"{'PASS' if clean_pass else 'check'}")

    print()
    if passes >= 3 and cap_ok:
        print("  CALIBRATED — the Critic responds to real defects on the right criterion.")
        print("  The revision loop will fire for reasons, not reflexively.")
    elif passes == 0:
        print("  FAIL — scores did not move on any degraded draft. The Critic is reading")
        print("  surface fluency, not content. Fixes, in order:")
        print("    1. Set GEMINI_CRITIC_MODEL to a stronger model than the writer")
        print("    2. Score one criterion per call instead of all five at once")
        print("    3. Drop temperature to 0.0")
    else:
        print(f"  PARTIAL — {passes}/4 degradations detected. Look at which ones failed;")
        print("  a critic blind to one criterion still contaminates that column in Phase 9.")

    if not clean_pass:
        print("\n  Note: the clean draft did not pass. If that repeats across topics, the")
        print("  Critic is over-eager and every run will burn the full revision budget.")


if __name__ == "__main__":
    main()
