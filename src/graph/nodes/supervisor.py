"""Supervisor and finalize — the routing brain and the exit.

Neither is a stub. Both are deterministic and neither makes a model call.

The supervisor is pure if/else on purpose. Paying a model to evaluate
`if gaps and research_loops < 2` adds cost, latency, and a nondeterministic
failure mode in exchange for nothing. "We used an LLM supervisor" is a common
unforced error in agent projects; being able to say why you didn't is the better
answer.

Every path out of here either advances the run or terminates it. No branch loops
without incrementing a counter — verify by reading the function, not by trusting
this docstring.
"""

from __future__ import annotations

from typing import Any

from src.graph.state import (
    CRITIC_TARGETS,
    MAX_RESEARCH_LOOPS,
    MAX_REVISIONS,
    MAX_SEARCHES,
    MAX_WRITE_ATTEMPTS,
    ReportState,
    trace_event,
)

NODE = "supervisor"


def supervisor(state: ReportState) -> dict[str, Any]:
    gaps = list(state.get("gaps") or [])
    # Gaps the Researcher could not afford. They arrive on their own field because the
    # Analyst overwrites gaps on the way here, and they are retired rather than retried:
    # the budget that would have paid for them is already gone.
    unaddressed = list(state.get("unaddressed_gaps") or [])
    # Empty string counts as no draft. A Writer that degraded returns "", and
    # treating that as a real draft sends it to the Critic, which skips, which
    # returns no critique, which routes back to the Writer — a spin. The
    # write-attempt bound below is what makes the retry terminate.
    draft = state.get("draft") or None
    crit = state.get("critique")
    loops = state.get("research_loops", 0)
    revs = state.get("revision_count", 0)
    searches = state.get("searches_used", 0)
    writes = state.get("write_attempts", 0)

    updates: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []

    # 0. Retire the Researcher's handover first, so it is recorded whether the run
    #    then loops, writes or finishes.
    if unaddressed:
        updates["unaddressed_gaps"] = []
        updates["unclosed_gaps"] = list(state.get("unclosed_gaps") or []) + unaddressed
        trace.append(
            trace_event(NODE, "retire_gaps", count=len(unaddressed),
                        why="researcher had no search budget left",
                        searches=f"{searches}/{MAX_SEARCHES}")
        )

    # 1. Evidence gaps outrank everything, including an existing draft. This is the
    #    loop that justifies the architecture, so it gets first refusal.
    #
    #    Both budgets are checked. A loop budget with no searches left is a wasted
    #    hop to a node that will decline to do anything, and before the handover
    #    existed it also lost the gap on the way.
    if gaps and loops < MAX_RESEARCH_LOOPS and searches < MAX_SEARCHES:
        return {
            **updates,
            "route": "researcher",
            "research_loops": loops + 1,
            "trace": trace + [
                trace_event(NODE, "route", to="researcher", why="evidence gaps",
                            gaps=len(gaps), loop=f"{loops + 1}/{MAX_RESEARCH_LOOPS}",
                            searches=f"{searches}/{MAX_SEARCHES}")
            ],
        }

    # 2. Gaps remain but a budget is spent. Retire them into `unclosed_gaps`, clear
    #    `gaps`, and FALL THROUGH to the normal decision.
    #
    #    Falling through matters. An earlier version routed straight to finalize
    #    here, which meant any run with one unclosed gap got ZERO quality revisions
    #    — an evidence problem silently cancelled the entire critic loop. The two
    #    loops are independent and must stay that way.
    if gaps:
        why = (
            "search budget spent"
            if searches >= MAX_SEARCHES
            else "research loop budget spent"
        )
        updates["gaps"] = []
        updates["unclosed_gaps"] = list(updates.get("unclosed_gaps")
                                        or state.get("unclosed_gaps") or []) + gaps
        trace.append(
            trace_event(NODE, "retire_gaps", count=len(gaps), why=why,
                        loops=f"{loops}/{MAX_RESEARCH_LOOPS}",
                        searches=f"{searches}/{MAX_SEARCHES}")
        )

    def out(route: str, why: str, **detail: Any) -> dict[str, Any]:
        return {
            **updates,
            "route": route,
            "trace": trace + [trace_event(NODE, "route", to=route, why=why, **detail)],
        }

    # 3. No draft yet, or the Writer returned an empty one. Bounded: a Writer that
    #    keeps failing must not spin the graph.
    #
    #    Without a bound this loops — empty draft, Critic skips, no critique, rule 4
    #    routes back to the Writer, same empty draft, until the recursion limit kills
    #    the run. Found by the contract suite rather than in production, which is the
    #    first time that has happened in this project.
    if draft is None:
        if writes >= MAX_WRITE_ATTEMPTS:
            return out("finalize", "writer produced no draft", degraded=True)
        res = out("writer", "no draft", attempt=f"{writes + 1}/{MAX_WRITE_ATTEMPTS}")
        res["write_attempts"] = writes + 1
        return res

    # 4. Draft exists but no critique. Two ways to get here: a Critic parse failure,
    #    or an upstream revision just completed (see rule 7, which clears the stale
    #    critique). Either way the Writer is the right next stop — it will rewrite
    #    against the new outline or evidence. This cannot loop: writer -> critic
    #    always produces a critique.
    if crit is None:
        # The upstream node has run by now, so its brief has been consumed. Clearing
        # it here rather than in the reader keeps one writer on the field.
        if state.get("revision_brief"):
            updates["revision_brief"] = None
        if writes >= MAX_WRITE_ATTEMPTS:
            return out("finalize", "write attempts exhausted", degraded=True)
        res = out("writer", "draft awaiting fresh critique",
                  attempt=f"{writes + 1}/{MAX_WRITE_ATTEMPTS}")
        res["write_attempts"] = writes + 1
        return res

    # 5. Critic passed.
    if crit.get("verdict") == "pass":
        scores = crit.get("scores") or {}
        return out("finalize", "critic passed",
                   min_score=min(scores.values()) if scores else None)

    # 6. Critic wants changes but the revision budget is spent. Ship degraded.
    #    Never loop forever, never return nothing.
    if revs >= MAX_REVISIONS:
        return out("finalize", "revision budget exhausted", degraded=True)

    # 7. Semantic retry, routed to whoever can actually fix the problem. Sending a
    #    structural complaint to the Writer produces cosmetic edits; the Analyst
    #    owns structure.
    #
    #    CRITIC_TARGETS is derived from ROUTES in state.py rather than written out
    #    here. A hardcoded tuple happens to hold today and stops holding the moment
    #    a node is added or renamed, which is Phase 2's error 10.
    target = crit.get("target", "writer")
    if target not in CRITIC_TARGETS:
        trace.append(trace_event(NODE, "retarget", requested=target, to="writer",
                                 why="not a routable repair target"))
        target = "writer"

    res = out(target, "critic requested revision",
              revision=f"{revs + 1}/{MAX_REVISIONS}")
    res["revision_count"] = revs + 1

    # Routing upstream (analyst/researcher) must clear the critique.
    #
    # The bug this fixes: the upstream node re-runs and control returns here, but
    # the OLD critique is still in state with verdict="revise". The supervisor
    # routes upstream again, burns the second revision, hits the cap, and ships the
    # ORIGINAL draft — the improved outline or new evidence never reaches the Writer
    # at all. The loop ran, cost tokens, and changed nothing.
    #
    # Clearing it means rule 4 catches the return trip and sends it to the Writer,
    # which is the only node that can turn a better plan into a better draft.
    #
    # But clearing it also deleted the criticism before the Analyst could read it —
    # its revision branch keys off critique["target"], so it re-planned from the same
    # findings and never learned what was wrong. Measured end to end: two full loops,
    # structural_coherence stuck at 1, section counts wandering 4-3-4-3, the run
    # shipped degraded. The same "ran, cost tokens, changed nothing" this clearing
    # was introduced to prevent, recreated from the other side.
    #
    # So the message travels on its own field and the routing signal travels on
    # `critique`. Same shape as the unaddressed_gaps fix: the thing that kept getting
    # wiped gets a field nobody else writes.
    if target in ("analyst", "researcher"):
        res["critique"] = None
        res["revision_brief"] = {
            "target": target,
            "issues": crit.get("issues") or [],
            "summary": crit.get("summary", ""),
            "scores": crit.get("scores") or {},
        }
    return res


def finalize(state: ReportState) -> dict[str, Any]:
    """Ship it.

    If the run ended with unresolved criticism or unclosed evidence gaps, append
    them as declared limitations rather than shipping silently as if it passed. An
    honest "here is what this report does not establish" is more useful than a
    clean-looking report that quietly omits the same information — and it is the
    difference between a bounded system and one that just gave up.
    """
    draft = (state.get("draft") or "").rstrip()
    crit = state.get("critique") or {}
    unclosed = state.get("unclosed_gaps") or []
    degraded = crit.get("verdict") == "revise"

    if degraded or unclosed:
        lines = ["", "", "---", "", "## Known limitations", ""]
        for issue in crit.get("issues", []):
            problem = (issue.get("problem") or "").strip()
            name = (issue.get("criterion") or "").replace("_", " ")
            if problem:
                lines.append(f"- {problem} ({name})")
        for g in unclosed:
            lines.append(f"- Evidence gap not closed within the search budget: {g}")
        draft = draft + "\n".join(lines)

    return {
        "draft": draft,
        "route": "done",
        "trace": [
            trace_event("finalize", "shipped", degraded=degraded,
                        unclosed_gaps=len(unclosed), words=len(draft.split()))
        ],
    }
