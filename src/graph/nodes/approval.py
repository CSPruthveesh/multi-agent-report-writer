"""Approval gate — Phase 7.

    Reads:  outline, findings, gaps
    Writes: outline (if the human edits it), gaps (if they reject)
    Runs:   only when the graph is built with hitl=True

WHERE TO PUT A HUMAN, AND WHY HERE
----------------------------------
The gating decision is about reversibility and leverage, not about how "important"
a step feels.

  Outline approval        HIGH leverage, LOW review cost. Ten lines to read, and it
  (this node)             determines every downstream token. Catching a wrong frame
                          here costs one glance; catching it in the finished report
                          costs a full rewrite.

  Per-search approval     LOW leverage, HIGH review cost. Five interruptions per
  (deliberately absent)   run, each asking about a decision that is cheap to undo —
                          a bad search just wastes one round. Gating cheap reversible
                          actions is how HITL systems get switched off.

  Final report approval   Zero leverage. The work is already paid for. Approval at
  (deliberately absent)   the end is not a gate, it is a receipt.

The rule: gate where the decision is expensive to reverse and cheap to review.

OFF BY DEFAULT
--------------
The six-topic evaluation runs unattended, and a graph that blocks on human input
cannot be evaluated. So HITL is opt-in at build time and the approval node is only
added to the topology when requested — the graph genuinely has two shapes, and
export_mermaid can render both.

NOT WRAPPED IN resilient()
--------------------------
interrupt() works by raising, and langgraph's GraphInterrupt subclasses Exception.
A retry wrapper would catch the pause, re-run this node, catch the second interrupt
and degrade — the gate would never open, and the symptom would be a node that
mysteriously fails twice. build() leaves this node unwrapped and retry.py re-raises
GraphBubbleUp; either alone would do, and the failure is silent enough to deserve
both.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from src.graph.state import ReportState, trace_event

NODE = "approval"


def approval(state: ReportState) -> dict[str, Any]:
    """Pause for outline review.

    interrupt() raises out of the node, LangGraph checkpoints, and control returns to
    the caller. Resuming with Command(resume=payload) re-executes this node from the
    top with the payload as interrupt()'s return value.

    That re-execution is why nothing expensive belongs above the interrupt call —
    everything before it runs twice.
    """
    outline = state.get("outline") or ""
    findings = state.get("findings") or []
    gaps = state.get("gaps") or []

    decision = interrupt(
        {
            "kind": "outline_approval",
            "outline": outline,
            "finding_count": len(findings),
            "unclosed_gaps": list(state.get("unclosed_gaps") or []),
            "open_gaps": gaps,
            "prompt": "Approve this outline, edit it, or reject to send it back to "
                      "the analyst.",
            "options": ["approve", "edit", "reject"],
        }
    )

    # Resumed. Normalise, because a caller may send a bare string.
    if isinstance(decision, str):
        decision = {"action": decision}
    decision = decision or {}
    action = decision.get("action", "approve")

    if action == "edit" and decision.get("outline"):
        return {
            "outline": decision["outline"],
            "trace": [trace_event(NODE, "edited", by="human",
                                  chars=len(decision["outline"]))],
        }

    if action == "reject":
        # Send it back to the Analyst on revision_brief — the field Phase 5 built to
        # carry criticism upstream, and the one the Analyst's revision branch reads.
        #
        # It was gaps at first, on the reasoning that a gap already routes through the
        # supervisor and needs no new machinery. Running it showed two reasons that is
        # wrong. A rejection is not an evidence gap, so with the search budget spent
        # the supervisor retired it straight into unclosed_gaps and the reviewer's note
        # appeared in the report's Known limitations as missing evidence. And the
        # rejection never reached the supervisor at all, because approval's only edge
        # went to the Writer — see the conditional edge in build().
        #
        # `note` and `feedback` are both accepted because the CLI and the demo script
        # each had their own name for it. One field with two producers is the drift
        # this project keeps meeting; accepting both is the cheap fix, and the trace
        # records what arrived.
        note = (decision.get("note") or decision.get("feedback")
                or "Outline rejected by reviewer; rebuild it.")
        return {
            "outline": None,
            "revision_brief": {
                "target": "analyst",
                "issues": [{
                    "span": "",
                    "criterion": "structural_coherence",
                    "problem": f"A human reviewer rejected this outline: {note}",
                    "fix": "Rebuild the outline to address the objection. Do not simply "
                           "reorder the same sections.",
                }],
                "summary": note,
                "scores": {},
            },
            "trace": [trace_event(NODE, "rejected", by="human", note=note[:60])],
        }

    return {"trace": [trace_event(NODE, "approved", by="human")]}
