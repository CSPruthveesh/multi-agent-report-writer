from __future__ import annotations

import operator
import re
from typing import Annotated, Any, TypedDict

SUFFIX_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Was 2. Lowered after the matched-pair experiment, which compared 0 against 2 across six
# topics and found the loop bought 77 findings, no citation gain distinguishable from
# run-to-run noise, the same number of unclosed gaps declared, and 36.7% more tokens. The
# report is the binding constraint, not the evidence: a 1,000-word report carries 30 to 40
# citations however much is gathered, so the loop's findings crowd out first-pass findings
# rather than adding to them.
#
# 1 rather than 0, and the distinction is a judgement rather than a measurement. The
# experiment tested 0 and 2; it did not test 1. What supports 1 specifically is weaker
# evidence — the two topics that ran a SECOND loop scored worst of the six with the judge
# — so this cuts the round that has no support while keeping the one the architecture was
# built around. Going to 0 is defensible on the data and is one flag away:
#
#     uv run python -m src.run --system multiagent --all --max-research-loops 0
MAX_RESEARCH_LOOPS = 1
MAX_REVISIONS = 2

# Guard against a failing Writer spinning the graph. Distinct from MAX_REVISIONS:
# that bounds how many times the Critic may ask for changes, this bounds how many
# times the supervisor will send a run to a Writer that keeps coming back empty.
MAX_WRITE_ATTEMPTS = 2

MAX_SEARCHES = 5

ROUTES = ("researcher", "analyst", "writer", "finalize")

CRITIC_TARGETS = tuple(r for r in ROUTES if r != "finalize")


class ReportState(TypedDict, total=False):
    topic: str

    findings: Annotated[list[dict[str, Any]], operator.add]
    searches_used: int
    research_loops: int
    # The gap-loop cap, carried in state rather than read from the module constant, so
    # that a run made with the loop disabled records the condition that produced it.
    # An experimental condition that is not in the artifact is one nobody can verify
    # afterwards — the same argument as code_version, one level up.
    max_research_loops: int
    # Where this arm's output lands: results/multiagent_<suffix>/. In state rather than
    # threaded as an argument because a resumed run persists through the same path as a
    # fresh one, and a thread parked before its arm was recorded would otherwise come
    # back into the wrong directory and overwrite the control.
    out_suffix: str
    write_attempts: int
    # Off by default so Phase 9 can batch six topics unattended. An approval gate
    # that cannot be turned off is an approval gate that blocks the evaluation.
    hitl: bool
    # Gaps the Researcher could not afford to search. It cannot write unclosed_gaps —
    # that field belongs to the supervisor — and it cannot hand them back through gaps,
    # because the Analyst sits between them on every path and overwrites that field.
    unaddressed_gaps: list[str]
    unclosed_gaps: list[str]

    outline: str | None
    gaps: list[str]
    draft: str | None
    critique: dict[str, Any] | None
    # What the Critic said, carried to an upstream node the supervisor routed to.
    # It cannot travel on `critique`: routing upstream has to clear that field, or
    # the stale verdict routes upstream again on the return trip. Clearing it also
    # deleted the message before the recipient read it, so the criticism gets a field
    # of its own — written by the supervisor, read by the Analyst, wiped by nobody.
    revision_brief: dict[str, Any] | None

    revision_count: int
    route: str

    token_log: Annotated[list[dict[str, Any]], operator.add]
    trace: Annotated[list[dict[str, Any]], operator.add]


def initial_state(
    topic: str,
    *,
    hitl: bool = False,
    max_research_loops: int = MAX_RESEARCH_LOOPS,
    out_suffix: str = "",
) -> ReportState:
    # Validated at the single choke point rather than at the CLI, so a programmatic
    # caller cannot get a directory name past it. This becomes a path segment; anything
    # with a separator or a dot in it would write outside results/<system>/.
    if out_suffix and not SUFFIX_RE.match(out_suffix):
        raise ValueError(
            f"out_suffix must match {SUFFIX_RE.pattern} — got {out_suffix!r}"
        )
    return ReportState(
        topic=topic,
        hitl=hitl,
        findings=[],
        searches_used=0,
        research_loops=0,
        max_research_loops=max_research_loops,
        out_suffix=out_suffix,
        write_attempts=0,
        unaddressed_gaps=[],
        unclosed_gaps=[],
        outline=None,
        gaps=[],
        draft=None,
        critique=None,
        revision_brief=None,
        revision_count=0,
        route="",
        token_log=[],
        trace=[],
    )


def trace_event(node: str, action: str, **detail: Any) -> dict[str, Any]:
    return {"node": node, "action": action, **detail}


def log_entries(ledger: Any, state: ReportState | None = None) -> list[dict[str, Any]]:
    """TokenLedger -> plain dicts for the token_log reducer, tagged with loop context.

    Nodes build a local ledger and dump it into shared state. Keeping the live ledger
    object out of state matters: state is checkpointed and serialised, and an
    arbitrary Python object in a checkpointed field is a problem discovered when a
    resumed run cannot be resumed.

    The loop tags are what make the cost analysis possible. Without them the project
    can say "multi-agent cost 26% more" and cannot say "the gap loop was 31% of the
    run" — and the second sentence is the one that attributes cost to a mechanism.

    Tags are read from the state the node SAW on entry, so a call made during the
    second research loop is attributed to loop 2 even though the supervisor moves the
    counter afterwards. state is optional so existing callers keep working; they just
    get everything filed under first_pass, which is wrong quietly rather than loudly
    and is why every caller passes it.
    """
    st = state or {}
    research_loop = st.get("research_loops", 0)
    revision = st.get("revision_count", 0)
    out = []
    for r in ledger.records:
        d = r.as_dict()
        d["research_loop"] = research_loop
        d["revision"] = revision
        # Coarse phase for the summary table. Evidence work and quality retries are
        # different products and adding them together destroys the only comparison
        # worth making.
        if research_loop > 0 and d["node"] in ("researcher", "analyst"):
            d["phase"] = "gap_loop"
        elif revision > 0:
            d["phase"] = "revision_loop"
        else:
            d["phase"] = "first_pass"
        out.append(d)
    return out
