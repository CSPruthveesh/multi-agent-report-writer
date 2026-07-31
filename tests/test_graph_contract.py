from __future__ import annotations

import copy
import importlib
from types import SimpleNamespace

import pytest

from src.graph import nodes
from src.graph.build import FREE_OVERRIDES, GRAPH, build, run
from src.graph.checkpoint import sqlite_checkpointer, thread_config
from src.graph.nodes import _fake
from src.graph.state import (
    CRITIC_TARGETS,
    MAX_RESEARCH_LOOPS,
    MAX_REVISIONS,
    MAX_SEARCHES,
    MAX_WRITE_ATTEMPTS,
    ROUTES,
    ReportState,
    initial_state,
)

REVISE = {"verdict": "revise", "target": "writer", "issues": []}
PASSED = {"verdict": "pass", "target": "writer", "issues": []}
FINDING = {"id": "F001", "claim": "a claim", "source_url": "https://x", "confidence": "high"}


def test_initial_state_covers_every_declared_field():
    assert set(initial_state("t")) == set(ReportState.__annotations__)


def test_critic_targets_derive_from_routes():
    assert set(CRITIC_TARGETS) <= set(ROUTES)
    assert "finalize" not in CRITIC_TARGETS


def test_conditional_edges_are_exactly_routes():
    g = GRAPH.get_graph()
    targets = {e.target for e in g.edges if e.conditional and e.source == "supervisor"}
    assert targets == set(ROUTES)


def test_build_rejects_an_unknown_override():
    with pytest.raises(ValueError, match="unknown node"):
        build(overrides={"reseacher": lambda s: {}})


def test_run_refuses_to_write_results_while_nodes_are_stubbed():
    if not nodes.STUBBED:
        pytest.skip("no stubs left; the gate is expected to be open")
    with pytest.raises(RuntimeError, match="still stubs"):
        run("t1", "a topic")


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"gaps": ["g"]}, "researcher"),
        ({"gaps": ["g"], "research_loops": MAX_RESEARCH_LOOPS}, "writer"),
        ({"gaps": ["g"], "searches_used": MAX_SEARCHES}, "writer"),
        ({}, "writer"),
        # A draft with no critique is now normal, not an error: routing upstream
        # clears the critique, and the Writer is the only node that turns a better
        # plan into a better draft.
        ({"draft": "d"}, "writer"),
        # An empty draft means the Writer declined. Retry it once...
        ({"draft": "   "}, "writer"),
        # ...but not forever.
        ({"draft": "   ", "write_attempts": MAX_WRITE_ATTEMPTS}, "finalize"),
        ({"draft": "d", "critique": PASSED}, "finalize"),
        ({"draft": "d", "critique": REVISE}, "writer"),
        ({"draft": "d", "critique": REVISE, "revision_count": MAX_REVISIONS}, "finalize"),
        ({"draft": "d", "critique": {**REVISE, "target": "analyst"}}, "analyst"),
        ({"draft": "d", "critique": {**REVISE, "target": "researcher"}}, "researcher"),
        ({"draft": "d", "critique": {**REVISE, "target": "editor"}}, "writer"),
        ({"draft": "d", "critique": {"verdict": "revise", "issues": []}}, "writer"),
    ],
)
def test_supervisor_routing_table(patch, expected):
    state = initial_state("t")
    state.update(patch)
    assert nodes.supervisor(state)["route"] == expected


def test_supervisor_retires_gaps_it_cannot_act_on_without_spending_a_revision():
    state = initial_state("t")
    state.update(gaps=["g"], searches_used=MAX_SEARCHES, draft="d", critique=REVISE)
    out = nodes.supervisor(state)
    assert out["unclosed_gaps"] == ["g"]
    assert out["gaps"] == []
    assert out["revision_count"] == 1


def test_finalize_declares_unclosed_gaps_and_ignores_stale_ones():
    state = initial_state("t")
    state.update(draft="# R\n\nbody.", gaps=["stale"], unclosed_gaps=["real"], critique=PASSED)
    draft = nodes.finalize(state)["draft"]
    assert "Known limitations" in draft
    assert "real" in draft
    assert "stale" not in draft


def test_finalize_leaves_a_clean_pass_untouched():
    state = initial_state("t")
    state.update(draft="# R\n\nbody.", critique=PASSED)
    assert nodes.finalize(state)["draft"] == "# R\n\nbody."


def test_routing_upstream_leaves_a_brief_the_analyst_can_read():
    """Clearing the critique must not delete the criticism along with it.

    The Analyst's revision branch keyed off critique["target"], and the supervisor
    clears `critique` when it routes upstream — so the message was deleted before the
    recipient read it. Measured end to end before the fix: the Analyst re-planned
    twice with revision=False, structural_coherence stayed at 1, and the run shipped
    degraded having spent both revisions.
    """
    issue = {"span": "s", "criterion": "structural_coherence",
             "problem": "sections do not connect", "fix": "add the causal link"}
    state = initial_state("t")
    state.update(draft="d",
                 critique={**REVISE, "target": "analyst", "issues": [issue]})

    out = nodes.supervisor(state)
    assert out["route"] == "analyst"
    assert out["critique"] is None
    assert out["revision_brief"]["target"] == "analyst"
    assert out["revision_brief"]["issues"] == [issue]


def test_the_analyst_acts_on_the_brief(offline_nodes):
    state = initial_state("t")
    state.update(
        findings=[FINDING],
        revision_brief={
            "target": "analyst",
            "issues": [{"span": "s", "criterion": "structural_coherence",
                        "problem": "sections do not connect", "fix": "add the link"}],
        },
    )
    ev = nodes.analyst(state)["trace"][0]
    assert ev["revision"] is True, "the Analyst did not see the brief it was sent"


def test_the_brief_reaches_the_analyst_through_the_graph():
    """The two tests above cover the halves. This covers the join.

    Every defect in this project that survived to a live run lived between
    components while both sides passed in isolation, so a criticism that leaves the
    supervisor and a node that reads one are not together evidence that the message
    arrives.
    """
    seen = []

    def researcher(state):
        return {"findings": [FINDING], "searches_used": MAX_SEARCHES, "gaps": [],
                "unaddressed_gaps": [], "trace": []}

    def analyst(state):
        seen.append(state.get("revision_brief"))
        return {"outline": "## A [F001]", "gaps": [], "trace": []}

    def writer(state):
        return {"draft": "# R\n\nbody [F001].", "trace": []}

    def critic(state):
        first = state.get("critique") is None
        return {
            "critique": {
                "scores": {"structural_coherence": 1},
                "verdict": "revise" if first else "pass",
                "target": "analyst",
                "issues": [{"span": "body", "criterion": "structural_coherence",
                            "problem": "no argument", "fix": "connect the sections"}],
                "summary": "structure",
            },
            "trace": [],
        }

    app = build(overrides={"researcher": researcher, "analyst": analyst,
                           "writer": writer, "critic": critic})
    app.invoke(initial_state("t"), config={"recursion_limit": 40})

    assert len(seen) >= 2, "the analyst was never re-run"
    assert seen[0] is None, "a brief existed before any criticism"
    assert seen[1], "the analyst was routed to and received nothing"
    assert seen[1]["issues"][0]["problem"] == "no argument"


def test_the_brief_is_cleared_once_consumed():
    """It exists only between routing upstream and the supervisor seeing the result."""
    state = initial_state("t")
    state.update(draft="d", revision_brief={"target": "analyst", "issues": []})
    out = nodes.supervisor(state)
    assert out["route"] == "writer"
    assert out["revision_brief"] is None


def test_routing_upstream_clears_the_critique():
    """Otherwise the stale verdict sends the run upstream twice and ships the original.

    The upstream node re-runs, control returns here, and the old critique is still in
    state saying "revise". Without clearing it the supervisor routes upstream again,
    burns the second revision, hits the cap and finalizes the draft that was never
    rewritten — the improved outline reaches nobody.
    """
    state = initial_state("t")
    state.update(draft="d", critique={**REVISE, "target": "analyst"})
    out = nodes.supervisor(state)
    assert out["route"] == "analyst"
    assert out["critique"] is None
    assert out["revision_count"] == 1


def test_routing_to_the_writer_keeps_the_critique():
    """The Writer needs the issues — they are what it turns into edits."""
    state = initial_state("t")
    state.update(draft="d", critique=REVISE)
    out = nodes.supervisor(state)
    assert out["route"] == "writer"
    assert "critique" not in out


def test_a_crashed_run_resumes_from_the_last_completed_node(tmp_path):
    """A checkpointer whose resume path has never executed is a backup nobody restored.

    Crash simulated with KeyboardInterrupt rather than an ordinary exception, because
    resilient() catches Exception and would degrade the node instead of killing the
    run — which is the point of it, and the wrong thing here.

    The assertion that matters is not that the resumed run finishes. It is that the
    nodes which already completed are not re-executed: replaying the Researcher would
    double the findings and spend the search budget twice, which on a real run is the
    difference between resuming and restarting.
    """
    calls = {"writer": 0}

    def crashing_writer(state):
        calls["writer"] += 1
        if calls["writer"] == 1:
            raise KeyboardInterrupt("simulated crash mid-run")
        return _fake.writer(state)

    cp = sqlite_checkpointer(tmp_path / "resume.sqlite")
    overrides = {**FREE_OVERRIDES, "writer": crashing_writer}
    cfg = thread_config("resume-test")

    with pytest.raises(KeyboardInterrupt):
        build(checkpointer=cp, overrides=overrides).invoke(initial_state("t"), config=cfg)

    # Work done before the crash is on disk.
    parked = build(checkpointer=cp, overrides=overrides).get_state(cfg)
    findings_at_crash = len(parked.values.get("findings") or [])
    assert findings_at_crash > 0, "nothing was checkpointed before the crash"

    # Resume: same thread, no new input, the writer works this time.
    final = build(checkpointer=cp, overrides=overrides).invoke(None, config=cfg)

    assert final["route"] == "done", "the resumed run did not finish"
    assert len(final["findings"]) == findings_at_crash, (
        "the Researcher re-ran on resume — that is a restart, not a resume"
    )
    # The node that crashed is re-entered; nothing upstream of it is. The exact count
    # is 1 crash plus however many times the revision loop calls the Writer, so the
    # property is "it ran again", not a number.
    assert calls["writer"] > 1, "the crashed node was never re-entered"
    assert final["revision_count"] == 1, "the revision loop did not survive the resume"


def test_the_free_routing_check_is_actually_free():
    """`python -m src.graph.build` is documented as free and has gone paid five times.

    Once per node that became real: the Researcher in Phase 2, the Analyst in 3, the
    Writer in 4, the Critic in 5. Each time the command kept printing "no API calls"
    while making them, and each time the remedy was to add one more double. _fake.py
    has carried a docstring telling the next person to do that since Phase 3, and it
    has not worked once — a written reminder is not a mechanism.

    conftest blocks llm.client(), so a node reaching the API fails here rather than on
    an invoice. FREE_OVERRIDES is imported rather than rebuilt, so this cannot pass
    while the command it describes does something else.
    """
    app = build(overrides=FREE_OVERRIDES)
    final = app.invoke(initial_state("t"), config={"recursion_limit": 40})

    assert final["route"] == "done", "the free path did not reach the end"
    models = {r.get("model") for r in (final.get("token_log") or [])}
    assert models <= {"fake"}, f"a real model was called on the free path: {models}"
    assert all(e.get("fake") or e["node"] in ("supervisor", "finalize")
               for e in final["trace"]), "a real node ran on the free path"


def test_an_empty_draft_is_reported_as_no_draft():
    """`"" or None` is about the record, not about termination.

    MAX_WRITE_ATTEMPTS bounds the spin on its own: without the empty-string check an
    empty draft falls past rule 3 to rule 4, which carries the same bound, so the run
    still ends. Verified by removing the check — chaos stayed 12/12 and the suite
    stayed green.

    What changes is what the trace says. Rule 4's reason is "draft awaiting fresh
    critique", which is false when there is no draft at all, and the trace is the
    artifact every failure in this project was diagnosed from.
    """
    state = initial_state("t")
    state.update(draft="   ")
    out = nodes.supervisor(state)
    reasons = [e.get("why") for e in out["trace"] if e.get("action") == "route"]
    assert reasons == ["no draft"], f"empty draft mis-reported as {reasons}"


def test_a_writer_that_never_produces_a_draft_terminates():
    """Routing back to a Writer that returns nothing must be bounded.

    Without a bound this spins: empty draft -> Critic skips -> no critique -> route
    back to the Writer -> same empty draft, until the recursion limit kills the run.
    The property is termination, not any particular number of attempts, so the test
    asserts the loop ends rather than asserting where.
    """
    state = initial_state("t")
    for step in range(MAX_WRITE_ATTEMPTS + 2):
        out = nodes.supervisor(state)
        if out["route"] == "finalize":
            assert step <= MAX_WRITE_ATTEMPTS, "took longer than the budget allows"
            return
        assert out["route"] == "writer"
        state.update(draft="", write_attempts=out["write_attempts"])
    raise AssertionError("supervisor never stopped routing to a failing Writer")


def test_supervisor_retires_unaddressed_gaps():
    state = initial_state("t")
    state.update(unaddressed_gaps=["could not afford this one"], searches_used=MAX_SEARCHES)
    out = nodes.supervisor(state)
    assert out["unclosed_gaps"] == ["could not afford this one"]
    assert out["unaddressed_gaps"] == []


def test_unaddressed_gaps_survive_the_analyst_and_reach_the_report():
    """The handover travels on its own field, because the Analyst is always in the way.

    START -> researcher -> analyst -> supervisor, on every path. Both the Researcher and
    the Analyst write `gaps`, and that field overwrites, so a gap handed back through it
    is wiped before the supervisor ever sees it. That is what happened: unclosed_gaps
    could never be non-empty in a real run and the report declared no limitations.

    The suite missed it because it tested the supervisor's retire logic directly, with
    the field already populated. Nothing checked that the value arrives.
    """
    def researcher(state):
        return {
            "gaps": [],
            "unaddressed_gaps": ["no budget left for this one"],
            "searches_used": MAX_SEARCHES,
            "trace": [],
        }

    def analyst(state):
        # What the real Analyst does on every path: writes gaps, wiping what was there.
        return {"outline": "an outline", "gaps": [], "trace": []}

    app = build(overrides={"researcher": researcher, "analyst": analyst})
    final = app.invoke(initial_state("t"), config={"recursion_limit": 40})

    assert final["unclosed_gaps"] == ["no budget left for this one"]
    assert "Known limitations" in final["draft"]
    assert "no budget left for this one" in final["draft"]


@pytest.fixture
def offline_nodes(monkeypatch):
    """Stop the model-calling nodes reaching the API from inside this suite.

    The registry binds each node's name to its function, which shadows the submodule,
    so importlib is the only way to reach the module and patch the generate() it
    closed over. That shadowing is recorded at the end of section 14 of the Phase 2
    write-up; it has now bitten twice, once per node that became real.
    """
    analyst_mod = importlib.import_module("src.graph.nodes.analyst")
    plan = analyst_mod.Analysis(
        sections=["Framing [F001]"],
        thesis="a thesis",
        tensions=[],
        gaps=[analyst_mod.Gap(missing="a missing thing", blocks="Framing")],
    )
    monkeypatch.setattr(analyst_mod, "generate",
                        lambda *a, **k: SimpleNamespace(parsed=plan))

    writer_mod = importlib.import_module("src.graph.nodes.writer")
    edits = writer_mod.Revision(edits=[writer_mod.Edit(find="d", replace="D", why="w")])
    monkeypatch.setattr(
        writer_mod, "generate",
        lambda *a, **k: SimpleNamespace(text="# R\n\nBody citing [F001].", parsed=edits),
    )

    critic_mod = importlib.import_module("src.graph.nodes.critic")
    verdict = critic_mod.RawCritique(
        scores=critic_mod.Scores(
            factual_grounding=4, structural_coherence=4, depth_of_analysis=4,
            citation_integrity=4, absence_of_filler=4,
        ),
        issues=[],
        summary="fine",
    )
    monkeypatch.setattr(critic_mod, "generate",
                        lambda *a, **k: SimpleNamespace(parsed=verdict))


@pytest.mark.parametrize("name", ["analyst", "writer", "critic", "supervisor", "finalize"])
def test_no_node_mutates_the_state_it_is_given(name, offline_nodes):
    state = initial_state("t")
    # outline matters: without it the Writer takes its no-outline path and the test
    # passes while exercising nothing. It did exactly that when first written.
    state.update(draft="d", critique=REVISE, findings=[FINDING], outline="## A [F001]")
    before = copy.deepcopy(state)
    getattr(nodes, name)(state)
    assert state == before
