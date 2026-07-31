from __future__ import annotations

import copy
import importlib
from types import SimpleNamespace

import pytest

from src.graph import nodes
from src.graph.build import GRAPH, build, run
from src.graph.state import (
    CRITIC_TARGETS,
    MAX_RESEARCH_LOOPS,
    MAX_REVISIONS,
    MAX_SEARCHES,
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
        ({"draft": "d"}, "finalize"),
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


@pytest.mark.parametrize("name", ["analyst", "writer", "critic", "supervisor", "finalize"])
def test_no_node_mutates_the_state_it_is_given(name, offline_nodes):
    state = initial_state("t")
    # outline matters: without it the Writer takes its no-outline path and the test
    # passes while exercising nothing. It did exactly that when first written.
    state.update(draft="d", critique=REVISE, findings=[FINDING], outline="## A [F001]")
    before = copy.deepcopy(state)
    getattr(nodes, name)(state)
    assert state == before
