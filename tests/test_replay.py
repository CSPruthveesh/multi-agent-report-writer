"""The replay endpoint, and the one assumption it rests on.

A recorded run has no per-event timing. What it has is a trace where each event names
the tokens its node spent, and a ledger listing every model call in order with its own
tokens and latency. The endpoint recovers per-event duration by walking the ledger and
cutting it wherever the running total matches the next event's figure.

That is an inference, not a stored fact, and it is only sound while the two agree
exactly. If a node ever writes a trace event whose token count is not the sum of a
contiguous run of its own ledger records, the replay would silently start showing
invented durations — which is the placeholder-price failure in a new place. So the
match is asserted rather than assumed, against every committed run.
"""

from __future__ import annotations

import json

import pytest

from api.main import _event_calls, _used, replay
from src.common.io import RESULTS, load_topics

RECORDED = [
    t["id"] for t in load_topics()
    if (RESULTS / "multiagent" / t["id"] / "run.json").exists()
]


def _run(topic_id: str) -> dict:
    p = RESULTS / "multiagent" / topic_id / "run.json"
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.mark.skipif(not RECORDED, reason="no recorded runs committed")
@pytest.mark.parametrize("topic_id", RECORDED)
def test_every_model_event_maps_onto_a_run_of_ledger_records(topic_id):
    """No event is left without a duration, and no record is left over."""
    run = _run(topic_id)
    trace = run["trace"]
    records = run["tokens"]["records"]

    calls = _event_calls(trace, records)
    spending = [i for i, e in enumerate(trace) if e.get("tokens")]

    assert set(calls) == set(spending), (
        "an event that spent tokens got no duration — the ledger did not line up"
    )
    assert sum(c["in"] + c["out"] for c in calls.values()) == run["tokens"]["total"], (
        "the recovered in/out split does not add up to the recorded total"
    )


@pytest.mark.skipif(not RECORDED, reason="no recorded runs committed")
@pytest.mark.parametrize("topic_id", RECORDED)
def test_the_replay_ends_on_the_numbers_the_record_states(topic_id):
    """The last frame has to agree with run.json, or the replay is fiction."""
    run = _run(topic_id)
    out = replay(topic_id)

    assert len(out["frames"]) == len(run["trace"])
    last = out["frames"][-1]
    assert last["cost"]["tokens"] == run["tokens"]["total"]
    assert last["counts"]["findings"] == run["findings_count"]
    assert last["budgets"]["searches"] == run["searches_used"]
    assert out["done"]["unclosed_gaps"] == run["unclosed_gaps"]
    # Provenance: the cap this run was made under, so the page can say when it differs
    # from the cap in force now. Backfilled by scripts/backfill_max_research_loops.py.
    assert out["max_research_loops"] is not None


@pytest.mark.skipif(not RECORDED, reason="no recorded runs committed")
def test_a_replay_frame_is_shaped_like_a_live_one():
    """The client drives both through one renderer. Divergence here is a broken page."""
    f = replay(RECORDED[0])["frames"][0]
    assert set(f) >= {"event", "cost", "budgets", "counts"}
    assert set(f["cost"]) == {"tokens", "usd", "multiple"}
    assert set(f["budgets"]) == {"searches", "research_loops", "revisions"}
    assert set(f["counts"]) == {"findings", "has_outline", "draft_words"}


def test_an_unrecorded_topic_reports_rather_than_raises():
    out = replay("nope")
    assert out["error"] and "frames" not in out


@pytest.mark.parametrize(
    ("value", "expected"),
    [("3/5", 3), ("2/2", 2), (None, None), ("", None), ("-", None), (0, 0)],
)
def test_budget_strings_parse_or_decline(value, expected):
    """"3/5" is a budget; "-" is the supervisor's placeholder for no value."""
    assert _used(value) == expected
