"""SQLite checkpointing — durable state between supersteps.

A run costs 30,000 to 50,000 tokens and takes over a minute. Losing it to a crash
in the Critic and starting from the first search is the difference between a system
you would deploy and a script. The checkpointer writes state after every node, so
resume_run() replays from the last completed one rather than from the top.

This is also what makes an approval gate possible at all: an interrupt parks the
thread on disk and the process can exit while it waits.

Phase 2 section 5 is why this works without a migration. log_entries() converts
CallRecords into plain dicts on the way into state, and initial_state was verified
JSON-serialisable at the time — because an arbitrary Python object in a
checkpointed field is a problem you discover when a resumed run cannot be resumed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.common.io import ROOT

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DB = CHECKPOINT_DIR / "runs.sqlite"


def sqlite_checkpointer(path: Path | None = None):
    """A checkpointer backed by a file, not memory.

    check_same_thread=False because the API in Phase 6 serves requests from a
    thread pool and LangGraph touches the connection from whichever thread is
    running the superstep.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    db = path or CHECKPOINT_DB
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    return SqliteSaver(conn)


async def async_sqlite_checkpointer(path: Path | None = None):
    """The same file, reachable from an async graph.

    SqliteSaver is sync only — graph.astream() against one raises NotImplementedError
    and says so, which is how the API found out. The web demo streams, so it needs the
    async saver, and both write the same runs.sqlite: a thread parked by the API can be
    inspected by `--status` from the CLI and vice versa.

    A connection per caller rather than one shared. The two sides of a gated run are
    separate HTTP requests, so nothing can be held open between them anyway, and the
    durable state is the file rather than the handle.
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db = path or CHECKPOINT_DB
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db))
    saver = AsyncSqliteSaver(conn)
    # SqliteSaver creates its tables on first use; the async one does not, and a saver
    # asked to read a thread from a database with no schema is a confusing failure.
    await saver.setup()
    return saver


def describe(graph: Any, thread_id: str) -> dict[str, Any]:
    """What a saved thread is waiting on, without resuming it.

    `--status` exists because a parked thread is invisible otherwise: the process
    that started it has exited, and the only record is a row in a SQLite file. Being
    able to ask "what is this waiting for" before deciding whether to answer it is
    the difference between a durable pause and a lost run.
    """
    snap = graph.get_state(thread_config(thread_id))
    values = snap.values or {}
    pending = [
        getattr(itr, "value", None)
        for task in (getattr(snap, "tasks", ()) or ())
        for itr in (getattr(task, "interrupts", ()) or ())
    ]
    return {
        "thread_id": thread_id,
        "exists": bool(values),
        "next": list(snap.next or ()),
        "awaiting_human": bool(pending),
        "interrupt": pending[0] if pending else None,
        "findings": len(values.get("findings") or []),
        "tokens_spent": sum(
            r.get("total_tokens", 0) for r in (values.get("token_log") or [])
        ),
        "research_loops": values.get("research_loops", 0),
        "revisions": values.get("revision_count", 0),
        "has_draft": bool((values.get("draft") or "").strip()),
    }


def thread_config(thread_id: str) -> dict[str, Any]:
    """Every checkpointed invocation needs a thread id — it is the key the saved
    state is filed under, and the handle resume_run() uses to find it again."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
