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


def thread_config(thread_id: str) -> dict[str, Any]:
    """Every checkpointed invocation needs a thread id — it is the key the saved
    state is filed under, and the handle resume_run() uses to find it again."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
