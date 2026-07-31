"""Enforce the suite's no-network guarantee structurally.

Section 16 of the Phase 2 write-up promises that every test in this directory runs
without an API call. That promise was checked by hand, and at one point checked
wrongly: running the suite with GEMINI_API_KEY set to a bad value proves nothing,
because llm.py calls load_dotenv(override=True) and .env beats the environment. The
suite looked offline and was not.

Blocking the client is the version that cannot be fooled. Any test that reaches a
real API fails, and says so, instead of quietly spending quota on every run.
"""

from __future__ import annotations

import pytest

from src.common import llm


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError(
            "this test tried to open a real API client. Tests in tests/ must run "
            "offline: patch the node's generate(), or put the test in scripts/ where "
            "the cost is explicit and the operator chooses to pay it."
        )

    monkeypatch.setattr(llm, "client", blocked)
