"""Node registry.

All five nodes are real. STUBBED is empty, which is the condition for trusting any
results this graph produces — check it before generating Phase 9 numbers.
"""

from src.graph.nodes.analyst import analyst
from src.graph.nodes.critic import critic
from src.graph.nodes.researcher import researcher
from src.graph.nodes.supervisor import finalize, supervisor
from src.graph.nodes.writer import writer

STUBBED: list[str] = []

__all__ = ["STUBBED", "analyst", "critic", "finalize", "researcher", "supervisor", "writer"]
