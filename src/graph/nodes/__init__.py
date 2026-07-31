"""Node registry.

One import line per node. As each phase lands, move that node's import from
_stubs to its own module and delete it from _stubs.py. The `STUBBED` list is the
honest record of what is still fake — check it before trusting any results.
"""

from src.graph.nodes._stubs import critic
from src.graph.nodes.analyst import analyst
from src.graph.nodes.researcher import researcher
from src.graph.nodes.supervisor import finalize, supervisor
from src.graph.nodes.writer import writer

# Update this as phases land. If it is non-empty, results are not real.
STUBBED = ["critic"]

__all__ = ["STUBBED", "analyst", "critic", "finalize", "researcher", "supervisor", "writer"]
