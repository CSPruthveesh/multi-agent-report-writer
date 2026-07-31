from src.graph.nodes._stubs import analyst, critic, writer
from src.graph.nodes.researcher import researcher
from src.graph.nodes.supervisor import finalize, supervisor

STUBBED = ["analyst", "writer", "critic"]

__all__ = ["STUBBED", "analyst", "critic", "finalize", "researcher", "supervisor", "writer"]
