from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.graph.state import AgentState
from app.graph.nodes import (
    moderation_node,
    rag_agent_node,
    tool_node,
    writer_node,
    refuse_node,
    route_after_moderation,
    route_after_rag,
)


def build_graph(checkpointer: bool = True):
    """
    Multi-agent graph:
      START -> moderation -> (rag_agent | refuse)
      rag_agent -> (tools | writer)
      tools -> writer
      writer / refuse -> END
    """
    builder = StateGraph(AgentState)

    builder.add_node("moderation", moderation_node)
    builder.add_node("rag_agent", rag_agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("writer", writer_node)
    builder.add_node("refuse", refuse_node)

    builder.set_entry_point("moderation")

    builder.add_conditional_edges(
        "moderation",
        route_after_moderation,
        {"rag_agent": "rag_agent", "refuse": "refuse"},
    )
    builder.add_conditional_edges(
        "rag_agent",
        route_after_rag,
        {"tools": "tools", "writer": "writer"},
    )
    builder.add_edge("tools", "writer")
    builder.add_edge("writer", END)
    builder.add_edge("refuse", END)

    memory = MemorySaver() if checkpointer else None
    return builder.compile(checkpointer=memory)


# Singleton for the service
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=True)
    return _graph
