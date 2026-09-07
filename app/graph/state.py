from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class DocumentInfo(TypedDict):
    content: str
    source: str
    score: float
    metadata: dict


class AgentState(TypedDict):
    """Internal state of the LangGraph multi-agent system."""

    # Conversation history (LangGraph message reducer)
    messages: Annotated[List[BaseMessage], add_messages]

    # Original user query
    query: str

    # Thread / session id for multi-turn
    thread_id: str

    # Moderation result
    is_relevant: bool
    moderation_reason: str

    # Retrieved documents (list of dicts for serializability)
    documents: List[DocumentInfo]

    # Final answer produced by writer
    answer: str

    # Intermediate flags
    needs_retrieval: bool
