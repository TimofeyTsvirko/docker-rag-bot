import json
import logging
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from app.graph.state import AgentState, DocumentInfo
from app.graph.tools import vector_search
from app.services.llm import get_llm, get_llm_with_structured_output

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()


# ---------- Structured output schemas ----------

class ModerationResult(BaseModel):
    is_relevant: bool = Field(description="Whether the query is relevant to Docker/containers")
    reason: str = Field(description="Short explanation")


# ---------- Nodes ----------

def moderation_node(state: AgentState) -> dict:
    """Guardrail: filter out irrelevant queries."""
    llm = get_llm_with_structured_output(ModerationResult)
    system = _load_prompt("moderation")
    query = state["query"]

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=query),
    ]
    try:
        result: ModerationResult = llm.invoke(messages)
        return {
            "is_relevant": result.is_relevant,
            "moderation_reason": result.reason,
            "messages": [AIMessage(content=f"[moderation] relevant={result.is_relevant}: {result.reason}")],
        }
    except Exception as e:
        logger.exception("Moderation failed, defaulting to relevant")
        return {
            "is_relevant": True,
            "moderation_reason": f"fallback: {e}",
            "messages": [AIMessage(content="[moderation] fallback to relevant")],
        }


def rag_agent_node(state: AgentState) -> dict:
    """
    RAG agent that decides to call the vector_search tool.
    Uses tool-calling LLM.
    """
    llm = get_llm().bind_tools([vector_search])
    system = _load_prompt("rag_agent")

    # Build conversation context
    history = state.get("messages", [])
    messages = [SystemMessage(content=system)] + list(history)
    if not any(isinstance(m, HumanMessage) for m in history):
        messages.append(HumanMessage(content=state["query"]))

    response = llm.invoke(messages)
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Execute tools requested by the RAG agent (ToolNode equivalent)."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {"documents": state.get("documents", [])}

    documents: list[DocumentInfo] = []
    tool_messages = []

    for call in last.tool_calls:
        if call["name"] == "vector_search":
            query = call["args"].get("query", state["query"])
            results = vector_search.invoke({"query": query})
            for r in results:
                documents.append(
                    DocumentInfo(
                        content=r["content"],
                        source=r["source"],
                        score=r["score"],
                        metadata=r.get("metadata", {}),
                    )
                )
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(results, ensure_ascii=False),
                    tool_call_id=call["id"],
                )
            )
        else:
            tool_messages.append(
                ToolMessage(content=f"Unknown tool {call['name']}", tool_call_id=call["id"])
            )

    return {
        "documents": documents,
        "messages": tool_messages,
    }


def writer_node(state: AgentState) -> dict:
    """Form final answer from query + retrieved documents."""
    llm = get_llm()
    system = _load_prompt("writer")

    docs = state.get("documents") or []
    if docs:
        context_parts = []
        for i, d in enumerate(docs, 1):
            context_parts.append(
                f"[Document {i} | source: {d['source']} | score: {d['score']:.3f}]\n{d['content']}"
            )
        context = "\n\n".join(context_parts)
    else:
        context = "No relevant documents were retrieved."

    user_content = (
        f"User query: {state['query']}\n\n"
        f"Retrieved documents:\n{context}"
    )

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    answer = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)],
    }


def refuse_node(state: AgentState) -> dict:
    """Produce polite refusal for irrelevant queries."""
    reason = state.get("moderation_reason", "query is outside the knowledge domain")
    answer = (
        f"Извините, я могу отвечать только на вопросы, связанные с Docker, контейнерами "
        f"и оркестрацией. ({reason})\n\n"
        f"Sorry, I can only answer questions related to Docker, containers and orchestration. ({reason})"
    )
    return {
        "answer": answer,
        "documents": [],
        "messages": [AIMessage(content=answer)],
    }


# ---------- Routing ----------

def route_after_moderation(state: AgentState) -> Literal["rag_agent", "refuse"]:
    if state.get("is_relevant", False):
        return "rag_agent"
    return "refuse"


def route_after_rag(state: AgentState) -> Literal["tools", "writer"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "writer"
