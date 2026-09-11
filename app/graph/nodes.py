import json
import logging
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph.state import AgentState, DocumentInfo
from app.graph.tools import vector_search
from app.services.llm import get_llm, get_llm_with_structured_output

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()


class ModerationResult(BaseModel):
    is_relevant: bool = Field(description="Whether the query is relevant to Docker/containers")
    reason: str = Field(description="Short explanation")


def _docs_from_search(query: str) -> list[DocumentInfo]:
    results = vector_search.invoke({"query": query})
    documents: list[DocumentInfo] = []
    for r in results:
        documents.append(
            DocumentInfo(
                content=r["content"],
                source=r["source"],
                score=r["score"],
                metadata=r.get("metadata", {}),
            )
        )
    logger.info("vector_search query=%r -> %d docs", query, len(documents))
    return documents


def moderation_node(state: AgentState) -> dict:
    """Guardrail: filter out irrelevant queries."""
    llm = get_llm_with_structured_output(ModerationResult)
    system = _load_prompt("moderation")
    query = state["query"]

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=query),
    ]
    reset = {
        "documents": [],
        "answer": "",
        "needs_retrieval": True,
    }
    try:
        result: ModerationResult = llm.invoke(messages)
        return {
            **reset,
            "is_relevant": result.is_relevant,
            "moderation_reason": result.reason,
            "messages": [AIMessage(content=f"[moderation] relevant={result.is_relevant}: {result.reason}")],
        }
    except Exception as e:
        logger.exception("Moderation failed, defaulting to relevant")
        return {
            **reset,
            "is_relevant": True,
            "moderation_reason": f"fallback: {e}",
            "messages": [AIMessage(content="[moderation] fallback to relevant")],
        }


def rag_agent_node(state: AgentState) -> dict:
    """
    RAG agent: asks the LLM to call vector_search.
    If the model does not emit tool_calls, we still force a search with the user query.
    """
    llm = get_llm().bind_tools([vector_search])
    system = _load_prompt("rag_agent")
    query = state["query"]

    history: list = []
    for m in state.get("messages") or []:
        if isinstance(m, HumanMessage):
            history.append(m)
        elif isinstance(m, AIMessage) and m.content and not str(m.content).startswith("[moderation]"):
            # Skip pure tool-call messages that have no text
            if not getattr(m, "tool_calls", None) or (isinstance(m.content, str) and m.content.strip()):
                history.append(m)

    max_msgs = get_settings().history_max_messages
    history = history[-max_msgs:] if max_msgs > 0 else history

    messages = [SystemMessage(content=system)] + history
    if not history or not isinstance(history[-1], HumanMessage) or history[-1].content != query:
        messages.append(HumanMessage(content=query))

    response = llm.invoke(messages)

    # Force tool call if model answered without one
    if not getattr(response, "tool_calls", None):
        logger.warning("RAG agent did not call tools — forcing vector_search")
        documents = _docs_from_search(query)
        return {
            "messages": [response],
            "documents": documents,
        }

    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Execute tools requested by the RAG agent."""
    last = state["messages"][-1]
    documents: list[DocumentInfo] = list(state.get("documents") or [])
    tool_messages = []

    if isinstance(last, AIMessage) and last.tool_calls:
        for call in last.tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", "")

            if name == "vector_search":
                q = (args or {}).get("query") or state["query"]
                results = vector_search.invoke({"query": q})
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
                        content=json.dumps(results, ensure_ascii=False)[:8000],
                        tool_call_id=call_id,
                    )
                )
            else:
                tool_messages.append(
                    ToolMessage(content=f"Unknown tool {name}", tool_call_id=call_id)
                )
    elif not documents:
        # Safety net
        documents = _docs_from_search(state["query"])

    return {
        "documents": documents,
        "messages": tool_messages,
    }


def writer_node(state: AgentState) -> dict:
    """Form final answer from query + retrieved documents."""
    docs = state.get("documents") or []
    query = state.get("query", "")

    if not docs:
        if any("а" <= c.lower() <= "я" or c in "ёЁ" for c in query):
            answer = (
                "В базе знаний не нашлось релевантных документов по вашему запросу. "
                "Проверьте, что /ingest завершился успешно, и попробуйте переформулировать вопрос."
            )
        else:
            answer = (
                "No relevant documents were found in the knowledge base for your query. "
                "Ensure /ingest succeeded and try rephrasing the question."
            )
        return {
            "answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    llm = get_llm()
    system = _load_prompt("writer")

    context_parts = []
    for i, d in enumerate(docs, 1):
        context_parts.append(
            f"[Document {i} | source: {d['source']} | score: {d['score']:.3f}]\n{d['content']}"
        )
    context = "\n\n".join(context_parts)

    user_content = f"User query: {query}\n\nRetrieved documents:\n{context}"

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


def route_after_moderation(state: AgentState) -> Literal["rag_agent", "refuse"]:
    if state.get("is_relevant", False):
        return "rag_agent"
    return "refuse"


def route_after_rag(state: AgentState) -> Literal["tools", "writer"]:
    last = state["messages"][-1] if state.get("messages") else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    # Forced-search path in rag_agent_node already set documents for this turn
    if state.get("documents"):
        return "writer"
    # Safety: still go to tools so tool_node can run a fallback search
    return "tools"
