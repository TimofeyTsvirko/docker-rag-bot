import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from app.config import get_settings
from app.graph.graph import get_graph
from app.services.ingestion import ingest_directory, ingest_files

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Docker RAG Multi-Agent Chatbot",
    description="LangGraph multi-agent RAG over Docker documentation (Qdrant)",
    version="1.0.0",
)


# ---------- Schemas ----------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    thread_id: Optional[str] = None


class DocumentOut(BaseModel):
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    answer: str
    documents: List[DocumentOut]
    thread_id: str
    is_relevant: bool


class IngestResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    ids_count: Optional[int] = None
    collection: Optional[str] = None


# ---------- Helpers ----------

def _run_graph(query: str, thread_id: str) -> Dict[str, Any]:
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "thread_id": thread_id,
    }
    result = graph.invoke(initial, config=config)
    return result


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Ordinary (non-streaming) user query."""
    thread_id = req.thread_id or str(uuid.uuid4())
    try:
        result = _run_graph(req.query, thread_id)
    except Exception as e:
        logger.exception("Graph failed")
        raise HTTPException(status_code=500, detail=str(e))

    docs = [
        DocumentOut(
            content=d["content"],
            source=d["source"],
            score=d["score"],
            metadata=d.get("metadata", {}),
        )
        for d in result.get("documents", [])
    ]
    return QueryResponse(
        answer=result.get("answer", ""),
        documents=docs,
        thread_id=thread_id,
        is_relevant=result.get("is_relevant", False),
    )


def _sse(event: str, data: Any) -> str:
    """Proper Server-Sent Event frame: event + JSON data."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """True token streaming of the writer answer via LangGraph astream_events + SSE."""
    thread_id = req.thread_id or str(uuid.uuid4())
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial = {
        "messages": [HumanMessage(content=req.query)],
        "query": req.query,
        "thread_id": thread_id,
    }

    async def event_generator():
        final_state: Dict[str, Any] = {}
        try:
            async for event in graph.astream_events(initial, config=config, version="v2"):
                kind = event.get("event")
                meta = event.get("metadata") or {}
                node = meta.get("langgraph_node")

                if kind == "on_chat_model_stream" and node == "writer":
                    chunk = (event.get("data") or {}).get("chunk")
                    content = getattr(chunk, "content", None) if chunk is not None else None
                    if content:
                        if isinstance(content, list):
                            text = "".join(
                                p.get("text", "") if isinstance(p, dict) else str(p)
                                for p in content
                            )
                        else:
                            text = str(content)
                        if text:
                            yield _sse("token", {"content": text})

                elif kind == "on_chain_end" and node == "writer":
                    # Capture writer output for final metadata
                    out = (event.get("data") or {}).get("output") or {}
                    if isinstance(out, dict):
                        final_state.update(out)

                elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                    # Full graph finished — merge final state if present
                    out = (event.get("data") or {}).get("output") or {}
                    if isinstance(out, dict):
                        final_state.update(out)

            docs = final_state.get("documents") or []
            yield _sse(
                "documents",
                [
                    {
                        "content": d.get("content", "") if isinstance(d, dict) else getattr(d, "content", ""),
                        "source": d.get("source", "") if isinstance(d, dict) else getattr(d, "source", ""),
                        "score": d.get("score", 0.0) if isinstance(d, dict) else getattr(d, "score", 0.0),
                    }
                    for d in docs
                ],
            )
            yield _sse(
                "done",
                {
                    "thread_id": thread_id,
                    "is_relevant": bool(final_state.get("is_relevant", False)),
                    "answer": final_state.get("answer", ""),
                },
            )
        except Exception as e:
            logger.exception("Streaming graph failed")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    clear_existing: bool = Form(False),
    files: Optional[List[UploadFile]] = File(None),
    directory: Optional[str] = Form(None),
):
    """
    Ingestion pipeline.
    - Upload files (multipart) OR
    - Point to a server-side directory (default: DATA_DIR from settings)
    """
    try:
        if files:
            # save uploaded files temporarily
            import tempfile
            from pathlib import Path

            tmp_dir = Path(tempfile.mkdtemp())
            saved = []
            for f in files:
                dest = tmp_dir / f.filename
                content = await f.read()
                dest.write_bytes(content)
                saved.append(str(dest))
            result = ingest_files(saved, clear_existing=clear_existing)
        else:
            result = ingest_directory(directory=directory, clear_existing=clear_existing)
        return IngestResponse(**result)
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
def startup():
    logger.info("Starting Docker RAG Multi-Agent service")
    # warm-up embeddings / collection if possible
    try:
        from app.services.qdrant import get_vector_store
        get_vector_store()
        logger.info("Qdrant collection ready")
    except Exception as e:
        logger.warning("Qdrant not ready yet: %s", e)
