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
        "is_relevant": False,
        "moderation_reason": "",
        "documents": [],
        "answer": "",
        "needs_retrieval": True,
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


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """Streaming response (token-by-token for the final answer)."""
    thread_id = req.thread_id or str(uuid.uuid4())
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial = {
        "messages": [HumanMessage(content=req.query)],
        "query": req.query,
        "thread_id": thread_id,
        "is_relevant": False,
        "moderation_reason": "",
        "documents": [],
        "answer": "",
        "needs_retrieval": True,
    }

    async def event_generator():
        try:
            # We stream only the final answer for simplicity.
            # Full event stream can be added later if needed.
            result = graph.invoke(initial, config=config)
            answer = result.get("answer", "")
            # naive token stream
            for token in answer.split(" "):
                yield f"data: {token} \n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
