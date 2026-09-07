from typing import List

from langchain_core.tools import tool
from langchain_core.documents import Document

from app.services.qdrant import similarity_search
from app.config import get_settings


@tool
def vector_search(query: str) -> List[dict]:
    """
    Search the vector database for the most relevant Docker / containerization documents.
    Uses embedding similarity (cosine distance). Call this tool whenever factual information
    from the knowledge base is required.
    """
    settings = get_settings()
    docs: List[Document] = similarity_search(
        query=query,
        k=settings.top_k,
        score_threshold=settings.score_threshold,
    )

    results = []
    for d in docs:
        results.append(
            {
                "content": d.page_content,
                "source": d.metadata.get("source", "unknown"),
                "score": float(d.metadata.get("score", 0.0)),
                "metadata": {k: v for k, v in d.metadata.items() if k not in ("score",)},
            }
        )
    return results
