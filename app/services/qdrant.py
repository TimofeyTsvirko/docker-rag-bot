from functools import lru_cache
from typing import List, Optional
import logging

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings
from app.services.embeddings import get_embeddings

logger = logging.getLogger(__name__)


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        prefer_grpc=False,
    )


def ensure_collection(vector_size: int) -> None:
    """Create collection if it does not exist (cosine distance)."""
    settings = get_settings()
    client = get_qdrant_client()

    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )


def get_vector_store() -> QdrantVectorStore:
    settings = get_settings()
    embeddings = get_embeddings()

    dummy = embeddings.embed_query("dimension probe")
    ensure_collection(len(dummy))

    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )


def similarity_search(
    query: str,
    k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> List[Document]:
    """
    Vector search with cosine similarity via Qdrant.
    langchain-qdrant may return either similarity (higher=better) or distance (lower=better).
    We accept top-k and apply a soft filter.
    """
    settings = get_settings()
    store = get_vector_store()
    k = k or settings.top_k
    score_threshold = (
        score_threshold if score_threshold is not None else settings.score_threshold
    )

    results = store.similarity_search_with_score(query, k=k)
    logger.info(
        "raw scores for %r: %s",
        query,
        [(d.metadata.get("source"), float(s)) for d, s in results],
    )

    docs: List[Document] = []
    if not results:
        return docs

    scores = [float(s) for _, s in results]
    # Heuristic: if most scores are small (< 1.5) and lower seems better, treat as distance
    # Cosine distance in Qdrant is often in [0, 2]; similarity in [0, 1].
    looks_like_distance = max(scores) <= 2.0 and min(scores) < score_threshold and max(scores) > 1.0

    for doc, score in results:
        score = float(score)
        if looks_like_distance:
            # lower distance = better; keep if distance <= (1 - threshold) roughly or simply keep all top-k
            ok = score <= (2.0 - score_threshold)
        else:
            # similarity: higher = better
            ok = score >= score_threshold or score_threshold <= 0

        if ok:
            doc.metadata["score"] = score
            docs.append(doc)

    # If threshold wiped everything, still return top-k (better than empty)
    if not docs and results:
        logger.warning("score threshold filtered all docs — returning raw top-%d", k)
        for doc, score in results:
            doc.metadata["score"] = float(score)
            docs.append(doc)

    return docs
