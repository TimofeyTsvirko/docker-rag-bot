from functools import lru_cache
from typing import List, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings
from app.services.embeddings import get_embeddings


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

    # determine vector size
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
    """Pure vector search with cosine similarity (via Qdrant)."""
    settings = get_settings()
    store = get_vector_store()
    k = k or settings.top_k
    score_threshold = score_threshold if score_threshold is not None else settings.score_threshold

    # langchain-qdrant similarity_search_with_score returns (doc, score)
    results = store.similarity_search_with_score(query, k=k)

    docs = []
    for doc, score in results:
        # Qdrant cosine score is similarity (higher = better). Filter by threshold.
        if score >= score_threshold:
            doc.metadata["score"] = float(score)
            docs.append(doc)
    return docs
