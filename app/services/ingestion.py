import hashlib
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)
from qdrant_client.http import models as qmodels

from app.config import get_settings
from app.services.qdrant import get_vector_store, ensure_collection, get_qdrant_client
from app.services.embeddings import get_embeddings

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_chunk_id(doc: Document, chunk_index: int) -> str:
    """Deterministic point id so re-ingest of the same chunk upserts instead of duplicating."""
    source = doc.metadata.get("source", "")
    file_hash = doc.metadata.get("file_hash", "")
    content_key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:16]
    raw = f"{source}|{file_hash}|{chunk_index}|{content_key}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _delete_points_for_sources(sources: Sequence[str]) -> int:
    """Remove existing vectors for the given document sources."""
    if not sources:
        return 0
    settings = get_settings()
    client = get_qdrant_client()
    unique = list({s for s in sources if s})
    deleted = 0

    for key in ("metadata.source", "source"):
        try:
            client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchAny(any=unique),
                            )
                        ]
                    )
                ),
            )
            deleted = len(unique)
            logger.info("Deleted existing points for sources via %s: %s", key, unique)
            break
        except Exception as e:
            logger.debug("Delete by %s failed (%s): %s", key, unique, e)
            continue

    if not deleted:
        logger.warning(
            "Could not delete by source filter for %s — "
            "re-ingest may still rely on stable chunk ids (upsert)",
            unique,
        )
    return deleted


def load_single_file(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        try:
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            return []
    elif suffix == ".pdf":
        try:
            loader = PyPDFLoader(str(path))
            docs = loader.load()
        except Exception as e:
            logger.warning("Failed to load PDF %s: %s", path, e)
            return []
    else:
        return []

    for d in docs:
        d.metadata["source"] = str(path.name)
        d.metadata["file_path"] = str(path)
        d.metadata["file_hash"] = _file_hash(path)
    return docs


def load_documents_from_dir(directory: Optional[str] = None) -> List[Document]:
    settings = get_settings()
    root = Path(directory or settings.data_dir)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return []

    all_docs: List[Document] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            all_docs.extend(load_single_file(path))
    return all_docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def _ingest_chunks(chunks: List[Document], clear_existing: bool) -> dict:
    """Shared embed + write path with deduplication."""
    settings = get_settings()
    embeddings = get_embeddings()
    vector_size = len(embeddings.embed_query("probe"))
    ensure_collection(vector_size)
    store = get_vector_store()

    if clear_existing:
        client = store.client
        try:
            client.delete_collection(settings.qdrant_collection)
        except Exception:
            pass
        ensure_collection(vector_size)
        store = get_vector_store()
    else:
        # Avoid duplicates: drop previous vectors for the same sources, then upsert
        sources = [c.metadata.get("source", "") for c in chunks]
        _delete_points_for_sources(sources)

    ids = [_stable_chunk_id(c, i) for i, c in enumerate(chunks)]
    written = store.add_documents(chunks, ids=ids)

    return {
        "status": "ok",
        "chunks": len(chunks),
        "ids_count": len(written),
        "collection": settings.qdrant_collection,
    }


def ingest_directory(directory: Optional[str] = None, clear_existing: bool = False) -> dict:
    """
    Full ingestion pipeline:
    1. Load files from directory
    2. Chunk
    3. Embed + upsert into Qdrant (deduplicated by source + stable chunk ids)
    """
    raw_docs = load_documents_from_dir(directory)
    if not raw_docs:
        return {"status": "empty", "documents": 0, "chunks": 0}

    chunks = chunk_documents(raw_docs)
    result = _ingest_chunks(chunks, clear_existing=clear_existing)
    result["documents"] = len(raw_docs)
    return result


def ingest_files(file_paths: List[str], clear_existing: bool = False) -> dict:
    """Ingest a list of explicit file paths (deduplicated by source + stable chunk ids)."""
    docs: List[Document] = []
    for p in file_paths:
        path = Path(p)
        if path.exists() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(load_single_file(path))

    if not docs:
        return {"status": "empty", "documents": 0, "chunks": 0}

    chunks = chunk_documents(docs)
    result = _ingest_chunks(chunks, clear_existing=clear_existing)
    result["documents"] = len(docs)
    return result
