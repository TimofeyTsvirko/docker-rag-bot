import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)

from app.config import get_settings
from app.services.qdrant import get_vector_store, ensure_collection
from app.services.embeddings import get_embeddings

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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


def ingest_directory(directory: Optional[str] = None, clear_existing: bool = False) -> dict:
    """
    Full ingestion pipeline:
    1. Load files from directory
    2. Chunk
    3. Embed + upsert into Qdrant
    """
    settings = get_settings()
    raw_docs = load_documents_from_dir(directory)
    if not raw_docs:
        return {"status": "empty", "documents": 0, "chunks": 0}

    chunks = chunk_documents(raw_docs)
    embeddings = get_embeddings()
    vector_size = len(embeddings.embed_query("probe"))
    ensure_collection(vector_size)

    store = get_vector_store()

    if clear_existing:
        # delete and recreate for simplicity
        client = store.client
        try:
            client.delete_collection(settings.qdrant_collection)
        except Exception:
            pass
        ensure_collection(vector_size)
        store = get_vector_store()

    # add documents (langchain handles embedding)
    ids = store.add_documents(chunks)

    return {
        "status": "ok",
        "documents": len(raw_docs),
        "chunks": len(chunks),
        "ids_count": len(ids),
        "collection": settings.qdrant_collection,
    }


def ingest_files(file_paths: List[str], clear_existing: bool = False) -> dict:
    """Ingest a list of explicit file paths."""
    docs: List[Document] = []
    for p in file_paths:
        path = Path(p)
        if path.exists() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(load_single_file(path))

    if not docs:
        return {"status": "empty", "documents": 0, "chunks": 0}

    chunks = chunk_documents(docs)
    embeddings = get_embeddings()
    vector_size = len(embeddings.embed_query("probe"))
    ensure_collection(vector_size)
    store = get_vector_store()

    if clear_existing:
        client = store.client
        try:
            client.delete_collection(get_settings().qdrant_collection)
        except Exception:
            pass
        ensure_collection(vector_size)
        store = get_vector_store()

    ids = store.add_documents(chunks)
    return {
        "status": "ok",
        "documents": len(docs),
        "chunks": len(chunks),
        "ids_count": len(ids),
    }
