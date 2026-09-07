from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> Embeddings:
    """
    Return embedding model according to current settings.
    Change provider / model name via .env without touching code.
    """
    settings = get_settings()

    if settings.embedding_provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model,
            model_kwargs={"device": "cpu"},  # safe default; change to cuda if preferred
            encode_kwargs={"normalize_embeddings": True},
        )

    return OpenAIEmbeddings(
        api_key=settings.openai_api_key or "sk-placeholder",
        base_url=settings.openai_base_url,
        model=settings.openai_embedding_model,
    )
