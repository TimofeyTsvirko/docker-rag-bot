from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ----- LLM -----
    llm_provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # ----- Embeddings -----
    embedding_provider: Literal["huggingface", "openai"] = "huggingface"
    hf_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # ----- Qdrant -----
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "docker_docs"
    qdrant_distance: str = "Cosine"

    # ----- RAG -----
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5
    score_threshold: float = 0.35

    # ----- App -----
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    data_dir: str = "./data/documents"


@lru_cache
def get_settings() -> Settings:
    return Settings()
