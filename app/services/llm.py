from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from app.config import get_settings


@lru_cache
def get_llm() -> BaseChatModel:
    """Return chat model according to current settings. Change provider/model via .env only."""
    settings = get_settings()

    if settings.llm_provider == "ollama":
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
        )

    # openai-compatible
    return ChatOpenAI(
        api_key=settings.openai_api_key or "sk-placeholder",
        base_url=settings.openai_base_url,
        model=settings.openai_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def get_llm_with_structured_output(schema: Any) -> BaseChatModel:
    """Convenience wrapper for structured output."""
    llm = get_llm()
    return llm.with_structured_output(schema)
