from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen3:4b"
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag"
    collection_name: str = "rag_minilm_v1"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"


@lru_cache
def get_settings() -> Settings:
    return Settings()