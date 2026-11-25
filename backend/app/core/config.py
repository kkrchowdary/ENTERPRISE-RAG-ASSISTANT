from functools import lru_cache
from pathlib import Path
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL")
    database_url: str = os.getenv("DATABASE_URL")
    documents_dir: Path = Path(os.getenv("DOCUMENTS_DIR"))
    collection_name: str = os.getenv("COLLECTION_NAME")
    reranker_model: str = os.getenv("RERANKER_MODEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()