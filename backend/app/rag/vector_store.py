from functools import lru_cache

from langchain_chroma import Chroma

from app.core.config import get_settings
from app.llm.embeddings import get_embeddings


@lru_cache
def get_vector_store() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.chroma_dir),
        embedding_function=get_embeddings(),
    )