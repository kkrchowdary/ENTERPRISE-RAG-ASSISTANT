from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.core.config import get_settings


@lru_cache
def get_reranker() -> CrossEncoder:
    return CrossEncoder(get_settings().reranker_model)


def rerank(query: str, documents: list[Document], limit: int = 4) -> list[Document]:
    if not documents:
        return []
    scores = get_reranker().predict([(query, document.page_content) for document in documents])
    ranked = sorted(zip(documents, scores, strict=True), key=lambda item: -float(item[1]))
    return [document for document, _ in ranked[:limit]]