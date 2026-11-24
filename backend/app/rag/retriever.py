import asyncio
import logging
import re
from collections import defaultdict
from uuid import UUID

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sqlalchemy import select

from app.core.config import get_settings
from app.core.logging import log_operation
from app.database.connection import get_session_factory
from app.database.models import Chunk, Collection, StoredDocument
from app.llm.embeddings import get_embeddings
from app.rag.reranker import rerank

Candidate = tuple[str, Document]
logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def bm25_rank(query: str, candidates: list[Candidate], limit: int = 20) -> list[Candidate]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    tokenized = [(candidate, tokenize(candidate[1].page_content)) for candidate in candidates]
    tokenized = [(candidate, tokens) for candidate, tokens in tokenized if tokens]
    if not tokenized:
        return []
    scores = BM25Okapi([tokens for _, tokens in tokenized]).get_scores(query_tokens)
    query_terms = set(query_tokens)
    matches = [
        (candidate, float(score))
        for (candidate, tokens), score in zip(tokenized, scores, strict=True)
        if query_terms.intersection(tokens)
    ]
    return [candidate for candidate, _ in sorted(matches, key=lambda item: -item[1])[:limit]]


def fuse(semantic: list[Candidate], lexical: list[Candidate]) -> list[Document]:
    scores: dict[str, float] = defaultdict(float)
    documents: dict[str, Document] = {}
    for ranking in (semantic, lexical):
        for position, (identifier, document) in enumerate(ranking, start=1):
            scores[identifier] += 1 / (60 + position)
            documents[identifier] = document
    return [documents[key] for key in sorted(scores, key=lambda key: -scores[key])[:20]]


def as_candidate(identifier: UUID, content: str, metadata: dict) -> Candidate:
    return str(identifier), Document(page_content=content, metadata=metadata)


async def search(query: str) -> list[Document]:
    with log_operation(logger, "retrieval"):
        settings = get_settings()
        with log_operation(logger, "embedding.query"):
            embeddings = await asyncio.to_thread(get_embeddings)
            vector = await embeddings.aembed_query(query)
            if len(vector) != 384:
                raise ValueError("The embedding model must return 384-dimensional vectors")
        with log_operation(logger, "vector_store.search"):
            async with get_session_factory()() as session:
                model = await session.scalar(
                    select(Collection.embedding_model).where(Collection.name == settings.collection_name)
                )
                if model is None:
                    logger.info("retrieval.empty")
                    return []
                if model != settings.embeddings_model:
                    raise ValueError("Collection uses a different embedding model; choose a new collection")
                columns = (Chunk.id, Chunk.content, Chunk.metadata_json)
                all_chunks = (await session.execute(
                    select(*columns)
                    .join(Chunk.document)
                    .where(StoredDocument.collection_name == settings.collection_name)
                )).all()
                semantic_chunks = (await session.execute(
                    select(*columns)
                    .join(Chunk.document)
                    .where(StoredDocument.collection_name == settings.collection_name)
                    .order_by(Chunk.embedding.cosine_distance(vector))
                    .limit(20)
                )).all()
        candidates = [as_candidate(*chunk) for chunk in all_chunks]
        semantic = [as_candidate(*chunk) for chunk in semantic_chunks]
        with log_operation(logger, "retrieval.bm25"):
            lexical = await asyncio.to_thread(bm25_rank, query, candidates)
        documents = fuse(semantic, lexical)
        logger.info("retrieval.candidates", extra={
            "chunk_count": len(candidates),
            "semantic_count": len(semantic),
            "lexical_count": len(lexical),
            "candidate_count": len(documents),
        })
        with log_operation(logger, "retrieval.rerank"):
            result = await asyncio.to_thread(rerank, query, documents)
        logger.info("retrieval.results", extra={"chunk_count": len(result)})
        return result