import asyncio

from langchain_core.documents import Document
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.config import get_settings
from app.database.connection import get_session_factory
from app.database.models import Chunk, Collection
from app.llm.embeddings import get_embeddings, validate_embedding_inputs


async def add_documents(documents: list[Document]) -> None:
    settings = get_settings()
    embeddings = await asyncio.to_thread(get_embeddings)
    await asyncio.to_thread(validate_embedding_inputs, embeddings, documents)
    vectors = await embeddings.aembed_documents([doc.page_content for doc in documents])
    if len(vectors) != len(documents) or any(len(vector) != 384 for vector in vectors):
        raise ValueError("The embedding model must return one 384-dimensional vector per chunk")
    async with get_session_factory()() as session, session.begin():
        await session.execute(
            insert(Collection)
            .values(name=settings.collection_name, embedding_model=settings.embeddings_model)
            .on_conflict_do_nothing(index_elements=[Collection.name])
        )
        model = await session.scalar(
            select(Collection.embedding_model).where(Collection.name == settings.collection_name)
        )
        if model != settings.embeddings_model:
            raise ValueError("Collection uses a different embedding model; choose a new collection")
        session.add_all([
            Chunk(collection_name=settings.collection_name, content=doc.page_content,
                  metadata_json=doc.metadata, embedding=vector)
            for doc, vector in zip(documents, vectors, strict=True)
        ])