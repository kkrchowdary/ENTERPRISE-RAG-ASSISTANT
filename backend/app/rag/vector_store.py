import asyncio
import logging
from pathlib import Path
from uuid import UUID, uuid4

from langchain_core.documents import Document
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.config import get_settings
from app.core.logging import log_operation
from app.database.connection import get_session_factory
from app.database.models import Chunk, Collection, DocumentMetadata, StoredDocument
from app.llm.embeddings import get_embeddings, validate_embedding_inputs

logger = logging.getLogger(__name__)


def save_original(directory: Path, document_id: UUID, extension: str, content: bytes) -> str:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{document_id}{extension}"
    with path.open("xb") as file:
        file.write(content)
    return str(path)


async def add_documents(
    documents: list[Document], *, filename: str, content: bytes,
    page_count: int | None, extractor: str,
) -> UUID:
    settings = get_settings()
    with log_operation(logger, "embedding.documents"):
        embeddings = await asyncio.to_thread(get_embeddings)
        await asyncio.to_thread(validate_embedding_inputs, embeddings, documents)
        vectors = await embeddings.aembed_documents([doc.page_content for doc in documents])
        if len(vectors) != len(documents) or any(len(vector) != 384 for vector in vectors):
            raise ValueError("The embedding model must return one 384-dimensional vector per chunk")
    with log_operation(logger, "vector_store.write"):
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
            document_id = uuid4()
            extension = Path(filename).suffix.lower()
            storage_path = await asyncio.to_thread(
                save_original, settings.documents_dir, document_id, extension, content
            )
            document = StoredDocument(
                id=document_id, collection_name=settings.collection_name, name=filename,
                extension=extension, storage_path=storage_path, page_count=page_count,
                size_bytes=len(content),
                extraction_metadata=DocumentMetadata(extractor=extractor),
                chunks=[
                    Chunk(content=doc.page_content, metadata_json=doc.metadata, embedding=vector)
                    for doc, vector in zip(documents, vectors, strict=True)
                ],
            )
            session.add(document)
        logger.info("vector_store.inserted", extra={"chunk_count": len(documents)})
        return document_id