import asyncio
import hashlib
import logging
import math
import os
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from langchain_core.documents import Document
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.config import get_settings
from app.core.logging import log_operation
from app.database.connection import get_session_factory
from app.database.models import Chunk, Collection, DocumentMetadata, Folder, StoredDocument
from app.llm.embeddings import get_embeddings, validate_embedding_inputs

logger = logging.getLogger(__name__)


def save_original(directory: Path, document_id: UUID, extension: str, content: bytes) -> str:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{document_id}{extension}"
    with path.open("xb") as file:
        file.write(content)
    return str(path)


async def embed_documents(documents: list[Document]) -> list[list[float]]:
    if not documents:
        raise ValueError("Document must contain at least one extractable chunk")
    with log_operation(logger, "embedding.documents"):
        embeddings = await asyncio.to_thread(get_embeddings)
        await asyncio.to_thread(validate_embedding_inputs, embeddings, documents)
        vectors = await embeddings.aembed_documents([doc.page_content for doc in documents])
        if (len(vectors) != len(documents)
                or any(len(vector) != 384 or not all(math.isfinite(x) for x in vector)
                       for vector in vectors)):
            raise ValueError("The embedding model must return one 384-dimensional vector per chunk")
        return vectors


async def store_documents(
    documents: list[Document], vectors: list[list[float]], *, filename: str, content: bytes,
    page_count: int | None, extractor: str,
) -> UUID:
    if (not documents or len(vectors) != len(documents)
            or any(len(vector) != 384 or not all(math.isfinite(x) for x in vector)
                   for vector in vectors)):
        raise ValueError("Document chunks and embeddings must be nonempty and aligned")
    settings = get_settings()
    document_embedding = [
        math.fsum(values) / len(vectors) for values in zip(*vectors, strict=True)
    ]
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
            folder_path = settings.documents_dir.resolve()
            folder_id = uuid5(NAMESPACE_URL, os.path.normcase(str(folder_path)))
            await session.execute(
                insert(Folder).values(id=folder_id, path=str(folder_path))
                .on_conflict_do_nothing(index_elements=[Folder.id])
            )
            storage_path = await asyncio.to_thread(
                save_original, settings.documents_dir, document_id, extension, content
            )
            chunks = []
            for index, (doc, vector) in enumerate(zip(documents, vectors, strict=True)):
                chunk_id = uuid4()
                content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
                metadata = {
                    **doc.metadata, "document_id": str(document_id), "chunk_id": str(chunk_id),
                    "chunk_index": index, "chunk_type": "text", "content_hash": content_hash,
                }
                chunks.append(Chunk(
                    id=chunk_id, content=doc.page_content, metadata_json=metadata,
                    embedding=vector, chunk_index=index, chunk_type="text",
                    content_hash=content_hash,
                ))
            document = StoredDocument(
                id=document_id, collection_name=settings.collection_name, name=filename,
                extension=extension, storage_path=storage_path, page_count=page_count,
                size_bytes=len(content), folder_id=folder_id, embedding=document_embedding,
                extraction_metadata=DocumentMetadata(extractor=extractor, metadata_json={
                    "content_hash": hashlib.sha256(content).hexdigest(),
                    "embedding_model": settings.embeddings_model, "chunk_count": len(chunks),
                    "name": filename, "extension": extension, "size_bytes": len(content),
                    "page_count": page_count, "extractor": extractor,
                }),
                chunks=chunks,
            )
            session.add(document)
        logger.info("vector_store.inserted", extra={"chunk_count": len(documents)})
        return document_id


async def add_documents(
    documents: list[Document], *, filename: str, content: bytes,
    page_count: int | None, extractor: str,
) -> UUID:
    vectors = await embed_documents(documents)
    return await store_documents(
        documents, vectors, filename=filename, content=content,
        page_count=page_count, extractor=extractor,
    )