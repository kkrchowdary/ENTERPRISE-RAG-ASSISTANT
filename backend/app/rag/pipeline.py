import asyncio
import logging
import tempfile
from pathlib import Path
from uuid import UUID

from langchain_core.documents import Document

from app.core.logging import log_operation
from app.rag.chunking import get_splitter
from app.rag.loaders import LoaderOrchestrator
from app.rag.vector_store import add_documents

logger = logging.getLogger(__name__)


class InvalidDocumentError(ValueError):
    pass


def load_chunks(filename: str, content: bytes) -> tuple[list[Document], int | None, str]:
    suffix = Path(filename).suffix.lower()
    loader = LoaderOrchestrator()
    if suffix.lstrip(".") not in loader.loaders:
        raise InvalidDocumentError(f"Unsupported file type: {suffix or '(none)'}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"upload{suffix}"
        path.write_bytes(content)
        documents = loader.load_file(str(path))
    for document in documents:
        page = document.metadata.get("page")
        document.metadata = {"source": filename}
        if isinstance(page, int):
            document.metadata["page"] = page
    chunks = get_splitter().split_documents(documents)
    if not chunks or not any(chunk.page_content.strip() for chunk in chunks):
        raise InvalidDocumentError("File contains no extractable text")
    page_count = len(documents) if suffix == ".pdf" else None
    return (
        [chunk for chunk in chunks if chunk.page_content.strip()],
        page_count,
        loader.loaders[suffix.lstrip(".")].__name__,
    )


async def ingest(filename: str, content: bytes) -> tuple[UUID, int]:
    with log_operation(logger, "ingestion"):
        with log_operation(logger, "ingestion.load_and_split"):
            chunks, page_count, extractor = await asyncio.to_thread(load_chunks, filename, content)
        logger.info("ingestion.chunks", extra={"chunk_count": len(chunks)})
        document_id = await add_documents(
            chunks, filename=filename, content=content, page_count=page_count, extractor=extractor
        )
        return document_id, len(chunks)