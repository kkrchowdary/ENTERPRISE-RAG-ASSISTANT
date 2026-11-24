import asyncio
import tempfile
from pathlib import Path

from langchain_core.documents import Document

from app.rag.chunking import get_splitter
from app.rag.loaders import LoaderOrchestrator
from app.rag.vector_store import add_documents


class InvalidDocumentError(ValueError):
    pass


def load_chunks(filename: str, content: bytes) -> list[Document]:
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
    return [chunk for chunk in chunks if chunk.page_content.strip()]


async def ingest(filename: str, content: bytes) -> int:
    chunks = await asyncio.to_thread(load_chunks, filename, content)
    await add_documents(chunks)
    return len(chunks)