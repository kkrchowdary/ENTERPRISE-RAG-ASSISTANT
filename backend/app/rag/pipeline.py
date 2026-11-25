import asyncio
import logging
import tempfile
from pathlib import Path
from typing import TypedDict
from uuid import UUID

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.core.logging import log_operation
from app.rag.chunking import get_splitter
from app.rag.loaders import LoaderOrchestrator
from app.rag.vector_store import embed_documents, store_documents

logger = logging.getLogger(__name__)


class InvalidDocumentError(ValueError):
    pass


def load_source(filename: str, content: bytes) -> tuple[list[Document], int | None, str]:
    suffix = Path(filename).suffix.lower()
    loader = LoaderOrchestrator()
    if suffix.lstrip(".") not in loader.loaders:
        raise InvalidDocumentError(f"Unsupported file type: {suffix or '(none)'}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"upload{suffix}"
        path.write_bytes(content)
        documents = loader.load_file(str(path))
    for document in documents:
        document.metadata["source"] = filename
    page_count = len(documents) if suffix == ".pdf" else None
    return documents, page_count, loader.loaders[suffix.lstrip(".")].__name__


def split_source(documents: list[Document]) -> list[Document]:
    chunks = get_splitter().split_documents(documents)
    if not chunks or not any(chunk.page_content.strip() for chunk in chunks):
        raise InvalidDocumentError("File contains no extractable text")
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def load_chunks(filename: str, content: bytes) -> tuple[list[Document], int | None, str]:
    documents, page_count, extractor = load_source(filename, content)
    return split_source(documents), page_count, extractor


class IngestionState(TypedDict, total=False):
    filename: str
    content: bytes
    documents: list[Document]
    page_count: int | None
    extractor: str
    chunks: list[Document]
    vectors: list[list[float]]
    document_id: UUID


async def load_node(state: IngestionState) -> dict:
    with log_operation(logger, "ingestion.load"):
        documents, page_count, extractor = await asyncio.to_thread(
            load_source, state["filename"], state["content"]
        )
    return {"documents": documents, "page_count": page_count, "extractor": extractor}


async def split_node(state: IngestionState) -> dict:
    with log_operation(logger, "ingestion.split"):
        chunks = await asyncio.to_thread(split_source, state["documents"])
    logger.info("ingestion.chunks", extra={"chunk_count": len(chunks)})
    return {"chunks": chunks}


async def embed_node(state: IngestionState) -> dict:
    return {"vectors": await embed_documents(state["chunks"])}


async def store_node(state: IngestionState) -> dict:
    document_id = await store_documents(
        state["chunks"], state["vectors"], filename=state["filename"],
        content=state["content"], page_count=state["page_count"], extractor=state["extractor"],
    )
    return {"document_id": document_id}


ingestion_builder = StateGraph(IngestionState)
ingestion_builder.add_node("load", load_node)
ingestion_builder.add_node("split", split_node)
ingestion_builder.add_node("embed", embed_node)
ingestion_builder.add_node("store", store_node)
ingestion_builder.add_edge(START, "load")
ingestion_builder.add_edge("load", "split")
ingestion_builder.add_edge("split", "embed")
ingestion_builder.add_edge("embed", "store")
ingestion_builder.add_edge("store", END)
ingestion_graph = ingestion_builder.compile()

workflow_builder = StateGraph(IngestionState)
workflow_builder.add_node("ingestion", ingestion_graph)
workflow_builder.add_edge(START, "ingestion")
workflow_builder.add_edge("ingestion", END)
workflow = workflow_builder.compile()


async def ingest(filename: str, content: bytes) -> tuple[UUID, int]:
    with log_operation(logger, "ingestion"):
        result = await workflow.ainvoke({"filename": filename, "content": content})
        return result["document_id"], len(result["chunks"])