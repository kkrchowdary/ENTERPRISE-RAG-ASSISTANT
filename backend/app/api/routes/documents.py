import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.llm.embeddings import EmbeddingInputTooLong
from app.rag.pipeline import InvalidDocumentError, ingest
from app.schemas.document import DocumentResponse

router = APIRouter()
logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post("/documents", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentResponse:
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not filename or filename in {".", ".."}:
        logger.warning("upload.rejected", extra={"reason": "missing_filename"})
        raise HTTPException(status_code=400, detail="A filename is required")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        logger.warning("upload.rejected", extra={"reason": "too_large"})
        raise HTTPException(status_code=413, detail="File exceeds 10 MiB")
    if not content:
        logger.warning("upload.rejected", extra={"reason": "empty"})
        raise HTTPException(status_code=400, detail="File is empty")
    logger.info("upload.received", extra={"size_bytes": len(content)})
    try:
        document_id, chunks = await ingest(filename, content)
    except (InvalidDocumentError, EmbeddingInputTooLong) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DocumentResponse(document_id=document_id, filename=filename, chunks=chunks)