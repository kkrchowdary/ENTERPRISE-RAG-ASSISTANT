from fastapi import APIRouter, File, HTTPException, UploadFile

from app.llm.embeddings import EmbeddingInputTooLong
from app.rag.pipeline import InvalidDocumentError, ingest
from app.schemas.document import DocumentResponse

router = APIRouter()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@router.post("/documents", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentResponse:
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="A filename is required")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MiB")
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")
    try:
        chunks = await ingest(filename, content)
    except (InvalidDocumentError, EmbeddingInputTooLong) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DocumentResponse(filename=filename, chunks=chunks)