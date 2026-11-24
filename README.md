# Minimal RAG assistant

The backend indexes uploaded documents with LangChain and Chroma. An outer
LangGraph workflow runs a LangChain agent with a vector-retriever tool. Answers
use retrieved text and include source filenames and PDF page numbers (zero-based).
BM25, hybrid search, and reranking are not enabled.

## Setup

Use Python 3.12 and install [Ollama](https://ollama.com/download). From
`backend`, create a virtual environment, install dependencies, and start the API:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull qwen3:4b
.\.venv\Scripts\python -m uvicorn app.main:app --reload
```

Ollama must be running. The first upload downloads the embedding model from
Hugging Face. The defaults are
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
a small English model producing 384-dimensional vectors, and
[`qwen3:4b`](https://ollama.com/library/qwen3:4b), a 2.5 GB quantized chat model.
Qwen thinking is disabled for shorter responses. Actual speed and answer quality
depend on the hardware and documents.

## API

API documentation is available at `http://localhost:8000/docs`.

```powershell
curl.exe -F "file=@example.pdf" http://localhost:8000/documents
curl.exe -H "Content-Type: application/json" -d '{"question":"What does the document say?"}' http://localhost:8000/chat
```

`POST /documents` accepts one file, up to 10 MiB: PDF, TXT, Markdown, CSV,
PPTX, XLSX, XLS, or DOCX. It returns the filename and number of indexed chunks.
`POST /chat` accepts a question and returns an answer and a list of sources.
`GET /health` checks that the API is running, not that Ollama is available.

Only extracted chunks are saved in Chroma; uploaded originals are not retained.
The default `rag_minilm_v1` collection is separate from other collections, so
existing data is not deleted. Re-upload documents to index them in this
collection. If you change the embedding model, choose a new `COLLECTION_NAME`
and re-upload documents: embeddings from different models are not compatible.
Repeated uploads create additional chunks.

This local prototype has no authentication or tenant isolation. Do not expose it
to untrusted users or index sensitive documents without additional safeguards.