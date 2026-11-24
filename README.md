# Local hybrid RAG assistant

The backend indexes uploaded documents in local PostgreSQL with pgvector. An
outer LangGraph workflow runs a LangChain agent with document search, current
weather, and current time tools. Document search combines pgvector cosine
similarity with BM25 using reciprocal-rank fusion, then reranks the candidates
with a local cross-encoder. Answers include source filenames and PDF page
numbers (zero-based).

## Setup

Use Python 3.12, a local PostgreSQL server with
[pgvector installed](https://github.com/pgvector/pgvector#installation), and
[Ollama](https://ollama.com/download). Create a PostgreSQL database named `rag`
and give the configured database user permission to create the `vector`
extension and tables. Alternatively, an administrator can create the extension
first. The application creates `rag_collection`, `rag_document`,
`rag_document_metadata`, and `rag_chunk_v2` tables on startup; database
failures prevent startup.

From `backend`, create a virtual environment, install dependencies, configure
the database URL, and start the API:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set DATABASE_URL to your local PostgreSQL credentials.
ollama pull qwen3:4b
.\.venv\Scripts\python -m uvicorn app.main:app --reload
```

Ollama and PostgreSQL must be running. The first upload downloads the embedding
model and the first search downloads the reranker from Hugging Face. Defaults:
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
a small English model producing 384-dimensional vectors;
[`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
for reranking; and [`qwen3:4b`](https://ollama.com/library/qwen3:4b), a 2.5 GB
quantized chat model. Qwen thinking is disabled. Speed and quality depend on
the hardware and documents.
No MCP server is configured, so the backend does not enable an MCP adapter.
Document extraction runs locally using `pdfplumber`, `docx2txt`, `xlrd`,
Python's text/CSV support, and Unstructured's local format-specific partitioners.
It does not require the sunset `langchain-community` package.
Recursive folder loading uses LangChain's `BaseLoader` interface and local
file traversal; LangChain's `DirectoryLoader` is in the sunset community package.

For a standalone extraction preview, run `py .\app\rag\loaders.py` from
`backend`. It discovers the nearest `.env` above the script, including the
backend `.env`; existing environment variables take precedence. `DOCUMENTS_DIR`
relative paths are resolved from `backend`, and the directory used is reported.
The output JSON is written in the current directory; an existing
`loaded_documents.json` is preserved by writing a uniquely named file instead.
The output contains extracted document text. No output file is created when
there is no extractable text.

For each document-search tool call, retrieval takes up to 20 pgvector cosine
neighbors and up to 20 BM25 keyword matches. Reciprocal-rank fusion combines
them (rank constant 60), keeps at most 20 candidates, and the cross-encoder
scores each question/chunk pair. The highest-scoring 4 chunks, or fewer if
fewer are available, are passed to generation with source metadata. Multiple
tool calls can each contribute up to 4 chunks. There is no relevance-score
cutoff. These retrieval limits are fixed in the code and independent of model
sampling; `top_p` is not a retrieval setting.

Set `RERANKER_MODEL` to another compatible Hugging Face cross-encoder if
needed. The model must be cached locally or downloadable on first use; a
download or inference failure fails the search instead of silently skipping
reranking. Ollama generation uses `temperature=0`; this application does not
set generation `top_k` or `top_p`, so Ollama/model defaults apply.

The current chunk configuration permits 1,000 tokenizer tokens, while MiniLM
accepts 256 tokens including special tokens. Uploads with oversized chunks are
rejected before indexing rather than silently truncated. Reduce `chunk_size`
in `backend/app/rag/chunking.py` (for example, to 200), ensure `chunk_overlap`
is smaller than the new chunk size, and re-upload the document.
BM25 currently reads and scores all chunks in the collection per search, and
pgvector uses exact search without an approximate index. This favors simplicity
over large-corpus performance.

## API

API documentation is available at `http://localhost:8000/docs`.

```powershell
curl.exe -F "file=@example.pdf" http://localhost:8000/documents
curl.exe -H "Content-Type: application/json" -d '{"question":"What does the document say?"}' http://localhost:8000/chat
```

`POST /documents` accepts one file, up to 10 MiB: PDF, TXT, Markdown, CSV,
PPTX, XLSX, XLS, or DOCX. It returns the document UUID, filename, and number
of indexed chunks. It does not expose the server's stored-file path.
`POST /chat` accepts a question and returns an answer and a list of sources.
`GET /health` checks that the API is running, not that Ollama is available.
The weather tool sends the requested city to
[Open-Meteo](https://open-meteo.com/en/terms) and requires internet access.
Check its usage terms before commercial use. The time tool uses the server's
local timezone.

`POST /chat/stream` accepts the same JSON as `/chat` and returns Server-Sent
Events (`text/event-stream`). Each event's JSON includes a `type`:
`tool_start` and `tool_end` include the tool name, `sources` includes the
source list, `token` contains answer text, and `done` marks completion. Errors
produce an `error` event instead of `done`; once streaming starts the HTTP
status remains 200. Client disconnects cancel the workflow.

```powershell
curl.exe -N -H "Content-Type: application/json" -d '{"question":"What time is it?"}' http://localhost:8000/chat/stream
```

Only tool activity is streamed as agent progress; private reasoning is not
intentionally exposed. The agent selects tools first, then a separate model
call streams the final answer, which adds latency compared with one model call.
Qwen3 4B supports Ollama streaming and tool calling.

Uploaded originals are retained under generated UUID filenames in
`backend/data/documents` by default; `DOCUMENTS_DIR` overrides this location.
The files contain the original, potentially sensitive content and require
appropriate filesystem access controls and backups. PostgreSQL stores document
names, extensions, internal paths, byte sizes, PDF page counts, extractor
metadata, and chunks linked to their document by foreign keys. Page counts for
other formats are unknown (`NULL`). Repeated uploads create new documents and
chunks. A failed or ambiguous database commit can leave an unreferenced original
file; investigate before manually removing any such file.

Existing Chroma data and the legacy PostgreSQL `rag_chunk` table remain
untouched but are not queried or migrated. Re-upload original documents to
index them in the new tables; confirm retrieval before removing any old data.
Collection metadata prevents mixing embedding models; choose a new
`COLLECTION_NAME` and re-upload when changing models. The database schema
requires 384-dimensional embeddings.

Run tests with `python -m pytest -q tests` from `backend`. To include the
PostgreSQL round-trip test, set `TEST_DATABASE_URL` to a disposable local test
database with pgvector available. The test creates a uniquely named collection
and removes its own rows afterward.

## Logs

Application logs are written as JSON lines to the console and
`backend/logs/app.log`. The file rotates at 5 MiB and retains three older files;
older entries are overwritten. This local rotation is intended for a single
server process. Each HTTP response includes an `X-Request-ID` matching its
request, ingestion, retrieval, tool, and streaming log entries. Logs include
status, timings, counts, and error types, but not request contents, document
text, prompts, filenames, credentials, or exception messages. HTTP client and
server access logs are suppressed in favor of these request summaries.

This local prototype has no authentication or tenant isolation. Do not expose it
to untrusted users or index sensitive documents without additional safeguards.