from io import BytesIO
from uuid import UUID, uuid4

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi.testclient import TestClient
from docx import Document as WordDocument
from openpyxl import Workbook
from pptx import Presentation
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import xlwt

import app.agents.graph as graph_module
import app.agents.tools as tools
import app.main as main_module
import app.rag.chunking as chunking
import app.rag.pipeline as pipeline
from app.llm.embeddings import EmbeddingInputTooLong
from app.main import app


class ToolChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def model_with_search(answer: str = "The sky is blue.") -> ToolChatModel:
    return ToolChatModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "search_documents",
            "args": {"query": "What color is the sky?"},
            "id": "search-1",
            "type": "tool_call",
        }]),
        AIMessage(content=answer),
        AIMessage(content=answer),
    ])


def setup_store(monkeypatch, tmp_path):
    documents = []

    async def add_documents(chunks, **kwargs):
        documents.extend(chunks)
        return uuid4()

    async def search(query):
        return documents[:4]

    async def no_database():
        pass

    monkeypatch.setattr(pipeline, "add_documents", add_documents)
    monkeypatch.setattr(
        pipeline, "get_splitter",
        lambda: RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
    )
    monkeypatch.setattr(tools, "search", search)
    monkeypatch.setattr(main_module, "initialize_database", no_database)
    monkeypatch.setattr(main_module, "dispose_database", no_database)
    model = model_with_search()
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: model)
    return documents


def test_upload_retrieve_and_chat(monkeypatch, tmp_path):
    setup_store(monkeypatch, tmp_path)
    with TestClient(app) as client:
        upload = client.post(
            "/documents", files={"file": ("notes.txt", b"The sky is blue.", "text/plain")}
        )
        assert upload.status_code == 200
        assert upload.json()["filename"] == "notes.txt"
        assert upload.json()["chunks"] == 1
        assert UUID(upload.json()["document_id"])
        assert "path" not in upload.json()
        response = client.post("/chat", json={"question": "What color is the sky?"})
        assert response.status_code == 200
        assert response.json() == {
            "answer": "The sky is blue.",
            "sources": [{"source": "notes.txt", "page": None}],
        }


def test_empty_store_and_missing_tool_call(monkeypatch, tmp_path):
    setup_store(monkeypatch, tmp_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/chat", json={"question": "What color is the sky?"})
        assert response.json() == {"answer": graph_module.NO_CONTEXT, "sources": []}
        monkeypatch.setattr(
            graph_module, "get_chat_model",
            lambda: ToolChatModel(responses=[AIMessage(content="Unsupported claim")]),
        )
        response = client.post("/chat", json={"question": "What color is the sky?"})
        assert response.status_code == 500


def test_invalid_uploads_and_questions(monkeypatch, tmp_path):
    setup_store(monkeypatch, tmp_path)
    with TestClient(app) as client:
        assert client.post("/documents", files={"file": ("empty.txt", b"")}).status_code == 400
        assert client.post("/documents", files={"file": ("bad.xyz", b"text")}).status_code == 400
        assert client.post("/documents", files={"file": ("empty.txt", b"   ")}).status_code == 400
        assert client.post(
            "/documents", files={"file": ("large.txt", b"x" * (10 * 1024 * 1024 + 1))}
        ).status_code == 413
        assert client.post("/chat", json={"question": "   "}).status_code == 400
        assert client.post("/chat", json={"question": ""}).status_code == 422


def test_oversized_embedding_input_returns_actionable_error(monkeypatch, tmp_path):
    setup_store(monkeypatch, tmp_path)

    async def reject(chunks, **kwargs):
        raise EmbeddingInputTooLong("Reduce chunk_size in app/rag/chunking.py")

    monkeypatch.setattr(pipeline, "add_documents", reject)
    with TestClient(app) as client:
        response = client.post("/documents", files={"file": ("large.txt", b"words")})
    assert response.status_code == 400
    assert "Reduce chunk_size" in response.json()["detail"]


def test_pdf_page_count_includes_blank_pages(monkeypatch):
    class PDFLoader:
        pass

    class Loader:
        loaders = {"pdf": PDFLoader}

        def load_file(self, path):
            return [
                Document(page_content="first page", metadata={"page": 0}),
                Document(page_content="", metadata={"page": 1}),
                Document(page_content="third page", metadata={"page": 2}),
            ]

    monkeypatch.setattr(pipeline, "LoaderOrchestrator", Loader)
    monkeypatch.setattr(
        pipeline, "get_splitter",
        lambda: RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
    )
    chunks, pages, extractor = pipeline.load_chunks("document.pdf", b"pdf")
    assert pages == 3 and extractor == "PDFLoader"
    assert [chunk.metadata["page"] for chunk in chunks] == [0, 2]
    assert all(chunk.metadata["source"] == "document.pdf" for chunk in chunks)


def test_model_errors_are_not_reported_as_success(monkeypatch, tmp_path):
    setup_store(monkeypatch, tmp_path)

    def fail():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(graph_module, "get_chat_model", fail)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/chat", json={"question": "hello"}).status_code == 500


def test_legacy_excel_upload(monkeypatch, tmp_path):
    setup_store(monkeypatch, tmp_path)
    workbook = xlwt.Workbook()
    workbook.add_sheet("Sheet1").write(0, 0, "legacy spreadsheet")
    data = BytesIO()
    workbook.save(data)
    with TestClient(app) as client:
        response = client.post(
            "/documents", files={"file": ("sample.xls", data.getvalue())}
        )
        assert response.status_code == 200
        assert response.json()["chunks"] == 1


def test_supported_document_formats(monkeypatch, tmp_path):
    documents = setup_store(monkeypatch, tmp_path)
    spreadsheet = Workbook()
    spreadsheet.active["A1"] = "spreadsheet text"
    slides = Presentation()
    slides.slides.add_slide(slides.slide_layouts[1]).shapes.title.text = "slide text"
    word = WordDocument()
    word.add_paragraph("word text")
    files = [("notes.md", b"# Markdown text"), ("notes.csv", b"name\nAlice\n")]
    for filename, document in [
        ("sheet.xlsx", spreadsheet),
        ("slides.pptx", slides),
        ("word.docx", word),
    ]:
        data = BytesIO()
        document.save(data)
        files.append((filename, data.getvalue()))
    with TestClient(app) as client:
        for filename, content in files:
            response = client.post("/documents", files={"file": (filename, content)})
            assert response.status_code == 200, (filename, response.text)
            assert response.json()["chunks"] > 0
    for source, expected in {
        "notes.md": "Markdown text", "notes.csv": "Alice",
        "sheet.xlsx": "spreadsheet text", "slides.pptx": "slide text",
        "word.docx": "word text",
    }.items():
        assert any(
            doc.metadata["source"] == source and expected in doc.page_content
            for doc in documents
        )


def test_splitter_uses_embedding_tokenizer(monkeypatch):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    monkeypatch.setattr(chunking.AutoTokenizer, "from_pretrained", lambda name: tokenizer)
    chunking.get_splitter.cache_clear()
    try:
        chunks = chunking.get_splitter().split_text(" ".join(["word"] * 1500))
        assert len(chunks) > 1
        assert all(len(tokenizer.tokenize(chunk)) <= 1000 for chunk in chunks)
    finally:
        chunking.get_splitter.cache_clear()