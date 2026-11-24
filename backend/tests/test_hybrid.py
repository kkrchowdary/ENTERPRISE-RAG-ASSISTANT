import asyncio
import json
import os
from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from sqlalchemy import delete, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import selectinload
from sqlalchemy.schema import CreateTable

import app.agents.graph as graph_module
import app.agents.tools as tools
import app.main as main_module
import app.rag.reranker as reranker
import app.rag.retriever as retriever
import app.rag.vector_store as vector_store
from app.core.config import get_settings
from app.database.connection import dispose_database, get_engine, get_session_factory, initialize_database
from app.database.models import Chunk, Collection, DocumentMetadata, StoredDocument
from app.llm.embeddings import EmbeddingInputTooLong
from app.main import app


def candidate(identifier: str, text: str):
    return identifier, Document(page_content=text, metadata={"source": identifier})


def test_bm25_and_reciprocal_rank_fusion():
    lexical_only = candidate("lexical", "special identifier xy123")
    semantic_only = candidate("semantic", "related meaning")
    shared = candidate("shared", "identifier xy123")
    assert retriever.bm25_rank("xy123", [semantic_only, lexical_only, shared])
    assert not retriever.bm25_rank("not-present", [semantic_only, lexical_only])
    fused = retriever.fuse([semantic_only, shared], [lexical_only, shared])
    assert fused[0].metadata["source"] == "shared"
    assert {doc.metadata["source"] for doc in fused} == {"lexical", "semantic", "shared"}


def test_cross_encoder_reranks_candidates(monkeypatch):
    class FakeReranker:
        def predict(self, pairs):
            assert pairs[0][0] == "question"
            return [0.1, 0.9]

    monkeypatch.setattr(reranker, "get_reranker", lambda: FakeReranker())
    docs = [candidate("a", "first")[1], candidate("b", "second")[1]]
    assert reranker.rerank("question", docs)[0].metadata["source"] == "b"
    assert reranker.rerank("question", []) == []


def test_retrieval_candidate_and_final_limits(monkeypatch):
    lexical = [candidate(f"lexical-{i}", f"term value {i}") for i in range(25)]
    semantic = [candidate(f"semantic-{i}", f"meaning {i}") for i in range(20)]
    lexical_ranked = retriever.bm25_rank("term", lexical)
    assert len(lexical_ranked) == 20
    fused = retriever.fuse(semantic, lexical_ranked)
    assert len(fused) == 20

    class FakeReranker:
        def predict(self, pairs):
            assert len(pairs) == 20
            return list(range(len(pairs)))

    monkeypatch.setattr(reranker, "get_reranker", lambda: FakeReranker())
    assert reranker.rerank("term", fused) == list(reversed(fused[-4:]))


@pytest.mark.parametrize("reranker_fails", [False, True])
def test_hybrid_reranking_reaches_generation_with_sources(monkeypatch, reranker_fails):
    settings = SimpleNamespace(embeddings_model="model", collection_name="collection")
    lexical = (uuid4(), "rare code xy123", {"source": "lexical.pdf", "page": 2})
    semantic = (uuid4(), "related concept", {"source": "semantic.txt"})
    statements = []
    prompts = []

    class FakeEmbeddings:
        async def aembed_query(self, query):
            assert query == "xy123"
            return [1.0] + [0.0] * 383

    class FakeReranker:
        def predict(self, pairs):
            assert pairs == [("xy123", "related concept"), ("xy123", "rare code xy123")]
            if reranker_fails:
                raise RuntimeError("model unavailable")
            return [0.1, 0.9]

    class Result:
        def __init__(self, rows):
            self.rows = rows

        def all(self):
            return self.rows

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def scalar(self, statement):
            return "model"

        async def execute(self, statement):
            compiled = statement.compile(dialect=postgresql.dialect())
            sql = str(compiled)
            if "ORDER BY" in sql:
                assert 20 in compiled.params.values()
            statements.append(sql)
            return Result([semantic] if "ORDER BY" in sql else [lexical, semantic])

    class CaptureModel(FakeMessagesListChatModel):
        def bind_tools(self, tool_list, **kwargs):
            return self

        def _generate(self, messages, *args, **kwargs):
            prompts.append(messages)
            return super()._generate(messages, *args, **kwargs)

    model = CaptureModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "search_documents", "args": {"query": "xy123"},
            "id": "search-1", "type": "tool_call",
        }]),
        AIMessage(content="The code is xy123."),
        AIMessage(content="The code is xy123."),
    ])
    monkeypatch.setattr(retriever, "get_settings", lambda: settings)
    monkeypatch.setattr(retriever, "get_embeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever, "get_session_factory", lambda: Session)
    monkeypatch.setattr(reranker, "get_reranker", lambda: FakeReranker())
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: model)
    if reranker_fails:
        with pytest.raises(RuntimeError, match="model unavailable"):
            asyncio.run(graph_module.answer_question("What is xy123?"))
        assert len(prompts) == 1
        return
    result = asyncio.run(graph_module.answer_question("What is xy123?"))
    assert result.answer == "The code is xy123."
    assert [(source.source, source.page) for source in result.sources] == [
        ("lexical.pdf", 2), ("semantic.txt", None),
    ]
    tool_results = json.loads(prompts[-1][-1].text.split("Tool results: ", 1)[1])
    ranked = json.loads(tool_results[0]["content"])
    assert [(item["source"], item["page"], item["text"]) for item in ranked] == [
        ("lexical.pdf", 2, "rare code xy123"),
        ("semantic.txt", None, "related concept"),
    ]
    assert "<=>" in statements[1]
    assert "rag_chunk_v2.embedding" not in statements[0]
    assert "JOIN rag_document" in statements[0]
    assert "rag_document.collection_name" in statements[1]

def test_reject_collection_with_different_embedding_model(monkeypatch):
    settings = SimpleNamespace(embeddings_model="new", collection_name="existing")

    class FakeEmbeddings:
        async def aembed_query(self, query):
            return [1.0] + [0.0] * 383

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def scalar(self, statement):
            return "old"

    monkeypatch.setattr(retriever, "get_settings", lambda: settings)
    monkeypatch.setattr(retriever, "get_embeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever, "get_session_factory", lambda: Session)
    with pytest.raises(ValueError, match="different embedding model"):
        asyncio.run(retriever.search("query"))


def test_ingestion_stores_vectors_and_checks_collection_model(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        embeddings_model="model", collection_name="collection", documents_dir=tmp_path
    )
    saved = []

    class FakeEmbeddings:
        _client = SimpleNamespace(
            max_seq_length=256,
            tokenizer=SimpleNamespace(encode=lambda text, **kwargs: [0] * (len(text.split()) + 2)),
        )

        async def aembed_documents(self, texts):
            return [[1.0] + [0.0] * 383 for _ in texts]

    class Session:
        model = "model"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def begin(self):
            return self

        async def execute(self, statement):
            assert "ON CONFLICT" in str(statement.compile(dialect=postgresql.dialect()))

        async def scalar(self, statement):
            return self.model

        def add(self, row):
            saved.append(row)

    monkeypatch.setattr(vector_store, "get_settings", lambda: settings)
    monkeypatch.setattr(vector_store, "get_embeddings", FakeEmbeddings)
    monkeypatch.setattr(vector_store, "get_session_factory", lambda: Session)
    doc = Document(page_content="content", metadata={"source": "file.txt"})
    document_id = asyncio.run(vector_store.add_documents(
        [doc], filename="file.txt", content=b"original", page_count=None, extractor="TextLoader"
    ))
    stored = saved[0]
    assert stored.id == document_id and stored.name == "file.txt" and stored.extension == ".txt"
    assert stored.size_bytes == 8 and stored.page_count is None
    assert stored.extraction_metadata.extractor == "TextLoader"
    assert stored.chunks[0].document is stored
    assert stored.chunks[0].content == "content"
    assert stored.chunks[0].metadata_json == {"source": "file.txt"}
    assert len(stored.chunks[0].embedding) == 384
    assert (tmp_path / f"{document_id}.txt").read_bytes() == b"original"
    assert stored.storage_path == str((tmp_path / f"{document_id}.txt").resolve())
    Session.model = "other"
    with pytest.raises(ValueError, match="different embedding model"):
        asyncio.run(vector_store.add_documents(
            [doc], filename="file.txt", content=b"original", page_count=None, extractor="TextLoader"
        ))
    assert len(list(tmp_path.iterdir())) == 1


def test_oversized_chunk_is_rejected_before_embedding(monkeypatch):
    class FakeEmbeddings:
        _client = SimpleNamespace(
            max_seq_length=256,
            tokenizer=SimpleNamespace(encode=lambda text, **kwargs: [0] * 257),
        )

        async def aembed_documents(self, texts):
            pytest.fail("Oversized chunks must not be embedded")

    monkeypatch.setattr(vector_store, "get_embeddings", FakeEmbeddings)
    with pytest.raises(EmbeddingInputTooLong, match="257 tokens.*256.*Reduce chunk_size"):
        asyncio.run(vector_store.add_documents(
            [Document(page_content="oversized", metadata={"source": "file.txt"})],
            filename="file.txt", content=b"original", page_count=None, extractor="TextLoader",
        ))


def test_document_and_chunk_schema_has_foreign_keys():
    dialect = postgresql.dialect()
    document_sql = str(CreateTable(StoredDocument.__table__).compile(dialect=dialect))
    metadata_sql = str(CreateTable(DocumentMetadata.__table__).compile(dialect=dialect))
    chunk_sql = str(CreateTable(Chunk.__table__).compile(dialect=dialect))
    assert "FOREIGN KEY(collection_name) REFERENCES rag_collection" in document_sql
    assert "FOREIGN KEY(document_id) REFERENCES rag_document" in metadata_sql
    assert "FOREIGN KEY(document_id) REFERENCES rag_document" in chunk_sql
    assert "VECTOR(384)" in chunk_sql and Chunk.__tablename__ == "rag_chunk_v2"


def test_database_failure_preserves_existing_and_saved_files(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        embeddings_model="model", collection_name="collection", documents_dir=tmp_path
    )
    existing = tmp_path / "existing.txt"
    existing.write_bytes(b"existing data")

    class FakeEmbeddings:
        _client = SimpleNamespace(
            max_seq_length=256,
            tokenizer=SimpleNamespace(encode=lambda text, **kwargs: [0] * 4),
        )

        async def aembed_documents(self, texts):
            return [[1.0] + [0.0] * 383 for _ in texts]

    class Transaction:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            raise RuntimeError("commit failed")

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def begin(self):
            return Transaction()

        async def execute(self, statement):
            pass

        async def scalar(self, statement):
            return "model"

        def add(self, document):
            pass

    monkeypatch.setattr(vector_store, "get_settings", lambda: settings)
    monkeypatch.setattr(vector_store, "get_embeddings", FakeEmbeddings)
    monkeypatch.setattr(vector_store, "get_session_factory", lambda: Session)
    with pytest.raises(RuntimeError, match="commit failed"):
        asyncio.run(vector_store.add_documents(
            [Document(page_content="content", metadata={"source": "file.txt"})],
            filename="file.txt", content=b"original", page_count=None, extractor="TextLoader",
        ))
    assert existing.read_bytes() == b"existing data"
    files = [path for path in tmp_path.iterdir() if path != existing]
    assert len(files) == 1 and files[0].read_bytes() == b"original"


def test_original_file_collision_never_overwrites(tmp_path):
    document_id = uuid4()
    existing = tmp_path / f"{document_id}.txt"
    existing.write_bytes(b"existing data")
    with pytest.raises(FileExistsError):
        vector_store.save_original(tmp_path, document_id, ".txt", b"new data")
    assert existing.read_bytes() == b"existing data"


class ToolChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tool_list, **kwargs):
        return self


def weather_model():
    return ToolChatModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "get_weather", "args": {"city": "London"},
            "id": "weather-1", "type": "tool_call",
        }]),
        AIMessage(content="It is 12 degrees Celsius in London."),
        AIMessage(content="It is 12 degrees Celsius in London."),
    ])


def test_weather_tool_and_agent(monkeypatch):
    async def no_database():
        pass

    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json={"results": [{
                "name": "London", "country": "UK", "latitude": 51.5, "longitude": -0.1,
            }]})
        return httpx.Response(200, json={
            "timezone": "Europe/London",
            "current": {"temperature_2m": 12, "wind_speed_10m": 5},
            "current_units": {"temperature_2m": "°C", "wind_speed_10m": "km/h"},
        })

    client_type = httpx.AsyncClient
    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda **kwargs: client_type(
        transport=httpx.MockTransport(respond), **kwargs
    ))
    monkeypatch.setattr(main_module, "initialize_database", no_database)
    monkeypatch.setattr(main_module, "dispose_database", no_database)
    model = weather_model()
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: model)
    with TestClient(app) as client:
        response = client.post("/chat", json={"question": "Weather in London?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "It is 12 degrees Celsius in London.", "sources": []}
    assert requests[0].url.params["name"] == "London"
    assert requests[1].url.params["current"] == "temperature_2m,wind_speed_10m"


def test_weather_service_error_is_not_silenced(monkeypatch):
    async def no_database():
        pass

    client_type = httpx.AsyncClient
    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda **kwargs: client_type(
        transport=httpx.MockTransport(lambda request: httpx.Response(503)), **kwargs
    ))
    monkeypatch.setattr(main_module, "initialize_database", no_database)
    monkeypatch.setattr(main_module, "dispose_database", no_database)
    model = weather_model()
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: model)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/chat", json={"question": "Weather in London?"}).status_code == 500


def test_current_time_is_passed_to_agent_prompt(monkeypatch):
    prompts = []

    class CaptureModel(ToolChatModel):
        def _generate(self, messages, *args, **kwargs):
            prompts.append(messages[0].text)
            return super()._generate(messages, *args, **kwargs)

    model = CaptureModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "current_time", "args": {}, "id": "time-1", "type": "tool_call",
        }]),
        AIMessage(content="Here is the current time."),
        AIMessage(content="Here is the current time."),
    ])
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: model)
    result = asyncio.run(graph_module.answer_question("What time is it?"))
    assert result.answer == "Here is the current time."
    assert datetime.now().astimezone().date().isoformat() in prompts[0]


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")
def test_real_postgres_roundtrip(monkeypatch, tmp_path):
    class FakeEmbeddings(Embeddings):
        _client = SimpleNamespace(
            max_seq_length=256,
            tokenizer=SimpleNamespace(encode=lambda text, **kwargs: [0] * (len(text.split()) + 2)),
        )

        def embed_documents(self, texts):
            return [[1.0] + [0.0] * 383 for _ in texts]

        def embed_query(self, text):
            return [1.0] + [0.0] * 383

    settings = get_settings()
    monkeypatch.setattr(settings, "database_url", os.environ["TEST_DATABASE_URL"])
    monkeypatch.setattr(settings, "collection_name", f"test_{uuid4().hex}")
    monkeypatch.setattr(settings, "documents_dir", tmp_path)
    monkeypatch.setattr(vector_store, "get_embeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever, "get_embeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever, "rerank", lambda query, docs: docs)
    get_engine.cache_clear()

    async def check():
        await initialize_database()
        try:
            document_id = await vector_store.add_documents(
                [Document(page_content="The sky is blue", metadata={"source": "test.txt"})],
                filename="test.txt", content=b"The sky is blue", page_count=None,
                extractor="TextLoader",
            )
            async with get_session_factory()() as session:
                stored = await session.scalar(
                    select(StoredDocument)
                    .options(selectinload(StoredDocument.chunks), selectinload(StoredDocument.extraction_metadata))
                    .where(StoredDocument.id == document_id)
                )
                assert stored is not None
                assert stored.name == "test.txt" and stored.size_bytes == len(b"The sky is blue")
                assert stored.extraction_metadata.document_id == document_id
                assert stored.chunks[0].document_id == document_id
                assert stored.chunks[0].id is not None
                assert (tmp_path / f"{document_id}.txt").read_bytes() == b"The sky is blue"
            result = await retriever.search("blue sky")
            assert result[0].page_content == "The sky is blue"
        finally:
            async with get_session_factory()() as session, session.begin():
                document_ids = select(StoredDocument.id).where(
                    StoredDocument.collection_name == settings.collection_name
                )
                await session.execute(delete(Chunk).where(Chunk.document_id.in_(document_ids)))
                await session.execute(
                    delete(DocumentMetadata).where(DocumentMetadata.document_id.in_(document_ids))
                )
                await session.execute(
                    delete(StoredDocument).where(StoredDocument.collection_name == settings.collection_name)
                )
                await session.execute(delete(Collection).where(Collection.name == settings.collection_name))
            await dispose_database()
            get_engine.cache_clear()

    asyncio.run(check())
