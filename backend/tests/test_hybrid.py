import asyncio
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
from sqlalchemy import delete
from sqlalchemy.dialects import postgresql

import app.agents.graph as graph_module
import app.agents.tools as tools
import app.main as main_module
import app.rag.reranker as reranker
import app.rag.retriever as retriever
import app.rag.vector_store as vector_store
from app.core.config import get_settings
from app.database.connection import dispose_database, get_engine, get_session_factory, initialize_database
from app.database.models import Chunk, Collection
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


def test_search_combines_pgvector_and_bm25(monkeypatch):
    settings = SimpleNamespace(embeddings_model="model", collection_name="collection")
    lexical = (uuid4(), "rare code xy123", {"source": "lexical"})
    semantic = (uuid4(), "related concept", {"source": "semantic"})
    statements = []

    class FakeEmbeddings:
        async def aembed_query(self, query):
            return [1.0] + [0.0] * 383

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
            sql = str(statement.compile(dialect=postgresql.dialect()))
            statements.append(sql)
            return Result([semantic] if "ORDER BY" in sql else [lexical, semantic])

    monkeypatch.setattr(retriever, "get_settings", lambda: settings)
    monkeypatch.setattr(retriever, "get_embeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever, "get_session_factory", lambda: Session)
    monkeypatch.setattr(retriever, "rerank", lambda query, docs: docs)
    result = asyncio.run(retriever.search("xy123"))
    assert {doc.metadata["source"] for doc in result} == {"lexical", "semantic"}
    assert "<=>" in statements[1]
    assert "rag_chunk.embedding" not in statements[0]


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


def test_ingestion_stores_vectors_and_checks_collection_model(monkeypatch):
    settings = SimpleNamespace(embeddings_model="model", collection_name="collection")
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

        def add_all(self, rows):
            saved.extend(rows)

    monkeypatch.setattr(vector_store, "get_settings", lambda: settings)
    monkeypatch.setattr(vector_store, "get_embeddings", FakeEmbeddings)
    monkeypatch.setattr(vector_store, "get_session_factory", lambda: Session)
    doc = Document(page_content="content", metadata={"source": "file.txt"})
    asyncio.run(vector_store.add_documents([doc]))
    assert saved[0].content == "content"
    assert saved[0].metadata_json == {"source": "file.txt"}
    assert len(saved[0].embedding) == 384
    Session.model = "other"
    with pytest.raises(ValueError, match="different embedding model"):
        asyncio.run(vector_store.add_documents([doc]))


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
        asyncio.run(vector_store.add_documents([
            Document(page_content="oversized", metadata={"source": "file.txt"})
        ]))


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
def test_real_postgres_roundtrip(monkeypatch):
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
    monkeypatch.setattr(vector_store, "get_embeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever, "get_embeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever, "rerank", lambda query, docs: docs)
    get_engine.cache_clear()

    async def check():
        await initialize_database()
        try:
            await vector_store.add_documents([
                Document(page_content="The sky is blue", metadata={"source": "test.txt"})
            ])
            result = await retriever.search("blue sky")
            assert result[0].page_content == "The sky is blue"
        finally:
            async with get_session_factory()() as session, session.begin():
                await session.execute(delete(Chunk).where(Chunk.collection_name == settings.collection_name))
                await session.execute(delete(Collection).where(Collection.name == settings.collection_name))
            await dispose_database()
            get_engine.cache_clear()

    asyncio.run(check())
