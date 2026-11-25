import asyncio
import json
import logging
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

import app.agents.graph as graph_module
import app.agents.tools as tools
import app.api.routes.chat as chat_route
import app.core.logging as app_logging
import app.main as main_module
import app.rag.pipeline as pipeline
import app.rag.retriever as retriever
from app.main import app
from app.schemas.chat import ChatResponse


@pytest.fixture
def logs():
    entries = []

    class Capture(logging.Handler):
        def emit(self, record):
            entries.append(json.loads(self.format(record)))

    handler = Capture()
    handler.addFilter(app_logging.RequestContextFilter())
    handler.setFormatter(app_logging.JsonFormatter())
    logger = logging.getLogger("app")
    logger.addHandler(handler)
    try:
        yield entries
    finally:
        logger.removeHandler(handler)


def no_database(monkeypatch):
    async def noop():
        pass

    monkeypatch.setattr(main_module, "initialize_database", noop)
    monkeypatch.setattr(main_module, "dispose_database", noop)


def test_request_correlation_and_sensitive_data_not_logged(monkeypatch, logs):
    no_database(monkeypatch)
    secret = "PRIVATE_QUESTION_AND_TOKEN"

    async def answer(question):
        await asyncio.to_thread(logging.getLogger("app.test").info, "test.worker")
        return ChatResponse(answer=secret, sources=[])

    monkeypatch.setattr(chat_route, "answer_question", answer)
    with TestClient(app) as client:
        response = client.post(
            "/chat?token=" + secret,
            headers={"X-Request-ID": secret},
            json={"question": secret},
        )
    request_id = response.headers["X-Request-ID"]
    assert len(request_id) == 32 and request_id != secret
    events = [entry for entry in logs if entry.get("request_id") == request_id]
    assert {entry["event"] for entry in events} >= {
        "request.started", "test.worker", "request.completed"
    }
    completed = next(entry for entry in events if entry["event"] == "request.completed")
    assert completed["route"] == "/chat"
    assert completed["status_code"] == 200 and completed["completed"] is True
    assert completed["duration_ms"] >= 0
    assert secret not in json.dumps(logs)


def test_ingestion_logs_counts_without_filename_or_content(monkeypatch, logs):
    no_database(monkeypatch)
    secret = "PRIVATE_DOCUMENT_CONTENT"

    async def embed(chunks):
        return [[0.0] * 384 for chunk in chunks]

    async def save(chunks, vectors, **kwargs):
        assert chunks[0].page_content == secret
        return uuid4()

    monkeypatch.setattr(pipeline, "embed_documents", embed)
    monkeypatch.setattr(pipeline, "store_documents", save)
    monkeypatch.setattr(
        pipeline, "get_splitter",
        lambda: RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
    )
    with TestClient(app) as client:
        response = client.post(
            "/documents", files={"file": ("PRIVATE_FILENAME.txt", secret.encode())}
        )
    request_id = response.headers["X-Request-ID"]
    assert response.status_code == 200
    events = [entry for entry in logs if entry.get("request_id") == request_id]
    assert {entry["event"] for entry in events} >= {
        "upload.received", "ingestion.started", "ingestion.load.completed", "ingestion.split.completed",
        "ingestion.chunks", "ingestion.completed", "request.completed",
    }
    assert next(item for item in events if item["event"] == "ingestion.chunks")["chunk_count"] == 1
    assert secret not in json.dumps(logs)
    assert "PRIVATE_FILENAME" not in json.dumps(logs)


def test_retrieval_logs_stages_and_counts(monkeypatch, logs):
    settings = SimpleNamespace(embeddings_model="model", collection_name="collection")
    chunk = (uuid4(), "PRIVATE_RETRIEVED_CONTENT", {"source": "PRIVATE_FILENAME"})

    class Embeddings:
        async def aembed_query(self, query):
            return [1.0] + [0.0] * 383

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def scalar(self, statement):
            return "model"

        async def execute(self, statement):
            return SimpleNamespace(all=lambda: [chunk])

    monkeypatch.setattr(retriever, "get_settings", lambda: settings)
    monkeypatch.setattr(retriever, "get_embeddings", Embeddings)
    monkeypatch.setattr(retriever, "get_session_factory", lambda: Session)
    monkeypatch.setattr(retriever, "rerank", lambda query, documents: documents)
    token = app_logging.request_id.set("retrieval-test")
    try:
        asyncio.run(retriever.search("PRIVATE_QUERY"))
    finally:
        app_logging.request_id.reset(token)
    events = [entry for entry in logs if entry.get("request_id") == "retrieval-test"]
    assert {entry["event"] for entry in events} >= {
        "embedding.query.completed", "vector_store.search.completed",
        "retrieval.bm25.completed", "retrieval.candidates",
        "retrieval.rerank.completed", "retrieval.completed",
    }
    counts = next(item for item in events if item["event"] == "retrieval.candidates")
    assert counts["chunk_count"] == counts["semantic_count"] == counts["candidate_count"] == 1
    assert "PRIVATE_" not in json.dumps(logs)


def test_stream_logs_agent_tools_and_completion_without_content(monkeypatch, logs):
    no_database(monkeypatch)

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tool_list, **kwargs):
            return self

    class AnswerModel:
        async def astream(self, messages):
            yield AIMessageChunk(content="PRIVATE_ANSWER")

    agent = ToolModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "current_time", "args": {}, "id": "time-1", "type": "tool_call",
        }]),
        AIMessage(content="PRIVATE_INTERNAL_RESPONSE"),
    ])
    models = iter([agent, AnswerModel()])
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: next(models))
    with TestClient(app) as client:
        response = client.post("/chat/stream", json={"question": "PRIVATE_QUESTION"})
    assert response.status_code == 200
    events = [entry for entry in logs if entry.get("request_id") == response.headers["X-Request-ID"]]
    assert {entry["event"] for entry in events} >= {
        "stream.started", "agent.started", "agent.tool_started", "tool.current_time.completed",
        "agent.tool_completed", "agent.completed", "answer.generate.completed",
        "stream.completed", "request.completed",
    }
    assert next(item for item in events if item["event"] == "agent.tool_started")["tool"] == "current_time"
    assert "PRIVATE_" not in json.dumps(logs)


def test_failures_logged_safely_and_request_id_preserved(monkeypatch, logs):
    no_database(monkeypatch)
    secret = "postgresql://user:PRIVATE_PASSWORD@localhost/db"

    async def fail(question):
        raise RuntimeError(secret)

    monkeypatch.setattr(chat_route, "answer_question", fail)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/chat", json={"question": "PRIVATE_QUESTION"})
    assert response.status_code == 500
    events = [entry for entry in logs if entry.get("request_id") == response.headers["X-Request-ID"]]
    failed = next(entry for entry in events if entry["event"] == "request.failed")
    completed = next(entry for entry in events if entry["event"] == "request.completed")
    assert failed["error_type"] == "RuntimeError"
    assert completed["status_code"] == 500 and completed["completed"] is True
    assert secret not in json.dumps(logs)
    assert "PRIVATE_QUESTION" not in json.dumps(logs)


def test_startup_failure_is_logged_without_error_details(monkeypatch, logs):
    async def fail():
        raise RuntimeError("PRIVATE_DATABASE_CREDENTIAL")

    async def noop():
        pass

    monkeypatch.setattr(main_module, "initialize_database", fail)
    monkeypatch.setattr(main_module, "dispose_database", noop)
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass
    failure = next(entry for entry in logs if entry["event"] == "application.startup.failed")
    assert failure["error_type"] == "RuntimeError"
    assert any(entry["event"] == "application.shutdown.completed" for entry in logs)
    assert "PRIVATE_DATABASE_CREDENTIAL" not in json.dumps(logs)


def test_weather_failure_logs_type_not_city(monkeypatch, logs):
    client_type = httpx.AsyncClient
    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda **kwargs: client_type(
        transport=httpx.MockTransport(lambda request: httpx.Response(503)), **kwargs
    ))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(tools.get_weather.ainvoke({"city": "PRIVATE_CITY"}))
    failure = next(entry for entry in logs if entry["event"] == "tool.weather.failed")
    assert failure["error_type"] == "HTTPStatusError"
    assert "PRIVATE_CITY" not in json.dumps(logs)


def test_stream_failure_and_disconnect_are_logged(monkeypatch, logs):
    class BrokenGraph:
        async def astream(self, state, stream_mode):
            raise RuntimeError("PRIVATE_EXCEPTION")
            yield

    class Request:
        async def is_disconnected(self):
            return False

    monkeypatch.setattr(chat_route, "graph", BrokenGraph())
    events = asyncio.run(_collect_stream(Request()))
    assert "event: error" in events[0]
    failure = next(entry for entry in logs if entry["event"] == "stream.failed")
    assert failure["error_type"] == "RuntimeError"
    assert "PRIVATE_EXCEPTION" not in json.dumps(logs)

    class Disconnected:
        async def is_disconnected(self):
            return True

    class FakeGraph:
        async def astream(self, state, stream_mode):
            yield {"type": "token", "text": "PRIVATE_RESPONSE"}

    monkeypatch.setattr(chat_route, "graph", FakeGraph())
    assert asyncio.run(_collect_stream(Disconnected())) == []
    assert any(entry["event"] == "stream.disconnected" for entry in logs)
    assert "PRIVATE_RESPONSE" not in json.dumps(logs)


def test_closed_stream_is_logged(monkeypatch, logs):
    class FakeGraph:
        async def astream(self, state, stream_mode):
            yield {"type": "token", "text": "PRIVATE_RESPONSE"}
            yield {"type": "token", "text": "more"}

    class Request:
        async def is_disconnected(self):
            return False

    async def close_stream():
        stream = chat_route.stream_events("PRIVATE_QUESTION", Request())
        await anext(stream)
        await stream.aclose()

    monkeypatch.setattr(chat_route, "graph", FakeGraph())
    asyncio.run(close_stream())
    assert any(entry["event"] == "stream.closed" for entry in logs)
    assert "PRIVATE_" not in json.dumps(logs)


async def _collect_stream(request):
    return [event async for event in chat_route.stream_events("PRIVATE_QUESTION", request)]


def test_file_logging_rotates_and_does_not_duplicate_handlers(monkeypatch, tmp_path):
    logger = logging.getLogger("app")
    server_logger = logging.getLogger("uvicorn.error")
    old_handlers = [item for item in logger.handlers if isinstance(item, app_logging.RagLogFileHandler)]
    old_server_handlers = [item for item in old_handlers if item in server_logger.handlers]
    for handler in old_handlers:
        logger.removeHandler(handler)
        server_logger.removeHandler(handler)
    monkeypatch.setattr(app_logging, "LOG_FILE", tmp_path / "app.log")
    try:
        app_logging.configure_logging()
        app_logging.configure_logging()
        handlers = [item for item in logger.handlers if isinstance(item, app_logging.RagLogFileHandler)]
        assert len(handlers) == 1 and server_logger.handlers.count(handlers[0]) == 1
        assert handlers[0].maxBytes == 5 * 1024 * 1024
        assert handlers[0].backupCount == 3
        logging.getLogger("app.test").info("test.file_event", extra={"chunk_count": 2})
        try:
            raise RuntimeError("PRIVATE_EXCEPTION_CONTENT")
        except RuntimeError:
            server_logger.exception("Exception in ASGI application")
        handlers[0].flush()
        content = (tmp_path / "app.log").read_text(encoding="utf-8")
        entries = [json.loads(line) for line in content.splitlines()]
        assert [entry["event"] for entry in entries].count("test.file_event") == 1
        assert any("Server exception (RuntimeError)" in entry["event"] for entry in entries)
        assert "PRIVATE_EXCEPTION_CONTENT" not in content
        handlers[0].maxBytes = 250
        for _ in range(10):
            logging.getLogger("app.test").info("test.rotation")
        handlers[0].flush()
        assert (tmp_path / "app.log.1").exists()
        assert len(list(tmp_path.glob("app.log*"))) <= 4
    finally:
        for handler in list(logger.handlers):
            if isinstance(handler, app_logging.RagLogFileHandler):
                logger.removeHandler(handler)
                server_logger.removeHandler(handler)
                handler.close()
        for handler in old_handlers:
            logger.addHandler(handler)
        for handler in old_server_handlers:
            server_logger.addHandler(handler)
