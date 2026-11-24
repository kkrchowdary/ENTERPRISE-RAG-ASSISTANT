import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk

import app.agents.graph as graph_module
import app.agents.tools as tools
import app.api.routes.chat as chat_route
import app.main as main_module
from app.main import app


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, tool_list, **kwargs):
        return self


class StreamingAnswer:
    async def astream(self, messages):
        for content in ["<thi", "nk>private reasoning</thi", "nk>Answer", " text."]:
            yield AIMessageChunk(content=content)


def no_database(monkeypatch):
    async def noop():
        pass

    monkeypatch.setattr(main_module, "initialize_database", noop)
    monkeypatch.setattr(main_module, "dispose_database", noop)


def parse_events(body):
    return [json.loads(part.split("data: ", 1)[1]) for part in body.strip().split("\n\n")]


def test_sse_streams_answer_and_safe_tool_progress(monkeypatch):
    no_database(monkeypatch)
    agent = ToolModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "current_time", "args": {}, "id": "time-1", "type": "tool_call",
        }]),
        AIMessage(content="Intermediate private reasoning must not appear."),
    ])
    models = iter([agent, StreamingAnswer()])
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: next(models))
    with TestClient(app) as client:
        response = client.post("/chat/stream", json={"question": "What time is it?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_events(response.text)
    assert [event["type"] for event in events] == [
        "tool_start", "tool_end", "sources", "token", "token", "done",
    ]
    assert "".join(event["text"] for event in events if event["type"] == "token") == "Answer text."
    assert events[2]["sources"] == []
    assert "private" not in response.text


def test_tokens_arrive_before_generation_finishes(monkeypatch):
    async def run():
        gate = asyncio.Event()

        class GatedAnswer:
            async def astream(self, messages):
                yield AIMessageChunk(content="first ")
                await gate.wait()
                yield AIMessageChunk(content="second")

        agent = ToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "current_time", "args": {}, "id": "time-1", "type": "tool_call",
            }]),
            AIMessage(content="intermediate"),
        ])
        models = iter([agent, GatedAnswer()])
        monkeypatch.setattr(graph_module, "get_chat_model", lambda: next(models))
        stream = graph_module.graph.astream({"question": "time"}, stream_mode="custom")
        try:
            while True:
                event = await anext(stream)
                if event["type"] == "token":
                    break
            assert event["text"] == "first "
            pending = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            assert not pending.done()
            gate.set()
            assert (await pending)["text"] == "second"
        finally:
            await stream.aclose()

    asyncio.run(run())


def test_stream_empty_retrieval_returns_no_context(monkeypatch):
    no_database(monkeypatch)

    async def empty_search(query):
        return []

    monkeypatch.setattr(tools, "search", empty_search)
    agent = ToolModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "search_documents", "args": {"query": "missing"},
            "id": "search-1", "type": "tool_call",
        }]),
        AIMessage(content="unsupported"),
    ])
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: agent)
    with TestClient(app) as client:
        response = client.post("/chat/stream", json={"question": "What is in the documents?"})
    events = parse_events(response.text)
    assert [event for event in events if event["type"] == "token"] == [
        {"type": "token", "text": graph_module.NO_CONTEXT}
    ]
    assert events[-1]["type"] == "done"


def test_stream_includes_document_sources(monkeypatch):
    no_database(monkeypatch)

    async def search(query):
        return [Document(page_content="Answer text.", metadata={"source": "doc.pdf", "page": 2})]

    monkeypatch.setattr(tools, "search", search)
    agent = ToolModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "search_documents", "args": {"query": "answer"},
            "id": "search-1", "type": "tool_call",
        }]),
        AIMessage(content="intermediate"),
    ])
    models = iter([agent, StreamingAnswer()])
    monkeypatch.setattr(graph_module, "get_chat_model", lambda: next(models))
    with TestClient(app) as client:
        response = client.post("/chat/stream", json={"question": "What is in the document?"})
    events = parse_events(response.text)
    assert next(event for event in events if event["type"] == "sources")["sources"] == [
        {"source": "doc.pdf", "page": 2}
    ]


def test_stream_errors_are_explicit_and_never_report_done(monkeypatch):
    no_database(monkeypatch)

    class BrokenGraph:
        async def astream(self, state, stream_mode):
            yield {"type": "tool_start", "tool": "get_weather"}
            raise RuntimeError("private failure detail")

    monkeypatch.setattr(chat_route, "graph", BrokenGraph())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/chat/stream", json={"question": "Weather?"})
    events = parse_events(response.text)
    assert [event["type"] for event in events] == ["tool_start", "error"]
    assert "private failure detail" not in response.text


def test_disconnect_closes_graph_stream(monkeypatch):
    closed = []

    class FakeGraph:
        async def astream(self, state, stream_mode):
            try:
                yield {"type": "tool_start", "tool": "current_time"}
                yield {"type": "token", "text": "not delivered"}
            finally:
                closed.append(True)

    class Request:
        calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return self.calls > 1

    async def run():
        return [event async for event in chat_route.stream_events("time", Request())]

    monkeypatch.setattr(chat_route, "graph", FakeGraph())
    events = asyncio.run(run())
    assert len(events) == 1
    assert closed == [True]


def test_cancellation_closes_graph_without_error_event(monkeypatch):
    closed = []

    class FakeGraph:
        async def astream(self, state, stream_mode):
            try:
                yield {"type": "tool_start", "tool": "current_time"}
                raise asyncio.CancelledError()
            finally:
                closed.append(True)

    class Request:
        async def is_disconnected(self):
            return False

    async def run():
        return [event async for event in chat_route.stream_events("time", Request())]

    monkeypatch.setattr(chat_route, "graph", FakeGraph())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    assert closed == [True]
