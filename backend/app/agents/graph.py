import json
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agents.state import RAGState
from app.agents.tools import current_time, get_weather, search_documents
from app.llm.model import get_chat_model
from app.llm.prompts import ANSWER_PROMPT, SYSTEM_PROMPT
from app.schemas.chat import ChatResponse, Source

NO_CONTEXT = "I could not find relevant information in the uploaded documents."


class ReasoningFilter:
    def __init__(self) -> None:
        self.pending = ""
        self.closing: str | None = None

    def feed(self, text: str, final: bool = False) -> str:
        self.pending += text
        output = ""
        while self.pending:
            markers = [self.closing] if self.closing else ["<think>", "<analysis>"]
            positions = [(self.pending.find(marker), marker) for marker in markers]
            positions = [(index, marker) for index, marker in positions if index >= 0]
            if positions:
                index, marker = min(positions)
                if self.closing is None:
                    output += self.pending[:index]
                    self.closing = "</think>" if marker == "<think>" else "</analysis>"
                else:
                    self.closing = None
                self.pending = self.pending[index + len(marker):]
                continue
            keep = 0 if final else max(
                (length for marker in markers for length in range(1, len(marker))
                 if self.pending.endswith(marker[:length])),
                default=0,
            )
            if self.closing is None:
                output += self.pending[:-keep] if keep else self.pending
            self.pending = self.pending[-keep:] if keep else ""
            break
        return output


async def rag_agent(state: RAGState) -> dict:
    writer = get_stream_writer()
    agent = create_agent(
        model=get_chat_model(),
        tools=[search_documents, get_weather, current_time],
        system_prompt=SYSTEM_PROMPT.format(
            current_datetime=datetime.now().astimezone().isoformat()
        ),
    )
    tool_messages: list[ToolMessage] = []
    async for update in agent.astream(
        {"messages": [{"role": "user", "content": state["question"]}]},
        {"recursion_limit": 20},
        stream_mode="updates",
    ):
        for node in update.values():
            for message in node.get("messages", []):
                if isinstance(message, AIMessage):
                    for call in message.tool_calls:
                        writer({"type": "tool_start", "tool": call["name"]})
                elif isinstance(message, ToolMessage):
                    if message.status == "error":
                        raise RuntimeError(f"Tool failed: {message.name}")
                    writer({"type": "tool_end", "tool": message.name})
                    tool_messages.append(message)
    if not tool_messages:
        raise RuntimeError("The chat model did not call a tool")
    documents = [
        document
        for message in tool_messages if message.name == "search_documents"
        if isinstance(message.artifact, list)
        for document in message.artifact
        if isinstance(document, Document)
    ]
    sources = list(dict.fromkeys(
        (str(doc.metadata.get("source", "unknown")), doc.metadata.get("page"))
        for doc in documents
    ))
    source_models = [Source(source=source, page=page) for source, page in sources]
    writer({"type": "sources", "sources": [source.model_dump() for source in source_models]})
    return {
        "tool_results": [
            {"tool": message.name, "content": message.content} for message in tool_messages
        ],
        "sources": source_models,
        "has_context": bool(documents) or any(
            message.name in {"get_weather", "current_time"} for message in tool_messages
        ),
    }


async def generate_answer(state: RAGState) -> dict:
    writer = get_stream_writer()
    if not state["has_context"]:
        writer({"type": "token", "text": NO_CONTEXT})
        return {"response": ChatResponse(answer=NO_CONTEXT, sources=[])}
    prompt = ANSWER_PROMPT.format_messages(
        current_datetime=datetime.now().astimezone().isoformat(),
        question=state["question"],
        tool_results=json.dumps(state["tool_results"], ensure_ascii=False),
    )
    reasoning_filter = ReasoningFilter()
    parts = []
    async for chunk in get_chat_model().astream(prompt):
        text = reasoning_filter.feed(chunk.text)
        if text:
            parts.append(text)
            writer({"type": "token", "text": text})
    text = reasoning_filter.feed("", final=True)
    if text:
        parts.append(text)
        writer({"type": "token", "text": text})
    answer = "".join(parts)
    if not answer.strip():
        raise RuntimeError("The chat model returned no answer")
    return {"response": ChatResponse(answer=answer, sources=state["sources"])}


builder = StateGraph(RAGState)
builder.add_node("rag_agent", rag_agent)
builder.add_node("generate_answer", generate_answer)
builder.add_edge(START, "rag_agent")
builder.add_edge("rag_agent", "generate_answer")
builder.add_edge("generate_answer", END)
graph = builder.compile()


async def answer_question(question: str) -> ChatResponse:
    result = await graph.ainvoke({"question": question})
    return result["response"]
