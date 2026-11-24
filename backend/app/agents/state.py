from typing import TypedDict

from app.schemas.chat import ChatResponse, Source


class RAGState(TypedDict, total=False):
    question: str
    tool_results: list[dict]
    sources: list[Source]
    has_context: bool
    response: ChatResponse