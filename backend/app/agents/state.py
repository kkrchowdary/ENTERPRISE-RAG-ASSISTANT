from typing import TypedDict

from app.schemas.chat import ChatResponse


class RAGState(TypedDict, total=False):
    question: str
    response: ChatResponse