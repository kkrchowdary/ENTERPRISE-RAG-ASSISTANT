import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.agents.graph import answer_question, graph
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def validated_question(request: ChatRequest) -> str:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be blank")
    return question


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await answer_question(validated_question(request))


async def stream_events(question: str, request: Request):
    events = graph.astream({"question": question}, stream_mode="custom")
    try:
        async for event in events:
            if await request.is_disconnected():
                return
            yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        if not await request.is_disconnected():
            yield 'event: done\ndata: {"type":"done"}\n\n'
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Chat streaming failed")
        if not await request.is_disconnected():
            yield 'event: error\ndata: {"type":"error","message":"Chat failed; check server logs."}\n\n'
    finally:
        await events.aclose()


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request) -> StreamingResponse:
    question = validated_question(body)
    return StreamingResponse(
        stream_events(question, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )