from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import chat, documents, health
from app.database.connection import dispose_database, initialize_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await initialize_database()
        yield
    finally:
        await dispose_database()


app = FastAPI(title="RAG Assistant", lifespan=lifespan)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)