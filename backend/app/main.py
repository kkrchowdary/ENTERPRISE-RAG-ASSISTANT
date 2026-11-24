from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from app.api.routes import chat, documents, health
from app.core.logging import RequestLoggingMiddleware, configure_logging, log_operation
from app.database.connection import dispose_database, initialize_database

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with log_operation(logger, "application.startup"):
            await initialize_database()
        yield
    finally:
        with log_operation(logger, "application.shutdown"):
            await dispose_database()


app = FastAPI(title="RAG Assistant", lifespan=lifespan)
app.add_middleware(RequestLoggingMiddleware)
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)