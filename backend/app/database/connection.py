import logging
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.database.models import Base

logger = logging.getLogger(__name__)


@lru_cache
def get_engine() -> AsyncEngine:
    return create_async_engine(get_settings().database_url, pool_pre_ping=True)


def get_session_factory() -> async_sessionmaker:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def initialize_database() -> None:
    async with get_engine().begin() as connection:
        await connection.execute(text("SELECT pg_advisory_xact_lock(435981203014)"))
        installed = await connection.scalar(text(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        ))
        if not installed:
            available = await connection.scalar(text(
                "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'vector')"
            ))
            if not available:
                message = "pgvector is unavailable; install pgvector on the PostgreSQL server"
                logger.error(message)
                raise RuntimeError(message)
            try:
                await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except SQLAlchemyError as exc:
                message = (
                    "Cannot enable pgvector; check its server installation, grant CREATE privileges, "
                    "or ask a database administrator to run CREATE EXTENSION vector"
                )
                logger.error(message)
                raise RuntimeError(message) from exc
        await connection.run_sync(Base.metadata.create_all)
        for statement in (
            "ALTER TABLE rag_document ADD COLUMN IF NOT EXISTS folder_id UUID REFERENCES rag_folder(id)",
            "ALTER TABLE rag_document ADD COLUMN IF NOT EXISTS embedding vector(384)",
            "ALTER TABLE rag_document_metadata ADD COLUMN IF NOT EXISTS metadata_json JSONB",
            "ALTER TABLE rag_chunk_v2 ADD COLUMN IF NOT EXISTS chunk_index INTEGER",
            "ALTER TABLE rag_chunk_v2 ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)",
            "ALTER TABLE rag_chunk_v2 ADD COLUMN IF NOT EXISTS chunk_type VARCHAR",
        ):
            await connection.execute(text(statement))
        for table in ("rag_chunk_v2", "rag_document_metadata"):
            constraint = f"{table}_document_id_fkey"
            action = await connection.scalar(text(
                "SELECT CAST(confdeltype AS text) FROM pg_constraint WHERE conrelid = "
                f"'{table}'::regclass AND conname = '{constraint}' AND contype = 'f'"
            ))
            if action is None:
                raise RuntimeError(f"Expected foreign key {constraint} was not found; inspect schema")
            if action != "c":
                await connection.execute(text(
                    f"ALTER TABLE {table} DROP CONSTRAINT {constraint}, "
                    f"ADD CONSTRAINT {constraint} FOREIGN KEY (document_id) "
                    "REFERENCES rag_document(id) ON DELETE CASCADE"
                ))


async def dispose_database() -> None:
    await get_engine().dispose()