from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.database.models import Base


@lru_cache
def get_engine() -> AsyncEngine:
    return create_async_engine(get_settings().database_url, pool_pre_ping=True)


def get_session_factory() -> async_sessionmaker:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def initialize_database() -> None:
    async with get_engine().begin() as connection:
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.run_sync(Base.metadata.create_all)


async def dispose_database() -> None:
    await get_engine().dispose()