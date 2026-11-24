from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Collection(Base):
    __tablename__ = "rag_collection"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)


class Chunk(Base):
    __tablename__ = "rag_chunk"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    collection_name: Mapped[str] = mapped_column(
        ForeignKey("rag_collection.name"), index=True, nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)