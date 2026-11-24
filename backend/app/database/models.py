from __future__ import annotations

from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Collection(Base):
    __tablename__ = "rag_collection"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)
    documents: Mapped[list[StoredDocument]] = relationship(back_populates="collection")


class StoredDocument(Base):
    __tablename__ = "rag_document"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    collection_name: Mapped[str] = mapped_column(
        ForeignKey("rag_collection.name"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    extension: Mapped[str] = mapped_column(String, nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    collection: Mapped[Collection] = relationship(back_populates="documents")
    extraction_metadata: Mapped[DocumentMetadata] = relationship(
        back_populates="document", uselist=False
    )
    chunks: Mapped[list[Chunk]] = relationship(back_populates="document")


class DocumentMetadata(Base):
    __tablename__ = "rag_document_metadata"

    document_id: Mapped[UUID] = mapped_column(ForeignKey("rag_document.id"), primary_key=True)
    extractor: Mapped[str] = mapped_column(String, nullable=False)
    document: Mapped[StoredDocument] = relationship(back_populates="extraction_metadata")


class Chunk(Base):
    __tablename__ = "rag_chunk_v2"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(ForeignKey("rag_document.id"), index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    document: Mapped[StoredDocument] = relationship(back_populates="chunks")