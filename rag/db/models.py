"""
SQLAlchemy models for RAG audit tracking and document metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from rag.db.base import Base


class RAGDocument(Base):
    """Tracks ingested documents and chunking metadata."""

    __tablename__ = "rag_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    total_pages: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunks_created: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="ingested", nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    file_hash: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )


class RAGQueryLog(Base):
    """Full audit log of questions asked, retrieved sources, and latency."""

    __tablename__ = "rag_query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    filters_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    answer_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sources_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )


class SOPDocument(Base):
    """Tracks ingested Standard Operating Procedures (SOPs) and versions."""

    __tablename__ = "rag_sop_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(64), default="general", index=True, nullable=False)
    department: Mapped[str] = mapped_column(String(128), default="general", index=True, nullable=False)
    version: Mapped[str] = mapped_column(String(32), default="1.0", nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
