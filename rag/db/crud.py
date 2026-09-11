"""
CRUD operations for RAG documents, query logs, and SOP documents.
"""

from __future__ import annotations

from typing import Any, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from rag.db.models import RAGDocument, RAGQueryLog, SOPDocument


async def create_document(
    session: AsyncSession,
    document_id: str,
    filename: str,
    total_pages: int,
    chunks_created: int,
    metadata_json: Optional[dict[str, Any]] = None,
    file_hash: Optional[str] = None,
    status: str = "ingested",
) -> RAGDocument:
    doc = RAGDocument(
        document_id=document_id,
        filename=filename,
        total_pages=total_pages,
        chunks_created=chunks_created,
        metadata_json=metadata_json or {},
        file_hash=file_hash,
        status=status,
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    return doc


async def get_document_by_id(
    session: AsyncSession,
    document_id: str,
) -> Optional[RAGDocument]:
    stmt = select(RAGDocument).where(RAGDocument.document_id == document_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_documents(
    session: AsyncSession,
    limit: int = 100,
) -> list[RAGDocument]:
    stmt = select(RAGDocument).order_by(RAGDocument.ingested_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def delete_document(
    session: AsyncSession,
    document_id: str,
) -> bool:
    doc = await get_document_by_id(session, document_id)
    if doc is None:
        return False
    await session.delete(doc)
    await session.commit()
    return True


async def log_query(
    session: AsyncSession,
    query_text: str,
    top_k: int,
    filters_json: Optional[dict[str, Any]] = None,
    answer_text: Optional[str] = None,
    sources_json: Optional[list[dict[str, Any]]] = None,
    latency_ms: float = 0.0,
) -> RAGQueryLog:
    log_entry = RAGQueryLog(
        query_text=query_text,
        top_k=top_k,
        filters_json=filters_json,
        answer_text=answer_text,
        sources_json=sources_json or [],
        latency_ms=latency_ms,
    )
    session.add(log_entry)
    await session.commit()
    await session.refresh(log_entry)
    return log_entry


async def list_query_logs(
    session: AsyncSession,
    limit: int = 100,
) -> list[RAGQueryLog]:
    stmt = select(RAGQueryLog).order_by(RAGQueryLog.created_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def create_sop_document(
    session: AsyncSession,
    document_id: str,
    filename: str,
    category: str = "general",
    department: str = "general",
    version: str = "1.0",
    metadata_json: Optional[dict[str, Any]] = None,
) -> SOPDocument:
    sop = SOPDocument(
        document_id=document_id,
        filename=filename,
        category=category,
        department=department,
        version=version,
        metadata_json=metadata_json or {},
    )
    session.add(sop)
    await session.commit()
    await session.refresh(sop)
    return sop


async def list_sop_documents(
    session: AsyncSession,
    category: Optional[str] = None,
    department: Optional[str] = None,
    limit: int = 100,
) -> list[SOPDocument]:
    stmt = select(SOPDocument).where(SOPDocument.is_active.is_(True))
    if category:
        stmt = stmt.where(SOPDocument.category == category)
    if department:
        stmt = stmt.where(SOPDocument.department == department)
    stmt = stmt.order_by(SOPDocument.ingested_at.desc()).limit(limit)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_sop_document(
    session: AsyncSession,
    document_id: str,
) -> Optional[SOPDocument]:
    stmt = select(SOPDocument).where(SOPDocument.document_id == document_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
