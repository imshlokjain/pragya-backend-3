"""
Database session management for RAG relational tracking.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag.config import get_config

_engine = None
_sessionmaker = None


def get_engine():
    global _engine, _sessionmaker
    if _engine is None:
        config = get_config()
        url = config.database_url
        if not url:
            # Default to local sqlite for audit logs
            url = "sqlite+aiosqlite:///./rag_audit.db"
        _engine = create_async_engine(url, echo=False)
        _sessionmaker = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _engine


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager yielding a database session."""
    get_engine()
    assert _sessionmaker is not None
    async with _sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
