"""
Unit tests for RAG database layer and CRUD operations.
"""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from rag.db.base import Base
from rag.db import crud


@pytest_asyncio.fixture
async def async_session(tmp_path):
    """Create an in-memory SQLite database session for testing."""
    db_path = tmp_path / "test_rag.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
class TestDatabaseCRUD:
    """Test suite for RAG relational models."""

    async def test_create_and_get_document(self, async_session):
        doc = await crud.create_document(
            session=async_session,
            document_id="doc-test-1",
            filename="budget_2025.pdf",
            total_pages=10,
            chunks_created=25,
            metadata_json={"department": "Finance"},
            file_hash="abc123hash",
        )
        assert doc.id is not None
        assert doc.document_id == "doc-test-1"

        retrieved = await crud.get_document_by_id(async_session, "doc-test-1")
        assert retrieved is not None
        assert retrieved.filename == "budget_2025.pdf"
        assert retrieved.metadata_json["department"] == "Finance"

    async def test_list_and_delete_document(self, async_session):
        await crud.create_document(
            session=async_session,
            document_id="doc-test-2",
            filename="policy_a.pdf",
            total_pages=5,
            chunks_created=12,
        )
        docs = await crud.list_documents(async_session)
        assert len(docs) >= 1

        deleted = await crud.delete_document(async_session, "doc-test-2")
        assert deleted is True
        assert await crud.get_document_by_id(async_session, "doc-test-2") is None

    async def test_log_and_list_queries(self, async_session):
        log = await crud.log_query(
            session=async_session,
            query_text="What is the fiscal deficit?",
            top_k=5,
            filters_json={"year": 2025},
            answer_text="The fiscal deficit was 4.9 percent.",
            sources_json=[{"doc": "budget.pdf", "page": 1}],
            latency_ms=125.5,
        )
        assert log.id is not None
        assert log.latency_ms == 125.5

        logs = await crud.list_query_logs(async_session)
        assert len(logs) >= 1
        assert logs[0].query_text == "What is the fiscal deficit?"

    async def test_sop_document_lifecycle(self, async_session):
        sop = await crud.create_sop_document(
            session=async_session,
            document_id="sop-flood-01",
            filename="flood_response_sop.pdf",
            category="disaster_management",
            department="Revenue & Disaster",
            version="2.1",
            metadata_json={"level": "Tier-1"},
        )
        assert sop.id is not None
        assert sop.category == "disaster_management"

        sops = await crud.list_sop_documents(async_session, category="disaster_management")
        assert len(sops) == 1
        assert sops[0].document_id == "sop-flood-01"

        retrieved = await crud.get_sop_document(async_session, "sop-flood-01")
        assert retrieved is not None
        assert retrieved.version == "2.1"
