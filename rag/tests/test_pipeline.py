"""
End-to-end pipeline test.

Tests the full flow: PDF → ingest → query → answer + citations.
"""

import pytest
from rag.pipeline import RAGPipeline
from rag.config import RAGConfig
from rag.schemas import QueryResponse, IngestResponse


@pytest.fixture
def pipeline(tmp_path):
    """Create a pipeline with a temp vector store (no LLM needed for retrieval tests)."""
    config = RAGConfig(
        embedding_provider="sentence-transformer",
        embedding_model="all-MiniLM-L6-v2",
        vector_store_path=str(tmp_path / "chroma_e2e"),
        vector_store_collection="e2e_test",
        reranking_enabled=False,
    )
    return RAGPipeline(config=config)


@pytest.mark.asyncio
class TestEndToEndPipeline:
    """Full pipeline integration tests."""

    async def test_ingest_document(self, pipeline, sample_pdf_bytes):
        """Test: PDF → extract → chunk → embed → store."""
        result = await pipeline.ingest_document(
            source=sample_pdf_bytes,
            filename="Economic Survey 2025-26.pdf",
            metadata={"year": 2025},
        )

        assert isinstance(result, IngestResponse)
        assert result.chunks_created > 0
        assert result.filename == "Economic Survey 2025-26.pdf"
        assert result.total_pages > 0

    async def test_retrieve_without_generation(self, pipeline, sample_pdf_bytes):
        """Test: query → retrieve chunks (no LLM call)."""
        # Ingest first
        await pipeline.ingest_document(
            source=sample_pdf_bytes,
            filename="Economic Survey 2025-26.pdf",
        )

        # Retrieve
        result = pipeline.retrieve(
            question="What was the GDP growth rate?",
            top_k=3,
        )

        assert len(result.chunks) > 0
        assert result.query == "What was the GDP growth rate?"

        # Should find GDP-related content
        all_text = " ".join(sc.chunk.content for sc in result.chunks)
        # Check that we got relevant content (GDP or growth or percent)
        assert any(
            keyword in all_text.lower()
            for keyword in ["gdp", "growth", "percent", "6.4"]
        )

    async def test_query_without_llm(self, pipeline, sample_pdf_bytes):
        """Test: query with generate=False returns raw context."""
        await pipeline.ingest_document(
            source=sample_pdf_bytes,
            filename="Economic Survey 2025-26.pdf",
        )

        response = await pipeline.query(
            question="unemployment rate in Punjab",
            top_k=3,
            generate=False,
        )

        assert isinstance(response, QueryResponse)
        assert len(response.sources) > 0
        assert response.answer  # should contain raw context
        assert response.retrieval_count > 0

        # Citations should include document info
        for source in response.sources:
            assert source.document
            assert source.page > 0
            assert source.chunk_id

    async def test_document_management(self, pipeline, sample_pdf_bytes):
        """Test: ingest → exists → list → delete → verify deleted."""
        result = await pipeline.ingest_document(
            source=sample_pdf_bytes,
            filename="test_doc.pdf",
        )
        doc_id = result.document_id

        # Exists
        assert pipeline.document_exists(doc_id) is True

        # List
        docs = pipeline.list_documents()
        assert len(docs) >= 1
        assert any(d["document_id"] == doc_id for d in docs)

        # Delete
        await pipeline.delete_document(doc_id)
        assert pipeline.document_exists(doc_id) is False

    async def test_retrieve_for_ml_model(self, pipeline, sample_pdf_bytes):
        """Test: retrieve() returns structured data suitable for ML model input."""
        await pipeline.ingest_document(
            source=sample_pdf_bytes,
            filename="Economic Survey.pdf",
        )

        result = pipeline.retrieve("literacy rate", top_k=5)

        # Should be usable for ML models
        assert hasattr(result, "documents")
        assert hasattr(result, "scores")
        assert hasattr(result, "metadata")

        # Metadata should have all required fields
        for meta in result.metadata:
            assert "document_id" in meta
            assert "document_name" in meta
            assert "page_number" in meta
            assert "section" in meta
            assert "chunk_id" in meta
            assert "score" in meta
