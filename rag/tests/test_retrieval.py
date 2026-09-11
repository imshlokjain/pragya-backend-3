"""
Tests for retrieval — vector store operations and semantic search.
"""

import pytest
from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.embeddings.sentence_transformer import SentenceTransformerProvider
from rag.retrieval.retriever import Retriever
from rag.schemas import Chunk, ContentType


@pytest.fixture(scope="module")
def embedding_provider():
    return SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def vector_store(tmp_path):
    """Create a ChromaDB store in a temp directory."""
    return ChromaVectorStore(
        persist_directory=str(tmp_path / "chroma_test"),
        collection_name="test_collection",
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            document_name="Economic Survey 2025-26",
            page_number=3,
            section="Macroeconomic Overview",
            content_type=ContentType.TEXT,
            content="India's GDP growth rate in 2024-25 was estimated at 6.4 percent, "
                    "driven by domestic consumption and infrastructure investment.",
            char_count=120,
        ),
        Chunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            document_name="Economic Survey 2025-26",
            page_number=5,
            section="Employment",
            content_type=ContentType.TEXT,
            content="The unemployment rate in Punjab stood at 7.3 percent in 2024-25, "
                    "higher than the national average of 4.1 percent.",
            char_count=110,
        ),
        Chunk(
            chunk_id="chunk-3",
            document_id="doc-1",
            document_name="Economic Survey 2025-26",
            page_number=5,
            section="Employment",
            content_type=ContentType.TABLE,
            content="| State | Unemployment Rate (%) |\n| --- | --- |\n"
                    "| Punjab | 7.3 |\n| Haryana | 6.1 |\n| National Average | 4.1 |",
            char_count=90,
        ),
        Chunk(
            chunk_id="chunk-4",
            document_id="doc-1",
            document_name="Economic Survey 2025-26",
            page_number=10,
            section="Education",
            content_type=ContentType.TEXT,
            content="The national literacy rate was recorded at 77.7 percent. "
                    "Female literacy rate was 70.3 percent while male literacy rate was 84.7 percent.",
            char_count=130,
        ),
    ]


class TestChromaVectorStore:
    """Tests for chroma_store.py."""

    def test_add_and_count(self, vector_store, sample_chunks, embedding_provider):
        texts = [c.content for c in sample_chunks]
        embeddings = embedding_provider.embed_documents(texts)

        vector_store.add_documents(sample_chunks, embeddings)
        assert vector_store.count() == 4

    def test_similarity_search(self, vector_store, sample_chunks, embedding_provider):
        # First add documents
        texts = [c.content for c in sample_chunks]
        embeddings = embedding_provider.embed_documents(texts)
        vector_store.add_documents(sample_chunks, embeddings)

        # Search for GDP-related content
        query_emb = embedding_provider.embed_query("What was India's GDP growth rate?")
        results = vector_store.similarity_search(query_emb, top_k=2)

        assert len(results) == 2
        assert results[0].score > 0  # should have positive similarity
        # Top result should be about GDP
        assert "GDP" in results[0].chunk.content or "growth" in results[0].chunk.content

    def test_metadata_filtering(self, vector_store, sample_chunks, embedding_provider):
        texts = [c.content for c in sample_chunks]
        embeddings = embedding_provider.embed_documents(texts)
        vector_store.add_documents(sample_chunks, embeddings)

        query_emb = embedding_provider.embed_query("employment data")
        results = vector_store.similarity_search(
            query_emb,
            top_k=10,
            filters={"section": "Employment"},
        )

        # All results should be from the Employment section
        for r in results:
            assert r.chunk.section == "Employment"

    def test_document_exists(self, vector_store, sample_chunks, embedding_provider):
        texts = [c.content for c in sample_chunks]
        embeddings = embedding_provider.embed_documents(texts)
        vector_store.add_documents(sample_chunks, embeddings)

        assert vector_store.document_exists("doc-1") is True
        assert vector_store.document_exists("doc-nonexistent") is False

    def test_list_documents(self, vector_store, sample_chunks, embedding_provider):
        texts = [c.content for c in sample_chunks]
        embeddings = embedding_provider.embed_documents(texts)
        vector_store.add_documents(sample_chunks, embeddings)

        docs = vector_store.list_documents()
        assert len(docs) == 1
        assert docs[0]["document_id"] == "doc-1"
        assert docs[0]["chunk_count"] == 4

    def test_delete_document(self, vector_store, sample_chunks, embedding_provider):
        texts = [c.content for c in sample_chunks]
        embeddings = embedding_provider.embed_documents(texts)
        vector_store.add_documents(sample_chunks, embeddings)

        vector_store.delete_document("doc-1")
        assert vector_store.count() == 0
        assert vector_store.document_exists("doc-1") is False


class TestRetriever:
    """Tests for retriever.py."""

    def test_retrieve(self, vector_store, sample_chunks, embedding_provider):
        texts = [c.content for c in sample_chunks]
        embeddings = embedding_provider.embed_documents(texts)
        vector_store.add_documents(sample_chunks, embeddings)

        retriever = Retriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            default_top_k=3,
        )

        result = retriever.retrieve("What was the unemployment rate in Punjab?")

        assert len(result.chunks) == 3
        assert result.query == "What was the unemployment rate in Punjab?"

        # Should retrieve Punjab-related content with high score
        contents = [sc.chunk.content for sc in result.chunks]
        assert any("Punjab" in c for c in contents)

    def test_retrieve_with_filters(self, vector_store, sample_chunks, embedding_provider):
        texts = [c.content for c in sample_chunks]
        embeddings = embedding_provider.embed_documents(texts)
        vector_store.add_documents(sample_chunks, embeddings)

        retriever = Retriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )

        result = retriever.retrieve(
            "data about education",
            top_k=5,
            filters={"section": "Education"},
        )

        for sc in result.chunks:
            assert sc.chunk.section == "Education"

    def test_retrieval_result_properties(self, vector_store, sample_chunks, embedding_provider):
        texts = [c.content for c in sample_chunks]
        embeddings = embedding_provider.embed_documents(texts)
        vector_store.add_documents(sample_chunks, embeddings)

        retriever = Retriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
        )

        result = retriever.retrieve("GDP growth", top_k=2)

        # Test convenience properties
        assert len(result.documents) == 2
        assert len(result.scores) == 2
        assert len(result.metadata) == 2
        assert all(isinstance(s, float) for s in result.scores)
