"""
Tests for embedding providers.
"""

import math
import pytest

from rag.embeddings.sentence_transformer import SentenceTransformerProvider
from rag.exceptions import EmbeddingError


class TestSentenceTransformerProvider:
    """Tests for the sentence-transformers embedding provider."""

    @pytest.fixture(scope="class")
    def provider(self):
        """Create a provider instance (shared across tests in this class)."""
        return SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")

    def test_embed_query(self, provider):
        embedding = provider.embed_query("What is the GDP growth rate?")
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)
        assert len(embedding) == 384

    def test_embed_documents(self, provider):
        texts = [
            "India's GDP grew by 6.4 percent.",
            "Unemployment rate in Punjab was 7.3 percent.",
            "The fiscal deficit was 4.9 percent of GDP.",
        ]
        embeddings = provider.embed_documents(texts)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == 384
            assert all(isinstance(x, float) for x in emb)

    def test_embed_empty_list(self, provider):
        result = provider.embed_documents([])
        assert result == []

    def test_dimension(self, provider):
        assert provider.dimension == 384

    def test_model_name(self, provider):
        assert provider.model_name == "all-MiniLM-L6-v2"

    def test_semantic_similarity(self, provider):
        """Verify that semantically similar texts have closer embeddings."""
        t1 = "GDP growth rate of India"
        t2 = "economic growth percentage of India"
        t3 = "recipe for chocolate cake"

        embeddings = provider.embed_documents([t1, t2, t3])
        e1, e2, e3 = embeddings

        # Cosine similarity
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        sim_12 = dot(e1, e2)
        sim_13 = dot(e1, e3)

        assert sim_12 > sim_13
