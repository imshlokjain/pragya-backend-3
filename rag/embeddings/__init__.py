"""Embeddings subpackage — embedding provider abstraction and implementations."""

from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.sentence_transformer import SentenceTransformerProvider

__all__ = ["EmbeddingProvider", "SentenceTransformerProvider"]
