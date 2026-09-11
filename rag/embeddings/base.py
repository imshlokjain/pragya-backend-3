"""
Abstract base class for embedding providers.

All embedding backends implement this interface so they can be
swapped without changing any downstream code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract embedding provider.

    Implementations must provide:
    - ``embed_documents``: batch-embed a list of texts
    - ``embed_query``: embed a single query string
    - ``dimension``: the embedding vector size
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of documents.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query.

        Args:
            query: The query text.

        Returns:
            A single embedding vector.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name (for logging/debugging)."""
        ...
