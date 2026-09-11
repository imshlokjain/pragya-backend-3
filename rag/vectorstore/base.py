"""
Abstract base class for vector stores.

All vector database backends implement this interface so they can
be swapped without changing retrieval or pipeline code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.schemas import Chunk, ScoredChunk


class VectorStore(ABC):
    """Abstract vector store interface.

    Implementations must support:
    - Adding document chunks with embeddings
    - Similarity search with optional metadata filtering
    - Document-level deletion
    - Existence checks
    """

    @abstractmethod
    def add_documents(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks and their embeddings to the store.

        Args:
            chunks: List of chunk objects with metadata.
            embeddings: Corresponding embedding vectors.
        """
        ...

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        keyword_filter: str | None = None,
    ) -> list[ScoredChunk]:
        """Find the most similar chunks to a query embedding.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            filters: Optional metadata filters (e.g. ``{"year": "2025"}``).
            keyword_filter: Optional keyword for full-text filtering.

        Returns:
            List of ``ScoredChunk`` ordered by descending similarity.
        """
        ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete all chunks belonging to a document.

        Args:
            document_id: The document ID to remove.
        """
        ...

    @abstractmethod
    def document_exists(self, document_id: str) -> bool:
        """Check if a document has been ingested.

        Args:
            document_id: The document ID to check.

        Returns:
            True if at least one chunk with this document_id exists.
        """
        ...

    @abstractmethod
    def list_documents(self) -> list[dict]:
        """List all unique documents in the store.

        Returns:
            List of dicts with document metadata.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks in the store."""
        ...
