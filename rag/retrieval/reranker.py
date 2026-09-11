"""
Reranker — optional cross-encoder reranking for retrieval results.

Reranking takes the initial top-N candidates from vector search and
re-scores them using a cross-encoder model that processes (query, document)
pairs together for higher accuracy.

This is disabled by default and toggled via ``RAG_RERANKING_ENABLED=true``.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.schemas import ScoredChunk
from rag.exceptions import RetrievalError

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker.

    Args:
        model_name: Hugging Face cross-encoder model name.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model: Any = None

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.info("Reranker model loaded")
        except ImportError:
            raise RetrievalError(
                message="sentence-transformers is not installed",
                details="Install with: pip install sentence-transformers",
            )
        except Exception as exc:
            raise RetrievalError(
                message=f"Failed to load reranker model: {self._model_name}",
                details=str(exc),
            ) from exc

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """Rerank chunks using the cross-encoder.

        Args:
            query: The user's query.
            chunks: Candidate chunks from vector search.
            top_k: Number of top results to return after reranking.

        Returns:
            Reranked list of ``ScoredChunk``, sorted by descending score.
        """
        if not chunks:
            return []

        self._load_model()

        try:
            # Create (query, document) pairs
            pairs = [(query, sc.chunk.content) for sc in chunks]

            # Score all pairs
            scores = self._model.predict(pairs)

            # Attach new scores to chunks
            reranked = []
            for sc, score in zip(chunks, scores):
                reranked.append(
                    ScoredChunk(
                        chunk=sc.chunk,
                        score=round(float(score), 4),
                    )
                )

            # Sort by descending reranker score
            reranked.sort(key=lambda x: x.score, reverse=True)

            return reranked[:top_k]

        except Exception as exc:
            logger.warning("Reranking failed, returning original order: %s", exc)
            # Graceful degradation: return original results
            return chunks[:top_k]
