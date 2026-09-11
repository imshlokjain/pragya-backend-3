"""
Retriever — semantic retrieval with optional metadata and keyword filtering.

This module is the core retrieval engine. It embeds the query, searches
the vector store, and optionally applies reranking.

The retriever is **independent of generation** — it can be used alone
to feed context into any downstream model (LLM, ML, custom).
"""

from __future__ import annotations

import logging

from rag.embeddings.base import EmbeddingProvider
from rag.vectorstore.base import VectorStore
from rag.retrieval.reranker import Reranker
from rag.schemas import RetrievalResult, ScoredChunk
from rag.exceptions import RetrievalError

logger = logging.getLogger(__name__)


class Retriever:
    """Semantic retriever with optional reranking.

    Args:
        embedding_provider: The embedding backend to use for query encoding.
        vector_store: The vector store to search.
        reranker: Optional reranker for second-stage scoring.
        default_top_k: Default number of results to return.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        default_top_k: int = 5,
    ):
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._reranker = reranker
        self._default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
        keyword_filter: str | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        Pipeline:
            1. Embed the query
            2. Search vector store for top candidates
            3. Optionally rerank candidates
            4. Return top_k results

        Args:
            query: The user's question.
            top_k: Number of results to return (overrides default).
            filters: Metadata filters (e.g. ``{"year": 2025}``).
            keyword_filter: Optional keyword for full-text matching.

        Returns:
            A ``RetrievalResult`` containing scored chunks.

        Raises:
            RetrievalError: If retrieval fails.
        """
        effective_top_k = top_k or self._default_top_k

        try:
            # Step 1: Embed the query
            query_embedding = self._embedding_provider.embed_query(query)

            # Step 2: Search vector store
            # If reranking, retrieve more candidates first
            search_k = effective_top_k
            if self._reranker is not None:
                search_k = max(effective_top_k * 4, 20)

            results = self._vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=search_k,
                filters=filters,
                keyword_filter=keyword_filter,
            )

            logger.info(
                "Retrieved %d candidates for query: '%s'",
                len(results),
                query[:80],
            )

            # Step 3: Rerank if enabled
            if self._reranker is not None and len(results) > 0:
                results = self._reranker.rerank(
                    query=query,
                    chunks=results,
                    top_k=effective_top_k,
                )
                logger.info("Reranked to %d results", len(results))
            else:
                results = results[:effective_top_k]

            return RetrievalResult(
                query=query,
                chunks=results,
            )

        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                message="Retrieval failed",
                details=str(exc),
            ) from exc
