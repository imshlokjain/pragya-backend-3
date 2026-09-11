"""
ChromaDB vector store implementation.

Uses ``chromadb.PersistentClient`` for disk-based storage that survives
server restarts. Supports metadata filtering and keyword search.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.exceptions import VectorStoreError
from rag.schemas import Chunk, ContentType, ScoredChunk
from rag.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


def _build_where_clause(filters: dict | None) -> dict | None:
    """Convert user-friendly filters to ChromaDB where clause.

    Handles single and multiple filters. ChromaDB requires ``$and``
    for multiple conditions.
    """
    if not filters:
        return None

    conditions = []
    for key, value in filters.items():
        if isinstance(value, (int, float)):
            conditions.append({key: value})
        else:
            conditions.append({key: str(value)})

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store.

    Args:
        persist_directory: Path where ChromaDB stores data.
        collection_name: Name of the ChromaDB collection.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_data",
        collection_name: str = "rag_documents",
    ):
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    def _get_collection(self):
        """Lazy-initialise the ChromaDB client and collection."""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection '%s' ready (%d existing chunks)",
                self._collection_name,
                self._collection.count(),
            )
            return self._collection
        except Exception as exc:
            raise VectorStoreError(
                message="Failed to initialise ChromaDB",
                details=str(exc),
            ) from exc

    def add_documents(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                message="Chunk count does not match embedding count",
                details=f"Got {len(chunks)} chunks but {len(embeddings)} embeddings.",
            )

        collection = self._get_collection()

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "document_name": c.document_name,
                "page_number": c.page_number,
                "section": c.section,
                "content_type": c.content_type.value,
                "char_count": c.char_count,
            }
            for c in chunks
        ]

        batch_size = 500
        try:
            for i in range(0, len(ids), batch_size):
                end = min(i + batch_size, len(ids))
                collection.upsert(
                    ids=ids[i:end],
                    embeddings=embeddings[i:end],
                    documents=documents[i:end],
                    metadatas=metadatas[i:end],
                )
            logger.info("Upserted %d chunks into ChromaDB", len(chunks))
        except Exception as exc:
            raise VectorStoreError(
                message="Failed to add documents to ChromaDB",
                details=str(exc),
            ) from exc

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        keyword_filter: str | None = None,
    ) -> list[ScoredChunk]:
        collection = self._get_collection()

        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        where_clause = _build_where_clause(filters)
        if where_clause:
            query_params["where"] = where_clause

        if keyword_filter:
            query_params["where_document"] = {"$contains": keyword_filter}

        try:
            results = collection.query(**query_params)
        except Exception as exc:
            raise VectorStoreError(
                message="ChromaDB similarity search failed",
                details=str(exc),
            ) from exc

        scored_chunks: list[ScoredChunk] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return scored_chunks

        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            content = results["documents"][0][i]
            distance = results["distances"][0][i]
            # Cosine distance to similarity: 1.0 - distance
            similarity = 1.0 - distance

            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=meta.get("document_id", ""),
                document_name=meta.get("document_name", ""),
                page_number=meta.get("page_number", 0),
                section=meta.get("section", ""),
                content_type=ContentType(meta.get("content_type", "text")),
                content=content,
                char_count=meta.get("char_count", len(content)),
            )
            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    score=round(similarity, 4),
                )
            )

        return scored_chunks

    def delete_document(self, document_id: str) -> None:
        collection = self._get_collection()
        try:
            results = collection.get(
                where={"document_id": document_id},
                include=[],
            )
            if results and results.get("ids"):
                collection.delete(ids=results["ids"])
                logger.info(
                    "Deleted %d chunks for document %s",
                    len(results["ids"]),
                    document_id,
                )
        except Exception as exc:
            raise VectorStoreError(
                message=f"Failed to delete document {document_id}",
                details=str(exc),
            ) from exc

    def document_exists(self, document_id: str) -> bool:
        collection = self._get_collection()
        try:
            results = collection.get(
                where={"document_id": document_id},
                include=[],
                limit=1,
            )
            return bool(results.get("ids"))
        except Exception:
            return False

    def list_documents(self) -> list[dict]:
        collection = self._get_collection()
        try:
            results = collection.get(include=["metadatas"])
            if not results or not results.get("metadatas"):
                return []

            docs: dict[str, dict] = {}
            for meta in results["metadatas"]:
                doc_id = meta.get("document_id", "")
                if doc_id not in docs:
                    docs[doc_id] = {
                        "document_id": doc_id,
                        "filename": meta.get("document_name", ""),
                        "chunk_count": 0,
                    }
                docs[doc_id]["chunk_count"] += 1

            return list(docs.values())
        except Exception as exc:
            raise VectorStoreError(
                message="Failed to list documents",
                details=str(exc),
            ) from exc

    def count(self) -> int:
        collection = self._get_collection()
        return collection.count()
