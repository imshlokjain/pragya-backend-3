"""
SOP Pipeline managing ingestion, dedicated vector storage, and search for Standard Operating Procedures.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union, BinaryIO

from rag.config import get_config
from rag.schemas import Source, ScoredChunk
from rag.sop.schemas import SOPIngestResponse, SOPIngestRequest
from rag.sop.chunker import SOPChunker
from rag.ingestion.document_processor import process_pdf
from rag.preprocessing.cleaner import clean_document
from rag.embeddings.base import EmbeddingProvider
from rag.vectorstore.chroma_store import ChromaVectorStore
from rag.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

PDFSource = Union[str, bytes, BinaryIO]


class SOPPipeline:
    """Dedicated pipeline for Standard Operating Procedures (SOPs)."""

    def __init__(
        self,
        rag_pipeline: Optional[Any] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        sop_collection_name: Optional[str] = None,
    ):
        self.rag_pipeline = rag_pipeline
        config = get_config()

        # Share embedding provider with main RAG pipeline if available
        if rag_pipeline is not None and hasattr(rag_pipeline, "_embedding_provider"):
            self.embedding_provider = rag_pipeline._embedding_provider
        elif embedding_provider is not None:
            self.embedding_provider = embedding_provider
        else:
            from rag.embeddings.sentence_transformer import SentenceTransformerProvider
            self.embedding_provider = SentenceTransformerProvider(model_name=config.embedding_model)

        collection = sop_collection_name or getattr(config, "sop_collection", "sop_documents")
        self.vector_store = ChromaVectorStore(
            persist_directory=config.vector_store_path,
            collection_name=collection,
        )
        self.chunker = SOPChunker(target_chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        self.retriever = Retriever(
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            default_top_k=5,
        )

    async def ingest_sop(
        self,
        source: PDFSource,
        filename: str = "sop.pdf",
        category: str = "general",
        department: str = "General",
        version: str = "1.0",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> SOPIngestResponse:
        """Process, chunk, and store an SOP document in the dedicated SOP vector store."""
        meta = {
            "category": category,
            "department": department,
            "version": version,
            **(extra_metadata or {}),
        }

        doc = process_pdf(source, filename=filename, metadata=meta)
        doc = clean_document(doc)
        chunks = self.chunker.chunk_document(doc)

        if not chunks:
            return SOPIngestResponse(
                document_id=doc.metadata.document_id,
                filename=filename,
                category=category,
                department=department,
                version=version,
                total_pages=doc.metadata.total_pages,
                chunks_created=0,
                message="Processed SOP but no text chunks were extracted.",
            )

        # Generate embeddings & store in SOP collection
        texts = [c.content for c in chunks]
        embeddings = self.embedding_provider.embed_documents(texts)
        self.vector_store.add_documents(chunks, embeddings)

        logger.info(
            "Ingested SOP '%s' [%s]: %d pages → %d procedural chunks",
            filename, category, doc.metadata.total_pages, len(chunks)
        )

        return SOPIngestResponse(
            document_id=doc.metadata.document_id,
            filename=filename,
            category=category,
            department=department,
            version=version,
            total_pages=doc.metadata.total_pages,
            chunks_created=len(chunks),
        )

    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve procedural chunks matching query from the SOP store."""
        filters = {"category": category} if category else None
        retrieval = self.retriever.retrieve(query=query, top_k=top_k, filters=filters)
        return [
            {
                "document": sc.chunk.document_name,
                "page": sc.chunk.page_number,
                "section": sc.chunk.section,
                "content": sc.chunk.content,
                "score": sc.score,
                "chunk_id": sc.chunk.chunk_id,
            }
            for sc in retrieval.chunks
        ]

    def list_sops(self) -> List[Dict[str, Any]]:
        """List all SOP documents currently stored."""
        return self.vector_store.list_documents()
