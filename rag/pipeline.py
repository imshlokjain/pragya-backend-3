"""
RAG Pipeline — the single entry point for the entire RAG system.

This class wires together all components (ingestion, cleaning, chunking,
embedding, vector store, retrieval, reranking, generation) behind a
simple interface that your backend can call.

Usage:
    pipeline = RAGPipeline()

    # Ingest a document (once)
    doc_id = await pipeline.ingest_document(pdf_bytes, filename="report.pdf")

    # Query with generation
    response = await pipeline.query("What was India's GDP growth?")
    print(response.answer)
    print(response.sources)

    # Retrieval only (for custom ML models)
    result = await pipeline.retrieve("unemployment rate")
    for chunk in result.documents:
        print(chunk.content)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union, BinaryIO

from rag.config import RAGConfig, get_config
from rag.schemas import (
    QueryResponse,
    RetrievalResult,
    Source,
    Chunk,
    IngestResponse,
)
from rag.ingestion.document_processor import process_pdf
from rag.preprocessing.cleaner import clean_document
from rag.preprocessing.chunker import chunk_document
from rag.embeddings.base import EmbeddingProvider
from rag.vectorstore.base import VectorStore
from rag.retrieval.retriever import Retriever
from rag.retrieval.reranker import Reranker
from rag.generation.base import ModelProvider
from rag.generation.prompt import build_rag_prompt
from rag.exceptions import (
    RAGError,
    ConfigurationError,
    GenerationError,
)

logger = logging.getLogger(__name__)

# Type alias for PDF input sources
PDFSource = Union[str, Path, bytes, BinaryIO]


class RAGPipeline:
    """Main RAG pipeline — wire once, call from anywhere.

    The pipeline auto-configures from environment variables (``RAG_*``
    prefix) or accepts explicit component overrides for testing/custom setups.

    Args:
        config: Optional ``RAGConfig`` (reads env vars by default).
        embedding_provider: Optional custom embedding provider.
        vector_store: Optional custom vector store.
        model_provider: Optional custom LLM provider.
        retriever: Optional pre-built retriever.
    """

    def __init__(
        self,
        config: RAGConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        model_provider: ModelProvider | None = None,
        retriever: Retriever | None = None,
    ):
        self._config = config or get_config()

        # ── Build components ─────────────────────────────────────────
        self._embedding_provider = embedding_provider or self._build_embedding_provider()
        self._vector_store = vector_store or self._build_vector_store()
        self._model_provider = model_provider  # lazy — only needed for generation
        self._reranker = self._build_reranker() if self._config.reranking_enabled else None

        self._retriever = retriever or Retriever(
            embedding_provider=self._embedding_provider,
            vector_store=self._vector_store,
            reranker=self._reranker,
            default_top_k=self._config.top_k,
        )

        logger.info(
            "RAG Pipeline initialised | embedding=%s | reranking=%s | llm=%s",
            self._config.embedding_model,
            self._config.reranking_enabled,
            self._config.llm_provider,
        )

    # ── Public API ───────────────────────────────────────────────────

    async def ingest_document(
        self,
        source: PDFSource,
        filename: str | None = None,
        metadata: dict | None = None,
    ) -> IngestResponse:
        """Ingest a PDF document into the vector store.

        Pipeline: PDF → extract → clean → chunk → embed → store

        This should be called **once per document**. Subsequent queries
        reuse the stored embeddings.

        Args:
            source: File path, bytes, or file-like object.
            filename: Optional filename for the document.
            metadata: Optional metadata (year, department, etc.).

        Returns:
            ``IngestResponse`` with document ID and chunk count.
        """
        # Step 1: Process PDF (extract text + tables)
        logger.info("Ingesting document: %s", filename or "unknown")
        doc = process_pdf(source, filename=filename, metadata=metadata)

        # Step 2: Clean extracted text
        doc = clean_document(doc)

        # Step 3: Chunk the document
        chunks = chunk_document(
            doc,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )

        if not chunks:
            logger.warning("No chunks produced from document: %s", filename)
            return IngestResponse(
                document_id=doc.metadata.document_id,
                filename=doc.metadata.filename,
                total_pages=doc.metadata.total_pages,
                chunks_created=0,
                message="Document processed but no text chunks were extracted.",
            )

        # Step 4: Generate embeddings
        texts = [c.content for c in chunks]
        embeddings = self._embedding_provider.embed_documents(texts)

        # Step 5: Store in vector database
        self._vector_store.add_documents(chunks, embeddings)

        logger.info(
            "Ingested '%s': %d pages → %d chunks → stored",
            doc.metadata.filename,
            doc.metadata.total_pages,
            len(chunks),
        )

        return IngestResponse(
            document_id=doc.metadata.document_id,
            filename=doc.metadata.filename,
            total_pages=doc.metadata.total_pages,
            chunks_created=len(chunks),
        )

    async def query(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict | None = None,
        generate: bool = True,
    ) -> QueryResponse:
        """Query the RAG pipeline — retrieve context and optionally generate an answer.

        Args:
            question: The user's question.
            top_k: Number of chunks to retrieve (overrides config default).
            filters: Optional metadata filters.
            generate: If False, returns only retrieved context (no LLM call).

        Returns:
            ``QueryResponse`` with answer text and citation sources.
        """
        # Step 1: Retrieve relevant chunks
        retrieval_result = self.retrieve(question, top_k=top_k, filters=filters)

        # Build sources from retrieval
        sources = [
            Source(
                document=sc.chunk.document_name,
                page=sc.chunk.page_number,
                section=sc.chunk.section,
                chunk_id=sc.chunk.chunk_id,
                score=sc.score,
            )
            for sc in retrieval_result.chunks
        ]

        # If generation is not requested, return context only
        if not generate:
            context_text = "\n\n".join(
                sc.chunk.content for sc in retrieval_result.chunks
            )
            return QueryResponse(
                answer=context_text,
                sources=sources,
                query=question,
                retrieval_count=len(retrieval_result.chunks),
            )

        # Step 2: Build prompt with retrieved context
        system_prompt, user_prompt = build_rag_prompt(
            question=question,
            scored_chunks=retrieval_result.chunks,
        )

        # Step 3: Generate answer via LLM
        model_provider = self._get_model_provider()
        answer = await model_provider.generate(system_prompt, user_prompt)

        return QueryResponse(
            answer=answer,
            sources=sources,
            query=question,
            retrieval_count=len(retrieval_result.chunks),
        )

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict | None = None,
        keyword_filter: str | None = None,
    ) -> RetrievalResult:
        """Retrieve relevant chunks WITHOUT generation.

        Use this when you want to feed context to your own ML model
        instead of an LLM.

        Args:
            question: The query string.
            top_k: Number of chunks to retrieve.
            filters: Optional metadata filters.
            keyword_filter: Optional keyword for full-text matching.

        Returns:
            ``RetrievalResult`` with chunks, scores, and metadata.
        """
        return self._retriever.retrieve(
            query=question,
            top_k=top_k,
            filters=filters,
            keyword_filter=keyword_filter,
        )

    async def delete_document(self, document_id: str) -> None:
        """Remove a document and all its chunks from the vector store.

        Args:
            document_id: The document ID to delete.
        """
        self._vector_store.delete_document(document_id)
        logger.info("Deleted document: %s", document_id)

    def document_exists(self, document_id: str) -> bool:
        """Check if a document has been ingested.

        Args:
            document_id: The document ID to check.
        """
        return self._vector_store.document_exists(document_id)

    def list_documents(self) -> list[dict]:
        """List all ingested documents."""
        return self._vector_store.list_documents()

    # ── Component Builders ───────────────────────────────────────────

    def _build_embedding_provider(self) -> EmbeddingProvider:
        """Build the embedding provider from config."""
        provider = self._config.embedding_provider.lower()

        if provider == "sentence-transformer":
            from rag.embeddings.sentence_transformer import SentenceTransformerProvider
            return SentenceTransformerProvider(
                model_name=self._config.embedding_model,
            )
        elif provider == "openai":
            from rag.embeddings.openai_embeddings import OpenAIEmbeddingProvider
            return OpenAIEmbeddingProvider(
                model_name=self._config.embedding_model,
                api_key=self._config.openai_api_key,
            )
        else:
            raise ConfigurationError(
                message=f"Unknown embedding provider: {provider}",
                details="Supported: 'sentence-transformer', 'openai'",
            )

    def _build_vector_store(self) -> VectorStore:
        """Build the vector store from config."""
        from rag.vectorstore.chroma_store import ChromaVectorStore
        return ChromaVectorStore(
            persist_directory=self._config.vector_store_path,
            collection_name=self._config.vector_store_collection,
        )

    def _build_reranker(self) -> Reranker:
        """Build the reranker from config."""
        return Reranker(model_name=self._config.reranker_model)

    def _get_model_provider(self) -> ModelProvider:
        """Get or build the model provider (lazy, only for generation)."""
        if self._model_provider is not None:
            return self._model_provider

        provider = self._config.llm_provider.lower()

        if provider == "openai":
            from rag.generation.openai_provider import OpenAIModelProvider
            self._model_provider = OpenAIModelProvider(
                model_name=self._config.llm_model,
                api_key=self._config.llm_api_key,
                base_url=self._config.llm_base_url or None,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
            )
        elif provider == "huggingface":
            from rag.generation.huggingface_provider import HuggingFaceModelProvider
            self._model_provider = HuggingFaceModelProvider(
                model_name=self._config.llm_model,
                api_key=self._config.hf_api_key,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
            )
        else:
            raise ConfigurationError(
                message=f"Unknown LLM provider: {provider}",
                details="Supported: 'openai', 'huggingface'",
            )

        return self._model_provider
