"""
Data models for the RAG pipeline.

All inter-module communication uses these Pydantic models so that
data contracts are explicit and validated at runtime.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────


class ContentType(str, Enum):
    """Type of content within a chunk."""
    TEXT = "text"
    TABLE = "table"
    HEADING = "heading"


# ── Ingestion Models ─────────────────────────────────────────────────────


class ExtractedTable(BaseModel):
    """A single table extracted from a PDF page."""
    headers: list[str] = Field(default_factory=list, description="Column headers")
    rows: list[list[str]] = Field(default_factory=list, description="Row data")
    markdown: str = Field(default="", description="Table rendered as Markdown")
    page_number: int = Field(description="1-indexed page number")


class PageContent(BaseModel):
    """Content extracted from a single PDF page."""
    page_number: int = Field(description="1-indexed page number")
    text: str = Field(default="", description="Extracted plain text")
    tables: list[ExtractedTable] = Field(
        default_factory=list, description="Tables found on this page"
    )
    is_scanned: bool = Field(
        default=False, description="True if the page appears to be a scanned image"
    )


class DocumentMetadata(BaseModel):
    """Metadata about a processed document."""
    document_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique document identifier",
    )
    filename: str = Field(default="unknown.pdf", description="Original filename")
    total_pages: int = Field(default=0, description="Total page count")
    ingested_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of ingestion",
    )
    extra: dict = Field(
        default_factory=dict,
        description="Additional user-supplied metadata (year, department, etc.)",
    )


class ProcessedDocument(BaseModel):
    """A fully processed PDF document, ready for chunking."""
    metadata: DocumentMetadata
    pages: list[PageContent] = Field(default_factory=list)
    scanned_pages: list[int] = Field(
        default_factory=list,
        description="Page numbers that appear to be scanned images",
    )


# ── Chunk Models ─────────────────────────────────────────────────────────


class Chunk(BaseModel):
    """A single chunk of text with full provenance metadata."""
    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique chunk identifier",
    )
    document_id: str = Field(description="Parent document ID")
    document_name: str = Field(default="", description="Original filename")
    page_number: int = Field(description="Source page number")
    section: str = Field(default="", description="Section heading (if detected)")
    content_type: ContentType = Field(
        default=ContentType.TEXT, description="Type of content"
    )
    content: str = Field(description="The actual text content")
    char_count: int = Field(default=0, description="Character count of content")


# ── Retrieval Models ─────────────────────────────────────────────────────


class ScoredChunk(BaseModel):
    """A chunk with its similarity score from retrieval."""
    chunk: Chunk
    score: float = Field(description="Similarity score (higher = more relevant)")


class RetrievalResult(BaseModel):
    """Result of a retrieval operation — usable independently of generation."""
    query: str = Field(description="The original query")
    chunks: list[ScoredChunk] = Field(default_factory=list)

    @property
    def documents(self) -> list[Chunk]:
        """Shortcut to access just the chunk objects."""
        return [sc.chunk for sc in self.chunks]

    @property
    def scores(self) -> list[float]:
        """Shortcut to access just the scores."""
        return [sc.score for sc in self.chunks]

    @property
    def metadata(self) -> list[dict]:
        """Shortcut to access chunk metadata as dicts."""
        return [
            {
                "document_id": sc.chunk.document_id,
                "document_name": sc.chunk.document_name,
                "page_number": sc.chunk.page_number,
                "section": sc.chunk.section,
                "content_type": sc.chunk.content_type.value,
                "chunk_id": sc.chunk.chunk_id,
                "score": sc.score,
            }
            for sc in self.chunks
        ]


# ── Generation Models ────────────────────────────────────────────────────


class Source(BaseModel):
    """A citation source for the generated answer."""
    document: str = Field(description="Document filename")
    page: int = Field(description="Page number")
    section: str = Field(default="", description="Section heading")
    chunk_id: str = Field(description="Chunk identifier")
    score: float = Field(description="Retrieval similarity score")


class QueryResponse(BaseModel):
    """Complete response from the RAG pipeline including answer and citations."""
    answer: str = Field(description="Generated answer text")
    sources: list[Source] = Field(
        default_factory=list, description="Citation sources"
    )
    query: str = Field(default="", description="Original question")
    retrieval_count: int = Field(
        default=0, description="Number of chunks retrieved"
    )


# ── API Request/Response Models ──────────────────────────────────────────


class IngestRequest(BaseModel):
    """Request body for the /rag/ingest endpoint (metadata only; file is multipart)."""
    metadata: dict = Field(
        default_factory=dict,
        description="Optional metadata (year, department, etc.)",
    )


class IngestResponse(BaseModel):
    """Response from the /rag/ingest endpoint."""
    document_id: str
    filename: str
    total_pages: int
    chunks_created: int
    message: str = "Document ingested successfully"


class QueryRequest(BaseModel):
    """Request body for the /rag/query endpoint."""
    query: str = Field(description="The question to answer")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    filters: dict | None = Field(
        default=None, description="Metadata filters for retrieval"
    )
    generate: bool = Field(
        default=True,
        description="If False, returns only retrieved chunks without LLM generation",
    )


class RetrieveRequest(BaseModel):
    """Request body for the /rag/retrieve endpoint."""
    query: str = Field(description="The question to retrieve context for")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    filters: dict | None = Field(
        default=None, description="Metadata filters for retrieval"
    )


class DocumentInfo(BaseModel):
    """Summary info about an ingested document."""
    document_id: str
    filename: str
    total_pages: int
    ingested_at: str
    chunk_count: int = 0
