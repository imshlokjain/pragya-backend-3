"""
Custom exceptions for the RAG pipeline.

Each exception maps to a specific failure mode so that the API layer
can return meaningful HTTP error responses instead of raw stack traces.
"""


class RAGError(Exception):
    """Base exception for all RAG pipeline errors."""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class PDFLoadError(RAGError):
    """Raised when a PDF cannot be opened or is invalid."""
    pass


class EmptyPDFError(RAGError):
    """Raised when a PDF contains no extractable content."""
    pass


class ExtractionError(RAGError):
    """Raised when text or table extraction fails."""
    pass


class OCRRequiredError(RAGError):
    """Raised when a page appears to be scanned and OCR is not enabled."""
    pass


class ChunkingError(RAGError):
    """Raised when the chunking process fails."""
    pass


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""
    pass


class VectorStoreError(RAGError):
    """Raised when vector store operations fail."""
    pass


class RetrievalError(RAGError):
    """Raised when retrieval fails."""
    pass


class GenerationError(RAGError):
    """Raised when LLM generation fails."""
    pass


class InsufficientContextError(RAGError):
    """Raised when retrieved context is insufficient to answer the question."""
    pass


class ConfigurationError(RAGError):
    """Raised when required configuration is missing or invalid."""
    pass
