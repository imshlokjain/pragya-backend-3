"""
RAG Pipeline Configuration.

All configuration is read from environment variables (or .env file).
Every setting has a sensible default so the pipeline works out of the box
with zero configuration for local development.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field


class RAGConfig(BaseSettings):
    """Central configuration for the RAG pipeline.

    All fields are read from environment variables prefixed with ``RAG_``.
    For example, ``RAG_CHUNK_SIZE=1024`` in your ``.env`` file sets
    ``chunk_size`` to 1024.
    """

    model_config = {"env_prefix": "RAG_", "env_file": ".env", "extra": "ignore"}

    # ── Embedding ────────────────────────────────────────────────────────
    embedding_provider: str = Field(
        default="sentence-transformer",
        description="Embedding backend: 'sentence-transformer' or 'openai'.",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name for the chosen embedding provider.",
    )

    # ── Vector Store ─────────────────────────────────────────────────────
    vector_store_path: str = Field(
        default="./chroma_data",
        description="Directory where ChromaDB persists its data.",
    )
    vector_store_collection: str = Field(
        default="rag_documents",
        description="ChromaDB collection name.",
    )

    # ── LLM / Generation ────────────────────────────────────────────────
    llm_provider: str = Field(
        default="openai",
        description="LLM backend: 'openai' or 'huggingface'.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name for the chosen LLM provider.",
    )
    llm_api_key: str = Field(
        default="",
        description="API key for the LLM provider (required for generation).",
    )
    llm_base_url: str = Field(
        default="",
        description="Custom base URL for OpenAI-compatible APIs (vLLM, Ollama, etc.).",
    )
    llm_temperature: float = Field(
        default=0.1,
        description="Sampling temperature for generation (low = more factual).",
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens in the generated response.",
    )

    # ── Hugging Face ─────────────────────────────────────────────────────
    hf_api_key: str = Field(
        default="",
        description="Hugging Face API token (required for HF provider).",
    )

    # ── OpenAI Embeddings ────────────────────────────────────────────────
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (used when embedding_provider='openai').",
    )

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=512,
        description="Maximum number of characters per chunk.",
    )
    chunk_overlap: int = Field(
        default=64,
        description="Overlap in characters between adjacent chunks.",
    )

    # ── Retrieval ────────────────────────────────────────────────────────
    top_k: int = Field(
        default=5,
        description="Default number of chunks to retrieve.",
    )

    # ── Reranking ────────────────────────────────────────────────────────
    reranking_enabled: bool = Field(
        default=False,
        description="Whether to apply cross-encoder reranking.",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking.",
    )
    reranker_top_k: int = Field(
        default=20,
        description="Number of candidates to retrieve before reranking.",
    )

    # ── Database Tracking ────────────────────────────────────────────────
    database_url: str = Field(
        default="",
        description="Async connection URL for PostgreSQL (e.g. postgresql+asyncpg://user:pass@localhost:5432/sih_db).",
    )

    # ── Agent Orchestrator ───────────────────────────────────────────────
    agent_enabled: bool = Field(
        default=True,
        description="Whether the multi-step agent orchestrator endpoint is enabled.",
    )
    agent_max_iterations: int = Field(
        default=5,
        description="Default maximum ReAct iterations for agent reasoning.",
    )

    # ── What-If Scenarios ────────────────────────────────────────────────
    whatif_enabled: bool = Field(
        default=True,
        description="Whether the What-If scenario simulation endpoint is enabled.",
    )

    # ── SOP Pipeline ─────────────────────────────────────────────────────
    sop_collection: str = Field(
        default="sop_documents",
        description="Dedicated ChromaDB collection name for SOP documents.",
    )


# Singleton config instance — import this wherever needed.
def get_config() -> RAGConfig:
    """Return a RAGConfig instance (reads env vars / .env on each call)."""
    return RAGConfig()
