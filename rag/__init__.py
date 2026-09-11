"""
RAG (Retrieval-Augmented Generation) Pipeline for Government PDF Analysis.

This module provides a modular, self-contained RAG pipeline that can be
integrated into any FastAPI backend. It handles PDF ingestion, text/table
extraction, chunking, embedding, vector storage, semantic retrieval,
optional reranking, and LLM-powered generation with citations.

Usage:
    from rag import RAGPipeline

    pipeline = RAGPipeline()
    await pipeline.ingest_document(pdf_bytes, filename="report.pdf")
    response = await pipeline.query("What was the GDP growth rate?")

    # Retrieval-only (for custom ML models):
    result = await pipeline.retrieve("unemployment rate in Punjab")
    context = result.chunks  # feed to your own model
"""

from rag.pipeline import RAGPipeline
from rag.schemas import QueryResponse, RetrievalResult, Chunk, Source
from rag.agent import AgentOrchestrator, AgentRequest, AgentResponse
from rag.scenarios import WhatIfEngine, ScenarioRequest, ScenarioResponse
from rag.sop import SOPPipeline, ResponsePlanGenerator, ResponsePlanRequest, ResponsePlanResponse

__all__ = [
    "RAGPipeline",
    "QueryResponse",
    "RetrievalResult",
    "Chunk",
    "Source",
    "AgentOrchestrator",
    "AgentRequest",
    "AgentResponse",
    "WhatIfEngine",
    "ScenarioRequest",
    "ScenarioResponse",
    "SOPPipeline",
    "ResponsePlanGenerator",
    "ResponsePlanRequest",
    "ResponsePlanResponse",
]

__version__ = "0.2.0"

