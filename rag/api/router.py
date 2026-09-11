"""
FastAPI Router for the RAG pipeline.

Provides minimal REST endpoints for document ingestion and querying.
Include this router in your existing FastAPI app with:

    from rag.api.router import router as rag_router
    app.include_router(rag_router, prefix="/rag", tags=["RAG"])

Endpoints:
    POST /rag/ingest     — Upload and ingest a PDF
    POST /rag/query      — Ask a question (with LLM generation)
    POST /rag/retrieve   — Retrieve context only (for custom models)
    GET  /rag/documents  — List ingested documents
    DELETE /rag/documents/{document_id} — Delete a document
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from rag.pipeline import RAGPipeline
from rag.schemas import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrievalResult,
    DocumentInfo,
)
from rag.agent.schemas import AgentRequest, AgentResponse
from rag.scenarios.schemas import ScenarioRequest, ScenarioResponse
from rag.sop.schemas import (
    ResponsePlanRequest,
    ResponsePlanResponse,
    SOPIngestResponse,
)
from rag.exceptions import (
    RAGError,
    PDFLoadError,
    EmptyPDFError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton pipeline instance — initialised on first use.
_pipeline: RAGPipeline | None = None
_sop_pipeline: Any | None = None
_agent_orchestrator: Any | None = None
_whatif_engine: Any | None = None
_response_plan_generator: Any | None = None


def get_pipeline() -> RAGPipeline:
    """Get or create the singleton RAG pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def set_pipeline(pipeline: RAGPipeline) -> None:
    """Override the pipeline instance (for testing or custom configs)."""
    global _pipeline
    _pipeline = pipeline


def get_sop_pipeline():
    """Get or create the singleton SOP pipeline."""
    global _sop_pipeline
    if _sop_pipeline is None:
        from rag.sop.pipeline import SOPPipeline
        _sop_pipeline = SOPPipeline(rag_pipeline=get_pipeline())
    return _sop_pipeline


def get_agent_orchestrator():
    """Get or create the singleton Agent Orchestrator."""
    global _agent_orchestrator
    if _agent_orchestrator is None:
        from rag.agent.orchestrator import AgentOrchestrator
        pipeline = get_pipeline()
        sop = get_sop_pipeline()
        model_provider = pipeline._get_model_provider()
        _agent_orchestrator = AgentOrchestrator(
            model_provider=model_provider,
            rag_pipeline=pipeline,
            sop_pipeline=sop,
        )
    return _agent_orchestrator


def get_whatif_engine():
    """Get or create the singleton What-If Scenario Engine."""
    global _whatif_engine
    if _whatif_engine is None:
        from rag.scenarios.what_if import WhatIfEngine
        _whatif_engine = WhatIfEngine(pipeline=get_pipeline())
    return _whatif_engine


def get_response_plan_generator():
    """Get or create the singleton Response Plan Generator."""
    global _response_plan_generator
    if _response_plan_generator is None:
        from rag.sop.plan_generator import ResponsePlanGenerator
        pipeline = get_pipeline()
        sop = get_sop_pipeline()
        _response_plan_generator = ResponsePlanGenerator(
            sop_pipeline=sop,
            rag_pipeline=pipeline,
        )
    return _response_plan_generator


async def _try_db_record_document(
    doc_id: str,
    filename: str,
    pages: int,
    chunks: int,
    metadata: dict,
) -> None:
    """Best-effort recording of ingested document in PostgreSQL/relational DB."""
    try:
        from rag.db.session import get_db
        from rag.db import crud
        async with get_db() as session:
            await crud.create_document(
                session=session,
                document_id=doc_id,
                filename=filename,
                total_pages=pages,
                chunks_created=chunks,
                metadata_json=metadata,
            )
    except Exception as exc:
        logger.debug("Database document record skipped: %s", exc)


async def _try_db_log_query(
    query: str,
    top_k: int,
    filters: Optional[dict],
    answer: Optional[str],
    sources: list,
    latency_ms: float,
) -> None:
    """Best-effort logging of queries into PostgreSQL/relational DB."""
    try:
        from rag.db.session import get_db
        from rag.db import crud
        async with get_db() as session:
            sources_dicts = [
                s.model_dump() if hasattr(s, "model_dump") else (s if isinstance(s, dict) else vars(s))
                for s in sources
            ]
            await crud.log_query(
                session=session,
                query_text=query,
                top_k=top_k,
                filters_json=filters,
                answer_text=answer,
                sources_json=sources_dicts,
                latency_ms=latency_ms,
            )
    except Exception as exc:
        logger.debug("Database query logging skipped: %s", exc)


# ── Ingest Endpoint ──────────────────────────────────────────────────────


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
    metadata: Optional[str] = Form(
        default=None,
        description='Optional JSON metadata, e.g. {"year": 2025, "department": "Finance"}',
    ),
):
    """Upload and ingest a PDF document into the RAG vector store.

    The document is processed once: text and tables are extracted,
    cleaned, chunked, embedded, and stored. Subsequent queries reuse
    the stored embeddings.
    """
    # Validate file type
    if file.filename and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    # Parse optional metadata
    extra_metadata = {}
    if metadata:
        import json
        try:
            extra_metadata = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON in metadata field.",
            )

    try:
        pipeline = get_pipeline()
        content = await file.read()
        result = await pipeline.ingest_document(
            source=content,
            filename=file.filename or "upload.pdf",
            metadata=extra_metadata,
        )
        # Asynchronously record in relational DB if configured
        await _try_db_record_document(
            doc_id=result.document_id,
            filename=result.filename,
            pages=result.total_pages,
            chunks=result.chunks_created,
            metadata=extra_metadata,
        )
        return result

    except (PDFLoadError, EmptyPDFError) as exc:
        raise HTTPException(status_code=400, detail=exc.message)
    except RAGError as exc:
        logger.error("Ingestion failed: %s", exc.message)
        raise HTTPException(status_code=500, detail=exc.message)
    except Exception as exc:
        logger.exception("Unexpected error during ingestion")
        raise HTTPException(status_code=500, detail="Internal server error during ingestion.")


# ── Query Endpoint ───────────────────────────────────────────────────────


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question and receive an answer with citations.

    The pipeline retrieves relevant chunks from ingested documents,
    builds a context-grounded prompt, and generates an answer via the
    configured LLM.

    Set ``generate=false`` in the request to retrieve context without
    calling the LLM.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_time = time.time()
    try:
        pipeline = get_pipeline()
        response = await pipeline.query(
            question=request.query,
            top_k=request.top_k,
            filters=request.filters,
            generate=request.generate,
        )
        latency_ms = round((time.time() - start_time) * 1000, 2)
        # Asynchronously log query in relational DB if configured
        await _try_db_log_query(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters,
            answer=response.answer,
            sources=response.sources,
            latency_ms=latency_ms,
        )
        return response

    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=exc.message)
    except RAGError as exc:
        logger.error("Query failed: %s", exc.message)
        raise HTTPException(status_code=500, detail=exc.message)
    except Exception as exc:
        logger.exception("Unexpected error during query")
        raise HTTPException(status_code=500, detail="Internal server error during query.")


# ── Retrieve Endpoint ───────────────────────────────────────────────────


@router.post("/retrieve")
async def retrieve_context(request: RetrieveRequest):
    """Retrieve relevant chunks without LLM generation.

    Use this endpoint when you want to feed the retrieved context
    into your own ML model or custom processing pipeline.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        pipeline = get_pipeline()
        result = pipeline.retrieve(
            question=request.query,
            top_k=request.top_k,
            filters=request.filters,
        )

        return {
            "query": result.query,
            "chunks": [
                {
                    "chunk_id": sc.chunk.chunk_id,
                    "document_id": sc.chunk.document_id,
                    "document_name": sc.chunk.document_name,
                    "page_number": sc.chunk.page_number,
                    "section": sc.chunk.section,
                    "content_type": sc.chunk.content_type.value,
                    "content": sc.chunk.content,
                    "score": sc.score,
                }
                for sc in result.chunks
            ],
            "total_results": len(result.chunks),
        }

    except RAGError as exc:
        logger.error("Retrieval failed: %s", exc.message)
        raise HTTPException(status_code=500, detail=exc.message)
    except Exception as exc:
        logger.exception("Unexpected error during retrieval")
        raise HTTPException(status_code=500, detail="Internal server error during retrieval.")


# ── Document Management ─────────────────────────────────────────────────


@router.get("/documents")
async def list_documents():
    """List all ingested documents in the vector store."""
    try:
        pipeline = get_pipeline()
        docs = pipeline.list_documents()
        return {"documents": docs, "total": len(docs)}
    except RAGError as exc:
        raise HTTPException(status_code=500, detail=exc.message)


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks from the vector store."""
    try:
        pipeline = get_pipeline()
        if not pipeline.document_exists(document_id):
            raise HTTPException(status_code=404, detail="Document not found.")
        await pipeline.delete_document(document_id)
        return {"message": f"Document {document_id} deleted successfully."}
    except HTTPException:
        raise
    except RAGError as exc:
        raise HTTPException(status_code=500, detail=exc.message)


# ── Agent Orchestrator Endpoint ──────────────────────────────────────────


@router.post("/agent", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """Execute a multi-step reasoning agent with tool calling.

    The agent reasons step-by-step using tools:
      - rag_search: Search government documents
      - data_extract: Extract figures and metrics
      - compare: Compare entities/metrics
      - calculate: Safe arithmetic calculations
      - sop_search: Search emergency SOP protocols

    Returns the answer and full step-by-step reasoning trace.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        orchestrator = get_agent_orchestrator()
        response = await orchestrator.run(request)
        return response
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=exc.message)
    except Exception as exc:
        logger.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(exc)}")


# ── What-If Scenario Endpoint ────────────────────────────────────────────


@router.post("/what-if", response_model=ScenarioResponse)
async def run_what_if_scenario(request: ScenarioRequest):
    """Analyze a hypothetical policy or socio-economic scenario.

    Grounds hypothetical projections in actual baseline data retrieved
    from government documents. Produces assumptions, sector-by-sector
    impacts, and citations.
    """
    if not request.scenario.strip():
        raise HTTPException(status_code=400, detail="Scenario cannot be empty.")

    try:
        engine = get_whatif_engine()
        response = await engine.analyze(request)
        return response
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=exc.message)
    except Exception as exc:
        logger.exception("What-if analysis failed")
        raise HTTPException(status_code=500, detail=f"What-If analysis failed: {str(exc)}")


# ── SOP Pipeline Endpoints ───────────────────────────────────────────────


@router.post("/sop/ingest", response_model=SOPIngestResponse)
async def ingest_sop_document(
    file: UploadFile = File(..., description="SOP PDF document to ingest"),
    category: str = Form(default="general", description="SOP category (e.g. 'disaster_management')"),
    department: str = Form(default="General Administration", description="Governing department"),
    version: str = Form(default="1.0", description="SOP version"),
    metadata: Optional[str] = Form(default=None, description="Optional extra metadata JSON string"),
):
    """Ingest a Standard Operating Procedure (SOP) into the dedicated SOP repository."""
    if file.filename and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    extra_meta = {}
    if metadata:
        import json
        try:
            extra_meta = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in metadata field.")

    try:
        sop_pipe = get_sop_pipeline()
        content = await file.read()
        res = await sop_pipe.ingest_sop(
            source=content,
            filename=file.filename or "sop.pdf",
            category=category,
            department=department,
            version=version,
            extra_metadata=extra_meta,
        )

        # Record in relational database if available
        try:
            from rag.db.session import get_db
            from rag.db import crud
            async with get_db() as session:
                await crud.create_sop_document(
                    session=session,
                    document_id=res.document_id,
                    filename=res.filename,
                    category=category,
                    department=department,
                    version=version,
                    metadata_json=extra_meta,
                )
        except Exception as exc:
            logger.debug("Database SOP record skipped: %s", exc)

        return res

    except (PDFLoadError, EmptyPDFError) as exc:
        raise HTTPException(status_code=400, detail=exc.message)
    except RAGError as exc:
        raise HTTPException(status_code=500, detail=exc.message)
    except Exception as exc:
        logger.exception("Unexpected error during SOP ingestion")
        raise HTTPException(status_code=500, detail="Internal server error during SOP ingestion.")


@router.post("/sop/response-plan", response_model=ResponsePlanResponse)
async def generate_response_plan(request: ResponsePlanRequest):
    """Generate a structured Operational Crisis/Incident Response Plan.

    Retrieves applicable SOP protocols and government policy reports, then
    synthesizes immediate actions, short-term actions, resource requirements,
    and escalation criteria.
    """
    if not request.situation.strip():
        raise HTTPException(status_code=400, detail="Situation description cannot be empty.")

    try:
        generator = get_response_plan_generator()
        plan = await generator.generate_plan(request)
        return plan
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=exc.message)
    except Exception as exc:
        logger.exception("Response plan generation failed")
        raise HTTPException(status_code=500, detail=f"Response plan generation failed: {str(exc)}")


@router.get("/sop/documents")
async def list_sop_documents():
    """List all ingested SOP documents."""
    try:
        sop_pipe = get_sop_pipeline()
        docs = sop_pipe.list_sops()
        return {"documents": docs, "total": len(docs)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

