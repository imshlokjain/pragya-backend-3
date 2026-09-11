from typing import Optional

from sqlalchemy.orm import Session

from backend.services import (
    rainfall_service,
    river_service,
    risk_service,
    scenario_service,
)
from backend.services.sop_provider import sop_provider


def get_current_risk(
    db: Session,
    zone_id: str,
) -> dict:
    """
    Get current risk using the same pipeline
    as the REST risk endpoint.
    """

    risk = risk_service.get_risk(
        db,
        zone_id,
    )

    if risk is None:
        return {
            "error": f"Unknown zone_id: {zone_id}"
        }

    return risk


def get_rainfall(
    db: Session,
    zone_id: str,
) -> dict:

    rainfall = rainfall_service.get_rainfall(
        db,
        zone_id,
    )

    if rainfall is None:
        return {
            "error": f"Unknown zone_id: {zone_id}"
        }

    return rainfall.model_dump()


def get_river_level(
    db: Session,
    zone_id: str,
) -> dict:

    river = river_service.get_river(
        db,
        zone_id,
    )

    if river is None:
        return {
            "error": f"Unknown zone_id: {zone_id}"
        }

    return river.model_dump()


def get_satellite_evidence(
    db: Session,
    zone_id: str,
) -> dict:

    return {
        "zone_id": zone_id,
        "status": "NOT_YET_AVAILABLE",
        "note": (
            "Satellite/NDWI water-change analysis "
            "is not implemented yet."
        ),
    }


def run_scenario(
    db: Session,
    zone_id: str,
    rainfall_multiplier: float,
    river_level_increase: Optional[float] = None,
) -> dict:
    """
    Run a hypothetical scenario using the same
    scenario service as the REST API.
    """

    result = scenario_service.run_scenario(
        db=db,
        zone_id=zone_id,
        rainfall_multiplier=rainfall_multiplier,
        river_level_increase=river_level_increase,
    )

    if result is None:
        return {
            "error": f"Unknown zone_id: {zone_id}"
        }

    return result


def search_sop(
    query: str,
    context: Optional[dict] = None,
) -> dict:
    """
    Search SOPs through the RAG pipeline's SOP collection,
    falling back seamlessly to local prototype SOP provider.
    """
    try:
        from rag.api.router import get_sop_pipeline
        import asyncio
        sop_pipe = get_sop_pipeline()

        # Check vector store
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    results = pool.submit(asyncio.run, sop_pipe.search(query=query, top_k=4)).result()
            else:
                results = loop.run_until_complete(sop_pipe.search(query=query, top_k=4))
        except Exception:
            results = []

        if results:
            return {
                "query": query,
                "context": context or {},
                "status": "FOUND",
                "clauses": [
                    {
                        "id": f"SOP-RAG-{i+1}",
                        "title": r.get("section") or r.get("document", "NDMA SOP Manual"),
                        "content": r.get("content", ""),
                        "source": f"{r.get('document', 'NDMA SOP')} (Page {r.get('page', 1)})",
                    }
                    for i, r in enumerate(results)
                ],
                "sources": list(set(f"{r.get('document', 'SOP')} p.{r.get('page', 1)}" for r in results)),
                "note": "Retrieved from ChromaDB NDMA SOP Vector Store",
            }
    except Exception:
        pass

    return sop_provider.search(
        query=query,
        context=context,
    )


def search_rag_documents(
    query: str,
) -> dict:
    """
    Retrieve grounding context from general RAG documents (policy, manuals, surveys).
    """
    try:
        from rag.api.router import get_pipeline
        pipeline = get_pipeline()
        retrieval = pipeline.retrieve(question=query, top_k=3)
        if retrieval and retrieval.chunks:
            return {
                "status": "FOUND",
                "chunks": [
                    {
                        "document": sc.chunk.document_name,
                        "page": sc.chunk.page_number,
                        "content": sc.chunk.content,
                        "score": sc.score,
                    }
                    for sc in retrieval.chunks
                ],
                "sources": list(set(f"{sc.chunk.document_name} (p.{sc.chunk.page_number})" for sc in retrieval.chunks)),
            }
    except Exception:
        pass
    return {"status": "NO_DOCUMENTS"}

