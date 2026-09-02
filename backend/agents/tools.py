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
    Search SOPs through the provider interface.

    The real RAG system can replace sop_provider
    without changing the chat orchestrator.
    """

    return sop_provider.search(
        query=query,
        context=context,
    )
