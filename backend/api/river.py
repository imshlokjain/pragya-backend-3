from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.schemas.risk import RiverOut
from backend.services import river_service
from backend.services import river_forecast_service
from backend.core.auth import require_roles


router = APIRouter(
    prefix="/api/v1/river",
    tags=["river"],
)


@router.get("/{zone_id}/forecast")
def get_zone_river_forecast(
    zone_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(
        require_roles("ADMIN", "OPERATOR", "ANALYST", "VIEWER")
    ),
):
    result = river_forecast_service.get_river_forecast(
        db,
        zone_id,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Zone '{zone_id}' not found or insufficient data",
        )

    return result


@router.get("/{zone_id}", response_model=RiverOut)
def get_zone_river(
    zone_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(
        require_roles("ADMIN", "OPERATOR", "ANALYST", "VIEWER")
    ),
):
    result = river_service.get_river(db, zone_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Zone '{zone_id}' not found or river data unavailable",
        )

    return result
