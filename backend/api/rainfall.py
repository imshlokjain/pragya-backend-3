from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.schemas.risk import RainfallOut
from backend.services import rainfall_service
from backend.core.auth import require_roles


router = APIRouter(
    prefix="/api/v1/rainfall",
    tags=["rainfall"],
)


@router.get("/{zone_id}", response_model=RainfallOut)
def get_zone_rainfall(
    zone_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(
        require_roles("ADMIN", "OPERATOR", "ANALYST", "VIEWER")
    ),
):
    result = rainfall_service.get_rainfall(db, zone_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Zone '{zone_id}' not found or rainfall data unavailable",
        )

    return result