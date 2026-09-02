from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.services import feature_service
from backend.core.auth import require_roles


router = APIRouter(
    prefix="/api/v1/features",
    tags=["features"],
)


@router.get("/{zone_id}")
def get_zone_features(
    zone_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(
        require_roles("ADMIN", "OPERATOR", "ANALYST", "VIEWER")
    ),
):
    result = feature_service.get_feature_snapshot(
        db,
        zone_id,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Zone '{zone_id}' not found",
        )

    return result
