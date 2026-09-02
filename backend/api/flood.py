from fastapi import APIRouter, Depends, HTTPException, Query, Query
from sqlalchemy.orm import Session

from backend.core.auth import require_roles
from backend.database.session import get_db
from backend.schemas.risk import FloodDetectionOut
from backend.services import flood_service


router = APIRouter(
    prefix="/api/v1/flood",
    tags=["flood"],
)


@router.get(
    "/{zone_id}",
    response_model=FloodDetectionOut,
)
def get_latest_flood_detection(
    zone_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(
        require_roles(
            "ADMIN",
            "OPERATOR",
            "ANALYST",
            "VIEWER",
        )
    ),
):
    result = flood_service.get_latest_flood_detection(
        db,
        zone_id,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No flood detection found "
                f"for zone '{zone_id}'"
            ),
        )

    return FloodDetectionOut(
        id=result.id,
        zone_id=zone_id,
        before_scene_id=result.before_scene_id,
        after_scene_id=result.after_scene_id,
        detected_at=result.detected_at,
        affected_area=result.affected_area,
        confidence=result.confidence,
        method=result.method,
        model_version=result.model_version,
    )


@router.get(
    "/{zone_id}/history",
    response_model=list[FloodDetectionOut],
)
def get_flood_history(
    zone_id: str,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    db: Session = Depends(get_db),
    current_user: dict = Depends(
        require_roles(
            "ADMIN",
            "OPERATOR",
            "ANALYST",
            "VIEWER",
        )
    ),
):
    detections = flood_service.list_flood_detections(
        db,
        zone_id,
        limit,
    )

    if detections is None:
        raise HTTPException(
            status_code=404,
            detail=f"Zone '{zone_id}' not found",
        )

    return [
        FloodDetectionOut(
            id=item.id,
            zone_id=zone_id,
            before_scene_id=item.before_scene_id,
            after_scene_id=item.after_scene_id,
            detected_at=item.detected_at,
            affected_area=item.affected_area,
            confidence=item.confidence,
            method=item.method,
            model_version=item.model_version,
        )
        for item in detections
    ]
