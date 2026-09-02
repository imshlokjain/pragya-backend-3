from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.schemas.risk import SatelliteObservationOut
from backend.services import satellite_service
from backend.core.auth import require_roles


router = APIRouter(
    prefix="/api/v1/satellite",
    tags=["satellite"],
)


@router.get(
    "/latest",
    response_model=SatelliteObservationOut,
)
def get_latest_satellite(
    db: Session = Depends(get_db),
    current_user: dict = Depends(
        require_roles("ADMIN", "OPERATOR", "ANALYST", "VIEWER")
    ),
):
    result = satellite_service.get_latest_satellite_observation(db)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No satellite observations available",
        )

    return SatelliteObservationOut(
        id=result.id,
        source=result.source,
        scene_id=result.scene_id,
        acquisition_time=result.acquisition_time,
        cloud_percentage=result.cloud_percentage,
        image_uri=result.image_uri,
        processing_status=result.processing_status,
    )


@router.get(
    "",
    response_model=list[SatelliteObservationOut],
)
def list_satellite(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    db: Session = Depends(get_db),
    current_user: dict = Depends(
        require_roles("ADMIN", "OPERATOR", "ANALYST", "VIEWER")
    ),
):
    observations = satellite_service.list_satellite_observations(
        db,
        limit,
    )

    return [
        SatelliteObservationOut(
            id=item.id,
            source=item.source,
            scene_id=item.scene_id,
            acquisition_time=item.acquisition_time,
            cloud_percentage=item.cloud_percentage,
            image_uri=item.image_uri,
            processing_status=item.processing_status,
        )
        for item in observations
    ]
