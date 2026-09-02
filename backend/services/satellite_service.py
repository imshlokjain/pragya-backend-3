from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.database.models import SatelliteObservation


def create_satellite_observation(
    db: Session,
    source: str,
    scene_id: Optional[str],
    acquisition_time: datetime,
    cloud_percentage: Optional[float] = None,
    image_uri: Optional[str] = None,
    processing_status: str = "PENDING",
):
    observation = SatelliteObservation(
        source=source,
        scene_id=scene_id,
        acquisition_time=acquisition_time,
        cloud_percentage=cloud_percentage,
        image_uri=image_uri,
        processing_status=processing_status,
    )

    db.add(observation)
    db.commit()
    db.refresh(observation)

    return observation


def get_latest_satellite_observation(
    db: Session,
) -> Optional[SatelliteObservation]:
    return (
        db.query(SatelliteObservation)
        .order_by(SatelliteObservation.acquisition_time.desc())
        .first()
    )


def list_satellite_observations(
    db: Session,
    limit: int = 20,
):
    return (
        db.query(SatelliteObservation)
        .order_by(SatelliteObservation.acquisition_time.desc())
        .limit(limit)
        .all()
    )
