from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.database.models import FloodDetection
from backend.services import zone_service


def create_flood_detection(
    db: Session,
    zone_id: str,
    before_scene_id: Optional[str] = None,
    after_scene_id: Optional[str] = None,
    affected_area: Optional[float] = None,
    confidence: Optional[float] = None,
    geometry=None,
    method: str = "NDWI",
    model_version: Optional[str] = None,
):
    code = zone_service.resolve_zone_code(db, zone_id)

    if code is None:
        return None

    zone = zone_service.zone_exists_by_code(
        db,
        code,
        zone_service.KNOWN_ZONES,
    )

    if zone is None:
        return None

    detection = FloodDetection(
        zone_id=zone.id,
        before_scene_id=before_scene_id,
        after_scene_id=after_scene_id,
        detected_at=datetime.utcnow(),
        affected_area=affected_area,
        confidence=confidence,
        geometry=geometry,
        method=method,
        model_version=model_version,
    )

    db.add(detection)
    db.commit()
    db.refresh(detection)

    return detection


def get_latest_flood_detection(
    db: Session,
    zone_id: str,
):
    code = zone_service.resolve_zone_code(db, zone_id)

    if code is None:
        return None

    zone = zone_service.zone_exists_by_code(
        db,
        code,
        zone_service.KNOWN_ZONES,
    )

    if zone is None:
        return None

    return (
        db.query(FloodDetection)
        .filter(FloodDetection.zone_id == zone.id)
        .order_by(FloodDetection.detected_at.desc())
        .first()
    )


def list_flood_detections(
    db: Session,
    zone_id: str,
    limit: int = 20,
):
    code = zone_service.resolve_zone_code(db, zone_id)

    if code is None:
        return None

    zone = zone_service.zone_exists_by_code(
        db,
        code,
        zone_service.KNOWN_ZONES,
    )

    if zone is None:
        return None

    return (
        db.query(FloodDetection)
        .filter(FloodDetection.zone_id == zone.id)
        .order_by(FloodDetection.detected_at.desc())
        .limit(limit)
        .all()
    )
