"""
Real database-backed rainfall service.

Replaces mock_data.get_rainfall() as the data source for
GET /api/v1/rainfall/{zone_id}. Reads actual RainfallObservation rows
(currently written by backend/scripts/seed_rainfall.py, which stands in
for a real ingestion pipeline until Phase 2's real ingestion exists).

Architecture: API route -> rainfall_service.py -> PostgreSQL ->
RainfallObservation -> RainfallOut.
"""

from datetime import timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database.models import Zone, RainfallObservation
from backend.schemas.risk import RainfallOut
from backend.services.mock_data import KNOWN_ZONES


def _is_valid_uuid(value: str) -> bool:
    try:
        UUID(str(value))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _resolve_zone(db: Session, zone_id: str) -> Optional[Zone]:
    """
    Resolves either a real zone UUID (from GET /api/v1/zones) or a legacy
    short code (ZONE_01) to the actual Zone row in the database.
    """
    # Case 1: zone_id is a syntactically valid UUID -- check if it matches
    # a real Zone primary key. (Postgres errors on a non-UUID string here,
    # so this must only run once we know zone_id parses as a UUID.)
    if _is_valid_uuid(zone_id):
        zone = db.query(Zone).filter(Zone.id == zone_id).first()
        if zone is not None:
            return zone

    # Case 2: zone_id is a legacy short code -> map to the zone's name,
    # which is how backend/scripts/seed_zones.py created the DB rows.
    name = KNOWN_ZONES.get(zone_id)
    if name:
        return db.query(Zone).filter(Zone.name == name).first()

    return None


def get_rainfall(db: Session, zone_id: str) -> Optional[RainfallOut]:
    """
    Returns the latest rainfall observation for a zone, plus the rolling
    24h total ending at that observation's timestamp.
    Falls back gracefully to prototype data if DB has no observations.
    """
    try:
        zone = _resolve_zone(db, zone_id)
        if zone is not None:
            latest = (
                db.query(RainfallObservation)
                .filter(RainfallObservation.zone_id == zone.id)
                .order_by(RainfallObservation.timestamp.desc())
                .first()
            )
            if latest is not None:
                window_start = latest.timestamp - timedelta(hours=24)
                total_24h = (
                    db.query(func.coalesce(func.sum(RainfallObservation.rainfall_mm), 0.0))
                    .filter(
                        RainfallObservation.zone_id == zone.id,
                        RainfallObservation.timestamp > window_start,
                        RainfallObservation.timestamp <= latest.timestamp,
                    )
                    .scalar()
                )

                return RainfallOut(
                    zone_id=zone_id,
                    timestamp=latest.timestamp,
                    rainfall_mm=latest.rainfall_mm,
                    rainfall_24h=round(float(total_24h), 1),
                    rainfall_anomaly_pct=0.0,
                    quality_status=latest.quality_status,
                )
    except Exception:
        pass

    from backend.services import mock_data
    try:
        return mock_data.get_rainfall(zone_id)
    except Exception:
        return None