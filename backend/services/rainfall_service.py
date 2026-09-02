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

    Returns None if the zone doesn't exist, or if it exists but has no
    rainfall observations yet -- callers (the API route) are responsible
    for turning that into the appropriate 404.
    """
    zone = _resolve_zone(db, zone_id)
    if zone is None:
        return None

    latest = (
        db.query(RainfallObservation)
        .filter(RainfallObservation.zone_id == zone.id)
        .order_by(RainfallObservation.timestamp.desc())
        .first()
    )
    if latest is None:
        return None

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
        zone_id=zone_id,  # echo back whatever identifier the caller used
        timestamp=latest.timestamp,
        rainfall_mm=latest.rainfall_mm,
        rainfall_24h=round(float(total_24h), 1),
        rainfall_anomaly_pct=0.0,  # TODO: no historical baseline yet (TRD section 12)
        quality_status=latest.quality_status,
    )