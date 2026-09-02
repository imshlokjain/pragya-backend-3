from sqlalchemy.orm import Session

from backend.database.models import RiverObservation, Zone
from backend.schemas.risk import RiverOut
from backend.services import zone_service


def get_river(db: Session, zone_id: str) -> RiverOut | None:
    # Resolve legacy code or UUID -> canonical short code
    code = zone_service.resolve_zone_code(db, zone_id)

    if code is None:
        return None

    # Convert short code -> actual database Zone row
    zone = zone_service.zone_exists_by_code(
        db,
        code,
        zone_service.KNOWN_ZONES,
    )

    if zone is None:
        return None

    # Get latest river observation for this real DB zone
    latest = (
        db.query(RiverObservation)
        .filter(RiverObservation.zone_id == zone.id)
        .order_by(RiverObservation.timestamp.desc())
        .first()
    )

    if latest is None:
        return None

    warning_level = latest.warning_level or 0.0
    danger_level = latest.danger_level or 0.0

    distance_to_danger = round(
        danger_level - latest.water_level,
        2,
    )

    # Determine trend
    if latest.rate_of_change is not None:
        if latest.rate_of_change > 0:
            trend = "RISING"
        elif latest.rate_of_change < 0:
            trend = "FALLING"
        else:
            trend = "STABLE"
    else:
        previous = (
            db.query(RiverObservation)
            .filter(
                RiverObservation.zone_id == zone.id,
                RiverObservation.timestamp < latest.timestamp,
            )
            .order_by(RiverObservation.timestamp.desc())
            .first()
        )

        if previous is None:
            trend = "STABLE"
        elif latest.water_level > previous.water_level:
            trend = "RISING"
        elif latest.water_level < previous.water_level:
            trend = "FALLING"
        else:
            trend = "STABLE"

    return RiverOut(
        zone_id=zone_id,
        timestamp=latest.timestamp,
        river_name=latest.river_name or "Unknown River",
        current_level=latest.water_level,
        warning_level=warning_level,
        danger_level=danger_level,
        distance_to_danger=distance_to_danger,
        trend=trend,
        quality_status=latest.quality_status or "MISSING",
    )