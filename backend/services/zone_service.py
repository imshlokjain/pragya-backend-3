"""
Real database-backed zone lookups.

Zone identity, names, and geometry come from PostgreSQL.
Legacy ZONE_01-style codes are supported as a bridge for the MVP.
"""

import uuid
from typing import Optional

from geoalchemy2.shape import to_shape
from sqlalchemy.orm import Session

from backend.database.models import Zone
from backend.services.mock_data import KNOWN_ZONES


NAME_TO_CODE = {
    name: code
    for code, name in KNOWN_ZONES.items()
}


def resolve_zone_code(
    db: Session,
    zone_id: str,
) -> Optional[str]:
    """
    Accept either:

        ZONE_01

    or:

        real database UUID

    and return the corresponding short zone code.
    """

    if zone_id in KNOWN_ZONES:
        return zone_id

    # Never send an arbitrary string into a PostgreSQL UUID column.
    try:
        uuid.UUID(zone_id)
    except (ValueError, AttributeError, TypeError):
        return None

    zone = (
        db.query(Zone)
        .filter(Zone.id == zone_id)
        .first()
    )

    if zone and zone.name in NAME_TO_CODE:
        return NAME_TO_CODE[zone.name]

    return None


def list_zones(db: Session) -> list[dict]:
    zones = (
        db.query(Zone)
        .order_by(Zone.name)
        .all()
    )

    return [
        _serialize(zone)
        for zone in zones
    ]


def get_zone(
    db: Session,
    zone_id: str,
) -> dict | None:

    # Support only real UUIDs for direct DB lookup.
    try:
        uuid.UUID(zone_id)
    except (ValueError, AttributeError, TypeError):
        return None

    zone = (
        db.query(Zone)
        .filter(Zone.id == zone_id)
        .first()
    )

    if zone is None:
        return None

    return _serialize(zone)


def zone_exists_by_code(
    db: Session,
    zone_code: str,
    code_to_name: dict,
) -> Zone | None:
    """
    Bridge legacy short codes such as ZONE_01 to
    real database Zone rows.
    """

    name = code_to_name.get(zone_code)

    if not name:
        return None

    return (
        db.query(Zone)
        .filter(Zone.name == name)
        .first()
    )


def _serialize(zone: Zone) -> dict:
    geom = (
        to_shape(zone.geometry)
        if zone.geometry is not None
        else None
    )

    return {
        "id": zone.id,
        "district_id": zone.district_id,
        "name": zone.name,
        "population": zone.population,
        "vulnerability_index": zone.vulnerability_index,
        "bounds": (
            geom.bounds
            if geom
            else None
        ),
    }
