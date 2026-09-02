"""
Seed the pilot district's zones into the real database.

Run once after migrations:
    python -m backend.scripts.seed_zones

This replaces the in-memory KNOWN_ZONES dict in mock_data.py as the source
of truth for zone identity/geometry. mock_data.py still generates the
risk/rainfall/river *values* for now (that's the ML teammate's job to
replace later) — but which zones exist, their names, and their boundaries
now live in Postgres where they belong.

Coordinates are illustrative placeholders around a generic Indian district
centroid — swap in your pilot district's real zone boundaries when available.
"""

from shapely.geometry import Polygon, MultiPolygon
from geoalchemy2.shape import from_shape

from backend.database.session import SessionLocal
from backend.database.models import Zone

PILOT_DISTRICT_ID = "D001"

# zone_id -> (name, population, vulnerability_index, polygon bounding box)
SEED_ZONES = {
    "ZONE_01": ("Riverside North", 18000, 0.55, (76.34, 30.36, 76.38, 30.40)),
    "ZONE_02": ("Market District", 42000, 0.35, (76.36, 30.33, 76.40, 30.36)),
    "ZONE_03": ("Lowland South", 9500, 0.78, (76.32, 30.28, 76.36, 30.32)),
    "ZONE_04": ("Upstream Colony", 12500, 0.40, (76.30, 30.34, 76.34, 30.38)),
    "ZONE_05": ("Embankment East", 21000, 0.62, (76.40, 30.31, 76.44, 30.35)),
}


def bbox_to_multipolygon(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    poly = Polygon([
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
    ])
    return MultiPolygon([poly])


def seed():
    db = SessionLocal()
    try:
        created, skipped = 0, 0
        for zone_key, (name, population, vulnerability, bbox) in SEED_ZONES.items():
            existing = db.query(Zone).filter(
                Zone.district_id == PILOT_DISTRICT_ID, Zone.name == name
            ).first()
            if existing:
                skipped += 1
                continue

            geom = bbox_to_multipolygon(bbox)
            zone = Zone(
                district_id=PILOT_DISTRICT_ID,
                name=name,
                geometry=from_shape(geom, srid=4326),
                population=population,
                vulnerability_index=vulnerability,
            )
            db.add(zone)
            created += 1

        db.commit()
        print(f"Seed complete: {created} zones created, {skipped} already existed.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
