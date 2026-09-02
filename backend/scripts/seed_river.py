from datetime import datetime, timedelta

from backend.database.models import RiverObservation, Zone
from backend.database.session import SessionLocal


RIVER_CONFIG = {
    "Riverside North": {
        "river_name": "Ghaggar River",
        "warning_level": 9.0,
        "danger_level": 9.5,
        "base_level": 8.4,
    },
    "Market District": {
        "river_name": "Ghaggar River",
        "warning_level": 8.5,
        "danger_level": 9.0,
        "base_level": 7.9,
    },
    "Lowland South": {
        "river_name": "Ghaggar River",
        "warning_level": 8.0,
        "danger_level": 8.5,
        "base_level": 7.4,
    },
    "Upstream Colony": {
        "river_name": "Ghaggar River",
        "warning_level": 9.5,
        "danger_level": 10.0,
        "base_level": 8.8,
    },
    "Embankment East": {
        "river_name": "Ghaggar River",
        "warning_level": 8.8,
        "danger_level": 9.3,
        "base_level": 8.2,
    },
}


def seed_river():
    db = SessionLocal()

    created = 0
    skipped = 0

    try:
        zones = db.query(Zone).all()

        for zone in zones:
            config = RIVER_CONFIG.get(zone.name)

            if config is None:
                continue

            now = datetime.utcnow()

            levels = [
                config["base_level"],
                config["base_level"] + 0.15,
                config["base_level"] + 0.30,
                config["warning_level"] - 0.10,
                config["warning_level"] + 0.10,
                config["danger_level"] - 0.15,
            ]

            for i, level in enumerate(levels):
                timestamp = now - timedelta(hours=(len(levels) - i) * 4)

                existing = (
                    db.query(RiverObservation)
                    .filter(
                        RiverObservation.source == "DEMO_SEED",
                        RiverObservation.station_id == f"RIVER_{zone.id}",
                        RiverObservation.zone_id == zone.id,
                        RiverObservation.timestamp == timestamp,
                    )
                    .first()
                )

                if existing:
                    skipped += 1
                    continue

                previous_level = levels[i - 1] if i > 0 else level
                rate_of_change = round(level - previous_level, 3)

                observation = RiverObservation(
                    source="DEMO_SEED",
                    station_id=f"RIVER_{zone.id}",
                    river_name=config["river_name"],
                    zone_id=zone.id,
                    timestamp=timestamp,
                    water_level=round(level, 2),
                    warning_level=config["warning_level"],
                    danger_level=config["danger_level"], 
                    rate_of_change=rate_of_change,
                    quality_status="FRESH",
                )

                db.add(observation)
                created += 1

        db.commit()

        print(f"River seed complete: created={created}, skipped={skipped}")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    seed_river()