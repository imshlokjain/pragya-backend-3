from datetime import datetime, timedelta

from backend.database.models import RainfallObservation, Zone
from backend.database.session import SessionLocal


RAINFALL_CONFIG = {
    "Riverside North": [12.4, 18.7, 22.1, 31.5, 28.6],
    "Market District": [10.2, 16.8, 20.4, 27.3, 24.9],
    "Lowland South": [15.1, 21.6, 26.3, 34.2, 30.8],
    "Upstream Colony": [8.7, 14.5, 19.8, 25.6, 29.4],
    "Embankment East": [11.3, 17.9, 23.5, 29.1, 26.7],
}


def seed_rainfall():
    db = SessionLocal()

    created = 0
    skipped = 0

    try:
        zones = db.query(Zone).all()

        for zone in zones:
            rainfall_values = RAINFALL_CONFIG.get(zone.name)

            if rainfall_values is None:
                continue

            now = datetime.utcnow()

            for i, rainfall_mm in enumerate(rainfall_values):
                timestamp = now - timedelta(hours=(len(rainfall_values) - i) * 6)

                existing = (
                    db.query(RainfallObservation)
                    .filter(
                        RainfallObservation.source == "DEMO_SEED",
                        RainfallObservation.station_id == f"RAIN_{zone.id}",
                        RainfallObservation.zone_id == zone.id,
                        RainfallObservation.timestamp == timestamp,
                    )
                    .first()
                )

                if existing:
                    skipped += 1
                    continue

                observation = RainfallObservation(
                    source="DEMO_SEED",
                    station_id=f"RAIN_{zone.id}",
                    zone_id=zone.id,
                    timestamp=timestamp,
                    rainfall_mm=rainfall_mm,
                    duration_minutes=60,
                    quality_status="FRESH",
                )

                db.add(observation)
                created += 1

        db.commit()

        print(
            f"Rainfall seed complete: "
            f"created={created}, skipped={skipped}"
        )

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    seed_rainfall()
