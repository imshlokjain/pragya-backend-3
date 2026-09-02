from datetime import datetime, timedelta

from backend.database.models import SatelliteObservation
from backend.database.session import SessionLocal


def main():
    db = SessionLocal()

    try:
        existing = db.query(SatelliteObservation).count()

        if existing > 0:
            print(
                f"Satellite seed skipped: "
                f"{existing} observations already exist"
            )
            return

        now = datetime.utcnow()

        observations = [
            SatelliteObservation(
                source="DEMO_SENTINEL",
                scene_id="S2_DEMO_001",
                acquisition_time=now - timedelta(days=2),
                cloud_percentage=8.5,
                image_uri="demo://sentinel/S2_DEMO_001",
                processing_status="PROCESSED",
            ),
            SatelliteObservation(
                source="DEMO_SENTINEL",
                scene_id="S2_DEMO_002",
                acquisition_time=now - timedelta(days=1),
                cloud_percentage=12.0,
                image_uri="demo://sentinel/S2_DEMO_002",
                processing_status="PROCESSED",
            ),
            SatelliteObservation(
                source="DEMO_SENTINEL",
                scene_id="S2_DEMO_003",
                acquisition_time=now,
                cloud_percentage=5.2,
                image_uri="demo://sentinel/S2_DEMO_003",
                processing_status="PROCESSED",
            ),
        ]

        db.add_all(observations)
        db.commit()

        print(
            f"Satellite seed complete: "
            f"created={len(observations)}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
