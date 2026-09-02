from datetime import datetime, timedelta

from backend.database.models import FloodDetection, Zone
from backend.database.session import SessionLocal


FLOOD_CONFIG = {
    "Riverside North": {
        "affected_area": 2.8,
        "confidence": 0.91,
    },
    "Market District": {
        "affected_area": 1.4,
        "confidence": 0.84,
    },
    "Lowland South": {
        "affected_area": 4.6,
        "confidence": 0.94,
    },
    "Upstream Colony": {
        "affected_area": 0.9,
        "confidence": 0.78,
    },
    "Embankment East": {
        "affected_area": 2.1,
        "confidence": 0.88,
    },
}


def main():
    db = SessionLocal()

    try:
        existing = db.query(FloodDetection).count()

        if existing > 0:
            print(
                f"Flood seed skipped: "
                f"{existing} detections already exist"
            )
            return

        zones = db.query(Zone).all()

        detections = []

        for zone in zones:
            config = FLOOD_CONFIG.get(zone.name)

            if config is None:
                continue

            detections.append(
                FloodDetection(
                    zone_id=zone.id,
                    before_scene_id="S2_DEMO_002",
                    after_scene_id="S2_DEMO_003",
                    detected_at=datetime.utcnow()
                    - timedelta(minutes=10),
                    affected_area=config["affected_area"],
                    confidence=config["confidence"],
                    method="NDWI",
                    model_version="prototype-flood-v0.1",
                )
            )

        db.add_all(detections)
        db.commit()

        print(
            f"Flood seed complete: "
            f"created={len(detections)}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
