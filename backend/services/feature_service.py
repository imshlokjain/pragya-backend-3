from datetime import timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database.models import (
    FloodDetection,
    RainfallObservation,
    RiverObservation,
    SatelliteObservation,
)
from backend.services import zone_service


def get_feature_snapshot(
    db: Session,
    zone_id: str,
):
    """
    Build the ML-ready feature snapshot for a zone.

    This service is responsible only for collecting and
    normalizing backend data.

    It does NOT perform ML prediction.
    """

    code = zone_service.resolve_zone_code(
        db,
        zone_id,
    )

    if code is None:
        return None

    zone = zone_service.zone_exists_by_code(
        db,
        code,
        zone_service.KNOWN_ZONES,
    )

    if zone is None:
        return None

    # ---------------------------------------------------------
    # Latest rainfall
    # ---------------------------------------------------------

    latest_rainfall = (
        db.query(RainfallObservation)
        .filter(
            RainfallObservation.zone_id == zone.id
        )
        .order_by(
            RainfallObservation.timestamp.desc()
        )
        .first()
    )

    rainfall_24h = 0.0
    rainfall_anomaly = 0.0

    if latest_rainfall is not None:

        window_start = (
            latest_rainfall.timestamp
            - timedelta(hours=24)
        )

        rainfall_24h = (
            db.query(
                func.coalesce(
                    func.sum(
                        RainfallObservation.rainfall_mm
                    ),
                    0.0,
                )
            )
            .filter(
                RainfallObservation.zone_id == zone.id,
                RainfallObservation.timestamp
                > window_start,
                RainfallObservation.timestamp
                <= latest_rainfall.timestamp,
            )
            .scalar()
        )

        rainfall_24h = float(
            rainfall_24h or 0.0
        )

    # ---------------------------------------------------------
    # Latest river observation
    # ---------------------------------------------------------

    latest_river = (
        db.query(RiverObservation)
        .filter(
            RiverObservation.zone_id == zone.id
        )
        .order_by(
            RiverObservation.timestamp.desc()
        )
        .first()
    )

    river_level = None
    river_level_change = None
    warning_level = None
    danger_level = None
    distance_to_danger = None

    if latest_river is not None:

        river_level = latest_river.water_level

        river_level_change = (
            latest_river.rate_of_change
        )

        warning_level = (
            latest_river.warning_level
        )

        danger_level = (
            latest_river.danger_level
        )

        if danger_level is not None:
            distance_to_danger = round(
                danger_level - river_level,
                2,
            )

        # If rate_of_change is missing,
        # compare with previous observation.
        if river_level_change is None:

            previous_river = (
                db.query(RiverObservation)
                .filter(
                    RiverObservation.zone_id
                    == zone.id,
                    RiverObservation.timestamp
                    < latest_river.timestamp,
                )
                .order_by(
                    RiverObservation.timestamp.desc()
                )
                .first()
            )

            if previous_river is not None:
                river_level_change = round(
                    river_level
                    - previous_river.water_level,
                    2,
                )

    # ---------------------------------------------------------
    # Latest flood detection
    # ---------------------------------------------------------

    latest_flood = (
        db.query(FloodDetection)
        .filter(
            FloodDetection.zone_id == zone.id
        )
        .order_by(
            FloodDetection.detected_at.desc()
        )
        .first()
    )

    flood_affected_area = None
    flood_confidence = None

    if latest_flood is not None:

        flood_affected_area = (
            latest_flood.affected_area
        )

        flood_confidence = (
            latest_flood.confidence
        )

    # ---------------------------------------------------------
    # Historical flood frequency
    # ---------------------------------------------------------

    historical_flood_frequency = (
        db.query(
            func.count(FloodDetection.id)
        )
        .filter(
            FloodDetection.zone_id == zone.id
        )
        .scalar()
    )

    historical_flood_frequency = int(
        historical_flood_frequency or 0
    )

    # ---------------------------------------------------------
    # Latest satellite observation
    # ---------------------------------------------------------

    latest_satellite = (
        db.query(SatelliteObservation)
        .order_by(
            SatelliteObservation.acquisition_time.desc()
        )
        .first()
    )

    satellite_cloud_percentage = None
    satellite_processing_status = None
    satellite_scene_id = None

    if latest_satellite is not None:

        satellite_cloud_percentage = (
            latest_satellite.cloud_percentage
        )

        satellite_processing_status = (
            latest_satellite.processing_status
        )

        satellite_scene_id = (
            latest_satellite.scene_id
        )

    # ---------------------------------------------------------
    # ML-ready feature snapshot
    # ---------------------------------------------------------

    return {
        "zone_id": zone_id,

        "zone": {
            "name": zone.name,
            "district_id": zone.district_id,
            "population": zone.population,
            "vulnerability_index": (
                zone.vulnerability_index
            ),
        },

        "rainfall": {
            "rainfall_mm": (
                latest_rainfall.rainfall_mm
                if latest_rainfall is not None
                else None
            ),
            "rainfall_24h": round(
                rainfall_24h,
                2,
            ),
            "rainfall_anomaly_pct": (
                rainfall_anomaly
            ),
            "quality_status": (
                latest_rainfall.quality_status
                if latest_rainfall is not None
                else "MISSING"
            ),
        },

        "river": {
            "water_level": river_level,
            "river_level_change": (
                river_level_change
            ),
            "warning_level": warning_level,
            "danger_level": danger_level,
            "distance_to_danger": (
                distance_to_danger
            ),
            "quality_status": (
                latest_river.quality_status
                if latest_river is not None
                else "MISSING"
            ),
        },

        "satellite": {
            "scene_id": satellite_scene_id,
            "cloud_percentage": (
                satellite_cloud_percentage
            ),
            "processing_status": (
                satellite_processing_status
            ),
            "zone_specific": False,
        },

        "flood": {
            "affected_area": (
                flood_affected_area
            ),
            "confidence": flood_confidence,
            "historical_flood_frequency": (
                historical_flood_frequency
            ),
            "method": (
                latest_flood.method
                if latest_flood is not None
                else None
            ),
            "model_version": (
                latest_flood.model_version
                if latest_flood is not None
                else None
            ),
        },

        "feature_status": {
            "rainfall_available": (
                latest_rainfall is not None
            ),
            "river_available": (
                latest_river is not None
            ),
            "satellite_available": (
                latest_satellite is not None
            ),
            "flood_detection_available": (
                latest_flood is not None
            ),
        },
    }
