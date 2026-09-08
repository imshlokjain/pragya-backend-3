from datetime import timedelta

from sqlalchemy.orm import Session

from backend.database.models import (
    RiverObservation,
    RainfallObservation,
)
from backend.services import zone_service
from backend.services.ml_predictor import predictor


def get_river_forecast(db: Session, zone_id: str):

    # Resolve zone
    code = zone_service.resolve_zone_code(db, zone_id)

    if code is None:
        return None

    zone = zone_service.zone_exists_by_code(
        db,
        code,
        zone_service.KNOWN_ZONES,
    )

    if zone is None:
        return None

    # --------------------------------------------------
    # GET LATEST REAL RIVER OBSERVATION
    # --------------------------------------------------

    river_records = (
        db.query(RiverObservation)
        .filter(RiverObservation.zone_id == zone.id)
        .order_by(RiverObservation.timestamp.desc())
        .limit(6)
        .all()
    )

    if not river_records:
        return None

    latest = river_records[0]

    # Reverse so records are oldest -> newest
    river_history = list(reversed(river_records))

    river_levels = [r.water_level for r in river_history]

    # --------------------------------------------------
    # RIVER FEATURES
    # --------------------------------------------------

    river_level = latest.water_level

    if len(river_levels) >= 2:
        river_level_change = (
            river_levels[-1] - river_levels[-2]
        )
    else:
        river_level_change = 0.0

    river_level_rolling_mean_3 = (
        sum(river_levels[-3:]) /
        len(river_levels[-3:])
    )

    river_level_rolling_max_6 = max(river_levels)

    # --------------------------------------------------
    # RAINFALL FEATURES
    # --------------------------------------------------

    rainfall_records = (
        db.query(RainfallObservation)
        .filter(
            RainfallObservation.zone_id == zone.id,
            RainfallObservation.timestamp <= latest.timestamp,
        )
        .order_by(RainfallObservation.timestamp.desc())
        .limit(6)
        .all()
    )

    rainfall_values = [
        r.rainfall_mm
        for r in rainfall_records
    ]

    rainfall_mm = (
        rainfall_values[0]
        if len(rainfall_values) > 0
        else 0.0
    )

    rainfall_previous = (
        rainfall_values[1]
        if len(rainfall_values) > 1
        else 0.0
    )

    rainfall_3_obs = sum(rainfall_values[:3])

    rainfall_6_obs = sum(rainfall_values[:6])

    # --------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------

    timestamp = latest.timestamp

    month = timestamp.month
    day_of_year = timestamp.timetuple().tm_yday

    # --------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------

    predicted_level = predictor.predict(
        month=month,
        day_of_year=day_of_year,
        rainfall_mm=rainfall_mm,
        rainfall_previous=rainfall_previous,
        rainfall_3_obs=rainfall_3_obs,
        rainfall_6_obs=rainfall_6_obs,
        river_level=river_level,
        river_level_change=river_level_change,
        river_level_rolling_mean_3=river_level_rolling_mean_3,
        river_level_rolling_max_6=river_level_rolling_max_6,
    )

    return {
        "zone_id": zone_id,
        "timestamp": timestamp,
        "current_river_level": round(river_level, 3),
        "predicted_river_level": round(predicted_level, 3),
        "forecast_horizon_hours": 6,

        "features": {
            "rainfall_mm": rainfall_mm,
            "rainfall_3_obs": rainfall_3_obs,
            "rainfall_6_obs": rainfall_6_obs,
            "river_level_change": river_level_change,
        },
    }
