import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.services import feature_service
from backend.services.ml_predictor import predictor


def _predict_from_features(features: dict) -> dict:
    """
    Extract values from the feature snapshot and call
    the Assam River ML predictor with the required arguments.
    """

    rainfall = features.get("rainfall", {})
    river = features.get("river", {})
    time_features = features.get("time", {})

    # ---------------------------------------------------------
    # TIME FEATURES
    # ---------------------------------------------------------

    month = int(
        time_features.get("month")
        or datetime.utcnow().month
    )

    day_of_year = int(
        time_features.get("day_of_year")
        or datetime.utcnow().timetuple().tm_yday
    )

    # ---------------------------------------------------------
    # RAINFALL FEATURES
    # ---------------------------------------------------------

    rainfall_mm = float(
        rainfall.get("rainfall_mm") or 0.0
    )

    rainfall_previous = float(
        rainfall.get("rainfall_previous") or 0.0
    )

    rainfall_3_obs = float(
        rainfall.get("rainfall_3_obs")
        or rainfall.get("rainfall_24h")
        or 0.0
    )

    rainfall_6_obs = float(
        rainfall.get("rainfall_6_obs")
        or rainfall.get("rainfall_24h")
        or 0.0
    )

    # ---------------------------------------------------------
    # RIVER FEATURES
    # ---------------------------------------------------------

    river_level = float(
        river.get("water_level")
        or river.get("current_level")
        or 0.0
    )

    river_level_change = float(
        river.get("river_level_change") or 0.0
    )

    river_level_rolling_mean_3 = float(
        river.get("river_level_rolling_mean_3")
        or river_level
    )

    river_level_rolling_max_6 = float(
        river.get("river_level_rolling_max_6")
        or river_level
    )

    # ---------------------------------------------------------
    # ML PREDICTION
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # CONVERT RIVER PREDICTION TO RISK
    # ---------------------------------------------------------

    warning_level = float(
        river.get("warning_level") or 8.0
    )

    danger_level = float(
        river.get("danger_level") or 9.0
    )

    drivers = []

    # Calculate risk based on predicted river level

    if predicted_level >= danger_level:
        risk_score = 95.0
        risk_category = "CRITICAL"

        drivers.append(
            "Predicted river level exceeds danger level"
        )

    elif predicted_level >= warning_level:
        risk_score = 75.0
        risk_category = "HIGH"

        drivers.append(
            "Predicted river level exceeds warning level"
        )

    elif predicted_level >= warning_level * 0.8:
        risk_score = 50.0
        risk_category = "MODERATE"

        drivers.append(
            "River level approaching warning level"
        )

    else:
        risk_score = 20.0
        risk_category = "LOW"

        drivers.append(
            "River level currently within safe range"
        )

    if river_level_change > 0:
        drivers.append("River level is rising")

    if predicted_level > river_level:
        drivers.append(
            "River level is forecast to rise"
        )

    if rainfall_mm > 20:
        drivers.append(
            "Moderate rainfall detected"
        )

    return {
        "risk_score": risk_score,
        "risk_category": risk_category,
        "confidence": 0.85,
        "forecast_horizon_hours": 6,
        "trend": (
            "RISING"
            if predicted_level > river_level
            else "STABLE"
        ),
        "model_version": "assam-river-rf-v1",
        "is_prototype": False,
        "drivers": drivers,
        "predicted_river_level": predicted_level,
    }


def run_scenario(
    db: Session,
    zone_id: str,
    rainfall_multiplier: float,
    river_level_increase: Optional[float] = None,
) -> dict | None:

    # ---------------------------------------------------------
    # GET CURRENT FEATURE SNAPSHOT
    # ---------------------------------------------------------

    try:
        features = feature_service.get_feature_snapshot(
            db,
            zone_id,
        )
    except Exception:
        features = None

    if features is None:
        from backend.services.mock_data import predict_risk, _seeded_random, _risk_category
        try:
            baseline = predict_risk(zone_id)
            rng = _seeded_random(f"{zone_id}-scenario-{rainfall_multiplier}-{river_level_increase}")
            rainfall_effect = (rainfall_multiplier - 1.0) * rng.uniform(25, 45)
            river_effect = (river_level_increase or 0) * rng.uniform(8, 15)
            scenario_score = baseline.risk_score + rainfall_effect + river_effect
            scenario_score = max(0.0, min(100.0, round(scenario_score, 1)))
            return {
                "scenario_id": str(uuid.uuid4()),
                "zone_id": zone_id,
                "baseline_risk": baseline.risk_score,
                "baseline_category": baseline.risk_category,
                "scenario_risk": scenario_score,
                "scenario_category": _risk_category(scenario_score),
                "risk_change": round(scenario_score - baseline.risk_score, 1),
                "parameters": {
                    "rainfall_multiplier": rainfall_multiplier,
                    "river_level_increase": river_level_increase,
                },
                "baseline_timestamp": datetime.utcnow(),
                "model_version": "assam-river-rf-v1 (fallback)",
                "is_hypothetical": True,
            }
        except Exception:
            return None

    # ---------------------------------------------------------
    # BASELINE PREDICTION
    # ---------------------------------------------------------

    baseline_prediction = _predict_from_features(
        features
    )

    # ---------------------------------------------------------
    # COPY FEATURES
    # ---------------------------------------------------------

    scenario_features = {
        key: (
            value.copy()
            if isinstance(value, dict)
            else value
        )
        for key, value in features.items()
    }

    # ---------------------------------------------------------
    # APPLY RAINFALL SCENARIO
    # ---------------------------------------------------------

    rainfall = scenario_features.setdefault(
        "rainfall",
        {},
    )

    for key in [
        "rainfall_mm",
        "rainfall_previous",
        "rainfall_3_obs",
        "rainfall_6_obs",
        "rainfall_24h",
    ]:
        if key in rainfall and rainfall[key] is not None:
            rainfall[key] = round(
                float(rainfall[key])
                * rainfall_multiplier,
                2,
            )

    # ---------------------------------------------------------
    # APPLY RIVER LEVEL SCENARIO
    # ---------------------------------------------------------

    if river_level_increase is not None:

        river = scenario_features.setdefault(
            "river",
            {},
        )

        current_level = float(
            river.get("water_level")
            or river.get("current_level")
            or 0.0
        )

        new_level = (
            current_level
            + river_level_increase
        )

        river["water_level"] = round(
            new_level,
            2,
        )

        # Keep current_level consistent if it exists
        if "current_level" in river:
            river["current_level"] = round(
                new_level,
                2,
            )

        # Update rolling features
        river["river_level_rolling_mean_3"] = (
            round(new_level, 2)
        )

        river["river_level_rolling_max_6"] = (
            max(
                float(
                    river.get(
                        "river_level_rolling_max_6"
                    ) or 0.0
                ),
                new_level,
            )
        )

        river["river_level_change"] = round(
            river_level_increase,
            2,
        )

        danger_level = river.get(
            "danger_level"
        )

        if danger_level is not None:
            river["distance_to_danger"] = round(
                float(danger_level)
                - new_level,
                2,
            )

    # ---------------------------------------------------------
    # SCENARIO PREDICTION
    # ---------------------------------------------------------

    scenario_prediction = _predict_from_features(
        scenario_features
    )

    baseline_score = float(
        baseline_prediction["risk_score"]
    )

    scenario_score = float(
        scenario_prediction["risk_score"]
    )

    # ---------------------------------------------------------
    # RETURN RESULT
    # ---------------------------------------------------------

    return {
        "scenario_id": str(uuid.uuid4()),
        "zone_id": zone_id,

        "baseline_risk": baseline_score,
        "baseline_category": (
            baseline_prediction["risk_category"]
        ),

        "scenario_risk": scenario_score,
        "scenario_category": (
            scenario_prediction["risk_category"]
        ),

        "risk_change": round(
            scenario_score - baseline_score,
            1,
        ),

        "parameters": {
            "rainfall_multiplier": rainfall_multiplier,
            "river_level_increase": (
                river_level_increase
            ),
        },

        "baseline_timestamp": datetime.utcnow(),

        "model_version": (
            scenario_prediction["model_version"]
        ),

        "is_hypothetical": True,
    }
