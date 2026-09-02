import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.services import feature_service
from backend.services import zone_service
from backend.services.ml_predictor import predictor


def run_scenario(
    db: Session,
    zone_id: str,
    rainfall_multiplier: float,
    river_level_increase: Optional[float] = None,
) -> dict | None:
    """
    Run a hypothetical flood-risk scenario.

    The scenario uses the same feature aggregation and
    ML predictor used by the normal risk endpoint.

    No database prediction is persisted because this
    represents a hypothetical scenario.
    """

    # ---------------------------------------------------------
    # Get current feature snapshot
    # ---------------------------------------------------------

    features = feature_service.get_feature_snapshot(
        db,
        zone_id,
    )

    if features is None:
        return None

    # ---------------------------------------------------------
    # Calculate baseline using the same predictor
    # ---------------------------------------------------------

    baseline_prediction = predictor.predict(
        features
    )

    # ---------------------------------------------------------
    # Copy features before modifying them
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
    # Apply rainfall scenario
    # ---------------------------------------------------------

    rainfall = scenario_features.setdefault(
        "rainfall",
        {},
    )

    current_rainfall_24h = float(
        rainfall.get("rainfall_24h") or 0.0
    )

    rainfall["rainfall_24h"] = round(
        current_rainfall_24h
        * rainfall_multiplier,
        2,
    )

    current_rainfall = float(
        rainfall.get("rainfall_mm") or 0.0
    )

    rainfall["rainfall_mm"] = round(
        current_rainfall
        * rainfall_multiplier,
        2,
    )

    # ---------------------------------------------------------
    # Apply river-level scenario
    # ---------------------------------------------------------

    if river_level_increase is not None:
        river = scenario_features.setdefault(
            "river",
            {},
        )

        current_level = float(
            river.get("water_level") or 0.0
        )

        new_level = (
            current_level
            + river_level_increase
        )

        river["water_level"] = round(
            new_level,
            2,
        )

        warning_level = river.get(
            "warning_level"
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

        # Recalculate river pressure through the
        # predictor using the updated water level.
        if (
            warning_level is not None
            and new_level >= float(warning_level)
        ):
            river["quality_status"] = (
                "SCENARIO_WARNING"
            )

    # ---------------------------------------------------------
    # Run same predictor on hypothetical features
    # ---------------------------------------------------------

    scenario_prediction = predictor.predict(
        scenario_features
    )

    baseline_score = float(
        baseline_prediction["risk_score"]
    )

    scenario_score = float(
        scenario_prediction["risk_score"]
    )

    # ---------------------------------------------------------
    # Return scenario result
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
