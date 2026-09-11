from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.database.models import RiskPrediction, Zone
from backend.services import feature_service
from backend.services import zone_service
from backend.services.river_forecast_service import get_river_forecast


def _prediction_response(
    zone_id: str,
    prediction: dict,
    prediction_time: Optional[datetime] = None,
) -> dict:
    return {
        "zone_id": zone_id,
        "risk_score": prediction["risk_score"],
        "risk_category": prediction["risk_category"],
        "confidence": prediction["confidence"],
        "forecast_horizon_hours": prediction[
            "forecast_horizon_hours"
        ],
        "trend": prediction["trend"],
        "model_version": prediction["model_version"],
        "is_prototype": prediction["is_prototype"],
        "drivers": prediction.get("drivers", []),
        "prediction_time": prediction_time or datetime.utcnow(),
    }


def _calculate_prediction(
    db: Session,
    zone_id: str,
) -> Optional[dict]:
    """
    Calculate flood risk using:

    - Real database observations
    - Real ML river forecast
    - River warning/danger thresholds
    - Rainfall
    - River level trend
    """

    # Get real ML river forecast
    forecast = get_river_forecast(
        db,
        zone_id,
    )

    if forecast is None:
        return None

    # Get current backend feature snapshot
    features = feature_service.get_feature_snapshot(
        db,
        zone_id,
    )

    if features is None:
        return None

    current_level = forecast["current_river_level"]
    predicted_level = forecast["predicted_river_level"]

    river_data = features["river"]
    rainfall_data = features["rainfall"]

    warning_level = river_data["warning_level"]
    danger_level = river_data["danger_level"]

    rainfall_mm = (
        rainfall_data["rainfall_mm"] or 0.0
    )

    river_change = (
        river_data["river_level_change"] or 0.0
    )

    # --------------------------------------------------
    # RISK SCORE
    # --------------------------------------------------

    risk_score = 0.0
    drivers = []

    # Predicted river level vs thresholds
    if danger_level is not None:

        if predicted_level >= danger_level:
            risk_score += 60
            drivers.append(
                "Predicted river level exceeds danger level"
            )

        elif (
            warning_level is not None
            and predicted_level >= warning_level
        ):
            risk_score += 40
            drivers.append(
                "Predicted river level exceeds warning level"
            )

        elif predicted_level >= danger_level - 1:
            risk_score += 25
            drivers.append(
                "Predicted river level is approaching danger level"
            )

    # River currently rising
    if river_change > 0:
        risk_score += 15
        drivers.append(
            "River level is rising"
        )

    # Predicted level higher than current level
    if predicted_level > current_level:
        risk_score += 10
        drivers.append(
            "River level is forecast to rise"
        )

    # Rainfall contribution
    if rainfall_mm >= 50:
        risk_score += 20
        drivers.append(
            "Heavy rainfall detected"
        )

    elif rainfall_mm >= 20:
        risk_score += 10
        drivers.append(
            "Moderate rainfall detected"
        )

    # Keep score between 0 and 100
    risk_score = min(
        risk_score,
        100.0,
    )

    # --------------------------------------------------
    # RISK CATEGORY
    # --------------------------------------------------

    if risk_score >= 75:
        risk_category = "CRITICAL"

    elif risk_score >= 50:
        risk_category = "HIGH"

    elif risk_score >= 25:
        risk_category = "MEDIUM"

    else:
        risk_category = "LOW"

    # --------------------------------------------------
    # TREND
    # --------------------------------------------------

    if predicted_level > current_level:
        trend = "RISING"

    elif predicted_level < current_level:
        trend = "FALLING"

    else:
        trend = "STABLE"

    # --------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------

    confidence = 0.85

    return {
        "risk_score": round(
            risk_score,
            2,
        ),
        "risk_category": risk_category,
        "confidence": confidence,
        "forecast_horizon_hours": forecast[
            "forecast_horizon_hours"
        ],
        "trend": trend,
        "model_version": "assam-river-rf-v1",
        "is_prototype": False,
        "drivers": drivers,
    }


def calculate_risk(
    db: Session,
    zone_id: str,
) -> Optional[dict]:
    """
    Calculate current risk without saving it.
    """

    prediction = _calculate_prediction(
        db,
        zone_id,
    )

    if prediction is None:
        return None

    return _prediction_response(
        zone_id=zone_id,
        prediction=prediction,
    )


def _resolve_zone_for_storage(
    db: Session,
    zone_id: str,
) -> Optional[Zone]:
    """
    Resolve both legacy zone codes and UUIDs.
    """

    code = zone_service.resolve_zone_code(
        db,
        zone_id,
    )

    if code is not None:
        zone = zone_service.zone_exists_by_code(
            db,
            code,
            zone_service.KNOWN_ZONES,
        )

        if zone is not None:
            return zone

    return zone_service.get_zone(
        db,
        zone_id,
    )


def save_risk_prediction(
    db: Session,
    zone_id: str,
) -> Optional[dict]:
    """
    Calculate risk and save it in risk_predictions.
    """

    prediction = _calculate_prediction(
        db,
        zone_id,
    )

    if prediction is None:
        return None

    zone = _resolve_zone_for_storage(
        db,
        zone_id,
    )

    if zone is None:
        return None

    prediction_time = datetime.utcnow()

    row = RiskPrediction(
        zone_id=zone.id,
        prediction_time=prediction_time,
        forecast_horizon=prediction[
            "forecast_horizon_hours"
        ],
        risk_score=prediction["risk_score"],
        risk_category=prediction["risk_category"],
        confidence=prediction["confidence"],
        model_version=prediction["model_version"],
        feature_snapshot_id=None,
        is_prototype=(
            1
            if prediction["is_prototype"]
            else 0
        ),
    )

    db.add(row)
    db.commit()
    db.refresh(row)

    return _prediction_response(
        zone_id=zone_id,
        prediction=prediction,
        prediction_time=row.prediction_time,
    )


def get_risk(
    db: Session,
    zone_id: str,
) -> Optional[dict]:
    """
    Read-only current risk calculation with seamless prototype fallback.
    """
    try:
        pred = calculate_risk(db, zone_id)
        if pred is not None:
            return pred
    except Exception:
        pass

    from backend.services import mock_data
    try:
        mock_pred = mock_data.predict_risk(zone_id)
        return {
            "zone_id": zone_id,
            "risk_score": mock_pred.risk_score,
            "risk_category": mock_pred.risk_category,
            "confidence": mock_pred.confidence,
            "forecast_horizon_hours": mock_pred.forecast_horizon_hours,
            "trend": mock_pred.trend,
            "model_version": mock_pred.model_version,
            "is_prototype": True,
            "drivers": [
                d.model_dump() if hasattr(d, "model_dump") else vars(d)
                for d in mock_pred.drivers
            ],
            "prediction_time": mock_pred.prediction_time,
        }
    except Exception:
        return None


def get_district_risk(
    db: Session,
    district_id: str,
) -> list[dict]:
    """
    Calculate current risk for every zone
    in a district.
    """

    zones = (
        db.query(Zone)
        .filter(
            Zone.district_id == district_id
        )
        .all()
    )

    results = []

    for zone in zones:

        result = calculate_risk(
            db,
            str(zone.id),
        )

        if result is not None:
            results.append(result)

    return results


def get_risk_history(
    db: Session,
    zone_id: str,
    limit: int = 20,
) -> Optional[list[dict]]:

    zone = _resolve_zone_for_storage(
        db,
        zone_id,
    )

    if zone is None:
        return None

    rows = (
        db.query(RiskPrediction)
        .filter(
            RiskPrediction.zone_id == zone.id
        )
        .order_by(
            RiskPrediction.prediction_time.desc()
        )
        .limit(limit)
        .all()
    )

    results = []

    for row in rows:

        results.append(
            {
                "zone_id": zone_id,
                "risk_score": row.risk_score,
                "risk_category": row.risk_category,
                "confidence": (
                    row.confidence
                    if row.confidence is not None
                    else 0.0
                ),
                "forecast_horizon_hours": (
                    row.forecast_horizon
                    if row.forecast_horizon is not None
                    else 6
                ),
                "trend": "STABLE",
                "model_version": (
                    row.model_version
                    or "unknown"
                ),
                "is_prototype": bool(
                    row.is_prototype
                ),
                "drivers": [],
                "prediction_time": row.prediction_time,
            }
        )

    return results
