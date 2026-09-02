from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from backend.database.models import RiskPrediction, Zone
from backend.services import feature_service
from backend.services.ml_predictor import predictor
from backend.services import zone_service


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


def calculate_risk(
    db: Session,
    zone_id: str,
) -> Optional[dict]:
    """
    Calculate current risk without persisting it.
    """

    features = feature_service.get_feature_snapshot(
        db,
        zone_id,
    )

    if features is None:
        return None

    prediction = predictor.predict(features)

    return _prediction_response(
        zone_id=zone_id,
        prediction=prediction,
    )


def _resolve_zone_for_storage(
    db: Session,
    zone_id: str,
) -> Optional[Zone]:
    """
    Resolve both supported zone identifiers:

    - Legacy codes such as ZONE_01
    - Real PostgreSQL UUID zone IDs
    """

    # First try the legacy ZONE_XX code.
    zone = zone_service.zone_exists_by_code(
        db,
        zone_id,
        {
            "ZONE_01": "Riverside North",
            "ZONE_02": "Market District",
            "ZONE_03": "Lowland South",
            "ZONE_04": "Upstream Colony",
            "ZONE_05": "Embankment East",
        },
    )

    if zone is not None:
        return zone

    # Then try a real UUID.
    return zone_service.get_zone(
        db,
        zone_id,
    )


def save_risk_prediction(
    db: Session,
    zone_id: str,
) -> Optional[dict]:
    """
    Calculate current risk and persist the prediction.

    Supports both legacy zone codes and real UUID zone IDs.
    """

    features = feature_service.get_feature_snapshot(
        db,
        zone_id,
    )

    if features is None:
        return None

    prediction = predictor.predict(features)

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
            1 if prediction["is_prototype"] else 0
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
    Read-only current risk calculation.

    Does not create a risk_predictions row.
    """

    return calculate_risk(
        db,
        zone_id,
    )


def get_district_risk(
    db: Session,
    district_id: str,
) -> list[dict]:
    """
    Calculate current risk for every zone in a district.

    Read-only. Does not persist predictions.
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
    """
    Return persisted predictions for a zone.
    """

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
