from abc import ABC, abstractmethod
from typing import Any, Dict

from backend.core.config import settings


class RiskPredictor(ABC):
    """
    Common interface for every risk prediction model.

    Both the prototype model and the future XGBoost model
    must implement this interface.
    """

    @abstractmethod
    def predict(
        self,
        features: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class PrototypeRiskPredictor(RiskPredictor):
    """
    Temporary rule-based predictor.

    This exists only until the real XGBoost + SHAP
    model is connected.
    """

    MODEL_VERSION = "prototype-scorer-v0.3"

    def predict(
        self,
        features: Dict[str, Any],
    ) -> Dict[str, Any]:

        zone = features.get("zone", {})
        rainfall = features.get("rainfall", {})
        river = features.get("river", {})
        flood = features.get("flood", {})

        rainfall_24h = float(
            rainfall.get("rainfall_24h") or 0.0
        )

        rainfall_anomaly = float(
            rainfall.get("rainfall_anomaly_pct") or 0.0
        )

        river_level = float(
            river.get("water_level") or 0.0
        )

        river_level_change = float(
            river.get("river_level_change") or 0.0
        )

        distance_to_danger = river.get(
            "distance_to_danger"
        )

        flood_confidence = float(
            flood.get("confidence") or 0.0
        )

        affected_area = float(
            flood.get("affected_area") or 0.0
        )

        historical_frequency = float(
            flood.get(
                "historical_flood_frequency"
            ) or 0.0
        )

        vulnerability = float(
            zone.get(
                "vulnerability_index"
            ) or 0.0
        )

        # -----------------------------------------------------
        # River pressure
        # -----------------------------------------------------

        if distance_to_danger is None:
            river_pressure = 0.0
            distance_value = 0.0
        else:
            distance_value = float(
                distance_to_danger
            )

            if distance_value <= 0:
                river_pressure = 1.0
            elif distance_value <= 0.5:
                river_pressure = 0.8
            elif distance_value <= 1.0:
                river_pressure = 0.5
            else:
                river_pressure = 0.2

        # -----------------------------------------------------
        # Feature components
        # -----------------------------------------------------

        rainfall_component = min(
            rainfall_24h / 150.0,
            1.0,
        )

        anomaly_component = min(
            abs(rainfall_anomaly) / 100.0,
            1.0,
        )

        river_component = min(
            river_level / 12.0,
            1.0,
        )

        river_change_component = min(
            max(river_level_change, 0.0) / 2.0,
            1.0,
        )

        flood_component = min(
            (
                flood_confidence * 0.6
                + min(
                    affected_area / 5.0,
                    1.0,
                ) * 0.4
            ),
            1.0,
        )

        history_component = min(
            historical_frequency / 5.0,
            1.0,
        )

        vulnerability_component = min(
            vulnerability,
            1.0,
        )

        # -----------------------------------------------------
        # Prototype score
        # -----------------------------------------------------

        score = (
            rainfall_component * 22.0
            + anomaly_component * 8.0
            + river_component * 18.0
            + river_pressure * 18.0
            + river_change_component * 8.0
            + flood_component * 14.0
            + history_component * 6.0
            + vulnerability_component * 6.0
        )

        score = round(
            max(0.0, min(score, 100.0)),
            1,
        )

        # -----------------------------------------------------
        # Category
        # -----------------------------------------------------

        if score < 25:
            category = "LOW"
        elif score < 50:
            category = "MODERATE"
        elif score < 75:
            category = "HIGH"
        else:
            category = "CRITICAL"

        # -----------------------------------------------------
        # Confidence
        # -----------------------------------------------------

        available = 0

        for section in (
            rainfall,
            river,
            features.get("satellite", {}),
            flood,
        ):
            if section:
                available += 1

        confidence = round(
            0.55 + (available / 4.0) * 0.35,
            2,
        )

        # -----------------------------------------------------
        # Drivers
        # -----------------------------------------------------

        drivers = [
            {
                "feature": "rainfall_24h",
                "value": rainfall_24h,
                "contribution": round(
                    rainfall_component * 0.22,
                    2,
                ),
                "direction": (
                    "INCREASES_RISK"
                    if rainfall_24h > 0
                    else "DECREASES_RISK"
                ),
            },
            {
                "feature": "river_level",
                "value": river_level,
                "contribution": round(
                    river_component * 0.18,
                    2,
                ),
                "direction": (
                    "INCREASES_RISK"
                    if river_level > 0
                    else "DECREASES_RISK"
                ),
            },
            {
                "feature": "distance_to_danger",
                "value": distance_value,
                "contribution": round(
                    river_pressure * 0.18,
                    2,
                ),
                "direction": (
                    "INCREASES_RISK"
                    if river_pressure > 0
                    else "DECREASES_RISK"
                ),
            },
            {
                "feature": "flood_confidence",
                "value": flood_confidence,
                "contribution": round(
                    flood_component * 0.14,
                    2,
                ),
                "direction": (
                    "INCREASES_RISK"
                    if flood_confidence > 0
                    else "DECREASES_RISK"
                ),
            },
            {
                "feature": "vulnerability_index",
                "value": vulnerability,
                "contribution": round(
                    vulnerability_component * 0.06,
                    2,
                ),
                "direction": (
                    "INCREASES_RISK"
                    if vulnerability > 0
                    else "DECREASES_RISK"
                ),
            },
        ]

        return {
            "risk_score": score,
            "risk_category": category,
            "confidence": confidence,
            "forecast_horizon_hours": 6,
            "trend": (
                "RISING"
                if river_level_change > 0
                else "STABLE"
            ),
            "model_version": self.MODEL_VERSION,
            "is_prototype": True,
            "drivers": drivers,
        }


def create_predictor() -> RiskPredictor:
    """
    Select the configured prediction model.
    """

    provider = settings.model_provider.lower().strip()

    if provider == "prototype":
        return PrototypeRiskPredictor()

    if provider == "xgboost":
        raise RuntimeError(
            "MODEL_PROVIDER=xgboost is configured, "
            "but the XGBoost predictor has not been integrated yet."
        )

    raise ValueError(
        f"Unsupported MODEL_PROVIDER: {settings.model_provider}. "
        "Expected 'prototype' or 'xgboost'."
    )


predictor = create_predictor()
