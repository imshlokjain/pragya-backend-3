class RiskPredictor:
    """
    Prototype flood risk scoring model.

    Produces:
    - risk_score (0-100)
    - risk_category
    - confidence
    - trend
    - drivers
    """

    def predict(self, features: dict) -> dict:

        rainfall = features["rainfall"]
        river = features["river"]
        flood = features["flood"]
        zone = features["zone"]

        risk_score = 0.0
        drivers = []

        # -------------------------------------------------
        # RAINFALL RISK
        # -------------------------------------------------

        rainfall_24h = rainfall.get("rainfall_24h") or 0.0

        rainfall_score = min(rainfall_24h / 100 * 25, 25)

        risk_score += rainfall_score

        if rainfall_score > 0:
            drivers.append({
                "feature": "rainfall_24h",
                "value": rainfall_24h,
                "contribution": round(rainfall_score, 2),
                "direction": "INCREASE",
            })

        # -------------------------------------------------
        # RIVER LEVEL RISK
        # -------------------------------------------------

        water_level = river.get("water_level")
        danger_level = river.get("danger_level")

        river_score = 0.0

        if (
            water_level is not None
            and danger_level is not None
            and danger_level > 0
        ):
            ratio = water_level / danger_level

            river_score = min(ratio * 35, 35)

            risk_score += river_score

            drivers.append({
                "feature": "river_level",
                "value": water_level,
                "contribution": round(river_score, 2),
                "direction": "INCREASE",
            })

        # -------------------------------------------------
        # RIVER TREND
        # -------------------------------------------------

        river_change = river.get(
            "river_level_change"
        ) or 0.0

        trend_score = 0.0

        if river_change > 0:
            trend_score = min(
                river_change * 10,
                15,
            )

            risk_score += trend_score

            drivers.append({
                "feature": "river_level_change",
                "value": river_change,
                "contribution": round(trend_score, 2),
                "direction": "INCREASE",
            })

        # -------------------------------------------------
        # FLOOD DETECTION
        # -------------------------------------------------

        flood_confidence = flood.get(
            "confidence"
        )

        flood_score = 0.0

        if flood_confidence is not None:

            flood_score = min(
                flood_confidence * 20,
                20,
            )

            risk_score += flood_score

            drivers.append({
                "feature": "flood_confidence",
                "value": flood_confidence,
                "contribution": round(flood_score, 2),
                "direction": "INCREASE",
            })

        # -------------------------------------------------
        # VULNERABILITY
        # -------------------------------------------------

        vulnerability = zone.get(
            "vulnerability_index"
        )

        vulnerability_score = 0.0

        if vulnerability is not None:

            vulnerability_score = min(
                vulnerability * 5,
                5,
            )

            risk_score += vulnerability_score

            drivers.append({
                "feature": "vulnerability_index",
                "value": vulnerability,
                "contribution": round(
                    vulnerability_score,
                    2,
                ),
                "direction": "INCREASE",
            })

        # -------------------------------------------------
        # FINAL SCORE
        # -------------------------------------------------

        risk_score = min(
            round(risk_score, 2),
            100,
        )

        # -------------------------------------------------
        # RISK CATEGORY
        # -------------------------------------------------

        if risk_score < 25:
            risk_category = "LOW"
        elif risk_score < 50:
            risk_category = "MODERATE"
        elif risk_score < 75:
            risk_category = "HIGH"
        else:
            risk_category = "CRITICAL"

        # -------------------------------------------------
        # TREND
        # -------------------------------------------------

        if river_change > 0:
            trend = "RISING"
        elif river_change < 0:
            trend = "FALLING"
        else:
            trend = "STABLE"

        return {
            "risk_score": risk_score,
            "risk_category": risk_category,
            "confidence": 0.75,
            "forecast_horizon_hours": 6,
            "trend": trend,
            "model_version": "prototype-risk-v1",
            "is_prototype": True,
            "drivers": drivers,
        }


predictor = RiskPredictor()
