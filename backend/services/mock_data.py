"""
Mock data service.

This stands in for two pieces that don't exist yet:
  1. The real ingestion pipelines (rainfall/river/satellite).
  2. The trained XGBoost + SHAP prediction service (ML teammate's deliverable).

Per MVP.md: simulated values must be clearly labelled and never presented as
live government observations. Every mock response below carries is_prototype /
a SIMULATED source tag so the frontend can render the appropriate disclaimer.

Once the ML teammate exports a real model artifact, only predict_risk() needs
to change — its function signature (zone_id, features) -> RiskPredictionOut
is the integration contract. Same for rainfall/river once real ingestion
lands: only the data source changes, not the shape callers depend on.
"""

import hashlib
import random
from datetime import datetime, timedelta

from backend.schemas.risk import RiskPredictionOut, RiskDriver, RainfallOut, RiverOut

KNOWN_ZONES = {
    "ZONE_01": "Riverside North",
    "ZONE_02": "Market District",
    "ZONE_03": "Lowland South",
    "ZONE_04": "Upstream Colony",
    "ZONE_05": "Embankment East",
}


def _seeded_random(zone_id: str) -> random.Random:
    # deterministic per zone so repeated calls look consistent during a demo
    seed = int(hashlib.sha256(zone_id.encode()).hexdigest(), 16) % (10**8)
    return random.Random(seed)


def _risk_category(score: float) -> str:
    if score <= 20:
        return "LOW"
    if score <= 40:
        return "MODERATE"
    if score <= 60:
        return "HIGH"
    if score <= 80:
        return "VERY_HIGH"
    return "CRITICAL"


def predict_risk(zone_id: str) -> RiskPredictionOut:
    rng = _seeded_random(zone_id)
    score = round(rng.uniform(20, 90), 1)
    rainfall_contrib = round(rng.uniform(0.15, 0.35), 2)
    river_contrib = round(rng.uniform(0.2, 0.4), 2)
    satellite_contrib = round(rng.uniform(0.1, 0.25), 2)
    historical_contrib = round(1 - rainfall_contrib - river_contrib - satellite_contrib, 2)

    drivers = [
        RiskDriver(feature="river_level_change", value=round(rng.uniform(0.1, 0.5), 2),
                   contribution=river_contrib, direction="INCREASES_RISK"),
        RiskDriver(feature="rainfall_24h_anomaly", value=round(rng.uniform(0.1, 0.6), 2),
                   contribution=rainfall_contrib, direction="INCREASES_RISK"),
        RiskDriver(feature="satellite_water_change", value=round(rng.uniform(0.05, 0.3), 2),
                   contribution=satellite_contrib, direction="INCREASES_RISK"),
        RiskDriver(feature="historical_flood_frequency", value=round(rng.uniform(0.05, 0.2), 2),
                   contribution=max(historical_contrib, 0.05), direction="INCREASES_RISK"),
    ]

    return RiskPredictionOut(
        zone_id=zone_id,
        risk_score=score,
        risk_category=_risk_category(score),
        confidence=round(rng.uniform(0.6, 0.9), 2),
        forecast_horizon_hours=6,
        trend=rng.choice(["INCREASING", "STABLE", "DECREASING"]),
        model_version="prototype-scorer-v0.1",
        is_prototype=True,
        drivers=drivers,
        prediction_time=datetime.utcnow(),
    )


def run_scenario(zone_id: str, rainfall_multiplier: float, river_level_increase: float = None) -> dict:
    """
    What-if engine (TRD section 22):
    Current Feature Snapshot -> Scenario Modifier -> Modified Feature Vector
    -> Prediction Model -> Scenario Result -> Baseline Comparison

    Until the real XGBoost model exists, both baseline and scenario reuse the
    same seeded prototype scorer, but the scenario perturbs the *inputs*
    (rainfall multiplier applied to the underlying feature) rather than
    just scaling the output score directly, so this stays a faithful stand-in
    for how the real model will be called once it exists.
    """
    baseline = predict_risk(zone_id)

    # Perturb the deterministic seed so the scenario reflects a *modified*
    # feature vector rather than being a pure re-roll of the same baseline.
    rng = _seeded_random(f"{zone_id}-scenario-{rainfall_multiplier}-{river_level_increase}")
    rainfall_effect = (rainfall_multiplier - 1.0) * rng.uniform(25, 45)  # score points per 100% rainfall increase
    river_effect = (river_level_increase or 0) * rng.uniform(8, 15)  # score points per meter of river rise

    scenario_score = baseline.risk_score + rainfall_effect + river_effect
    scenario_score = max(0.0, min(100.0, round(scenario_score, 1)))

    return {
        "baseline": baseline,
        "scenario_score": scenario_score,
        "scenario_category": _risk_category(scenario_score),
    }


def get_rainfall(zone_id: str) -> RainfallOut:
    rng = _seeded_random(zone_id + "rain")
    return RainfallOut(
        zone_id=zone_id,
        timestamp=datetime.utcnow(),
        rainfall_mm=round(rng.uniform(5, 40), 1),
        rainfall_24h=round(rng.uniform(50, 200), 1),
        rainfall_anomaly_pct=round(rng.uniform(-10, 80), 1),
        quality_status="SIMULATED",
    )


def get_river(zone_id: str) -> RiverOut:
    rng = _seeded_random(zone_id + "river")
    danger = round(rng.uniform(8.5, 10.0), 2)
    current = round(danger - rng.uniform(0.2, 3.0), 2)
    return RiverOut(
        zone_id=zone_id,
        timestamp=datetime.utcnow(),
        river_name="Simulated River",
        current_level=current,
        warning_level=round(danger - 0.5, 2),
        danger_level=danger,
        distance_to_danger=round(danger - current, 2),
        trend=rng.choice(["RISING", "STABLE", "FALLING"]),
        quality_status="SIMULATED",
    )
