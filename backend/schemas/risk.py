from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict


class RiskDriver(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str


class RiskPredictionOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    zone_id: str
    risk_score: float
    risk_category: str
    confidence: float
    forecast_horizon_hours: int
    trend: str
    model_version: str
    is_prototype: bool
    drivers: List[str]
    prediction_time: datetime


class RainfallOut(BaseModel):
    zone_id: str
    timestamp: datetime
    rainfall_mm: float
    rainfall_24h: float
    rainfall_anomaly_pct: float
    quality_status: str


class RiverOut(BaseModel):
    zone_id: str
    timestamp: datetime
    river_name: str
    current_level: float
    warning_level: float
    danger_level: float
    distance_to_danger: float
    trend: str
    quality_status: str


class ScenarioRequest(BaseModel):
    zone_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    rainfall_multiplier: float = Field(
        ...,
        ge=1.0,
        le=1.5,
    )

    river_level_increase: Optional[float] = Field(
        default=None,
        ge=0.0,
    )


class ScenarioResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    scenario_id: str
    zone_id: str
    baseline_risk: float
    baseline_category: str
    scenario_risk: float
    scenario_category: str
    risk_change: float
    parameters: dict
    baseline_timestamp: datetime
    model_version: str
    is_hypothetical: bool = True


class SatelliteObservationOut(BaseModel):
    id: str
    source: str
    scene_id: Optional[str]
    acquisition_time: datetime
    cloud_percentage: Optional[float]
    image_uri: Optional[str]
    processing_status: str


class FloodDetectionOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    zone_id: str
    before_scene_id: Optional[str]
    after_scene_id: Optional[str]
    detected_at: datetime
    affected_area: Optional[float]
    confidence: Optional[float]
    method: str
    model_version: Optional[str]
