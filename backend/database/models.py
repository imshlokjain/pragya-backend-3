import uuid
from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    ForeignKey,
    JSON,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from backend.database.session import Base


def gen_uuid():
    return str(uuid.uuid4())


class Zone(Base):
    __tablename__ = "zones"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    district_id = Column(String, index=True, nullable=False)
    name = Column(String, nullable=False)
    geometry = Column(Geometry(geometry_type="MULTIPOLYGON", srid=4326))
    population = Column(Integer, nullable=True)
    vulnerability_index = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RainfallObservation(Base):
    __tablename__ = "rainfall_observations"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    source = Column(String, nullable=False)
    station_id = Column(String, nullable=True)
    zone_id = Column(UUID(as_uuid=False), ForeignKey("zones.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    rainfall_mm = Column(Float, nullable=False)
    duration_minutes = Column(Integer, nullable=True)
    quality_status = Column(String, default="FRESH")  # FRESH, STALE, MISSING, SUSPECT, VALIDATED
    ingested_at = Column(DateTime, default=datetime.utcnow)


class RiverObservation(Base):
    __tablename__ = "river_observations"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    source = Column(String, nullable=False)
    station_id = Column(String, nullable=True)
    river_name = Column(String, nullable=True)
    zone_id = Column(UUID(as_uuid=False), ForeignKey("zones.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    water_level = Column(Float, nullable=False)
    warning_level = Column(Float, nullable=True)
    danger_level = Column(Float, nullable=True)
    rate_of_change = Column(Float, nullable=True)
    quality_status = Column(String, default="FRESH")
    ingested_at = Column(DateTime, default=datetime.utcnow)


class SatelliteObservation(Base):
    __tablename__ = "satellite_observations"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    source = Column(String, nullable=False)
    scene_id = Column(String, nullable=True)
    acquisition_time = Column(DateTime, nullable=False)
    geometry = Column(Geometry(geometry_type="MULTIPOLYGON", srid=4326), nullable=True)
    cloud_percentage = Column(Float, nullable=True)
    image_uri = Column(String, nullable=True)
    processing_status = Column(String, default="PENDING")


class FloodDetection(Base):
    __tablename__ = "flood_detections"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    zone_id = Column(UUID(as_uuid=False), ForeignKey("zones.id"), nullable=False, index=True)
    before_scene_id = Column(String, nullable=True)
    after_scene_id = Column(String, nullable=True)
    detected_at = Column(DateTime, default=datetime.utcnow)
    affected_area = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    geometry = Column(Geometry(geometry_type="MULTIPOLYGON", srid=4326), nullable=True)
    method = Column(String, default="NDWI")
    model_version = Column(String, nullable=True)


class RiskPrediction(Base):
    __tablename__ = "risk_predictions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    zone_id = Column(UUID(as_uuid=False), ForeignKey("zones.id"), nullable=False, index=True)
    prediction_time = Column(DateTime, default=datetime.utcnow)
    forecast_horizon = Column(Integer, default=6)  # hours
    risk_score = Column(Float, nullable=False)
    risk_category = Column(String, nullable=False)  # LOW..CRITICAL
    confidence = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    feature_snapshot_id = Column(String, nullable=True)
    is_prototype = Column(Integer, default=1)  # 1 = prototype scorer, 0 = validated model


class SopDocument(Base):
    __tablename__ = "sop_documents"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    title = Column(String, nullable=False)
    department = Column(String, nullable=True)
    version = Column(String, nullable=True)
    publication_date = Column(DateTime, nullable=True)
    source = Column(String, nullable=True)
    document_uri = Column(String, nullable=True)
    checksum = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("SopChunk", back_populates="document")


class SopChunk(Base):
    __tablename__ = "sop_chunks"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    document_id = Column(UUID(as_uuid=False), ForeignKey("sop_documents.id"), nullable=False)
    section = Column(String, nullable=True)
    clause = Column(String, nullable=True)
    page = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    embedding_id = Column(String, nullable=True)  # pointer into FAISS/Chroma

    document = relationship("SopDocument", back_populates="chunks")


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    user_id = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String, nullable=False)
    input_reference = Column(JSON, nullable=True)
    output_reference = Column(JSON, nullable=True)
    model_version = Column(String, nullable=True)
    document_versions = Column(JSON, nullable=True)
