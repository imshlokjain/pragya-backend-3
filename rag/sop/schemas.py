"""
Schemas for Standard Operating Procedure (SOP) ingestion and automated Response Plan generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from rag.schemas import Source


class PlanStep(BaseModel):
    """An individual actionable step in a response plan."""
    step_number: int = Field(description="Step sequence number")
    phase: str = Field(
        default="Immediate",
        description="Phase name (e.g. 'Immediate (0-2h)', 'Short-Term (2-12h)', 'Recovery')",
    )
    action: str = Field(description="Concrete directive or operational action")
    responsible_role: str = Field(
        description="Designated authority or agency (e.g. 'District Magistrate', 'Health Officer')"
    )
    timeline: str = Field(
        default="Immediate", description="Target execution window or SLA"
    )
    sop_reference: str = Field(
        default="", description="Citation to specific SOP manual, section, or protocol"
    )
    checklist: List[str] = Field(
        default_factory=list, description="Sub-tasks or verification criteria"
    )


class ApplicableSOP(BaseModel):
    """Summary of an SOP identified as matching the situation."""
    sop_name: str = Field(description="Title or filename of the matched SOP")
    section: str = Field(default="", description="Relevant section, heading, or protocol")
    relevance_score: float = Field(default=0.0, description="Similarity or relevance score")
    trigger_condition: str = Field(
        default="", description="Documented condition that triggered this SOP"
    )


class ResponsePlanRequest(BaseModel):
    """Request to synthesize an incident/crisis response plan."""
    situation: str = Field(
        description="Description of the incident, disaster, or operational emergency",
        examples=["Severe urban flooding following 150mm rainfall in central district."],
    )
    department: Optional[str] = Field(
        default=None, description="Optional department filter (e.g. 'Disaster Management', 'Health')"
    )
    severity: str = Field(
        default="High", description="Severity level: 'Low', 'Medium', 'High', 'Critical'"
    )
    max_steps: int = Field(
        default=10, description="Maximum total response steps to generate"
    )
    include_general_context: bool = Field(
        default=True,
        description="Whether to also pull demographic and infrastructure context from general RAG documents",
    )


class ResponsePlanResponse(BaseModel):
    """Structured response plan for frontline administrators and decision makers."""
    situation: str = Field(description="The incident described in the request")
    severity: str = Field(description="Assessed or assigned severity level")
    applicable_sops: List[ApplicableSOP] = Field(
        default_factory=list, description="Relevant standard operating procedures identified"
    )
    immediate_actions: List[PlanStep] = Field(
        default_factory=list, description="First priority actions (0-4 hours)"
    )
    short_term_actions: List[PlanStep] = Field(
        default_factory=list, description="Secondary operational steps (4-24 hours)"
    )
    resource_requirements: List[str] = Field(
        default_factory=list, description="Key personnel, equipment, medical supplies, or transport required"
    )
    escalation_criteria: List[str] = Field(
        default_factory=list, description="Conditions under which the incident must be escalated to state/national level"
    )
    citations: List[Source] = Field(
        default_factory=list, description="Citations from ingested SOPs and policy reports"
    )


class SOPIngestRequest(BaseModel):
    """Metadata payload accompanying an SOP PDF upload."""
    category: str = Field(
        default="general",
        description="SOP category (e.g. 'disaster_management', 'public_health', 'law_enforcement')",
    )
    department: str = Field(
        default="General Administration",
        description="Governing department or ministry",
    )
    version: str = Field(default="1.0", description="Document revision version")
    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional custom metadata tags"
    )


class SOPIngestResponse(BaseModel):
    """Response returned upon successful SOP ingestion."""
    document_id: str
    filename: str
    category: str
    department: str
    version: str
    total_pages: int
    chunks_created: int
    message: str = "SOP document successfully ingested into SOP repository."
