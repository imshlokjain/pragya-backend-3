"""
Schemas for What-If scenario modeling and hypothetical policy analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from rag.schemas import Source


class BaselineMetric(BaseModel):
    """A factual baseline metric retrieved from government documents."""
    metric: str = Field(description="Name of the metric or indicator")
    current_value: str = Field(description="Documented current or historical value")
    source_document: str = Field(default="", description="Source document name")
    page_number: Optional[int] = Field(default=None, description="Page number")
    context: str = Field(default="", description="Snippet or context describing the metric")


class ProjectedImpact(BaseModel):
    """Projected impact of the scenario on a specific domain or sector."""
    area: str = Field(description="Affected domain (e.g. 'Employment', 'Fiscal Deficit')")
    impact_summary: str = Field(description="High-level summary of projected impact")
    direction: str = Field(
        default="neutral",
        description="Direction of change: 'positive', 'negative', 'mixed', or 'neutral'",
    )
    confidence: str = Field(
        default="medium",
        description="Confidence level: 'high', 'medium', or 'low'",
    )
    reasoning: str = Field(
        description="Step-by-step logic connecting baseline to the projected outcome"
    )
    supporting_evidence: str = Field(
        default="",
        description="Evidence or correlations found in the source documents",
    )


class ScenarioRequest(BaseModel):
    """Request payload for What-If scenario simulation."""
    scenario: str = Field(
        description="Hypothetical situation or policy change to analyze",
        examples=["What if agricultural subsidies are increased by 20%?"],
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Explicit parameters or variables (e.g. {'growth_rate': '4.0%'})",
    )
    focus_areas: Optional[List[str]] = Field(
        default=None,
        description="Specific sectors or areas to analyze (e.g. ['inflation', 'jobs'])",
    )
    top_k: int = Field(
        default=8,
        description="Number of document chunks to retrieve for baseline context",
    )


class ScenarioResponse(BaseModel):
    """Structured response from the What-If analysis engine."""
    scenario: str = Field(description="The evaluated scenario")
    baseline_metrics: List[BaselineMetric] = Field(
        default_factory=list, description="Real data points extracted from documents"
    )
    assumptions: List[str] = Field(
        default_factory=list, description="Key assumptions underlying the projection"
    )
    projected_impacts: List[ProjectedImpact] = Field(
        default_factory=list, description="Sector-by-sector impact analysis"
    )
    sources: List[Source] = Field(
        default_factory=list, description="Document citations supporting the baseline"
    )
    caveats: str = Field(
        default="",
        description="Uncertainty disclaimer, missing variables, and model limitations",
    )
