"""
What-If Scenario Engine for evaluating hypothetical policy and economic situations.
"""

from __future__ import annotations

import re
import json
import logging
from typing import Any, Dict, List, Optional

from rag.scenarios.schemas import (
    ScenarioRequest,
    ScenarioResponse,
    BaselineMetric,
    ProjectedImpact,
)
from rag.scenarios.prompts import WHAT_IF_SYSTEM_PROMPT, WHAT_IF_USER_PROMPT
from rag.schemas import Source
from rag.generation.base import ModelProvider

logger = logging.getLogger(__name__)


class WhatIfEngine:
    """Simulates hypothetical scenarios grounded in retrieved baseline data."""

    def __init__(self, pipeline: Any, model_provider: Optional[ModelProvider] = None):
        self.pipeline = pipeline
        self._model_provider = model_provider

    def _get_model_provider(self) -> ModelProvider:
        if self._model_provider:
            return self._model_provider
        return self.pipeline._get_model_provider()

    def _clean_json_text(self, text: str) -> str:
        """Strip markdown formatting from JSON output."""
        text = text.strip()
        # Remove ```json ... ``` or ``` ... ```
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return text

    async def analyze(self, request: ScenarioRequest) -> ScenarioResponse:
        """Run scenario simulation against RAG knowledge base."""
        # Step 1: Formulate search queries from scenario and focus areas
        search_query = request.scenario
        if request.focus_areas:
            search_query += " " + " ".join(request.focus_areas)

        retrieval = self.pipeline.retrieve(question=search_query, top_k=request.top_k)

        # Step 2: Build citations and context string
        sources: List[Source] = []
        context_parts: List[str] = []

        for sc in retrieval.chunks:
            source = Source(
                document=sc.chunk.document_name,
                page=sc.chunk.page_number,
                section=sc.chunk.section,
                chunk_id=sc.chunk.chunk_id,
                score=sc.score,
            )
            sources.append(source)
            context_parts.append(
                f"[{sc.chunk.document_name} | Page {sc.chunk.page_number} | Section: {sc.chunk.section or 'N/A'}]\n"
                f"{sc.chunk.content}"
            )

        context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No specific documents found."

        # Step 3: Prompt formatting
        user_prompt = WHAT_IF_USER_PROMPT.format(
            scenario=request.scenario,
            parameters=json.dumps(request.parameters) if request.parameters else "None provided (auto-detect from scenario)",
            focus_areas=", ".join(request.focus_areas) if request.focus_areas else "General policy and socio-economic outcomes",
            context=context_str,
        )

        model = self._get_model_provider()
        raw_response = await model.generate(
            system_prompt=WHAT_IF_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Step 4: Parse response into structured models
        cleaned_json = self._clean_json_text(raw_response)
        try:
            data = json.loads(cleaned_json)
            baseline_metrics = [
                BaselineMetric(
                    metric=bm.get("metric", "Unknown"),
                    current_value=str(bm.get("current_value", "Unknown")),
                    source_document=bm.get("source_document", ""),
                    page_number=bm.get("page_number"),
                    context=bm.get("context", ""),
                )
                for bm in data.get("baseline_metrics", [])
            ]
            assumptions = [str(a) for a in data.get("assumptions", [])]
            projected_impacts = [
                ProjectedImpact(
                    area=pi.get("area", "General"),
                    impact_summary=pi.get("impact_summary", ""),
                    direction=pi.get("direction", "neutral"),
                    confidence=pi.get("confidence", "medium"),
                    reasoning=pi.get("reasoning", ""),
                    supporting_evidence=pi.get("supporting_evidence", ""),
                )
                for pi in data.get("projected_impacts", [])
            ]
            caveats = data.get("caveats", "")

            return ScenarioResponse(
                scenario=request.scenario,
                baseline_metrics=baseline_metrics,
                assumptions=assumptions,
                projected_impacts=projected_impacts,
                sources=sources,
                caveats=caveats,
            )

        except Exception as exc:
            logger.warning("Failed to parse What-If JSON response: %s. Falling back to plain response.", exc)
            return ScenarioResponse(
                scenario=request.scenario,
                baseline_metrics=[],
                assumptions=["Model returned freeform analysis."],
                projected_impacts=[
                    ProjectedImpact(
                        area="Overall Projection",
                        impact_summary=raw_response[:300] + "...",
                        direction="mixed",
                        confidence="low",
                        reasoning=raw_response,
                    )
                ],
                sources=sources,
                caveats="Could not parse structured JSON from LLM output.",
            )
