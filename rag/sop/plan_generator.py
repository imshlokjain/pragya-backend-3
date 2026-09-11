"""
Response Plan Generator synthesizing operational crisis/incident action plans from SOPs.
"""

from __future__ import annotations

import re
import json
import logging
from typing import Any, Dict, List, Optional

from rag.sop.schemas import (
    ResponsePlanRequest,
    ResponsePlanResponse,
    ApplicableSOP,
    PlanStep,
)
from rag.sop.prompts import SOP_PLAN_SYSTEM_PROMPT, SOP_PLAN_USER_PROMPT
from rag.sop.pipeline import SOPPipeline
from rag.schemas import Source
from rag.generation.base import ModelProvider

logger = logging.getLogger(__name__)


class ResponsePlanGenerator:
    """Generates structured response plans combining SOP procedures and RAG context."""

    def __init__(
        self,
        sop_pipeline: SOPPipeline,
        rag_pipeline: Optional[Any] = None,
        model_provider: Optional[ModelProvider] = None,
    ):
        self.sop_pipeline = sop_pipeline
        self.rag_pipeline = rag_pipeline
        self._model_provider = model_provider

    def _get_model_provider(self) -> ModelProvider:
        if self._model_provider:
            return self._model_provider
        if self.rag_pipeline and hasattr(self.rag_pipeline, "_get_model_provider"):
            return self.rag_pipeline._get_model_provider()
        raise RuntimeError("No ModelProvider configured for ResponsePlanGenerator.")

    def _clean_json_text(self, text: str) -> str:
        """Extract valid JSON from LLM text."""
        text = text.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return text

    async def generate_plan(self, request: ResponsePlanRequest) -> ResponsePlanResponse:
        """Synthesize a complete response plan based on the incident situation."""
        citations: List[Source] = []

        # Step 1: Retrieve matching SOP procedures
        sop_results = await self.sop_pipeline.search(
            query=request.situation,
            category=request.department,
            top_k=6,
        )

        sop_context_lines = []
        for r in sop_results:
            sop_context_lines.append(
                f"[SOP Document: {r['document']} | Page: {r['page']} | Section: {r['section']}]\n{r['content']}"
            )
            citations.append(
                Source(
                    document=r["document"],
                    page=r["page"],
                    section=r["section"],
                    chunk_id=r.get("chunk_id", ""),
                    score=r.get("score", 0.0),
                )
            )
        sop_context = "\n\n---\n\n".join(sop_context_lines) if sop_context_lines else "No specific SOP found."

        # Step 2: Optionally retrieve general government policy & demographic context
        general_context = "No general policy context requested."
        if request.include_general_context and self.rag_pipeline:
            gen_retrieval = self.rag_pipeline.retrieve(question=request.situation, top_k=4)
            gen_lines = []
            for sc in gen_retrieval.chunks:
                gen_lines.append(
                    f"[{sc.chunk.document_name} | Page: {sc.chunk.page_number}]\n{sc.chunk.content}"
                )
                citations.append(
                    Source(
                        document=sc.chunk.document_name,
                        page=sc.chunk.page_number,
                        section=sc.chunk.section,
                        chunk_id=sc.chunk.chunk_id,
                        score=sc.score,
                    )
                )
            if gen_lines:
                general_context = "\n\n---\n\n".join(gen_lines)

        # Step 3: Format prompt and call LLM
        user_prompt = SOP_PLAN_USER_PROMPT.format(
            situation=request.situation,
            severity=request.severity,
            department=request.department or "All relevant emergency departments",
            sop_context=sop_context,
            general_context=general_context,
        )

        model = self._get_model_provider()
        raw_output = await model.generate(
            system_prompt=SOP_PLAN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Step 4: Parse into ResponsePlanResponse
        cleaned_json = self._clean_json_text(raw_output)
        try:
            data = json.loads(cleaned_json)

            applicable_sops = [
                ApplicableSOP(
                    sop_name=asop.get("sop_name", "General SOP"),
                    section=asop.get("section", ""),
                    relevance_score=float(asop.get("relevance_score", 0.9)),
                    trigger_condition=asop.get("trigger_condition", ""),
                )
                for asop in data.get("applicable_sops", [])
            ]

            immediate_actions = [
                PlanStep(
                    step_number=s.get("step_number", idx + 1),
                    phase=s.get("phase", "Immediate (0-2 hours)"),
                    action=s.get("action", ""),
                    responsible_role=s.get("responsible_role", "Incident Commander"),
                    timeline=s.get("timeline", "Immediate"),
                    sop_reference=s.get("sop_reference", ""),
                    checklist=s.get("checklist", []),
                )
                for idx, s in enumerate(data.get("immediate_actions", []))
            ]

            short_term_actions = [
                PlanStep(
                    step_number=s.get("step_number", idx + len(immediate_actions) + 1),
                    phase=s.get("phase", "Short-Term (2-12 hours)"),
                    action=s.get("action", ""),
                    responsible_role=s.get("responsible_role", "Assigned Officer"),
                    timeline=s.get("timeline", "2-12 hours"),
                    sop_reference=s.get("sop_reference", ""),
                    checklist=s.get("checklist", []),
                )
                for idx, s in enumerate(data.get("short_term_actions", []))
            ]

            return ResponsePlanResponse(
                situation=request.situation,
                severity=request.severity,
                applicable_sops=applicable_sops,
                immediate_actions=immediate_actions,
                short_term_actions=short_term_actions,
                resource_requirements=data.get("resource_requirements", []),
                escalation_criteria=data.get("escalation_criteria", []),
                citations=citations,
            )

        except Exception as exc:
            logger.warning("Failed to parse response plan JSON: %s. Returning fallback plan.", exc)
            return ResponsePlanResponse(
                situation=request.situation,
                severity=request.severity,
                applicable_sops=[],
                immediate_actions=[
                    PlanStep(
                        step_number=1,
                        phase="Immediate (0-2 hours)",
                        action="Activate Emergency Operations Center (EOC) and notify nodal officers.",
                        responsible_role="Incident Commander",
                        timeline="0-2 hours",
                        sop_reference="General Emergency Protocol",
                        checklist=["Log incident", "Notify department leads"],
                    )
                ],
                short_term_actions=[
                    PlanStep(
                        step_number=2,
                        phase="Short-Term (2-12 hours)",
                        action=raw_output[:400],
                        responsible_role="Designated Field Officer",
                        timeline="2-12 hours",
                        sop_reference="General Operational Guidance",
                    )
                ],
                resource_requirements=["Incident command post", "First response team"],
                escalation_criteria=["Loss of communications or escalation beyond district capacity"],
                citations=citations,
            )
