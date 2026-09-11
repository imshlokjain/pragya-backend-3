"""
Unit tests for SOP Chunker, SOP Pipeline, and Response Plan Generator.
"""

import json
import pytest
from unittest.mock import MagicMock, AsyncMock

from rag.sop.chunker import SOPChunker
from rag.sop.pipeline import SOPPipeline
from rag.sop.plan_generator import ResponsePlanGenerator
from rag.sop.schemas import ResponsePlanRequest
from rag.schemas import ProcessedDocument, DocumentMetadata, PageContent
from rag.generation.base import ModelProvider


class MockSOPModelProvider(ModelProvider):
    def __init__(self, plan_dict: dict):
        self.plan_dict = plan_dict

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        return f"```json\n{json.dumps(self.plan_dict)}\n```"


@pytest.mark.asyncio
class TestSOPComponents:
    """Test suite for SOP-related capabilities."""

    def test_sop_procedural_chunker(self):
        doc = ProcessedDocument(
            metadata=DocumentMetadata(
                document_id="sop-doc-01",
                filename="Flood_Response_Protocol.pdf",
                total_pages=1,
            ),
            pages=[
                PageContent(
                    page_number=1,
                    text=(
                        "Standard Operating Procedure: Flash Flood Management\n\n"
                        "Step 1: Activate Emergency Operations Center (EOC) within 30 minutes of red alert.\n\n"
                        "Step 2: Deploy State Disaster Response Force (SDRF) to vulnerable low-lying zones.\n\n"
                        "Step 3: Setup community shelters with potable water and emergency medical supplies."
                    ),
                )
            ],
        )

        chunker = SOPChunker(target_chunk_size=300, overlap=50)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 1
        all_text = " ".join(c.content for c in chunks)
        assert "Step 1" in all_text
        assert "Step 2" in all_text
        assert "Step 3" in all_text

    async def test_response_plan_generator(self):
        mock_sop_pipe = MagicMock()
        mock_sop_pipe.search = AsyncMock(return_value=[
            {
                "document": "Flood_Response_Protocol.pdf",
                "page": 1,
                "section": "EOC Activation",
                "content": "Step 1: Activate EOC within 30 minutes. Role: District Collector.",
                "score": 0.95,
                "chunk_id": "sop-chk-1",
            }
        ])

        plan_output = {
            "applicable_sops": [
                {
                    "sop_name": "Flood_Response_Protocol.pdf",
                    "section": "EOC Activation",
                    "relevance_score": 0.95,
                    "trigger_condition": "Heavy rainfall warning > 100mm",
                }
            ],
            "immediate_actions": [
                {
                    "step_number": 1,
                    "phase": "Immediate (0-2 hours)",
                    "action": "Convene district disaster committee and activate 24/7 EOC.",
                    "responsible_role": "District Magistrate",
                    "timeline": "0-2 hours",
                    "sop_reference": "Section 1.1",
                    "checklist": ["Alert SDRF", "Verify wireless communication channels"],
                }
            ],
            "short_term_actions": [
                {
                    "step_number": 2,
                    "phase": "Short-Term (2-12 hours)",
                    "action": "Commence pre-emptive evacuation in river basin tehsils.",
                    "responsible_role": "Sub-Divisional Magistrate",
                    "timeline": "2-6 hours",
                    "sop_reference": "Section 2.3",
                    "checklist": ["Arrange buses", "Open relief camp kitchens"],
                }
            ],
            "resource_requirements": [
                "4x Motorized rescue boats",
                "500x Life jackets",
            ],
            "escalation_criteria": [
                "Water level exceeding danger mark at barrage bridge",
            ],
        }

        generator = ResponsePlanGenerator(
            sop_pipeline=mock_sop_pipe,
            model_provider=MockSOPModelProvider(plan_output),
        )

        req = ResponsePlanRequest(
            situation="Flash floods in river basin following 120mm rain in 6 hours",
            department="Disaster Management",
            severity="High",
        )

        plan = await generator.generate_plan(req)

        assert plan.situation == req.situation
        assert len(plan.applicable_sops) == 1
        assert len(plan.immediate_actions) == 1
        assert plan.immediate_actions[0].responsible_role == "District Magistrate"
        assert len(plan.short_term_actions) == 1
        assert len(plan.resource_requirements) == 2
        assert len(plan.citations) == 1
        assert plan.citations[0].document == "Flood_Response_Protocol.pdf"
