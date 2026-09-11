"""
Unit tests for What-If Scenario Analysis Engine.
"""

import json
import pytest
from unittest.mock import MagicMock

from rag.scenarios.what_if import WhatIfEngine
from rag.scenarios.schemas import ScenarioRequest
from rag.schemas import RetrievalResult, ScoredChunk, Chunk, ContentType
from rag.generation.base import ModelProvider


class MockWhatIfModelProvider(ModelProvider):
    def __init__(self, json_payload: dict):
        self.payload = json_payload

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        return f"```json\n{json.dumps(self.payload)}\n```"


@pytest.mark.asyncio
class TestWhatIfEngine:
    """Test suite for What-If scenario simulations."""

    async def test_what_if_analysis_success(self):
        sample_chunk = Chunk(
            chunk_id="chk-1",
            document_id="doc-1",
            document_name="Economic_Survey_2025.pdf",
            page_number=1,
            section="Overview",
            content_type=ContentType.TEXT,
            content="India's GDP growth rate was 6.4 percent in 2024-25.",
        )
        mock_pipeline = MagicMock()
        mock_pipeline.retrieve.return_value = RetrievalResult(
            query="GDP",
            chunks=[ScoredChunk(chunk=sample_chunk, score=0.92)],
        )

        llm_response = {
            "baseline_metrics": [
                {
                    "metric": "GDP Growth Rate",
                    "current_value": "6.4%",
                    "source_document": "Economic_Survey_2025.pdf",
                    "page_number": 1,
                    "context": "Estimated at 6.4 percent in 2024-25",
                }
            ],
            "assumptions": [
                "Domestic consumption remains resilient",
                "Monetary policy remains accommodative",
            ],
            "projected_impacts": [
                {
                    "area": "Employment",
                    "impact_summary": "Slower job creation in manufacturing",
                    "direction": "negative",
                    "confidence": "high",
                    "reasoning": "Lower GDP expansion correlates with decelerated industrial hiring.",
                    "supporting_evidence": "Historical manufacturing correlation",
                }
            ],
            "caveats": "Analysis does not account for sudden oil price spikes.",
        }

        engine = WhatIfEngine(
            pipeline=mock_pipeline,
            model_provider=MockWhatIfModelProvider(llm_response),
        )

        req = ScenarioRequest(
            scenario="What if GDP growth slows to 4.0%?",
            focus_areas=["Employment"],
            top_k=5,
        )

        response = await engine.analyze(req)

        assert response.scenario == "What if GDP growth slows to 4.0%?"
        assert len(response.baseline_metrics) == 1
        assert response.baseline_metrics[0].current_value == "6.4%"
        assert len(response.projected_impacts) == 1
        assert response.projected_impacts[0].area == "Employment"
        assert response.projected_impacts[0].confidence == "high"
        assert len(response.sources) == 1
        assert response.sources[0].document == "Economic_Survey_2025.pdf"
