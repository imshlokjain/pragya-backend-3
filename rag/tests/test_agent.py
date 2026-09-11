"""
Unit tests for Agent Orchestrator, Tools, and ReAct Loop.
"""

import pytest
from unittest.mock import AsyncMock

from rag.agent.tools import (
    CalculateTool,
    CompareTool,
    DataExtractTool,
)
from rag.agent.tool_registry import ToolRegistry
from rag.agent.orchestrator import AgentOrchestrator
from rag.agent.schemas import ToolCall, AgentRequest
from rag.generation.base import ModelProvider


class DummyModelProvider(ModelProvider):
    """Mock model provider returning scripted turns for testing ReAct loop."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return "Final Answer: Done processing."


@pytest.mark.asyncio
class TestAgentTools:
    """Test individual agent tools."""

    async def test_calculate_tool(self):
        calc = CalculateTool()
        res1 = await calc.execute(expression="(150 - 100) / 100 * 100")
        assert res1["result"] == 50.0

        res2 = await calc.execute(expression="25 * 4 + 10")
        assert res2["result"] == 110.0

        err_res = await calc.execute(expression="import os; os.system('ls')")
        assert "error" in err_res

    async def test_compare_tool(self):
        comp = CompareTool()
        res = await comp.execute(
            entity_a="Punjab",
            val_a=7.3,
            entity_b="Gujarat",
            val_b=3.2,
            metric="Unemployment Rate",
        )
        assert res["higher"] == "Punjab"
        assert res["difference"] == 4.1
        assert "7.3" in str(res["Punjab"])

    async def test_data_extract_tool(self):
        extractor = DataExtractTool()
        sample_text = (
            "The total literacy rate was 77.7 percent in 2024.\n"
            "The fiscal deficit remained 4.9 percent."
        )
        res = await extractor.execute(text=sample_text, metric="literacy rate")
        assert len(res["extracted_lines"]) == 1
        assert "77.7 percent" in res["extracted_lines"][0]

    async def test_tool_registry(self):
        registry = ToolRegistry()
        calc = CalculateTool()
        registry.register(calc)

        assert "calculate" in registry.get_tool_names()
        assert registry.get("calculate") is calc

        tool_call = ToolCall(tool_name="calculate", arguments={"expression": "10 + 5"})
        res = await registry.execute_tool(tool_call)
        assert res.success is True
        assert res.output["result"] == 15.0


@pytest.mark.asyncio
class TestAgentOrchestrator:
    """Test ReAct orchestrator loop."""

    async def test_orchestrator_multi_step_loop(self):
        # 1st turn: decide to calculate
        # 2nd turn: arrive at Final Answer
        mock_llm = DummyModelProvider([
            'Thought: I need to calculate the difference.\nAction: calculate\nAction Input: {"expression": "7.3 - 3.2"}',
            "Thought: The calculation gave 4.1.\nFinal Answer: Punjab has 4.1 percentage points higher unemployment than Gujarat.",
        ])

        registry = ToolRegistry()
        registry.register(CalculateTool())

        orchestrator = AgentOrchestrator(model_provider=mock_llm, tool_registry=registry)
        req = AgentRequest(
            question="What is the difference between Punjab and Gujarat unemployment?",
            max_iterations=5,
        )

        response = await orchestrator.run(req)

        assert response.total_iterations == 2
        assert "4.1" in response.answer
        assert len(response.steps) == 2
        assert response.steps[0].tool_call.tool_name == "calculate"
        assert response.steps[0].tool_result.output["result"] == 4.1
