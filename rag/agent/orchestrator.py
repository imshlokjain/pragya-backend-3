"""
Agent Orchestrator implementing a robust ReAct (Reasoning + Acting) loop with tool calling.
"""

from __future__ import annotations

import re
import json
import time
import logging
from typing import Any, Dict, List, Optional

from rag.agent.schemas import (
    AgentRequest,
    AgentResponse,
    AgentStep,
    ToolCall,
    ToolResult,
)
from rag.agent.tool_registry import ToolRegistry
from rag.agent.tools import (
    RAGSearchTool,
    DataExtractTool,
    CompareTool,
    CalculateTool,
    SOPSearchTool,
)
from rag.schemas import Source
from rag.generation.base import ModelProvider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are an expert government research and policy analyst AI agent.
Your mission is to answer complex questions accurately by reasoning step-by-step and using available tools.

You have access to the following tools:
{tools_description}

Use the following format strictly:

Question: the user question you must answer
Thought: your reasoning about what step to take next and what information is needed
Action: the name of the tool to use (must be one of: {tool_names})
Action Input: a JSON object with the tool's arguments, e.g. {{"query": "unemployment rate 2024"}}
Observation: the result of the action (will be provided to you)
... (this Thought/Action/Action Input/Observation loop can repeat up to {max_iterations} times)
Thought: I now have all the necessary information to provide a comprehensive answer.
Final Answer: your final, well-structured answer to the user question, citing relevant sources, numbers, and facts.

Rules:
1. Always reason thoroughly in the Thought block before taking an Action.
2. Action Input MUST be valid JSON.
3. If you already have sufficient facts to answer the question, proceed directly to Final Answer.
4. Base your answer strictly on facts discovered during tool execution. Do not hallucinate statistics.
"""


class AgentOrchestrator:
    """ReAct Agent that chains tools, searches, calculations, and data extraction."""

    def __init__(
        self,
        model_provider: ModelProvider,
        tool_registry: Optional[ToolRegistry] = None,
        rag_pipeline: Optional[Any] = None,
        sop_pipeline: Optional[Any] = None,
    ):
        self.model_provider = model_provider
        self.registry = tool_registry or ToolRegistry()
        self.rag_pipeline = rag_pipeline
        self.sop_pipeline = sop_pipeline

        # Automatically register standard tools if pipeline is provided
        if rag_pipeline is not None and not self.registry.get("rag_search"):
            self.registry.register(RAGSearchTool(pipeline=rag_pipeline))
            self.registry.register(DataExtractTool())
            self.registry.register(CompareTool())
            self.registry.register(CalculateTool())
            self.registry.register(SOPSearchTool(sop_pipeline=sop_pipeline, rag_pipeline=rag_pipeline))

    def _parse_action(self, text: str) -> tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """Extract Thought, Action, and Action Input from LLM response."""
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else ""

        action_match = re.search(r"Action:\s*([a-zA-Z0-9_-]+)", text, re.IGNORECASE)
        action = action_match.group(1).strip() if action_match else None

        input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL | re.IGNORECASE)
        action_input = None
        if input_match:
            try:
                action_input = json.loads(input_match.group(1).strip())
            except json.JSONDecodeError:
                action_input = {"raw_input": input_match.group(1).strip()}

        return thought, action, action_input

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """Check if LLM returned a Final Answer."""
        match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    async def run(self, request: AgentRequest) -> AgentResponse:
        """Run the ReAct agent loop on the incoming question."""
        start_time = time.time()
        max_iterations = request.max_iterations

        tools_desc = self.registry.format_tools_for_prompt(request.tools_enabled)
        tool_names = ", ".join(
            request.tools_enabled or self.registry.get_tool_names()
        )

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tools_description=tools_desc,
            tool_names=tool_names,
            max_iterations=max_iterations,
        )

        conversation_history = f"Question: {request.question}\n"
        steps: List[AgentStep] = []
        collected_sources: List[Source] = []
        final_answer: Optional[str] = None

        for iteration in range(1, max_iterations + 1):
            prompt = f"{conversation_history}\nThought:"
            llm_response = await self.model_provider.generate(
                system_prompt=system_prompt,
                user_prompt=prompt,
            )

            # Prepend 'Thought:' if model omitted it in completion
            if not llm_response.strip().lower().startswith("thought:"):
                full_llm_text = f"Thought: {llm_response}"
            else:
                full_llm_text = llm_response

            # Check if model arrived at Final Answer
            possible_answer = self._extract_final_answer(full_llm_text)
            if possible_answer:
                final_answer = possible_answer
                thought, _, _ = self._parse_action(full_llm_text)
                steps.append(
                    AgentStep(
                        step_number=iteration,
                        thought=thought or "Reached conclusion.",
                        tool_call=None,
                        tool_result=None,
                    )
                )
                break

            # Parse action and input
            thought, action_name, action_input = self._parse_action(full_llm_text)

            if not action_name:
                # If neither action nor final answer is found, treat response as final answer
                final_answer = full_llm_text.replace("Thought:", "").strip()
                steps.append(
                    AgentStep(
                        step_number=iteration,
                        thought=thought or "Direct answer produced.",
                        tool_call=None,
                        tool_result=None,
                    )
                )
                break

            # Execute tool call
            tool_call = ToolCall(
                tool_name=action_name,
                arguments=action_input or {},
            )
            tool_result = await self.registry.execute_tool(tool_call)

            # Collect any citations produced by search tools
            tool_instance = self.registry.get(action_name)
            if hasattr(tool_instance, "last_retrieved_sources"):
                for s in getattr(tool_instance, "last_retrieved_sources"):
                    if s not in collected_sources:
                        collected_sources.append(s)

            # Record step
            step = AgentStep(
                step_number=iteration,
                thought=thought,
                tool_call=tool_call,
                tool_result=tool_result,
            )
            steps.append(step)

            # Format observation for next turn
            obs_str = json.dumps(tool_result.output, default=str) if tool_result.success else f"Error: {tool_result.error}"
            # Truncate very large observations to keep context bounded
            if len(obs_str) > 2500:
                obs_str = obs_str[:2500] + "... [truncated]"

            conversation_history += (
                f"\nThought: {thought}\n"
                f"Action: {action_name}\n"
                f"Action Input: {json.dumps(action_input or {})}\n"
                f"Observation: {obs_str}\n"
            )

        # If max iterations reached without explicit Final Answer, synthesize one
        if not final_answer:
            synthesis_prompt = (
                f"{conversation_history}\n\n"
                f"Synthesize all the above observations into a final comprehensive answer for the question: '{request.question}'."
            )
            final_answer = await self.model_provider.generate(
                system_prompt="You are a helpful government analyst. Provide a clear, factual answer based strictly on the observations provided.",
                user_prompt=synthesis_prompt,
            )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        return AgentResponse(
            question=request.question,
            answer=final_answer,
            steps=steps,
            sources=collected_sources,
            total_iterations=len(steps),
            execution_time_ms=elapsed_ms,
        )
