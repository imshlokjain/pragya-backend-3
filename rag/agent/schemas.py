"""
Pydantic schemas for the Agent Orchestrator and Tool Calling system.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from rag.schemas import Source


class ToolCall(BaseModel):
    """Represents a tool call initiated by the agent."""
    call_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique identifier for this tool call",
    )
    tool_name: str = Field(description="Name of the tool to invoke")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool"
    )


class ToolResult(BaseModel):
    """Result returned from executing a tool."""
    call_id: str = Field(description="Matches the ToolCall.call_id")
    tool_name: str = Field(description="Name of the tool that ran")
    output: Any = Field(description="Output returned by the tool")
    success: bool = Field(default=True, description="Whether execution succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class AgentStep(BaseModel):
    """A single reasoning step in the ReAct loop."""
    step_number: int = Field(description="1-indexed iteration number")
    thought: str = Field(description="Agent's reasoning thought for this step")
    tool_call: Optional[ToolCall] = Field(
        default=None, description="Tool called in this step, if any"
    )
    tool_result: Optional[ToolResult] = Field(
        default=None, description="Result of the tool execution, if any"
    )


class AgentRequest(BaseModel):
    """Request to run the agent orchestrator."""
    question: str = Field(description="User's complex or analytical question")
    max_iterations: int = Field(
        default=5, ge=1, le=15, description="Maximum ReAct reasoning steps"
    )
    tools_enabled: Optional[List[str]] = Field(
        default=None, description="Optional whitelist of tool names to enable"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata filters for search tools"
    )


class AgentResponse(BaseModel):
    """Response from the agent orchestrator."""
    question: str = Field(description="Original user question")
    answer: str = Field(description="Final synthesized answer")
    steps: List[AgentStep] = Field(
        default_factory=list, description="Step-by-step reasoning and tool call trace"
    )
    sources: List[Source] = Field(
        default_factory=list, description="Citations referenced during the search tools"
    )
    total_iterations: int = Field(
        default=0, description="Total reasoning loops executed"
    )
    execution_time_ms: float = Field(
        default=0.0, description="Total execution time in milliseconds"
    )
