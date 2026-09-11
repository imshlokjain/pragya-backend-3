"""
Tool Registry for managing and executing agent tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from rag.agent.tools import BaseTool
from rag.agent.schemas import ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry holding tools available to the agent orchestrator."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a new tool instance."""
        if tool.name in self._tools:
            logger.warning("Overwriting existing tool '%s'", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Optional[BaseTool]:
        """Look up tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_tool_names(self) -> List[str]:
        """Return list of all registered tool names."""
        return list(self._tools.keys())

    def format_tools_for_prompt(self, allowed_names: Optional[List[str]] = None) -> str:
        """Render a readable description of tools for LLM prompts."""
        tools = [
            t for t in self._tools.values()
            if allowed_names is None or t.name in allowed_names
        ]
        if not tools:
            return "No tools available."

        lines = []
        for tool in tools:
            param_str = json.dumps(tool.parameters.get("properties", {}), indent=2)
            lines.append(
                f"- Tool: `{tool.name}`\n"
                f"  Description: {tool.description}\n"
                f"  Parameters schema:\n{param_str}"
            )
        return "\n\n".join(lines)

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return a structured ToolResult."""
        tool = self.get(tool_call.tool_name)
        if not tool:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=None,
                success=False,
                error=f"Tool '{tool_call.tool_name}' is not registered. Available tools: {self.get_tool_names()}",
            )

        try:
            output = await tool.execute(**tool_call.arguments)
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=output,
                success=True,
            )
        except Exception as exc:
            logger.exception("Error executing tool %s", tool_call.tool_name)
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=None,
                success=False,
                error=f"Error executing {tool_call.tool_name}: {str(exc)}",
            )
