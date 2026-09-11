"""
Built-in tools for the RAG Agent Orchestrator.
"""

from __future__ import annotations

import ast
import operator
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from rag.schemas import Source, ScoredChunk

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with given arguments and return output."""
        pass


class RAGSearchTool(BaseTool):
    """Searches the RAG vector store for relevant document chunks."""

    name = "rag_search"
    description = (
        "Search government documents for factual context, statistics, policies, and reports. "
        "Inputs: 'query' (search string), optional 'top_k' (number of chunks, default 4)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Specific search query targeting facts, numbers, or policy details.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of chunks to retrieve (1 to 10, default 4).",
                "default": 4,
            },
        },
        "required": ["query"],
    }

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.last_retrieved_sources: List[Source] = []

    async def execute(self, query: str, top_k: int = 4, **kwargs: Any) -> Dict[str, Any]:
        retrieval = self.pipeline.retrieve(question=query, top_k=top_k)
        chunks_info = []
        for sc in retrieval.chunks:
            source = Source(
                document=sc.chunk.document_name,
                page=sc.chunk.page_number,
                section=sc.chunk.section,
                chunk_id=sc.chunk.chunk_id,
                score=sc.score,
            )
            self.last_retrieved_sources.append(source)
            chunks_info.append({
                "document": sc.chunk.document_name,
                "page": sc.chunk.page_number,
                "section": sc.chunk.section,
                "content": sc.chunk.content,
            })
        return {
            "query": query,
            "results_count": len(chunks_info),
            "results": chunks_info,
        }


class DataExtractTool(BaseTool):
    """Extracts specific numbers, metrics, or factual statements from provided text or search results."""

    name = "data_extract"
    description = (
        "Extract specific figures, statistics, or data points for a given metric or entity from text. "
        "Inputs: 'text' (source text), 'metric' (what to extract, e.g., 'unemployment rate', 'expenditure')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The text block or snippet containing the data.",
            },
            "metric": {
                "type": "string",
                "description": "The metric or attribute to extract (e.g. 'GDP growth', 'poverty rate').",
            },
        },
        "required": ["text", "metric"],
    }

    async def execute(self, text: str, metric: str, **kwargs: Any) -> Dict[str, Any]:
        # Simple extraction logic + pattern recognition
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        matches = []
        metric_lower = metric.lower()
        for line in lines:
            if any(term in line.lower() for term in metric_lower.split()):
                matches.append(line)

        return {
            "metric": metric,
            "extracted_lines": matches if matches else ["No direct match found in snippet."],
            "raw_text_length": len(text),
        }


class CompareTool(BaseTool):
    """Compares two metrics or entities and computes numerical and qualitative differences."""

    name = "compare"
    description = (
        "Compare two entities or metrics (e.g. State A vs State B, Year 2023 vs 2024). "
        "Inputs: 'entity_a', 'val_a', 'entity_b', 'val_b', 'metric'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "entity_a": {"type": "string", "description": "First entity or period name"},
            "val_a": {"type": "number", "description": "Value for first entity"},
            "entity_b": {"type": "string", "description": "Second entity or period name"},
            "val_b": {"type": "number", "description": "Value for second entity"},
            "metric": {"type": "string", "description": "Metric being compared"},
        },
        "required": ["entity_a", "val_a", "entity_b", "val_b", "metric"],
    }

    async def execute(
        self,
        entity_a: str,
        val_a: Any,
        entity_b: str,
        val_b: Any,
        metric: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            num_a = float(val_a)
            num_b = float(val_b)
            diff = round(num_a - num_b, 4)
            abs_diff = abs(diff)
            pct_change = round(((num_a - num_b) / num_b * 100), 2) if num_b != 0 else None

            higher = entity_a if num_a > num_b else (entity_b if num_b > num_a else "Equal")
            return {
                "metric": metric,
                entity_a: num_a,
                entity_b: num_b,
                "difference": diff,
                "absolute_difference": abs_diff,
                "percentage_difference": f"{pct_change}%" if pct_change is not None else "N/A",
                "higher": higher,
                "summary": f"{higher} has higher {metric} by {abs_diff}" if higher != "Equal" else f"Both {entity_a} and {entity_b} have equal {metric}.",
            }
        except (ValueError, TypeError) as exc:
            return {
                "error": f"Could not perform numerical comparison: {str(exc)}",
                "entity_a": f"{entity_a}: {val_a}",
                "entity_b": f"{entity_b}: {val_b}",
            }


class CalculateTool(BaseTool):
    """Safely evaluates arithmetic mathematical expressions without eval()."""

    name = "calculate"
    description = (
        "Safely compute mathematical or arithmetic expressions (addition, subtraction, multiplication, division, percentages). "
        "Inputs: 'expression' (e.g. '(124.5 - 98.2) / 98.2 * 100')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression string to compute safely.",
            },
        },
        "required": ["expression"],
    }

    # Safe operators whitelist
    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):  # <number>
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):  # <left> <op> <right>
            op_type = type(node.op)
            if op_type not in self._OPERATORS:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self._OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):  # -<operand> or +<operand>
            op_type = type(node.op)
            if op_type not in self._OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._eval_node(node.operand)
            return self._OPERATORS[op_type](operand)
        else:
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    async def execute(self, expression: str, **kwargs: Any) -> Dict[str, Any]:
        try:
            # Clean expression
            clean_expr = expression.replace("%", "* 0.01").strip()
            tree = ast.parse(clean_expr, mode="eval")
            result = self._eval_node(tree.body)
            return {
                "expression": expression,
                "result": round(result, 6) if isinstance(result, float) else result,
            }
        except Exception as exc:
            return {
                "expression": expression,
                "error": f"Calculation failed: {str(exc)}",
            }


class SOPSearchTool(BaseTool):
    """Searches the Standard Operating Procedures (SOP) repository."""

    name = "sop_search"
    description = (
        "Search Standard Operating Procedures (SOPs) for emergency response protocols, "
        "step-by-step instructions, roles, and escalation criteria. "
        "Inputs: 'query' (search topic), optional 'category' (e.g. 'disaster', 'cyber')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Incident type, procedural question, or emergency scenario.",
            },
            "category": {
                "type": "string",
                "description": "Optional category filter (e.g., flood, health, cyber).",
            },
        },
        "required": ["query"],
    }

    def __init__(self, sop_pipeline: Any = None, rag_pipeline: Any = None):
        self.sop_pipeline = sop_pipeline
        self.rag_pipeline = rag_pipeline
        self.last_retrieved_sources: List[Source] = []

    async def execute(self, query: str, category: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        if self.sop_pipeline is not None:
            results = await self.sop_pipeline.search(query=query, category=category)
            return {"query": query, "category": category, "results": results}
        elif self.rag_pipeline is not None:
            filters = {"category": category} if category else None
            retrieval = self.rag_pipeline.retrieve(question=query, top_k=3, filters=filters)
            return {
                "query": query,
                "results": [
                    {
                        "document": sc.chunk.document_name,
                        "page": sc.chunk.page_number,
                        "content": sc.chunk.content,
                    }
                    for sc in retrieval.chunks
                ],
            }
        return {"error": "SOP pipeline is not initialized."}
