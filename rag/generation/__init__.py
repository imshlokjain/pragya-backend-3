"""Generation subpackage — prompt engineering, LLM abstraction, and providers."""

from rag.generation.base import ModelProvider
from rag.generation.prompt import build_rag_prompt

__all__ = ["ModelProvider", "build_rag_prompt"]
