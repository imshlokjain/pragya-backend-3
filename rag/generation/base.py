"""
Abstract base class for LLM model providers.

All LLM backends implement this interface so they can be swapped
without changing any generation or pipeline code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelProvider(ABC):
    """Abstract LLM provider.

    Implementations must provide a ``generate`` method that takes
    system + user prompts and returns a generated text response.
    """

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User message with context and question.

        Returns:
            Generated text response.
        """
        ...

    @property
    def model_name(self) -> str:
        """Human-readable model name (for logging)."""
        return self.__class__.__name__
