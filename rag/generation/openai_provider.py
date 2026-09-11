"""
OpenAI-compatible LLM provider.

Works with:
- OpenAI API (GPT-4o, GPT-4o-mini, etc.)
- Azure OpenAI
- Local servers (vLLM, Ollama, llama.cpp) via custom base_url
- Any OpenAI-compatible API
"""

from __future__ import annotations

import logging

from rag.generation.base import ModelProvider
from rag.exceptions import GenerationError, ConfigurationError

logger = logging.getLogger(__name__)


class OpenAIModelProvider(ModelProvider):
    """OpenAI-compatible LLM provider.

    Args:
        model_name: Model name (e.g. ``gpt-4o-mini``).
        api_key: API key for the provider.
        base_url: Optional custom base URL for compatible APIs.
        temperature: Sampling temperature (lower = more factual).
        max_tokens: Maximum tokens in the response.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str = "",
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        if not api_key:
            raise ConfigurationError(
                message="API key is required for OpenAI provider",
                details="Set RAG_LLM_API_KEY in your .env file.",
            )

        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """Lazy-initialise the OpenAI client."""
        if self._client is not None:
            return self._client

        try:
            from openai import AsyncOpenAI

            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = AsyncOpenAI(**kwargs)
            logger.info("OpenAI LLM client initialised for %s", self._model_name)
            return self._client
        except ImportError:
            raise GenerationError(
                message="openai package is not installed",
                details="Install it with: pip install openai",
            )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        client = self._get_client()

        try:
            response = await client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            content = response.choices[0].message.content
            if not content:
                raise GenerationError(
                    message="LLM returned empty response",
                    details=f"Model: {self._model_name}",
                )

            logger.info(
                "Generated response (%d chars) using %s",
                len(content),
                self._model_name,
            )
            return content

        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(
                message="LLM generation failed",
                details=str(exc),
            ) from exc

    @property
    def model_name(self) -> str:
        return self._model_name
