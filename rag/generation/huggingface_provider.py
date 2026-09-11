"""
Hugging Face Inference API LLM provider.

Uses the Hugging Face serverless Inference API to generate text
from any model hosted on the Hub.
"""

from __future__ import annotations

import logging

from rag.generation.base import ModelProvider
from rag.exceptions import GenerationError, ConfigurationError

logger = logging.getLogger(__name__)


class HuggingFaceModelProvider(ModelProvider):
    """Hugging Face Inference API provider.

    Args:
        model_name: Hugging Face model ID (e.g. ``mistralai/Mistral-7B-Instruct-v0.3``).
        api_key: Hugging Face API token.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        api_key: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        if not api_key:
            raise ConfigurationError(
                message="Hugging Face API token is required",
                details="Set RAG_HF_API_KEY in your .env file.",
            )

        self._model_name = model_name
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        """Lazy-initialise the HF Inference client."""
        if self._client is not None:
            return self._client

        try:
            from huggingface_hub import AsyncInferenceClient

            self._client = AsyncInferenceClient(
                model=self._model_name,
                token=self._api_key,
            )
            logger.info("HF Inference client initialised for %s", self._model_name)
            return self._client
        except ImportError:
            raise GenerationError(
                message="huggingface-hub is not installed",
                details="Install it with: pip install huggingface-hub",
            )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        client = self._get_client()

        try:
            response = await client.chat_completion(
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
                    message="HF model returned empty response",
                    details=f"Model: {self._model_name}",
                )

            logger.info(
                "Generated response (%d chars) using HF %s",
                len(content),
                self._model_name,
            )
            return content

        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(
                message="HF generation failed",
                details=str(exc),
            ) from exc

    @property
    def model_name(self) -> str:
        return self._model_name
