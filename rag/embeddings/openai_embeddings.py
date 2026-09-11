"""
OpenAI embedding provider.

Uses the OpenAI API (or any OpenAI-compatible endpoint) for embeddings.
Requires an API key via ``RAG_OPENAI_API_KEY``.
"""

from __future__ import annotations

import logging

from rag.embeddings.base import EmbeddingProvider
from rag.exceptions import EmbeddingError, ConfigurationError

logger = logging.getLogger(__name__)

# Model → dimension mapping for known OpenAI models
_KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible embedding provider.

    Args:
        model_name: OpenAI model name (default: ``text-embedding-3-small``).
        api_key: OpenAI API key.
        base_url: Optional custom base URL for compatible APIs.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str = "",
        base_url: str | None = None,
    ):
        if not api_key:
            raise ConfigurationError(
                message="OpenAI API key is required for OpenAI embeddings",
                details="Set RAG_OPENAI_API_KEY in your .env file.",
            )

        self._model_name = model_name
        self._dimension = _KNOWN_DIMENSIONS.get(model_name, 1536)
        self._client = None
        self._api_key = api_key
        self._base_url = base_url

    def _get_client(self):
        """Lazy-initialise the OpenAI client."""
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI

            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = OpenAI(**kwargs)
            logger.info("OpenAI embedding client initialised for %s", self._model_name)
            return self._client
        except ImportError:
            raise EmbeddingError(
                message="openai package is not installed",
                details="Install it with: pip install openai",
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._get_client()

        try:
            # OpenAI supports batching natively
            response = client.embeddings.create(
                model=self._model_name,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise EmbeddingError(
                message="Failed to generate OpenAI embeddings",
                details=str(exc),
            ) from exc

    def embed_query(self, query: str) -> list[float]:
        results = self.embed_documents([query])
        return results[0]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name
