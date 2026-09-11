"""
Sentence Transformer embedding provider.

Default provider using ``all-MiniLM-L6-v2`` — runs locally, free,
~80 MB download on first use. Produces 384-dimensional embeddings.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.embeddings.base import EmbeddingProvider
from rag.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class SentenceTransformerProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers.

    Args:
        model_name: Hugging Face model name (default: ``all-MiniLM-L6-v2``).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model: Any = None
        self._dimension: int | None = None

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name, token=False)

            # Determine dimension by encoding a dummy text
            dummy = self._model.encode(["test"])
            self._dimension = len(dummy[0])
            logger.info(
                "Embedding model loaded: %s (dimension=%d)",
                self._model_name, self._dimension,
            )
        except ImportError:
            raise EmbeddingError(
                message="sentence-transformers is not installed",
                details="Install it with: pip install sentence-transformers",
            )
        except Exception as exc:
            raise EmbeddingError(
                message=f"Failed to load embedding model: {self._model_name}",
                details=str(exc),
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors.
        """
        self._load_model()

        if not texts:
            return []

        try:
            embeddings = self._model.encode(
                texts,
                show_progress_bar=len(texts) > 50,
                batch_size=64,
                normalize_embeddings=True,
            )
            return [emb.tolist() for emb in embeddings]
        except Exception as exc:
            raise EmbeddingError(
                message="Failed to generate document embeddings",
                details=str(exc),
            ) from exc

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Args:
            query: The query text.

        Returns:
            Embedding vector.
        """
        self._load_model()

        try:
            embedding = self._model.encode(
                [query],
                normalize_embeddings=True,
            )
            return embedding[0].tolist()
        except Exception as exc:
            raise EmbeddingError(
                message="Failed to generate query embedding",
                details=str(exc),
            ) from exc

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension  # type: ignore

    @property
    def model_name(self) -> str:
        return self._model_name
