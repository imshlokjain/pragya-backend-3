from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class SopProvider(ABC):
    """
    Contract for SOP retrieval.

    The backend does not care whether the implementation
    uses embeddings, vector search, an LLM, PostgreSQL,
    or another retrieval system.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class PrototypeSopProvider(SopProvider):
    """
    Temporary provider used until the real RAG/SOP
    system is integrated.
    """

    def search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        return {
            "query": query,
            "context": context or {},
            "status": "NOT_YET_AVAILABLE",
            "clauses": [],
            "sources": [],
            "note": (
                "SOP RAG pipeline is not implemented yet. "
                "No SOP clause was retrieved."
            ),
        }


sop_provider = PrototypeSopProvider()
