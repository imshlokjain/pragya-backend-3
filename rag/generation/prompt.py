"""
Prompt templates for RAG generation.

These prompts are designed specifically for government document analysis.
They enforce:
- Grounding in retrieved evidence only
- No hallucination of statistics
- Mandatory citations with source and page
- Explicit "insufficient information" when context is lacking
"""

from __future__ import annotations

from rag.schemas import ScoredChunk


# ── System Prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise and helpful assistant that answers questions using official government documents.

STRICT RULES:
1. Answer the user's question using ONLY the provided context below.
2. Do NOT invent, fabricate, or hallucinate any facts, statistics, numbers, or data.
3. Do NOT use your training knowledge to supplement the context. If the context does not contain enough information, say so explicitly.
4. For every important claim, statistic, or data point, cite the source document and page number in the format [Source: Document Name, Page X].
5. If data comes from a table, mention that it is from a table in the document.
6. Clearly distinguish between:
   - Facts directly stated in the documents
   - Calculations or inferences you derive from the data (label these as "Calculated from..." or "Based on...")
7. If the context contains conflicting information from different documents, mention both and note the discrepancy.
8. If the available documents do not contain sufficient information to answer the question, respond with: "The available documents do not contain sufficient information to answer this question."
9. Be concise and factual. Avoid unnecessary elaboration.
10. When citing numbers, use the exact figures from the documents."""


# ── User Prompt Template ────────────────────────────────────────────────

USER_PROMPT_TEMPLATE = """Context from government documents:

{context}

---

Question: {question}

Provide a detailed answer based ONLY on the context above. Cite sources with [Source: Document Name, Page X] for every key claim."""


def build_rag_prompt(
    question: str,
    scored_chunks: list[ScoredChunk],
) -> tuple[str, str]:
    """Build the system and user prompts for RAG generation.

    Args:
        question: The user's question.
        scored_chunks: Retrieved chunks with scores.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    context = _format_context(scored_chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )
    return SYSTEM_PROMPT, user_prompt


def _format_context(scored_chunks: list[ScoredChunk]) -> str:
    """Format retrieved chunks into a context string for the prompt.

    Each chunk is labeled with its source document, page, and section
    so the LLM can cite them properly.
    """
    if not scored_chunks:
        return "(No relevant context found)"

    parts = []
    for i, sc in enumerate(scored_chunks, start=1):
        chunk = sc.chunk
        header_parts = [f"Document: {chunk.document_name}"]
        header_parts.append(f"Page: {chunk.page_number}")
        if chunk.section:
            header_parts.append(f"Section: {chunk.section}")
        header_parts.append(f"Type: {chunk.content_type.value}")
        header_parts.append(f"Relevance: {sc.score:.2f}")

        header = " | ".join(header_parts)

        parts.append(
            f"[Source {i}] {header}\n"
            f"{chunk.content}"
        )

    return "\n\n---\n\n".join(parts)
