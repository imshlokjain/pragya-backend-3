"""
Structure-Aware Chunker — splits documents into retrieval-ready chunks.

Chunking strategy:
    Document → Sections → Paragraphs → Sized Chunks

Each chunk retains full provenance metadata (document_id, page, section,
content type). Tables get their own dedicated chunks.

Chunk size and overlap are configurable via RAGConfig.
"""

from __future__ import annotations

import logging
import re
import uuid

from rag.schemas import ProcessedDocument, PageContent, Chunk, ContentType
from rag.ingestion.text_extractor import is_likely_heading

logger = logging.getLogger(__name__)


def chunk_document(
    doc: ProcessedDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split a processed document into retrieval-ready chunks.

    The strategy is:
    1. Process each page
    2. Detect section boundaries from headings
    3. Split text into paragraph-aware chunks
    4. Create separate chunks for tables
    5. Attach metadata to every chunk

    Args:
        doc: A cleaned ``ProcessedDocument``.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap in characters between adjacent text chunks.

    Returns:
        List of ``Chunk`` objects with full metadata.
    """
    all_chunks: list[Chunk] = []
    current_section = ""

    for page in doc.pages:
        page_num = page.page_number
        doc_id = doc.metadata.document_id
        doc_name = doc.metadata.filename

        # ── Text chunks ──────────────────────────────────────────────
        if page.text.strip():
            text_chunks, current_section = _chunk_page_text(
                text=page.text,
                current_section=current_section,
                document_id=doc_id,
                document_name=doc_name,
                page_number=page_num,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(text_chunks)

        # ── Table chunks ─────────────────────────────────────────────
        for table in page.tables:
            if not table.markdown.strip():
                continue

            table_chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=doc_id,
                document_name=doc_name,
                page_number=page_num,
                section=current_section,
                content_type=ContentType.TABLE,
                content=table.markdown,
                char_count=len(table.markdown),
            )
            all_chunks.append(table_chunk)

    logger.info(
        "Created %d chunks from '%s' (text: %d, table: %d)",
        len(all_chunks),
        doc.metadata.filename,
        sum(1 for c in all_chunks if c.content_type == ContentType.TEXT),
        sum(1 for c in all_chunks if c.content_type == ContentType.TABLE),
    )

    return all_chunks


def _chunk_page_text(
    text: str,
    current_section: str,
    document_id: str,
    document_name: str,
    page_number: int,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[Chunk], str]:
    """Chunk the text from a single page using structure-aware splitting.

    Args:
        text: Cleaned text from a page.
        current_section: The current section heading (carries across pages).
        document_id: Parent document ID.
        document_name: Original filename.
        page_number: 1-indexed page number.
        chunk_size: Max chunk size in characters.
        chunk_overlap: Overlap between chunks.

    Returns:
        Tuple of (list of Chunks, updated current_section).
    """
    chunks: list[Chunk] = []
    paragraphs = _split_into_paragraphs(text)

    buffer = ""
    section = current_section

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check if this paragraph is a section heading
        if is_likely_heading(para):
            # Flush buffer before starting new section
            if buffer.strip():
                chunks.extend(
                    _split_buffer(
                        buffer.strip(), section, document_id, document_name,
                        page_number, chunk_size, chunk_overlap,
                    )
                )
                buffer = ""
            section = para
            continue

        # If adding this paragraph would exceed chunk_size, flush first
        if buffer and len(buffer) + len(para) + 2 > chunk_size:
            chunks.extend(
                _split_buffer(
                    buffer.strip(), section, document_id, document_name,
                    page_number, chunk_size, chunk_overlap,
                )
            )
            # Keep overlap from end of buffer
            if chunk_overlap > 0 and len(buffer) > chunk_overlap:
                buffer = buffer[-chunk_overlap:]
            else:
                buffer = ""

        buffer += "\n\n" + para if buffer else para

    # Flush remaining buffer
    if buffer.strip():
        chunks.extend(
            _split_buffer(
                buffer.strip(), section, document_id, document_name,
                page_number, chunk_size, chunk_overlap,
            )
        )

    return chunks, section


def _split_buffer(
    text: str,
    section: str,
    document_id: str,
    document_name: str,
    page_number: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split a text buffer into chunks that respect the size limit.

    If the buffer fits in one chunk, returns a single chunk.
    Otherwise, splits at sentence boundaries.
    """
    if len(text) <= chunk_size:
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                section=section,
                content_type=ContentType.TEXT,
                content=text,
                char_count=len(text),
            )
        ]

    # Split at sentence boundaries
    sentences = _split_into_sentences(text)
    chunks: list[Chunk] = []
    current = ""

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > chunk_size:
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_number,
                    section=section,
                    content_type=ContentType.TEXT,
                    content=current.strip(),
                    char_count=len(current.strip()),
                )
            )
            # Keep overlap
            if chunk_overlap > 0 and len(current) > chunk_overlap:
                current = current[-chunk_overlap:]
            else:
                current = ""

        current += " " + sentence if current else sentence

    if current.strip():
        chunks.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                document_name=document_name,
                page_number=page_number,
                section=section,
                content_type=ContentType.TEXT,
                content=current.strip(),
                char_count=len(current.strip()),
            )
        )

    return chunks


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double newlines."""
    return re.split(r"\n\s*\n", text)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Uses a two-pass approach: first split on period-space boundaries,
    then re-join splits caused by common abbreviations (Mr., Dr., etc.).
    This avoids variable-width lookbehinds which newer Python versions reject.
    """
    _ABBREVIATIONS = {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
        "approx", "rs", "no", "govt", "fig", "vol", "dept", "st",
    }

    # Split on ". " (period followed by whitespace)
    raw_parts = re.split(r"\.\s+", text)

    # Re-join parts where the split was caused by an abbreviation
    sentences: list[str] = []
    buffer = ""

    for i, part in enumerate(raw_parts):
        if buffer:
            buffer += ". " + part
        else:
            buffer = part

        # Check if this part ends with an abbreviation word
        is_last = i == len(raw_parts) - 1
        if not is_last:
            # Get the last word before the split point
            last_word = buffer.rstrip().rsplit(None, 1)[-1].lower() if buffer.strip() else ""
            if last_word in _ABBREVIATIONS:
                # This was an abbreviation split — keep accumulating
                continue

        # Flush the buffer as a complete sentence
        if buffer.strip():
            s = buffer.strip()
            if not is_last and not s.endswith("."):
                s += "."
            sentences.append(s)
        buffer = ""

    if buffer.strip():
        sentences.append(buffer.strip())

    return [s for s in sentences if s]
