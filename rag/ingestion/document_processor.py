"""
Document Processor — orchestrates the full ingestion pipeline.

    PDF → load → extract text → extract tables → merge → ProcessedDocument

This is the single entry point for turning a raw PDF into a structured
document ready for cleaning and chunking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union, BinaryIO

from rag.schemas import (
    ProcessedDocument,
    DocumentMetadata,
    PageContent,
)
from rag.ingestion.pdf_loader import load_pdf, load_pdf_from_upload, PDFInput
from rag.ingestion.text_extractor import extract_text_from_pdf
from rag.ingestion.table_extractor import extract_tables_from_pdf
from rag.exceptions import ExtractionError

logger = logging.getLogger(__name__)


def process_pdf(
    source: PDFInput,
    filename: str | None = None,
    metadata: dict | None = None,
) -> ProcessedDocument:
    """Process a PDF synchronously: load, extract text and tables.

    Args:
        source: File path, bytes, or file-like object.
        filename: Optional filename (inferred from path if not given).
        metadata: Optional extra metadata (year, department, etc.).

    Returns:
        A ``ProcessedDocument`` with all pages, text, tables, and metadata.

    Raises:
        PDFLoadError: If the PDF cannot be opened.
        EmptyPDFError: If the PDF has no pages.
        ExtractionError: If extraction fails unexpectedly.
    """
    # Infer filename from path if not provided
    if filename is None and isinstance(source, (str, Path)):
        filename = Path(source).name
    filename = filename or "unknown.pdf"

    try:
        pdf = load_pdf(source)
    except Exception:
        raise  # re-raise PDFLoadError / EmptyPDFError as-is

    try:
        # Extract text from all pages
        text_pages = extract_text_from_pdf(pdf)

        # Extract tables from all pages
        tables_by_page = extract_tables_from_pdf(pdf)

        total_pages = len(pdf.pages)
    except Exception as exc:
        raise ExtractionError(
            message="Failed to extract content from PDF",
            details=str(exc),
        ) from exc
    finally:
        pdf.close()

    # Merge tables into page content
    pages: list[PageContent] = []
    scanned_pages: list[int] = []

    for page_content in text_pages:
        page_num = page_content.page_number
        page_tables = tables_by_page.get(page_num, [])
        page_content.tables = page_tables

        if page_content.is_scanned:
            scanned_pages.append(page_num)

        pages.append(page_content)

    doc_metadata = DocumentMetadata(
        filename=filename,
        total_pages=total_pages,
        extra=metadata or {},
    )

    doc = ProcessedDocument(
        metadata=doc_metadata,
        pages=pages,
        scanned_pages=scanned_pages,
    )

    logger.info(
        "Processed '%s': %d pages, %d scanned, %d tables total",
        filename,
        total_pages,
        len(scanned_pages),
        sum(len(p.tables) for p in pages),
    )

    return doc


async def process_pdf_upload(
    upload_file,
    filename: str | None = None,
    metadata: dict | None = None,
) -> ProcessedDocument:
    """Process a FastAPI UploadFile asynchronously.

    Args:
        upload_file: A FastAPI ``UploadFile``.
        filename: Optional override for the filename.
        metadata: Optional extra metadata.

    Returns:
        A ``ProcessedDocument``.
    """
    content = await upload_file.read()
    fname = filename or getattr(upload_file, "filename", "upload.pdf")
    return process_pdf(content, filename=fname, metadata=metadata)
