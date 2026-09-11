"""
PDF Loader — accepts file paths, bytes, or FastAPI UploadFile objects.

Normalises all inputs into an opened pdfplumber PDF object for downstream
extraction. Validates that the PDF is readable and non-empty.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import BinaryIO, Union

import pdfplumber

from rag.exceptions import PDFLoadError, EmptyPDFError

logger = logging.getLogger(__name__)

# Type alias for all accepted PDF input types
PDFInput = Union[str, Path, bytes, BinaryIO]


def load_pdf(source: PDFInput) -> pdfplumber.PDF:
    """Open a PDF from various source types.

    Args:
        source: One of:
            - ``str`` or ``Path``: filesystem path to a PDF
            - ``bytes``: raw PDF bytes
            - file-like object with a ``.read()`` method (e.g. FastAPI UploadFile,
              ``io.BytesIO``, or an open file handle)

    Returns:
        An opened ``pdfplumber.PDF`` instance. The caller is responsible
        for closing it (or using it as a context manager).

    Raises:
        PDFLoadError: If the file cannot be opened or is not a valid PDF.
        EmptyPDFError: If the PDF contains zero pages.
    """
    try:
        pdf = _open_source(source)
    except (PDFLoadError, EmptyPDFError):
        raise
    except Exception as exc:
        raise PDFLoadError(
            message="Failed to open PDF",
            details=str(exc),
        ) from exc

    if not pdf.pages:
        pdf.close()
        raise EmptyPDFError(
            message="The PDF contains no pages",
            details="Page count is 0. The file may be corrupt or empty.",
        )

    logger.info("Loaded PDF with %d pages", len(pdf.pages))
    return pdf


def _open_source(source: PDFInput) -> pdfplumber.PDF:
    """Dispatch to the correct pdfplumber opener based on input type."""

    # Filesystem path
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise PDFLoadError(
                message=f"PDF file not found: {path}",
                details="Check the file path and try again.",
            )
        if not path.suffix.lower() == ".pdf":
            logger.warning("File does not have .pdf extension: %s", path)
        return pdfplumber.open(path)

    # Raw bytes
    if isinstance(source, bytes):
        if len(source) == 0:
            raise PDFLoadError(
                message="Received empty bytes",
                details="The PDF content is 0 bytes.",
            )
        return pdfplumber.open(io.BytesIO(source))

    # File-like object (UploadFile.file, BytesIO, etc.)
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            data = data.encode("latin-1")
        if len(data) == 0:
            raise PDFLoadError(
                message="Received empty file stream",
                details="The uploaded file has no content.",
            )
        return pdfplumber.open(io.BytesIO(data))

    raise PDFLoadError(
        message=f"Unsupported PDF source type: {type(source).__name__}",
        details="Provide a file path (str/Path), bytes, or a file-like object.",
    )


async def load_pdf_from_upload(upload_file) -> pdfplumber.PDF:
    """Async helper for FastAPI UploadFile objects.

    FastAPI's ``UploadFile.read()`` is async, so we await it first
    and then hand the bytes to the synchronous loader.

    Args:
        upload_file: A FastAPI ``UploadFile`` instance.

    Returns:
        An opened ``pdfplumber.PDF``.
    """
    try:
        content = await upload_file.read()
    except Exception as exc:
        raise PDFLoadError(
            message="Failed to read uploaded file",
            details=str(exc),
        ) from exc

    return load_pdf(content)
