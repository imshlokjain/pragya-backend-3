"""
Text Extractor — extracts text content from PDF pages.

Handles section heading detection and flags pages that appear to be
scanned images (no extractable text).
"""

from __future__ import annotations

import logging
import re

import pdfplumber

from rag.schemas import PageContent

logger = logging.getLogger(__name__)

# Heuristic: lines that are ALL CAPS or start with a number + period are likely headings
_HEADING_PATTERNS = [
    re.compile(r"^(?:CHAPTER|SECTION|PART)\s+[\dIVXLCDM]+", re.IGNORECASE),
    re.compile(r"^\d+\.\d*\s+[A-Z]"),                  # "1.2 Overview"
    re.compile(r"^[A-Z][A-Z\s]{4,}$"),                   # "MACROECONOMIC OVERVIEW"
    re.compile(r"^(?:Annexure|Appendix|Schedule)\s", re.IGNORECASE),
]

# Minimum characters on a page to consider it non-scanned
_MIN_TEXT_CHARS = 30


def extract_text_from_page(page: pdfplumber.page.Page, page_number: int) -> PageContent:
    """Extract text from a single pdfplumber page.

    Args:
        page: A ``pdfplumber.page.Page`` object.
        page_number: 1-indexed page number.

    Returns:
        A ``PageContent`` with extracted text and scanned-page flag.
    """
    raw_text = page.extract_text() or ""

    is_scanned = len(raw_text.strip()) < _MIN_TEXT_CHARS

    if is_scanned:
        logger.warning("Page %d appears scanned (only %d chars extracted)", page_number, len(raw_text.strip()))

    return PageContent(
        page_number=page_number,
        text=raw_text,
        is_scanned=is_scanned,
    )


def extract_text_from_pdf(pdf: pdfplumber.PDF) -> list[PageContent]:
    """Extract text from all pages of a PDF.

    Args:
        pdf: An opened ``pdfplumber.PDF``.

    Returns:
        List of ``PageContent`` objects, one per page.
    """
    pages: list[PageContent] = []
    for i, page in enumerate(pdf.pages, start=1):
        page_content = extract_text_from_page(page, page_number=i)
        pages.append(page_content)

    total_text = sum(len(p.text) for p in pages)
    scanned = sum(1 for p in pages if p.is_scanned)
    logger.info(
        "Extracted text from %d pages (%d chars total, %d scanned pages)",
        len(pages), total_text, scanned,
    )
    return pages


def detect_sections(text: str) -> list[dict]:
    """Detect section headings in extracted text.

    Args:
        text: Raw extracted text from a page.

    Returns:
        List of dicts with ``heading`` and ``start_pos`` keys.
    """
    sections = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in _HEADING_PATTERNS:
            if pattern.match(stripped):
                sections.append({
                    "heading": stripped,
                    "start_pos": text.index(line),
                })
                break
    return sections


def is_likely_heading(line: str) -> bool:
    """Check if a line of text looks like a section heading.

    Args:
        line: A single line of text.

    Returns:
        True if the line matches heading heuristics.
    """
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False

    # All caps with at least 3 words
    if stripped.isupper() and len(stripped.split()) >= 2:
        return True

    for pattern in _HEADING_PATTERNS:
        if pattern.match(stripped):
            return True

    return False
