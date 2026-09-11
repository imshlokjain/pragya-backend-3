"""
Text Cleaner — preprocesses extracted text for chunking.

Handles the messy reality of government PDF extraction: repeated headers,
footers, page numbers, broken lines, OCR artifacts, etc. — while
preserving meaningful statistical data.
"""

from __future__ import annotations

import re
import logging
from collections import Counter

from rag.schemas import ProcessedDocument, PageContent

logger = logging.getLogger(__name__)

# ── Patterns ─────────────────────────────────────────────────────────────

# Page number patterns (standalone numbers, "Page X", "- X -", etc.)
_PAGE_NUM_PATTERNS = [
    re.compile(r"^\s*[-–—]\s*\d+\s*[-–—]\s*$"),          # - 42 -
    re.compile(r"^\s*Page\s+\d+\s*(?:of\s+\d+)?$", re.I), # Page 42 of 100
    re.compile(r"^\s*\d+\s*$"),                            # standalone number
    re.compile(r"^\s*\(\s*\d+\s*\)\s*$"),                  # (42)
]

# Hyphenation at line end
_HYPHEN_BREAK = re.compile(r"(\w+)-\s*\n\s*(\w+)")

# Multiple blank lines
_MULTI_BLANK = re.compile(r"\n{3,}")

# Multiple spaces (but not at start of line — preserve indentation)
_MULTI_SPACE = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")

# Common OCR artifacts
_OCR_ARTIFACTS = re.compile(r"[|¦}{~`€£¥©®™§¶]+")


def clean_document(doc: ProcessedDocument) -> ProcessedDocument:
    """Clean all pages in a processed document.

    This mutates the document in place for efficiency and returns it.

    Args:
        doc: A ``ProcessedDocument`` from the ingestion stage.

    Returns:
        The same document with cleaned text on every page.
    """
    # Step 1: Detect repeated headers/footers across pages
    repeated_lines = _detect_repeated_lines(doc.pages)

    # Step 2: Clean each page
    for page in doc.pages:
        page.text = clean_text(page.text, repeated_lines=repeated_lines)

    logger.info(
        "Cleaned %d pages, removed %d repeated header/footer pattern(s)",
        len(doc.pages),
        len(repeated_lines),
    )
    return doc


def clean_text(
    text: str,
    repeated_lines: set[str] | None = None,
) -> str:
    """Clean a block of extracted text.

    Args:
        text: Raw extracted text.
        repeated_lines: Optional set of lines to remove (headers/footers).

    Returns:
        Cleaned text.
    """
    if not text or not text.strip():
        return ""

    lines = text.split("\n")

    # Remove repeated headers/footers
    if repeated_lines:
        lines = [
            line for line in lines
            if _normalise_for_comparison(line) not in repeated_lines
        ]

    # Remove page number lines
    lines = [line for line in lines if not _is_page_number(line)]

    text = "\n".join(lines)

    # Fix hyphenation breaks (e.g., "econo-\nmic" → "economic")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)

    # Remove OCR artifacts (but keep numbers and %)
    text = _OCR_ARTIFACTS.sub(" ", text)

    # Normalise whitespace
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)

    # Strip trailing whitespace from each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


def _detect_repeated_lines(
    pages: list[PageContent],
    min_occurrences: int = 3,
    max_line_length: int = 200,
) -> set[str]:
    """Detect lines that repeat across multiple pages (headers/footers).

    We look at the first 3 and last 3 lines of each page. Lines that
    appear on more than ``min_occurrences`` pages are considered
    repeated headers or footers.

    Args:
        pages: List of page contents.
        min_occurrences: Minimum number of pages a line must appear on.
        max_line_length: Ignore lines longer than this (they're likely content).

    Returns:
        Set of normalised line strings to remove.
    """
    if len(pages) < min_occurrences:
        return set()

    counter: Counter[str] = Counter()

    for page in pages:
        lines = [l.strip() for l in page.text.split("\n") if l.strip()]
        if not lines:
            continue

        # Check first 3 and last 3 lines
        candidates = lines[:3] + lines[-3:]
        seen_on_page: set[str] = set()

        for line in candidates:
            if len(line) > max_line_length:
                continue
            normalised = _normalise_for_comparison(line)
            if normalised and normalised not in seen_on_page:
                seen_on_page.add(normalised)
                counter[normalised] += 1

    repeated = {
        line for line, count in counter.items()
        if count >= min_occurrences
    }

    if repeated:
        logger.debug("Detected %d repeated header/footer lines", len(repeated))

    return repeated


def _normalise_for_comparison(line: str) -> str:
    """Normalise a line for header/footer comparison.

    Strips whitespace, removes page numbers, and lowercases.
    """
    line = line.strip().lower()
    # Remove standalone page numbers within the line
    line = re.sub(r"\b\d+\b", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _is_page_number(line: str) -> bool:
    """Check if a line is just a page number."""
    for pattern in _PAGE_NUM_PATTERNS:
        if pattern.match(line):
            return True
    return False
