"""
Table Extractor — extracts tables from PDF pages and converts to Markdown.

Government PDFs frequently contain statistical tables. This module
extracts them as structured data and renders them as Markdown so they
can be embedded in chunks for LLM retrieval.
"""

from __future__ import annotations

import logging

import pdfplumber

from rag.schemas import ExtractedTable

logger = logging.getLogger(__name__)


def extract_tables_from_page(
    page: pdfplumber.page.Page,
    page_number: int,
) -> list[ExtractedTable]:
    """Extract all tables from a single PDF page.

    Tries the default line-based strategy first; if no tables are found,
    falls back to a text-based strategy for borderless tables.

    Args:
        page: A ``pdfplumber.page.Page`` object.
        page_number: 1-indexed page number.

    Returns:
        List of ``ExtractedTable`` objects.
    """
    tables: list[ExtractedTable] = []

    # Strategy 1: default (line-based detection)
    raw_tables = page.extract_tables() or []

    # Strategy 2: text-based detection for borderless tables
    if not raw_tables:
        raw_tables = page.extract_tables(
            table_settings={
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 4,
            }
        ) or []

    for raw_table in raw_tables:
        if not raw_table or len(raw_table) < 2:
            continue  # skip degenerate tables

        table = _parse_raw_table(raw_table, page_number)
        if table is not None:
            tables.append(table)

    if tables:
        logger.info("Page %d: extracted %d table(s)", page_number, len(tables))

    return tables


def extract_tables_from_pdf(pdf: pdfplumber.PDF) -> dict[int, list[ExtractedTable]]:
    """Extract tables from every page of a PDF.

    Args:
        pdf: An opened ``pdfplumber.PDF``.

    Returns:
        Dict mapping page number → list of ``ExtractedTable``.
    """
    all_tables: dict[int, list[ExtractedTable]] = {}
    for i, page in enumerate(pdf.pages, start=1):
        page_tables = extract_tables_from_page(page, page_number=i)
        if page_tables:
            all_tables[i] = page_tables

    total = sum(len(t) for t in all_tables.values())
    logger.info("Extracted %d table(s) across %d page(s)", total, len(all_tables))
    return all_tables


def _parse_raw_table(
    raw_table: list[list[str | None]],
    page_number: int,
) -> ExtractedTable | None:
    """Convert a raw pdfplumber table to an ExtractedTable.

    Args:
        raw_table: 2D list of cell values (may contain None).
        page_number: Source page number.

    Returns:
        An ``ExtractedTable``, or None if the table is degenerate.
    """
    # Clean cells: replace None with empty string, strip whitespace
    cleaned = []
    for row in raw_table:
        cleaned_row = [_clean_cell(cell) for cell in row]
        # Skip rows that are entirely empty
        if any(cell for cell in cleaned_row):
            cleaned.append(cleaned_row)

    if len(cleaned) < 2:
        return None

    headers = cleaned[0]
    rows = cleaned[1:]

    # Normalise row widths to match header count
    header_count = len(headers)
    normalised_rows = []
    for row in rows:
        if len(row) < header_count:
            row = row + [""] * (header_count - len(row))
        elif len(row) > header_count:
            row = row[:header_count]
        normalised_rows.append(row)

    markdown = _table_to_markdown(headers, normalised_rows)

    return ExtractedTable(
        headers=headers,
        rows=normalised_rows,
        markdown=markdown,
        page_number=page_number,
    )


def _clean_cell(cell: str | None) -> str:
    """Clean a single table cell value."""
    if cell is None:
        return ""
    # Replace newlines within cells with spaces
    return " ".join(str(cell).split())


def _table_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Render a table as a Markdown table string.

    Args:
        headers: Column header strings.
        rows: 2D list of row values.

    Returns:
        Markdown-formatted table string.
    """
    if not headers:
        return ""

    # Build header row
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"

    # Build data rows
    data_lines = []
    for row in rows:
        data_lines.append("| " + " | ".join(row) + " |")

    return "\n".join([header_line, separator] + data_lines)
