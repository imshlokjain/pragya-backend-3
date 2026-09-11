"""
Tests for PDF ingestion — loading, text extraction, and table extraction.
"""

import io
import pytest
from rag.ingestion.pdf_loader import load_pdf
from rag.ingestion.text_extractor import extract_text_from_pdf, is_likely_heading
from rag.ingestion.table_extractor import extract_tables_from_pdf, _table_to_markdown
from rag.ingestion.document_processor import process_pdf
from rag.exceptions import PDFLoadError, EmptyPDFError


class TestPDFLoader:
    """Tests for pdf_loader.py."""

    def test_load_from_path(self, sample_pdf_path):
        pdf = load_pdf(sample_pdf_path)
        assert len(pdf.pages) > 0
        pdf.close()

    def test_load_from_bytes(self, sample_pdf_bytes):
        pdf = load_pdf(sample_pdf_bytes)
        assert len(pdf.pages) > 0
        pdf.close()

    def test_load_from_bytesio(self, sample_pdf_bytes):
        buf = io.BytesIO(sample_pdf_bytes)
        pdf = load_pdf(buf)
        assert len(pdf.pages) > 0
        pdf.close()

    def test_load_nonexistent_file(self):
        with pytest.raises(PDFLoadError, match="not found"):
            load_pdf("/nonexistent/path/to/file.pdf")

    def test_load_empty_bytes(self):
        with pytest.raises(PDFLoadError, match="empty"):
            load_pdf(b"")

    def test_load_invalid_pdf(self):
        with pytest.raises(PDFLoadError):
            load_pdf(b"not a pdf file at all")


class TestTextExtractor:
    """Tests for text_extractor.py."""

    def test_extract_text(self, sample_pdf_bytes):
        pdf = load_pdf(sample_pdf_bytes)
        pages = extract_text_from_pdf(pdf)
        pdf.close()

        assert len(pages) > 0
        # Should have some text content
        total_text = "".join(p.text for p in pages)
        assert len(total_text) > 50

    def test_heading_detection(self):
        assert is_likely_heading("MACROECONOMIC OVERVIEW") is True
        assert is_likely_heading("CHAPTER 1: INTRODUCTION") is True
        assert is_likely_heading("1.2 Overview of Economy") is True
        assert is_likely_heading("This is a normal sentence.") is False
        assert is_likely_heading("") is False


class TestTableExtractor:
    """Tests for table_extractor.py."""

    def test_table_to_markdown(self):
        headers = ["State", "Rate (%)"]
        rows = [["Punjab", "7.3"], ["Haryana", "6.1"]]
        md = _table_to_markdown(headers, rows)

        assert "State" in md
        assert "Punjab" in md
        assert "7.3" in md
        assert "|" in md
        assert "---" in md

    def test_extract_tables(self, sample_pdf_bytes):
        pdf = load_pdf(sample_pdf_bytes)
        tables = extract_tables_from_pdf(pdf)
        pdf.close()
        # tables is a dict[int, list[ExtractedTable]]
        assert isinstance(tables, dict)


class TestDocumentProcessor:
    """Tests for document_processor.py."""

    def test_process_pdf_from_bytes(self, sample_pdf_bytes):
        doc = process_pdf(sample_pdf_bytes, filename="test_report.pdf")

        assert doc.metadata.filename == "test_report.pdf"
        assert doc.metadata.total_pages > 0
        assert len(doc.pages) > 0
        assert doc.metadata.document_id  # should be a UUID

    def test_process_pdf_from_path(self, sample_pdf_path):
        doc = process_pdf(sample_pdf_path)

        assert doc.metadata.total_pages > 0
        assert "test_government_report.pdf" in doc.metadata.filename

    def test_process_pdf_with_metadata(self, sample_pdf_bytes):
        doc = process_pdf(
            sample_pdf_bytes,
            filename="survey.pdf",
            metadata={"year": 2025, "department": "Finance"},
        )

        assert doc.metadata.extra["year"] == 2025
        assert doc.metadata.extra["department"] == "Finance"
