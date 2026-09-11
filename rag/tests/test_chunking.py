"""
Tests for text cleaning and chunking.
"""

import pytest
from rag.preprocessing.cleaner import clean_text, _is_page_number, _detect_repeated_lines
from rag.preprocessing.chunker import chunk_document, _split_into_paragraphs
from rag.ingestion.document_processor import process_pdf
from rag.schemas import PageContent


class TestCleaner:
    """Tests for cleaner.py."""

    def test_remove_page_numbers(self):
        assert _is_page_number("42") is True
        assert _is_page_number("- 42 -") is True
        assert _is_page_number("Page 42") is True
        assert _is_page_number("Page 42 of 100") is True
        assert _is_page_number("(42)") is True
        assert _is_page_number("The GDP was 42 percent") is False

    def test_fix_hyphenation(self):
        text = "The econo-\nmic growth was strong."
        cleaned = clean_text(text)
        assert "economic" in cleaned
        assert "econo-" not in cleaned

    def test_normalise_whitespace(self):
        text = "Too    many     spaces    here"
        cleaned = clean_text(text)
        assert "  " not in cleaned

    def test_normalise_blank_lines(self):
        text = "First paragraph\n\n\n\n\nSecond paragraph"
        cleaned = clean_text(text)
        assert "\n\n\n" not in cleaned

    def test_preserve_statistics(self):
        text = "GDP growth was 6.4% in 2024-25. Rs. 1,00,000 crore allocated."
        cleaned = clean_text(text)
        assert "6.4%" in cleaned
        assert "1,00,000" in cleaned

    def test_empty_text(self):
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_detect_repeated_lines(self):
        pages = [
            PageContent(page_number=i, text=f"Government of India\nContent page {i}\nPage {i}")
            for i in range(1, 6)
        ]
        repeated = _detect_repeated_lines(pages, min_occurrences=3)
        assert len(repeated) > 0  # "government of india" should be detected


class TestChunker:
    """Tests for chunker.py."""

    def test_split_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird."
        paras = _split_into_paragraphs(text)
        assert len(paras) == 3

    def test_chunk_document(self, sample_pdf_bytes):
        doc = process_pdf(sample_pdf_bytes, filename="test.pdf")
        chunks = chunk_document(doc, chunk_size=256, chunk_overlap=32)

        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.document_id == doc.metadata.document_id
            assert chunk.page_number > 0
            assert chunk.chunk_id  # UUID
            assert chunk.content  # non-empty
            assert chunk.char_count > 0

    def test_chunk_size_respected(self, sample_pdf_bytes):
        doc = process_pdf(sample_pdf_bytes, filename="test.pdf")
        max_size = 200
        chunks = chunk_document(doc, chunk_size=max_size, chunk_overlap=20)

        # Most chunks should respect the size limit
        # (some may exceed due to sentence boundaries — that's acceptable)
        oversized = [c for c in chunks if c.char_count > max_size * 2]
        assert len(oversized) <= len(chunks) * 0.2  # < 20% oversized is acceptable

    def test_chunk_metadata_preserved(self, sample_pdf_bytes):
        doc = process_pdf(sample_pdf_bytes, filename="economic_survey.pdf")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=64)

        assert all(c.document_name == "economic_survey.pdf" for c in chunks)
        assert all(c.document_id == doc.metadata.document_id for c in chunks)
