"""
Shared test fixtures for RAG tests.

Creates a small sample PDF programmatically for testing so we don't
need to ship a binary fixture.
"""

from __future__ import annotations

import io
import pytest


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a small government-style PDF in memory for testing.

    Uses reportlab if available, otherwise falls back to a minimal
    valid PDF created manually.
    """
    try:
        return _create_pdf_with_reportlab()
    except ImportError:
        return _create_minimal_pdf()


@pytest.fixture
def sample_pdf_path(tmp_path, sample_pdf_bytes) -> str:
    """Write the sample PDF to a temp file and return the path."""
    path = tmp_path / "test_government_report.pdf"
    path.write_bytes(sample_pdf_bytes)
    return str(path)


def _create_pdf_with_reportlab() -> bytes:
    """Create a realistic multi-page PDF using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Page 1: Title and intro
    story.append(Paragraph("Economic Survey 2025-26", styles["Title"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph("CHAPTER 1: MACROECONOMIC OVERVIEW", styles["Heading1"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "India's GDP growth rate in 2024-25 was estimated at 6.4 percent, "
        "reflecting a robust economic performance driven by strong domestic "
        "consumption and infrastructure investment. The manufacturing sector "
        "grew by 5.2 percent while the services sector expanded by 7.1 percent.",
        styles["Normal"],
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "The fiscal deficit for 2024-25 was contained at 4.9 percent of GDP, "
        "meeting the government's target. Foreign direct investment inflows "
        "reached USD 71 billion during the year.",
        styles["Normal"],
    ))

    # Table on page 1
    story.append(Spacer(1, 15))
    story.append(Paragraph("Table 1.1: Key Economic Indicators", styles["Heading2"]))
    table_data = [
        ["Indicator", "2023-24", "2024-25"],
        ["GDP Growth (%)", "8.2", "6.4"],
        ["Inflation (%)", "5.4", "4.8"],
        ["Fiscal Deficit (% of GDP)", "5.6", "4.9"],
        ["FDI (USD Billion)", "44", "71"],
    ]
    t = Table(table_data)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t)

    # Page 2: Employment section
    story.append(Spacer(1, 40))
    story.append(Paragraph("CHAPTER 2: EMPLOYMENT AND LABOUR", styles["Heading1"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "The unemployment rate in Punjab stood at 7.3 percent in 2024-25, "
        "higher than the national average of 4.1 percent. The state's "
        "agricultural sector employed 42 percent of the workforce while "
        "the services sector accounted for 35 percent.",
        styles["Normal"],
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "The national literacy rate was recorded at 77.7 percent according to "
        "the latest census data. Female literacy rate was 70.3 percent while "
        "male literacy rate was 84.7 percent.",
        styles["Normal"],
    ))

    # Table on page 2
    story.append(Spacer(1, 15))
    story.append(Paragraph("Table 2.1: State-wise Unemployment Rates", styles["Heading2"]))
    table_data2 = [
        ["State", "Unemployment Rate (%)"],
        ["Punjab", "7.3"],
        ["Haryana", "6.1"],
        ["Rajasthan", "4.5"],
        ["Gujarat", "3.2"],
        ["National Average", "4.1"],
    ]
    t2 = Table(table_data2)
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(t2)

    doc.build(story)
    return buf.getvalue()


def _create_minimal_pdf() -> bytes:
    """Create a minimal valid PDF without external dependencies.

    This produces a simple single-page PDF with text content.
    """
    content = (
        "Economic Survey 2025-26\n"
        "CHAPTER 1: MACROECONOMIC OVERVIEW\n"
        "India's GDP growth rate in 2024-25 was estimated at 6.4 percent.\n"
        "The fiscal deficit was contained at 4.9 percent of GDP.\n"
        "CHAPTER 2: EMPLOYMENT AND LABOUR\n"
        "The unemployment rate in Punjab stood at 7.3 percent.\n"
        "The national literacy rate was 77.7 percent.\n"
    )

    # Minimal PDF structure
    pdf_lines = [
        b"%PDF-1.4",
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj",
    ]

    # Stream content
    stream_content = b"BT /F1 12 Tf 72 720 Td "
    for line in content.split("\n"):
        if line.strip():
            escaped = line.replace("(", "\\(").replace(")", "\\)")
            stream_content += f"({escaped}) Tj 0 -18 Td ".encode()
    stream_content += b"ET"

    stream_obj = f"4 0 obj << /Length {len(stream_content)} >> stream\n".encode()
    stream_obj += stream_content
    stream_obj += b"\nendstream endobj"

    pdf_lines.append(stream_obj)
    pdf_lines.append(b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")

    body = b"\n".join(pdf_lines)

    xref_offset = len(body) + 1
    xref = b"\nxref\n0 6\n"
    xref += b"0000000000 65535 f \n"

    offset = len(b"%PDF-1.4\n")
    for i in range(1, 6):
        xref += f"{offset:010d} 00000 n \n".encode()
        # Approximate offset
        offset += 60

    trailer = f"\ntrailer << /Root 1 0 R /Size 6 >>\nstartxref\n{xref_offset}\n%%EOF".encode()

    return body + xref + trailer
