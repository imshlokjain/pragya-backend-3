"""
SOP-Aware Chunker preserving step sequences, protocols, roles, and conditions.
"""

from __future__ import annotations

import re
import uuid
from typing import List

from rag.schemas import ProcessedDocument, Chunk, ContentType

STEP_PATTERN = re.compile(
    r"^(?:Step\s+\d+|Phase\s+\d+|Protocol\s+[A-Z0-9]+|\d+\.\d+|\d+\)\s+|Action\s+Item|Role:)",
    re.IGNORECASE | re.MULTILINE,
)


class SOPChunker:
    """Specialized chunker that preserves procedural blocks and operational steps."""

    def __init__(self, target_chunk_size: int = 600, overlap: int = 80):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def chunk_document(self, doc: ProcessedDocument) -> List[Chunk]:
        """Split processed SOP document into step-aware chunks."""
        chunks: List[Chunk] = []

        for page in doc.pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            # First, separate into candidate blocks by headings or numbered steps
            blocks = re.split(r"\n(?=(?:Step\s+\d+|Phase\s+\d+|Protocol\s+[A-Z0-9]+|\d+\.\s+[A-Z]|\#{1,3}\s+))", page_text)

            current_chunk_text = ""
            current_section = "Procedure"

            for block in blocks:
                block = block.strip()
                if not block:
                    continue

                # Detect if block starts with a section or step header
                first_line = block.split("\n")[0].strip()
                if len(first_line) < 100 and (first_line.startswith("#") or "step" in first_line.lower() or "phase" in first_line.lower()):
                    current_section = first_line.lstrip("#").strip()

                if len(current_chunk_text) + len(block) <= self.target_chunk_size:
                    current_chunk_text = f"{current_chunk_text}\n\n{block}".strip()
                else:
                    if current_chunk_text:
                        chunks.append(
                            Chunk(
                                chunk_id=str(uuid.uuid4()),
                                document_id=doc.metadata.document_id,
                                document_name=doc.metadata.filename,
                                page_number=page.page_number,
                                section=current_section,
                                content_type=ContentType.TEXT,
                                content=current_chunk_text,
                                char_count=len(current_chunk_text),
                            )
                        )
                    # If single block is huge, break it down
                    if len(block) > self.target_chunk_size:
                        sub_blocks = [block[i:i + self.target_chunk_size] for i in range(0, len(block), self.target_chunk_size - self.overlap)]
                        for sb in sub_blocks[:-1]:
                            chunks.append(
                                Chunk(
                                    chunk_id=str(uuid.uuid4()),
                                    document_id=doc.metadata.document_id,
                                    document_name=doc.metadata.filename,
                                    page_number=page.page_number,
                                    section=current_section,
                                    content_type=ContentType.TEXT,
                                    content=sb.strip(),
                                    char_count=len(sb.strip()),
                                )
                            )
                        current_chunk_text = sub_blocks[-1].strip()
                    else:
                        current_chunk_text = block

            if current_chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=doc.metadata.document_id,
                        document_name=doc.metadata.filename,
                        page_number=page.page_number,
                        section=current_section,
                        content_type=ContentType.TEXT,
                        content=current_chunk_text,
                        char_count=len(current_chunk_text),
                    )
                )

        return chunks
