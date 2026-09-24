# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""DOCX parser — converts Word documents to Markdown using python-docx."""

from __future__ import annotations

from chrys.service.tools.builtins.doc_converter.parsers._table import rows_to_markdown_table
from chrys.service.tools.builtins.doc_converter.parsers.base import (
    DocumentImageSink,
    ParsedDocument,
    VisualOccurrence,
)

# python-docx heading style names → Markdown heading levels
_HEADING_MAP: dict[str, int] = {
    "Heading 1": 1,
    "Heading 2": 2,
    "Heading 3": 3,
    "Heading 4": 4,
    "Heading 5": 5,
    "Heading 6": 6,
    "Title": 1,
    "Subtitle": 2,
}


class DocxParser:
    """Convert DOCX files to Markdown via ``python-docx``."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".docx"})

    def parse(self, path: str, *, image_sink: DocumentImageSink | None = None) -> ParsedDocument:
        from docx import Document
        from docx.table import Table as DocxTable

        doc = Document(path)
        parts: list[str] = []
        for block in doc.iter_inner_content():
            if isinstance(block, DocxTable):
                rows = []
                for row in block.rows:
                    cells = tuple(cell.text.strip() for cell in row.cells)
                    rows.append(cells)
                if rows:
                    parts.append(rows_to_markdown_table(rows))
            else:
                text = block.text.strip()
                if not text:
                    continue
                style_name = block.style.name if block.style else ""
                level = _HEADING_MAP.get(style_name)
                if level is not None:
                    parts.append(f"{'#' * level} {text}")
                elif style_name.startswith("List"):
                    parts.append(f"- {text}")
                else:
                    parts.append(text)

        visuals: list[VisualOccurrence] = []
        if image_sink is not None:
            image_parts = tuple(doc.part.package.image_parts)
            for candidate_index, image_part in enumerate(image_parts):
                if not image_sink.try_reserve_occurrence():
                    image_sink.record_unprocessed_occurrences(len(image_parts) - candidate_index)
                    break
                ordinal = candidate_index + 1
                try:
                    blob = image_part.blob
                    source_name = image_part.filename or str(image_part.partname)
                except Exception:
                    image_sink.record_candidate_failure()
                    continue
                occurrence = image_sink.save_image(
                    blob,
                    location="Document",
                    ordinal=ordinal,
                    source_name=source_name,
                )
                if occurrence is not None:
                    visuals.append(occurrence)

        return ParsedDocument(
            markdown="\n\n".join(parts),
            visuals=tuple(visuals),
            warnings=image_sink.warnings if image_sink is not None else (),
        )
