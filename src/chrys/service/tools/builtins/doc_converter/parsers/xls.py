# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""XLS parser — converts legacy Excel files to Markdown using xlrd."""

from __future__ import annotations

from chrys.service.tools.builtins.doc_converter.parsers._table import rows_to_markdown_table
from chrys.service.tools.builtins.doc_converter.parsers.base import DocumentImageSink, ParsedDocument


class XlsParser:
    """Convert XLS (legacy Excel) files to Markdown tables via ``xlrd``."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset({".xls"})

    def parse(self, path: str, *, image_sink: DocumentImageSink | None = None) -> ParsedDocument:
        import xlrd

        wb = xlrd.open_workbook(path)
        parts: list[str] = []
        for sheet in wb.sheets():
            parts.append(f"# Sheet: {sheet.name}")
            if sheet.nrows == 0:
                parts.append("(empty sheet)")
                continue
            rows = [
                tuple(sheet.cell_value(row_idx, col) for col in range(sheet.ncols)) for row_idx in range(sheet.nrows)
            ]
            parts.append(rows_to_markdown_table(rows))
        return ParsedDocument(markdown="\n\n".join(parts))
