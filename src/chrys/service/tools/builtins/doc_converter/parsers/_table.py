# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared Markdown table renderer for document parsers."""

from __future__ import annotations


def _escape_cell(value: object) -> str:
    """Stringify and escape a cell value for use inside a Markdown table."""
    text = str(value) if value is not None else ""
    return text.replace("|", "\\|")


def rows_to_markdown_table(rows: list[tuple]) -> str:
    """Convert a list of row tuples to a Markdown table.

    The first row is used as the header.  Pipe characters in cell values
    are escaped so they don't break the table structure.
    """
    if not rows:
        return ""
    header = rows[0]
    col_count = len(header)
    header_cells = [_escape_cell(c) for c in header]
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for row in rows[1:]:
        cells = [_escape_cell(c) for c in row]
        # Pad or truncate to match header column count
        cells = (cells + [""] * col_count)[:col_count]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
