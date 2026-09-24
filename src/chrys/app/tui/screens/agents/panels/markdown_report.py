# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared building blocks for remote-info Markdown reports.

Used by the agent-config Test dialogs (ACP handshake, MCP connection) to
render information supplied by an external process. Every remote-controlled
string must pass through :func:`inline_text` or :func:`code_span` so crafted
agent/server text cannot forge block structure in the rendered report.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from chrys.foundation.i18n import msg

SUPPORTED_MARK = "✓"
UNSUPPORTED_MARK = "—"
NOT_ADVERTISED = msg("tui.connection_report.not_advertised", fallback="*Not advertised.*")


def inline_text(value: Any) -> str:
    """Force a remote-controlled string to stay inline.

    Collapsing all whitespace runs (newlines included) means crafted remote
    strings cannot forge block structure — headings, bullets, table rows —
    in the rendered report. Non-printable characters are dropped too: the
    markdown pipeline passes raw text through to the terminal, so embedded
    CSI/OSC escape sequences would otherwise reach it verbatim (screen
    clears, OSC-52 clipboard writes). Inline markup is left alone: it can
    only restyle the string's own text, which plain content could fake
    anyway.

    One caveat remains for callers: the result is only block-inert when it
    is NOT placed at the start of a rendered line — leading ``#``/``-``/
    ``>`` are still block syntax there. Start report lines with your own
    marker (bullet, bold, quote + italic) before any remote text.
    """
    printable = "".join(ch if ch.isprintable() else " " for ch in str(value))
    return " ".join(printable.split())


def code_span(value: Any) -> str:
    """Inline code span that a remote string cannot break out of."""
    return "`{}`".format(inline_text(value).replace("`", "'"))


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def cell(value: Any) -> str:
        return inline_text(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(" --- " for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def definition_bullets(entries: Sequence[tuple[str, str, str]]) -> str:
    """Bullet list of (name, id code span, description) definitions.

    Prose-heavy sections use bullets instead of tables: the table layout
    splits width proportionally to content, so a long Description column
    crushes the name/ID columns into character-wrapped shreds.
    """
    lines = []
    for name, ident, description in entries:
        head = f"- **{inline_text(name)}**"
        if ident:
            head += f" ({ident})"
        if flat := inline_text(description):
            head += f" — {flat}"
        lines.append(head)
    return "\n".join(lines)
