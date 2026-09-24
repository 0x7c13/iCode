# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared bounded parsing and quote-aware text helpers."""

from __future__ import annotations

import html
import re
from dataclasses import replace
from decimal import Decimal

from chrys.app.tui.widgets.markdown.diagram.model import (
    ChartData,
    Diagnostic,
    DiagnosticCode,
    DiagnosticSeverity,
    DiagramEdge,
    DiagramIR,
    DiagramKind,
    DiagramNode,
    Direction,
    NodeShape,
)

MAX_SOURCE_BYTES = 64 * 1024


MAX_NODES = 200


MAX_EDGES = 500


MAX_DIAGNOSTICS = 32


_NUMBER_PATTERN = r"[+-]?(?:\d{1,12}(?:\.\d{1,6})?|\.\d{1,6})"


_ID_PATTERN = r"[^\W\d][\w.-]*"


_ID_RE = re.compile(_ID_PATTERN)


def _diagram_direction(raw: str) -> Direction:
    return {
        "TB": Direction.TOP_DOWN,
        "TD": Direction.TOP_DOWN,
        "BT": Direction.BOTTOM_UP,
        "LR": Direction.LEFT_RIGHT,
        "RL": Direction.RIGHT_LEFT,
    }[raw.upper()]


class _Builder:
    """Mutable, bounded construction helper for immutable diagram records."""

    def __init__(self, kind: DiagramKind, direction: Direction) -> None:
        self.kind = kind
        self.direction = direction
        self.title = ""
        self.simplified = False
        self.nodes: list[DiagramNode] = []
        self.edges: list[DiagramEdge] = []
        self.diagnostics: list[Diagnostic] = []
        self._node_indexes: dict[str, int] = {}
        self._has_fatal_error = False

    @property
    def has_fatal_error(self) -> bool:
        return self._has_fatal_error

    def error(self, line: int, code: DiagnosticCode, **parameters: str | int) -> None:
        self._has_fatal_error = True
        diagnostic = Diagnostic(line, code, tuple(sorted(parameters.items())))
        if len(self.diagnostics) < MAX_DIAGNOSTICS:
            self.diagnostics.append(diagnostic)
            return
        warning_index = next(
            (
                index
                for index in range(len(self.diagnostics) - 1, -1, -1)
                if self.diagnostics[index].severity is DiagnosticSeverity.WARNING
            ),
            None,
        )
        if warning_index is not None:
            self.diagnostics[warning_index] = diagnostic

    def warning(self, line: int, code: DiagnosticCode, **parameters: str | int) -> None:
        if len(self.diagnostics) < MAX_DIAGNOSTICS:
            self.diagnostics.append(
                Diagnostic(line, code, tuple(sorted(parameters.items())), DiagnosticSeverity.WARNING)
            )

    def node(
        self,
        node_id: str,
        label: str | None,
        line: int,
        *,
        shape: NodeShape = NodeShape.RECTANGLE,
        explicit: bool = False,
        sections: tuple[tuple[str, ...], ...] | None = None,
    ) -> DiagramNode | None:
        index = self._node_indexes.get(node_id)
        if index is None:
            if len(self.nodes) >= MAX_NODES:
                self.error(line, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
                return None
            node = DiagramNode(node_id, label or node_id, shape, sections or (), line)
            self._node_indexes[node_id] = len(self.nodes)
            self.nodes.append(node)
            return node
        current = self.nodes[index]
        if explicit:
            next_label = label if label is not None else current.label
            if label is not None and current.label != node_id and current.label != label:
                self.warning(line, DiagnosticCode.NODE_REDECLARED, node_id=node_id)
            current = replace(
                current,
                label=next_label,
                shape=shape,
                sections=current.sections if sections is None else sections,
            )
            self.nodes[index] = current
        elif sections is not None:
            current = replace(current, sections=sections)
            self.nodes[index] = current
        return current

    def add_member(self, node_id: str, member: str, line: int) -> None:
        node = self.node(node_id, None, line)
        if node is None:
            return
        attributes = list(node.sections[0]) if node.sections else []
        methods = list(node.sections[1]) if len(node.sections) > 1 else []
        (methods if "(" in member else attributes).append(member)
        self.node(node_id, None, line, sections=(tuple(attributes), tuple(methods)))

    def add_er_attribute(self, node_id: str, attribute: str, line: int) -> None:
        """Append one display-ready ER attribute to an entity section."""
        node = self.node(node_id, None, line)
        if node is None:
            return
        attributes = list(node.sections[0]) if node.sections else []
        attributes.append(attribute)
        self.node(node_id, None, line, sections=(tuple(attributes),))

    def annotate(self, node_id: str, annotation: str, line: int) -> None:
        node = self.node(node_id, None, line)
        if node is None:
            return
        self.nodes[self._node_indexes[node_id]] = replace(node, annotation=_clean_label(annotation))

    def add_note(self, node_id: str, note: str, line: int) -> None:
        node = self.node(node_id, None, line)
        if node is None:
            return
        cleaned = _clean_label(note)
        self.nodes[self._node_indexes[node_id]] = replace(node, notes=(*node.notes, cleaned))

    def edge(self, edge: DiagramEdge) -> None:
        if edge.source not in self._node_indexes or edge.target not in self._node_indexes:
            return
        if len(self.edges) >= MAX_EDGES:
            self.error(edge.line, DiagnosticCode.EDGE_LIMIT, limit=MAX_EDGES)
            return
        self.edges.append(edge)

    def finish(self, chart: ChartData | None = None) -> DiagramIR:
        return DiagramIR(
            self.kind,
            self.direction,
            tuple(self.nodes),
            tuple(self.edges),
            tuple(self.diagnostics),
            self._has_fatal_error,
            chart,
            self.title,
            self.simplified,
        )


def _strip_inline_comment(line: str) -> str:
    quote = ""
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if char in {'"', "'"}:
            if not quote:
                quote = char
            elif quote == char:
                quote = ""
            continue
        if not quote and line.startswith("%%", index):
            return line[:index]
    return line


def _source_lines(source: str, *, preserve_indent: bool = False) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    first_content = True
    in_frontmatter = False
    in_accessibility_description = False
    for number, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        if first_content and stripped == "---":
            first_content = False
            in_frontmatter = True
            continue
        first_content = False
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue
        if in_accessibility_description:
            if stripped == "}":
                in_accessibility_description = False
            continue
        if re.fullmatch(r"accdescr\s*\{", stripped, re.IGNORECASE):
            in_accessibility_description = True
            continue
        if re.match(r"acc(?:title|descr)\s*:", stripped, re.IGNORECASE):
            continue
        cleaned = line.rstrip() if stripped.startswith("%%") else _strip_inline_comment(line).rstrip()
        if not preserve_indent:
            cleaned = cleaned.strip()
        if cleaned.strip():
            lines.append((number, cleaned))
    return lines


def _clean_label(label: str, *, quote_chars: str = "\"'") -> str:
    label = label.strip()
    if len(label) >= 2 and label[0] == label[-1] and label[0] in quote_chars:
        label = label[1:-1]
    label = label.replace("<br/>", " ").replace("<br>", " ").replace("\\n", " ")

    def decode_numeric_entity(match: re.Match[str]) -> str:
        digits = match.group("html") or match.group("mermaid")
        if len(digits) > 7:
            return " "
        value = int(digits)
        return chr(value) if 0x20 <= value <= 0x10FFFF and not 0xD800 <= value <= 0xDFFF else " "

    decoded = re.sub(
        r"&#(?P<html>\d+);?|(?<!&)#(?P<mermaid>\d+);",
        decode_numeric_entity,
        label,
    )
    return html.unescape(decoded).replace("\r", " ").replace("\n", " ").replace("\t", " ")


def _parse_list_items(raw: str, *, quote_chars: str = "\"'") -> tuple[str, ...] | None:
    """Parse a small Mermaid comma-separated list with quoted labels."""
    values: list[str] = []
    current: list[str] = []
    quote = ""
    escaped = False
    for char in raw:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\" and quote:
            escaped = True
            continue
        if char in quote_chars:
            if not quote:
                quote = char
            elif quote == char:
                quote = ""
            else:
                current.append(char)
            continue
        if char == "," and not quote:
            value = _clean_label("".join(current), quote_chars="")
            if not value:
                return None
            values.append(value)
            current = []
            continue
        current.append(char)
    if quote or escaped:
        return None
    value = _clean_label("".join(current), quote_chars="")
    if not value:
        return None
    values.append(value)
    return tuple(values)


def _find_unquoted_delimiter(raw: str, delimiters: tuple[str, ...], *, quote_chars: str = "\"'") -> int | None:
    """Find syntax outside quotes; -1 means absent, None means unclosed quotes."""
    quote = ""
    escaped = False
    for index, char in enumerate(raw):
        if escaped:
            escaped = False
        elif char == "\\" and quote:
            escaped = True
        elif char in quote_chars:
            if not quote:
                quote = char
            elif quote == char:
                quote = ""
        elif not quote and raw.startswith(delimiters, index):
            return index
    return None if quote or escaped else -1


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text
