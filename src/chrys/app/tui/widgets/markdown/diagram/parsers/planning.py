# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded Mermaid journey, timeline, kanban, and mindmap parsers."""

from __future__ import annotations

import re
from dataclasses import replace

from chrys.app.tui.widgets.markdown.diagram.model import (
    DiagnosticCode,
    DiagramEdge,
    DiagramIR,
    DiagramKind,
    Direction,
    NodeShape,
)
from chrys.app.tui.widgets.markdown.diagram.specs.planning import (
    JourneyChart,
    JourneyTask,
    KanbanChart,
    KanbanColumn,
    KanbanTask,
    TimelineChart,
    TimelinePeriod,
)

from .common import (
    MAX_EDGES,
    MAX_NODES,
    _Builder,
    _clean_label,
    _find_unquoted_delimiter,
    _parse_list_items,
    _source_lines,
)

_MINDMAP_SHAPES = (
    ("((", "))", NodeShape.CIRCLE),
    ("{{", "}}", NodeShape.HEXAGON),
    ("))", "((", NodeShape.RECTANGLE),
    ("[", "]", NodeShape.RECTANGLE),
    ("(", ")", NodeShape.ROUNDED),
    (")", "(", NodeShape.ROUNDED),
)


def _comment(builder: _Builder, line: int, text: str) -> bool:
    if not text.startswith("%%"):
        return False
    if text.startswith("%%{"):
        builder.warning(line, DiagnosticCode.UNSUPPORTED_DIRECTIVE)
    return True


def _planning_lines(builder: _Builder, source: str, *, preserve_indent: bool = False) -> list[tuple[int, str]]:
    lines = _source_lines(source, preserve_indent=preserve_indent)
    for index, (line, text) in enumerate(lines):
        if not _comment(builder, line, text.strip()):
            if builder.kind is not DiagramKind.TIMELINE and text.strip() != builder.kind.value:
                builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            return lines[index:]
    return []


def _split_fields(text: str, separator: str) -> tuple[str, ...] | None:
    """Split punctuation outside quoted strings, retaining empty fields."""
    fields: list[str] = []
    while True:
        index = _find_unquoted_delimiter(text, (separator,))
        if index is None:
            return None
        if index < 0:
            return (*fields, _clean_label(text))
        fields.append(_clean_label(text[:index]))
        text = text[index + len(separator) :]


def parse_journey(source: str) -> DiagramIR:
    """Parse sections and ordered tasks with 1-5 scores and actor lists."""
    builder = _Builder(DiagramKind.JOURNEY, Direction.LEFT_RIGHT)
    title = section = ""
    tasks: list[JourneyTask] = []
    for line, text in _planning_lines(builder, source)[1:]:
        if _comment(builder, line, text):
            continue
        if text.startswith("title "):
            title = _clean_label(text[6:])
            continue
        if text.startswith("section "):
            section = _clean_label(text[8:])
            continue
        fields = tuple(part.strip() for part in text.split(":", 2))
        if len(fields) != 3 or not fields[0] or not re.fullmatch(r"[1-5]", fields[1]):
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        # Apostrophes are actor text; retain our quoted-comma extension.
        actors = _parse_list_items(fields[2], quote_chars='"')
        if actors is None:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        if len(tasks) >= MAX_NODES:
            builder.error(line, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
            break
        tasks.append(JourneyTask(section, _clean_label(fields[0], quote_chars=""), int(fields[1]), actors))
    if not tasks:
        builder.error(1, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
    return builder.finish(JourneyChart(title, tuple(tasks)))


def parse_timeline(source: str) -> DiagramIR:
    """Retain textual periods, multiple events, sections, and LR/TD order."""
    builder = _Builder(DiagramKind.TIMELINE, Direction.LEFT_RIGHT)
    lines = _planning_lines(builder, source)
    header = " ".join(lines[0][1].split()) if lines else ""
    if header == "timeline TD":
        builder.direction = Direction.TOP_DOWN
    if header not in {"timeline", "timeline LR", "timeline TD"}:
        builder.error(1, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
    title = section = ""
    periods: list[TimelinePeriod] = []
    event_count = 0
    may_continue = False
    for line, text in lines[1:]:
        if _comment(builder, line, text):
            continue
        if text.startswith("title "):
            title = _clean_label(text[6:])
            continue
        if text.startswith("section "):
            section = _clean_label(text[8:])
            may_continue = False
            continue
        fields = tuple(_clean_label(part) for part in text.split(":"))
        if len(fields) < 2 or any(not event for event in fields[1:]):
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        if event_count + len(fields) - 1 > MAX_EDGES:
            builder.error(line, DiagnosticCode.EDGE_LIMIT, limit=MAX_EDGES)
            break
        if fields[0]:
            if len(periods) >= MAX_NODES:
                builder.error(line, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
                break
            periods.append(TimelinePeriod(section, fields[0], fields[1:]))
            may_continue = True
        elif may_continue:
            periods[-1] = replace(periods[-1], events=(*periods[-1].events, *fields[1:]))
        else:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        event_count += len(fields) - 1
    if not periods:
        builder.error(1, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
    return builder.finish(TimelineChart(title, header == "timeline TD", tuple(periods)))


def _kanban_metadata(text: str) -> tuple[tuple[str, str], ...] | None:
    if not text:
        return ()
    if not text.startswith("@{") or not text.endswith("}"):
        return None
    # Split before removing quotes so commas and colons inside values stay literal.
    raw_items: list[str] = []
    body = text[2:-1]
    while body.strip():
        end = _find_unquoted_delimiter(body, (",",))
        if end is None:
            return None
        if end < 0:
            raw_items.append(body)
            break
        raw_items.append(body[:end])
        body = body[end + 1 :]
        if not body.strip():
            return None
    metadata: list[tuple[str, str]] = []
    for item in raw_items:
        fields = _split_fields(item, ":")
        if fields is None or len(fields) != 2 or not fields[1]:
            return None
        key, value = fields
        if key not in {"assigned", "ticket", "priority"} or any(previous == key for previous, _ in metadata):
            return None
        if key == "priority" and value not in {"Very High", "High", "Low", "Very Low"}:
            return None
        metadata.append((key, value))
    return tuple(metadata)


def _kanban_item(text: str, line: int) -> tuple[str, str, tuple[tuple[str, str], ...]] | None:
    bracket = text.find("[")
    if bracket < 0:
        metadata_start = text.find("@{")
        body = text[:metadata_start].strip() if metadata_start >= 0 else text
        metadata = _kanban_metadata(text[metadata_start:].strip()) if metadata_start >= 0 else ()
        if metadata is None:
            return None
        if any(char in body for char in "[]{}"):
            return None
        label = _clean_label(body)
        return (f"@kanban:{line}", label, metadata) if label else None
    identifier = text[:bracket].strip()
    if identifier and re.fullmatch(r"[\w.-]+", identifier) is None:
        return None
    remainder = text[bracket + 1 :]
    # Only a leading quote delimits a quoted node label. Apostrophes in ordinary
    # labels (for example "Can't reproduce") are literal Mermaid text.
    end = _find_unquoted_delimiter(remainder, ("]",)) if remainder.startswith(('"', "'")) else remainder.find("]")
    if end is None or end < 0:
        return None
    metadata = _kanban_metadata(remainder[end + 1 :].strip())
    if metadata is None:
        return None
    label = _clean_label(remainder[:end])
    return (identifier or f"@kanban:{line}", label, metadata) if label else None


def parse_kanban(source: str) -> DiagramIR:
    """Read a two-level board with full task text and common metadata."""
    builder = _Builder(DiagramKind.KANBAN, Direction.LEFT_RIGHT)
    columns: list[KanbanColumn] = []
    identifiers: set[str] = set()
    column_indent: int | None = None
    task_indent: int | None = None
    for line, raw in _planning_lines(builder, source, preserve_indent=True)[1:]:
        text = raw.strip()
        if _comment(builder, line, text):
            continue
        indent = len(raw.expandtabs(4)) - len(raw.expandtabs(4).lstrip())
        item = _kanban_item(text, line)
        if item is None:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        identifier, label, metadata = item
        if identifier in identifiers:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        if len(identifiers) >= MAX_NODES:
            builder.error(line, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
            break
        identifiers.add(identifier)
        if column_indent is None:
            column_indent = indent
        if indent == column_indent:
            if metadata:
                builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            columns.append(KanbanColumn(identifier, label, ()))
            task_indent = None
        elif indent > column_indent and columns and (task_indent is None or indent == task_indent):
            task_indent = indent
            task = KanbanTask(identifier, label, metadata)
            columns[-1] = replace(columns[-1], tasks=(*columns[-1].tasks, task))
        else:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
    if not columns:
        builder.error(1, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
    return builder.finish(KanbanChart(tuple(columns)))


def _mindmap_label(text: str) -> tuple[str, NodeShape] | None:
    # Shape IDs are local display syntax; indentation, not identifier equality,
    # determines the tree. This also preserves repeated plain-text siblings.
    for opener, closer, shape in _MINDMAP_SHAPES:
        start = text.find(opener)
        if start < 0:
            continue
        identifier = text[:start].strip()
        if identifier and re.fullmatch(r"[\w.-]+", identifier) is None:
            continue
        if not text.endswith(closer):
            return None
        label = _clean_label(text[start + len(opener) : -len(closer)])
        return (label, shape) if label else None
    if any(char in text for char in "[]{}"):
        return None
    return (_clean_label(text), NodeShape.RECTANGLE) if text else None


def parse_mindmap(source: str) -> DiagramIR:
    """Translate indentation into a left-to-right tree of terminal nodes."""
    builder = _Builder(DiagramKind.MINDMAP, Direction.LEFT_RIGHT)
    stack: list[tuple[int, str]] = []
    last_node = ""
    for line, raw in _planning_lines(builder, source, preserve_indent=True)[1:]:
        text = raw.strip()
        if _comment(builder, line, text):
            continue
        if text.startswith("::"):
            if last_node and re.fullmatch(r"::icon\([^()]+\)", text):
                builder.add_note(last_node, text[7:-1], line)
            elif last_node and re.fullmatch(r":::[\w -]+", text):
                builder.warning(line, DiagnosticCode.UNSUPPORTED_DIRECTIVE)
            else:
                builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        parsed = _mindmap_label(text)
        if parsed is None:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        indent = len(raw.expandtabs(4)) - len(raw.expandtabs(4).lstrip())
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack and builder.nodes:
            builder.error(line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        identifier = f"@mindmap:{line}"
        node = builder.node(identifier, parsed[0], line, shape=parsed[1])
        if node is None:
            break
        if stack:
            builder.edge(DiagramEdge(stack[-1][1], identifier, directed=False, line=line))
        stack.append((indent, identifier))
        last_node = identifier
    if not builder.nodes:
        builder.error(1, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
    return builder.finish()
