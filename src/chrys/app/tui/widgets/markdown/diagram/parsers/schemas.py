# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Requirement and flat C4 schemas lowered to the shared graph model."""

from __future__ import annotations

import re

from chrys.app.tui.widgets.markdown.diagram.model import (
    DiagnosticCode,
    DiagramEdge,
    DiagramIR,
    DiagramKind,
    Direction,
    EdgeStyle,
    NodeShape,
)

from .common import (
    MAX_EDGES,
    _Builder,
    _clean_label,
    _diagram_direction,
    _find_unquoted_delimiter,
    _parse_list_items,
    _source_lines,
)

_REQUIREMENT_TYPES = {
    "requirement",
    "functionalrequirement",
    "interfacerequirement",
    "performancerequirement",
    "physicalrequirement",
    "designconstraint",
    "element",
}
_REQUIREMENT_RELATIONS = "contains|copies|derives|satisfies|verifies|refines|traces"
_RELATION_RE = re.compile(
    rf"\s*(?:-\s*({_REQUIREMENT_RELATIONS})\s*->|<-\s*({_REQUIREMENT_RELATIONS})\s*-)", re.IGNORECASE
)
_C4_NODE_RE = re.compile(r"(?:Person|(?:System|Container|Component)(?:Db|Queue)?)(?:_Ext)?")
_ERROR = DiagnosticCode.UNSUPPORTED_FLOW_STATEMENT


def _name(raw: str) -> str | None:
    """Read a quoted or plain schema name without consuming surrounding syntax."""
    raw = raw.strip()
    if not raw or _find_unquoted_delimiter(raw, ("{", "}", ":", ";", "<", ">"), quote_chars='"') != -1:
        return None
    return _clean_label(raw, quote_chars='"') or None


def _body_lines(source: str, builder: _Builder, headers: set[str]) -> list[tuple[int, str]]:
    lines = [(number, line) for number, line in _source_lines(source) if not line.startswith("%%")]
    if not lines:
        builder.error(1, DiagnosticCode.EMPTY_SOURCE)
        return []
    if lines[0][1].casefold() not in headers:
        builder.error(lines[0][0], _ERROR)
        return []
    return lines[1:]


def _add_pending_edges(builder: _Builder, edges: list[DiagramEdge]) -> None:
    declared = {node.node_id for node in builder.nodes}
    for edge in edges:
        if edge.source not in declared or edge.target not in declared:
            builder.error(edge.line, _ERROR)
        else:
            builder.edge(edge)


def _queue_edge(builder: _Builder, edges: list[DiagramEdge], edge: DiagramEdge) -> None:
    if len(edges) >= MAX_EDGES:
        builder.error(edge.line, DiagnosticCode.EDGE_LIMIT, limit=MAX_EDGES)
    else:
        edges.append(edge)


def _requirement_relation(raw: str, line: int) -> DiagramEdge | None:
    position = _find_unquoted_delimiter(raw, ("<-", "-"), quote_chars='"')
    if position is None or position < 0:
        return None
    relation = _RELATION_RE.match(raw, position)
    if relation is None:
        return None
    first, second = _name(raw[:position]), _name(raw[relation.end() :])
    if not first or not second:
        return None
    source, target = (first, second) if relation[1] else (second, first)
    kind = (relation[1] or relation[2]).lower()
    return DiagramEdge(source, target, kind, style=EdgeStyle.DOTTED, line=line)


def parse_requirement(source: str) -> DiagramIR:
    """Parse the SysML requirement/element fields and all seven relation kinds."""
    builder = _Builder(DiagramKind.REQUIREMENT, Direction.TOP_DOWN)
    edges: list[DiagramEdge] = []
    active_name: str | None = None
    active_type = ""
    active_line = 0
    fields: dict[str, str] = {}
    declared: set[str] = set()
    for line, raw in _body_lines(source, builder, {"requirementdiagram"}):
        if raw.startswith("#"):
            continue
        if active_name is not None:
            if raw == "}":
                if active_name in declared:
                    builder.error(line, _ERROR)
                else:
                    node = builder.node(active_name, active_name, active_line, sections=(tuple(fields.values()),))
                    if node is not None:
                        builder.annotate(active_name, active_type, active_line)
                        declared.add(active_name)
                active_name = None
                continue
            field = re.fullmatch(r"(\w+)\s*:\s*(.+)", raw)
            allowed = {"type", "docref"} if active_type.lower() == "element" else {"id", "text", "risk", "verifymethod"}
            if field is None or field[1].lower() not in allowed:
                builder.error(line, _ERROR)
                continue
            key, value = field[1].lower(), field[2]
            if _find_unquoted_delimiter(value, ("{", "}"), quote_chars='"') != -1:
                builder.error(line, _ERROR)
                continue
            cleaned = _clean_label(value, quote_chars='"')
            if (
                not cleaned
                or (key == "risk" and cleaned.lower() not in {"low", "medium", "high"})
                or (
                    key == "verifymethod" and cleaned.lower() not in {"analysis", "inspection", "test", "demonstration"}
                )
                or key in fields
            ):
                builder.error(line, _ERROR)
            else:
                fields[key] = f"{field[1]}: {cleaned}"
            continue
        if raw.startswith("title "):
            builder.title = _clean_label(raw[6:])
        elif direction := re.fullmatch(r"direction\s+(TB|TD|BT|LR|RL)", raw, re.IGNORECASE):
            builder.direction = _diagram_direction(direction[1])
        elif declaration := re.fullmatch(r"(\w+)\s+(.+?)\s*\{\s*(\})?", raw):
            kind, name = declaration[1], _name(declaration[2])
            if kind.lower() not in _REQUIREMENT_TYPES or name is None:
                builder.error(line, _ERROR)
                continue
            if declaration[3]:
                if name in declared:
                    builder.error(line, _ERROR)
                elif builder.node(name, name, line) is not None:
                    builder.annotate(name, kind, line)
                    declared.add(name)
            else:
                active_name, active_type, active_line, fields = name, kind, line, {}
        elif edge := _requirement_relation(raw, line):
            _queue_edge(builder, edges, edge)
        elif re.match(r"(?:style|classDef|class)\s", raw):
            builder.warning(line, DiagnosticCode.UNSUPPORTED_DIRECTIVE)
        else:
            builder.error(line, _ERROR)
    if active_name is not None:
        builder.error(active_line, _ERROR)
    _add_pending_edges(builder, edges)
    return builder.finish()


def _c4_arguments(raw: str) -> tuple[str, ...] | None:
    """Allow empty optional C4 arguments and commas inside quoted text."""
    arguments: list[str] = []
    while len(arguments) < 6:
        delimiter = _find_unquoted_delimiter(raw, (",",))
        if delimiter is None:
            return None
        item = raw if delimiter < 0 else raw[:delimiter]
        if item.lstrip().startswith("$") or _find_unquoted_delimiter(item, ("(", ")", "{", "}")) != -1:
            return None
        value = _parse_list_items(item)
        if value is None and item.strip() not in {"", '""', "''"}:
            return None
        arguments.append(value[0] if value is not None else "")
        if delimiter < 0:
            return tuple(arguments)
        raw = raw[delimiter + 1 :]
    return None


def _c4_node(builder: _Builder, kind: str, args: tuple[str, ...], line: int) -> bool:
    technology = kind.startswith(("Container", "Component"))
    maximum = 4 if technology else 3
    if not 2 <= len(args) <= maximum or not args[0] or not args[1]:
        return False
    details = args[2:]
    shape = NodeShape.CYLINDER if "Db" in kind else NodeShape.RECTANGLE
    node = builder.node(
        args[0], args[1], line, shape=shape, explicit=True, sections=(tuple(value for value in details if value),)
    )
    if node is not None:
        builder.annotate(args[0], kind, line)
    return True


def _c4_edge(kind: str, args: tuple[str, ...], line: int) -> DiagramEdge | None:
    if not 3 <= len(args) <= 5 or not all(args[:3]):
        return None
    source, target = (args[1], args[0]) if kind == "Rel_Back" else (args[0], args[1])
    label = " / ".join(value for value in args[2:] if value)
    return DiagramEdge(source, target, label, source_marker="◀" if kind == "BiRel" else "", line=line)


def parse_c4(source: str) -> DiagramIR:
    """Parse flat Context, Container and Component diagrams; reject boundaries."""
    builder = _Builder(DiagramKind.C4, Direction.TOP_DOWN)
    edges: list[DiagramEdge] = []
    for line, raw in _body_lines(source, builder, {"c4context", "c4container", "c4component"}):
        if raw.startswith("title "):
            builder.title = _clean_label(raw[6:])
            continue
        if direction := re.fullmatch(r"direction\s+(TB|TD|BT|LR|RL)", raw, re.IGNORECASE):
            builder.direction = _diagram_direction(direction[1])
            continue
        call = re.fullmatch(r"(\w+)\s*\((.*)\)", raw)
        args = _c4_arguments(call[2]) if call is not None else None
        if call is None or args is None:
            builder.error(line, _ERROR)
            continue
        kind = call[1]
        if _C4_NODE_RE.fullmatch(kind):
            if not _c4_node(builder, kind, args, line):
                builder.error(line, _ERROR)
        elif kind in {"Rel", "BiRel", "Rel_Back"}:
            if edge := _c4_edge(kind, args, line):
                _queue_edge(builder, edges, edge)
            else:
                builder.error(line, _ERROR)
        else:
            builder.error(line, _ERROR)
    _add_pending_edges(builder, edges)
    return builder.finish()
