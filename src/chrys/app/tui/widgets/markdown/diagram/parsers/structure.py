# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Fixed-grid blocks and explicitly schematic architecture/weighted flows."""

from __future__ import annotations

import csv
import re
from decimal import Decimal

from chrys.app.tui.widgets.markdown.diagram.model import (
    DiagnosticCode,
    DiagramEdge,
    DiagramIR,
    DiagramKind,
    Direction,
)
from chrys.app.tui.widgets.markdown.diagram.specs.structure import BlockCell, BlockChart

from .common import _ID_PATTERN, _NUMBER_PATTERN, MAX_NODES, _Builder, _clean_label, _source_lines
from .graphs import _parse_flow_operator, _parse_node_ref


def _body(source: str, builder: _Builder, header: str) -> list[tuple[int, str]]:
    lines = [(line, text) for line, text in _source_lines(source) if not text.startswith("%%")]
    if not lines or lines[0][1] != header:
        builder.error(lines[0][0] if lines else 1, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
    return lines[1:]


def parse_block(source: str) -> DiagramIR:
    """Parse flat blocks, spans, spaces and ordinary labelled connections."""
    builder = _Builder(DiagramKind.BLOCK, Direction.TOP_DOWN)
    columns: int | None = None
    cells: list[BlockCell] = []
    seen: set[str] = set()
    pending: list[DiagramEdge] = []
    for number, text in _body(source, builder, "block-beta"):
        if match := re.fullmatch(r"columns\s+(auto|\d{1,3})", text):
            if columns is not None or cells:
                builder.error(number, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            columns = 0 if match[1] == "auto" else int(match[1])
            if match[1] != "auto" and not 1 <= columns <= MAX_NODES:
                builder.error(number, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        position = 0
        row_cells: list[BlockCell] = []
        while position < len(text):
            parsed = _parse_node_ref(text, position)
            if parsed is None:
                builder.error(number, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
                break
            node_id, label, shape, explicit, end = parsed
            # An edge line references already placed blocks; it never adds slots.
            operator = _parse_flow_operator(text, end)
            if operator is not None:
                target_start = operator.position
                while target_start < len(text) and text[target_start].isspace():
                    target_start += 1
                target = _parse_node_ref(text, target_start)
                if position or target is None or text[target[-1] :].strip():
                    builder.error(number, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
                    break
                target_id = target[0]
                builder.node(node_id, label, number, shape=shape, explicit=explicit)
                builder.node(target_id, target[1], number, shape=target[2], explicit=target[3])
                pending.append(
                    DiagramEdge(
                        target_id if operator.reverse else node_id,
                        node_id if operator.reverse else target_id,
                        operator.label,
                        operator.style,
                        operator.directed,
                        operator.source_marker,
                        operator.target_marker,
                        line=number,
                    )
                )
                break
            span = 1
            if match := re.match(r":(\d{1,3})(?=\s|$)", text[end:]):
                span = int(match[1])
                end += match.end()
            if span < 1 or span > MAX_NODES or (text[end:] and not text[end].isspace()):
                builder.error(number, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
                break
            if node_id == "space" and not explicit:
                row_cells.append(BlockCell(None, span))
            elif node_id in seen:
                builder.error(number, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            else:
                builder.node(node_id, label, number, shape=shape, explicit=explicit)
                seen.add(node_id)
                row_cells.append(BlockCell(node_id, span))
            position = end
            while position < len(text) and text[position].isspace():
                position += 1
        cells.extend(row_cells)
        if len(cells) > MAX_NODES:
            builder.error(number, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
            break
    columns = columns or sum(cell.span for cell in cells) or 1
    if columns > MAX_NODES or any(cell.span > columns for cell in cells):
        builder.error(1, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
    for edge in pending:
        if edge.source not in seen or edge.target not in seen:
            builder.error(edge.line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
        else:
            builder.edge(edge)
    if not builder.nodes:
        builder.error(1, DiagnosticCode.NO_NODES)
    return builder.finish(BlockChart(columns, tuple(cells)))


_ARCH_NODE = re.compile(
    rf"(group|service)\s+({_ID_PATTERN})(?:\(([^()]+)\))?(?:\[([^\[\]]+)\])?(?:\s+in\s+({_ID_PATTERN}))?"
)
_ARCH_EDGE = re.compile(rf"({_ID_PATTERN}):([TBLR])\s*(<)?--(>)?\s*([TBLR]):({_ID_PATTERN})")


def parse_architecture(source: str) -> DiagramIR:
    """Keep group ancestry and port names, with an explicitly schematic layout."""
    builder = _Builder(DiagramKind.ARCHITECTURE, Direction.LEFT_RIGHT)
    builder.simplified = True
    groups: dict[str, tuple[str, str, str | None]] = {}
    services: dict[str, tuple[str | None, int]] = {}
    pending: list[DiagramEdge] = []
    for number, text in _body(source, builder, "architecture-beta"):
        if match := _ARCH_NODE.fullmatch(text):
            kind, node_id, icon, label, parent = match.groups()
            if node_id in groups or node_id in services:
                builder.error(number, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
                continue
            if len(groups) + len(services) >= MAX_NODES:
                builder.error(number, DiagnosticCode.NODE_LIMIT, limit=MAX_NODES)
                break
            if kind == "group":
                groups[node_id] = (_clean_label(label or node_id), icon or "", parent)
            else:
                builder.node(node_id, _clean_label(label or node_id), number, explicit=True)
                if icon:
                    builder.annotate(node_id, icon, number)
                services[node_id] = (parent, number)
        elif match := _ARCH_EDGE.fullmatch(text):
            source_id, source_port, source_arrow, target_arrow, target_port, target_id = match.groups()
            pending.append(
                DiagramEdge(
                    source_id,
                    target_id,
                    directed=bool(target_arrow),
                    source_marker="◀" if source_arrow else "",
                    source_label=source_port,
                    target_label=target_port,
                    line=number,
                )
            )
        else:
            builder.error(number, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)

    def ancestry(group_id: str | None, number: int) -> tuple[str, ...]:
        path: list[str] = []
        visited: set[str] = set()
        while group_id is not None:
            if group_id not in groups or group_id in visited:
                builder.error(number, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
                break
            visited.add(group_id)
            label, icon, parent = groups[group_id]
            path.append(f"{group_id}[{label}]" + (f" ({icon})" if icon else ""))
            group_id = parent
        return tuple(reversed(path))

    for node_id, (group_id, number) in services.items():
        if path := ancestry(group_id, number):
            builder.node(node_id, None, number, sections=(("in " + " / ".join(path),),))
    # Empty groups remain visible too; a group-only diagram is still meaningful.
    used_groups = {parent for _, _, parent in groups.values()} | {parent for parent, _ in services.values()}
    for group_id in groups:
        path = ancestry(group_id, 1)
        if group_id not in used_groups:
            builder.node(f"@group:{group_id}", " / ".join(path), 1)
            builder.annotate(f"@group:{group_id}", "group", 1)
    for edge in pending:
        if edge.source not in services or edge.target not in services:
            builder.error(edge.line, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
        else:
            builder.edge(edge)
    if not builder.nodes:
        builder.error(1, DiagnosticCode.NO_NODES)
    return builder.finish()


def parse_sankey(source: str) -> DiagramIR:
    """Keep every CSV flow/value without claiming proportional link widths."""
    builder = _Builder(DiagramKind.SANKEY, Direction.LEFT_RIGHT)
    builder.simplified = True
    ids: dict[str, str] = {}
    for number, text in _body(source, builder, "sankey-beta"):
        try:
            fields = next(csv.reader([text], strict=True, skipinitialspace=True))
        except csv.Error:
            builder.error(number, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        if len(fields) != 3 or not re.fullmatch(_NUMBER_PATTERN, fields[-1].strip()):
            builder.error(number, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        # CSV has already decoded quoting. Use complete names as identities,
        # independently of display normalization.
        source_name, target_name = (name.strip() for name in fields[:2])
        value = Decimal(fields[2].strip())
        if not source_name or not target_name or value <= 0:
            builder.error(number, DiagnosticCode.UNSUPPORTED_CHART_STATEMENT)
            continue
        for name in (source_name, target_name):
            if name not in ids:
                ids[name] = f"@sankey:{len(ids)}"
            builder.node(ids[name], _clean_label(name, quote_chars=""), number)
        builder.edge(DiagramEdge(ids[source_name], ids[target_name], fields[2].strip(), line=number))
    if not builder.edges:
        builder.error(1, DiagnosticCode.NO_NODES)
    return builder.finish()
