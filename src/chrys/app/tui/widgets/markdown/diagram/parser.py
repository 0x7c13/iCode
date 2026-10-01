# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded Mermaid header dispatch; syntax lives in parsers/."""

from __future__ import annotations

import re
from collections.abc import Callable

from .model import (
    Diagnostic,
    DiagnosticCode,
    DiagramIR,
    DiagramKind,
    Direction,
)
from .parsers.charts import (
    _parse_pie,
    _parse_quadrant,
    _parse_treemap,
    _parse_xychart,
)
from .parsers.common import (
    MAX_DIAGNOSTICS,
    MAX_EDGES,
    MAX_NODES,
    MAX_SOURCE_BYTES,
    _source_lines,
)
from .parsers.graphs import (
    _FLOW_HEADER_RE,
    _parse_class,
    _parse_er,
    _parse_flow,
    _parse_sequence,
    _parse_state,
    _split_statements,
)
from .parsers.planning import parse_journey, parse_kanban, parse_mindmap, parse_timeline
from .parsers.schedule import parse_gantt, parse_git, parse_packet
from .parsers.schemas import parse_c4, parse_requirement
from .parsers.structure import parse_architecture, parse_block, parse_sankey

_FAMILY_PARSERS: dict[str, Callable[[str], DiagramIR]] = {
    "journey": parse_journey,
    "timeline": parse_timeline,
    "kanban": parse_kanban,
    "mindmap": parse_mindmap,
    "gantt": parse_gantt,
    "gitgraph": parse_git,
    "packet": parse_packet,
    "packet-beta": parse_packet,
    "block-beta": parse_block,
    "architecture-beta": parse_architecture,
    "sankey-beta": parse_sankey,
    "requirementdiagram": parse_requirement,
    "c4context": parse_c4,
    "c4container": parse_c4,
    "c4component": parse_c4,
}


def parse_mermaid(source: str) -> DiagramIR:
    """Parse supported Mermaid source without raising for expected user input."""
    if len(source.encode("utf-8", errors="replace")) > MAX_SOURCE_BYTES:
        return DiagramIR(
            DiagramKind.UNKNOWN,
            Direction.TOP_DOWN,
            (),
            (),
            (Diagnostic(1, DiagnosticCode.SOURCE_LIMIT, (("limit", MAX_SOURCE_BYTES),)),),
            True,
        )
    lines = _source_lines(source, join_labels=True)
    if not lines:
        return DiagramIR(
            DiagramKind.UNKNOWN,
            Direction.TOP_DOWN,
            (),
            (),
            (Diagnostic(1, DiagnosticCode.EMPTY_SOURCE),),
            True,
        )
    header_index = next((index for index, (_, line) in enumerate(lines) if not line.startswith("%%")), 0)
    line_number, header_text = lines[header_index]
    # A Mermaid statement separator is also valid immediately after its header.
    # Split only graph families here; chart labels have their own grammars.
    statements = _split_statements(header_text, flow=True, sequence=header_text.startswith("sequenceDiagram"))
    if (
        statements
        and statements[0] != header_text
        and (
            re.fullmatch(r"(?:flowchart|graph)\s+(?:TB|TD|BT|LR|RL)", statements[0], re.IGNORECASE)
            or statements[0] in {"classDiagram", "sequenceDiagram", "stateDiagram", "stateDiagram-v2", "erDiagram"}
        )
    ):
        header_text = statements[0]
        lines[header_index : header_index + 1] = [(line_number, item) for item in statements]
    if parser := _FAMILY_PARSERS.get(header_text.split()[0].rstrip(":").casefold()):
        # These bounded adapters do not interpret Mermaid configuration. Do not
        # silently ignore settings that change scheduling, bit widths or branches.
        if source.lstrip().startswith("---") or any(line.strip().startswith("%%{") for line in source.splitlines()):
            return DiagramIR(
                DiagramKind.UNKNOWN,
                Direction.TOP_DOWN,
                (),
                (),
                (Diagnostic(line_number, DiagnosticCode.UNSUPPORTED_DIRECTIVE),),
                True,
            )
        return parser(source)
    if header := _FLOW_HEADER_RE.fullmatch(header_text):
        direction = (header.group(2) or "").upper()
        if direction not in {"", "TB", "TD", "BT", "LR", "RL"}:
            return DiagramIR(
                DiagramKind.FLOWCHART,
                Direction.TOP_DOWN,
                (),
                (),
                (
                    Diagnostic(
                        line_number,
                        DiagnosticCode.UNSUPPORTED_FLOW_DIRECTION,
                        (("direction", direction),),
                    ),
                ),
                True,
            )
        return _parse_flow(lines, header_index, header)
    if header_text == "classDiagram":
        return _parse_class(lines, header_index)
    if header_text.casefold() == "erdiagram":
        return _parse_er(lines, header_index)
    if header_text in {"stateDiagram", "stateDiagram-v2"}:
        return _parse_state(lines, header_index)
    if header_text == "sequenceDiagram":
        return _parse_sequence(lines, header_index)
    if pie_header := re.fullmatch(
        r"pie(?:\s+(showData))?(?:\s+title\s+(.+))?",
        header_text,
        re.IGNORECASE,
    ):
        return _parse_pie(
            lines,
            header_index,
            pie_header.group(1) is not None,
            pie_header.group(2) or "",
        )
    if xy_header := re.fullmatch(
        r"xychart(?:-beta)?(?:\s+(horizontal|vertical))?",
        header_text,
        re.IGNORECASE,
    ):
        return _parse_xychart(
            lines,
            header_index,
            horizontal=(xy_header.group(1) or "").casefold() == "horizontal",
        )
    if header_text.casefold() == "quadrantchart":
        return _parse_quadrant(lines, header_index)
    if header_text.casefold() == "treemap-beta":
        return _parse_treemap(source)
    return DiagramIR(
        DiagramKind.UNKNOWN,
        Direction.TOP_DOWN,
        (),
        (),
        (Diagnostic(line_number, DiagnosticCode.UNSUPPORTED_DIAGRAM_TYPE),),
        True,
    )


__all__ = ["MAX_DIAGNOSTICS", "MAX_EDGES", "MAX_NODES", "MAX_SOURCE_BYTES", "parse_mermaid"]
