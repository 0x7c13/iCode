# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Parsers for node/edge Mermaid diagrams and sequences."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

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
    _ID_PATTERN,
    _ID_RE,
    _Builder,
    _clean_label,
    _decimal_text,
    _diagram_direction,
)

_FLOW_HEADER_RE = re.compile(r"^(flowchart|graph)(?:\s+([A-Za-z]+))?\s*$", re.IGNORECASE)


_CLASS_RELATION_RE = re.compile(
    rf'^({_ID_PATTERN}?)(?:\s+"([^"]*)")?\s*'
    rf"(<\|--|--\|>|<\|\.\.|\.\.\|>|\*--|--\*|o--|--o|\.\.>|<\.\.|-->|<--|--)\s*"
    rf'(?:"([^"]*)"\s*)?({_ID_PATTERN})(?:\s*:\s*(.*))?$'
)


_QUOTED_ER_NAME_PATTERN = r'"(?:\\.|[^"\\])*"'


_ER_CLASS_NAMES_PATTERN = rf"{_ID_PATTERN}(?:,{_ID_PATTERN})*"


_ER_ENTITY_TOKEN_PATTERN = (
    rf"(?:{_ID_PATTERN}|{_QUOTED_ER_NAME_PATTERN})"
    rf"(?:\[(?:{_QUOTED_ER_NAME_PATTERN}|[^\[\]]+)\])?"
    rf"(?:\:\:\:{_ER_CLASS_NAMES_PATTERN})?"
)


_ER_ENTITY_REF_RE = re.compile(
    rf"^(?P<name>{_ID_PATTERN}|{_QUOTED_ER_NAME_PATTERN})"
    rf"(?:\[(?P<alias>{_QUOTED_ER_NAME_PATTERN}|[^\[\]]+)\])?"
    rf"(?:\:\:\:{_ER_CLASS_NAMES_PATTERN})?$"
)


_ER_CARDINALITY_PATTERN = r"(?:\|\||\|o|o\||}o|o\{|}\||\|\{)"


_ER_SYMBOLIC_RELATION_RE = re.compile(
    rf"^({_ER_ENTITY_TOKEN_PATTERN})\s*({_ER_CARDINALITY_PATTERN})\s*(--|\.\.)\s*"
    rf"({_ER_CARDINALITY_PATTERN})\s*({_ER_ENTITY_TOKEN_PATTERN})\s*:\s*(.*)$"
)


_ER_CARDINALITY_ALIAS_PATTERN = (
    r"(?:one\s+or\s+zero|zero\s+or\s+one|one\s+or\s+more|one\s+or\s+many|"
    r"zero\s+or\s+more|zero\s+or\s+many|many\(1\)|many\(0\)|only\s+one|1\+|0\+|1)"
)


_ER_WORD_RELATION_RE = re.compile(
    rf"^({_ER_ENTITY_TOKEN_PATTERN})\s+({_ER_CARDINALITY_ALIAS_PATTERN})\s+"
    rf"(optionally\s+to|to)\s+({_ER_CARDINALITY_ALIAS_PATTERN})\s+"
    rf"({_ER_ENTITY_TOKEN_PATTERN})\s*:\s*(.*)$",
    re.IGNORECASE,
)


_STATE_RELATION_RE = re.compile(rf"^(\[\*\]|{_ID_PATTERN}?)\s*-->\s*(\[\*\]|{_ID_PATTERN})(?:\s*:\s*(.*))?$")


_SEQUENCE_MESSAGE_RE = re.compile(
    rf"^({_ID_PATTERN}?)\s*(-->>|->>|-->|->|--\)|-\)|--x|-x)\s*[+-]?\s*({_ID_PATTERN})(?:\s*:\s*(.*))?$"
)


_FLOW_CLASS_SUFFIX_RE = re.compile(r":::[^\W\d][\w-]*")


_FLOW_METADATA_SHAPES = {
    "circle": NodeShape.CIRCLE,
    "cyl": NodeShape.CYLINDER,
    "cylinder": NodeShape.CYLINDER,
    "database": NodeShape.CYLINDER,
    "dbl-circ": NodeShape.CIRCLE,
    "decision": NodeShape.DECISION,
    "diam": NodeShape.DECISION,
    "diamond": NodeShape.DECISION,
    "double-circle": NodeShape.CIRCLE,
    "event": NodeShape.ROUNDED,
    "fork": NodeShape.FORK_JOIN,
    "fr-rect": NodeShape.SUBROUTINE,
    "hex": NodeShape.HEXAGON,
    "hexagon": NodeShape.HEXAGON,
    "in-out": NodeShape.SLANTED,
    "join": NodeShape.FORK_JOIN,
    "lean-l": NodeShape.SLANTED,
    "lean-r": NodeShape.SLANTED,
    "parallelogram": NodeShape.SLANTED,
    "pill": NodeShape.STADIUM,
    "prepare": NodeShape.HEXAGON,
    "proc": NodeShape.RECTANGLE,
    "process": NodeShape.RECTANGLE,
    "question": NodeShape.DECISION,
    "rect": NodeShape.RECTANGLE,
    "rectangle": NodeShape.RECTANGLE,
    "rounded": NodeShape.ROUNDED,
    "stadium": NodeShape.STADIUM,
    "subroutine": NodeShape.SUBROUTINE,
    "terminal": NodeShape.STADIUM,
    "trap-b": NodeShape.SLANTED,
    "trap-t": NodeShape.SLANTED,
    "trapezoid": NodeShape.SLANTED,
}


def _state_pseudo_id(kind: str, counter: int) -> str:
    """Return an internal ID that cannot match the Mermaid identifier grammar."""
    return f"@state-{kind}:{counter}"


@dataclass(frozen=True, slots=True)
class _FlowOperator:
    style: EdgeStyle
    directed: bool
    label: str
    source_marker: str
    target_marker: str
    reverse: bool
    position: int


_FLOW_ID_OPERATOR_RE = re.compile(r"-\.+->|-\.+-|-{2,}>|-{3,}|={2,}>|={3,}|--(?=\s)|-\.(?=\s)")


def _parse_node_ref(text: str, start: int = 0) -> tuple[str, str | None, NodeShape, bool, int] | None:
    match = _ID_RE.match(text, start)
    if match is None:
        return None
    position = match.end()
    for operator in (
        "<-->",
        "o--o",
        "x--x",
        "o--x",
        "x--o",
        "o-->",
        "-.->",
        "<--",
        "-->",
        "---",
        "--o",
        "--x",
        "==>",
        "-.",
        "--",
    ):
        operator_at = text.find(operator, start + 1, match.end() + len(operator))
        if operator_at >= 0:
            position = min(position, operator_at)
    # Only an operator beginning in the candidate ID can shorten it. One
    # lookahead character includes a trailing arrowhead or whitespace delimiter.
    if operator := _FLOW_ID_OPERATOR_RE.search(text, start + 1, match.end() + 1):
        position = min(position, operator.start())
    node_id = text[start:position]
    if _ID_RE.fullmatch(node_id) is None:
        return None
    shape = NodeShape.RECTANGLE
    label: str | None = None
    explicit = False
    delimiters = (
        ("(((", ")))", NodeShape.CIRCLE),
        ("((", "))", NodeShape.CIRCLE),
        ("[(", ")]", NodeShape.CYLINDER),
        ("([", "])", NodeShape.STADIUM),
        ("[[", "]]", NodeShape.SUBROUTINE),
        ("{{", "}}", NodeShape.HEXAGON),
        ("[/", "/]", NodeShape.SLANTED),
        ("[\\", "\\]", NodeShape.SLANTED),
        ("[/", "\\]", NodeShape.SLANTED),
        ("[\\", "/]", NodeShape.SLANTED),
        (">", "]", NodeShape.SLANTED),
        ("[", "]", NodeShape.RECTANGLE),
        ("(", ")", NodeShape.ROUNDED),
        ("{", "}", NodeShape.DECISION),
    )
    if text.startswith("@{", position) and (end := text.find("}", position + 2)) >= 0:
        metadata = text[position + 2 : end]
        if shape_match := re.search(r'["\']?shape["\']?\s*:\s*["\']?([A-Za-z0-9_-]+)', metadata):
            shape = _FLOW_METADATA_SHAPES.get(shape_match.group(1).casefold(), NodeShape.RECTANGLE)
        if label_match := re.search(r'["\']?label["\']?\s*:\s*["\']([^"\']*)["\']', metadata):
            label = _clean_label(label_match.group(1))
        explicit = True
        position = end + 1
    else:
        candidates: list[tuple[int, int, str, str, NodeShape]] = []
        for opening, closing, candidate_shape in delimiters:
            if text.startswith(opening, position):
                end = text.find(closing, position + len(opening))
                if end >= 0:
                    candidates.append((end, -len(opening), opening, closing, candidate_shape))
        if candidates:
            end, _, opening, closing, shape = min(candidates)
            label = _clean_label(text[position + len(opening) : end])
            explicit = True
            position = end + len(closing)
    while class_suffix := _FLOW_CLASS_SUFFIX_RE.match(text, position):
        position = class_suffix.end()
    return node_id, label, shape, explicit, position


def _parse_flow_operator(text: str, start: int) -> _FlowOperator | None:
    position = start
    while position < len(text) and text[position].isspace():
        position += 1
    labelled_operators = (
        (r"--\s+(.+?)\s+-{2,}>", EdgeStyle.SOLID),
        (r"-\.\s+(.+?)\s+\.+->", EdgeStyle.DOTTED),
        (r"==\s+(.+?)\s+={2,}>", EdgeStyle.HEAVY),
    )
    for pattern, style in labelled_operators:
        labelled = re.match(pattern, text[position:])
        if labelled is not None:
            return _FlowOperator(
                style,
                True,
                _clean_label(labelled.group(1)),
                "",
                "",
                False,
                position + labelled.end(),
            )
    operators = (
        ("<-->", EdgeStyle.SOLID, True, "◀", "▶", False),
        ("o--o", EdgeStyle.SOLID, False, "○", "○", False),
        ("x--x", EdgeStyle.SOLID, False, "x", "x", False),
        ("o--x", EdgeStyle.SOLID, False, "○", "x", False),
        ("x--o", EdgeStyle.SOLID, False, "x", "○", False),
        ("o-->", EdgeStyle.SOLID, True, "○", "", False),
        ("<--", EdgeStyle.SOLID, True, "", "", True),
        ("--o", EdgeStyle.SOLID, False, "", "○", False),
        ("--x", EdgeStyle.SOLID, False, "", "x", False),
    )
    for operator, style, directed, source_marker, target_marker, reverse in operators:
        if not text.startswith(operator, position):
            continue
        position += len(operator)
        while position < len(text) and text[position].isspace():
            position += 1
        label = ""
        if position < len(text) and text[position] == "|":
            end = text.find("|", position + 1)
            if end < 0:
                return None
            label = _clean_label(text[position + 1 : end])
            position = end + 1
        return _FlowOperator(style, directed, label, source_marker, target_marker, reverse, position)
    variable_operators = (
        (r"-\.+->", EdgeStyle.DOTTED, True),
        (r"={2,}>", EdgeStyle.HEAVY, True),
        (r"-{2,}>", EdgeStyle.SOLID, True),
        (r"-\.+-", EdgeStyle.DOTTED, False),
        (r"={3,}", EdgeStyle.HEAVY, False),
        (r"-{3,}", EdgeStyle.SOLID, False),
    )
    for pattern, style, directed in variable_operators:
        operator = re.match(pattern, text[position:])
        if operator is None:
            continue
        position += operator.end()
        while position < len(text) and text[position].isspace():
            position += 1
        label = ""
        if position < len(text) and text[position] == "|":
            end = text.find("|", position + 1)
            if end < 0:
                return None
            label = _clean_label(text[position + 1 : end])
            position = end + 1
        return _FlowOperator(style, directed, label, "", "", False, position)
    return None


def _split_statements(line: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    quote = ""
    bracket_depth = 0
    for char in line:
        if char in {'"', "'"}:
            if not quote:
                quote = char
            elif quote == char:
                quote = ""
        if not quote and char in "[({":
            bracket_depth += 1
        elif not quote and char in "])}" and bracket_depth:
            bracket_depth -= 1
        if char == ";" and not quote and not bracket_depth:
            entity_suffix = re.search(r"(?:&(?:#\d+|[A-Za-z][A-Za-z0-9]+)|#\d+)$", "".join(current[-64:]))
            if entity_suffix is None:
                if statement := "".join(current).strip():
                    statements.append(statement)
                current = []
                continue
        current.append(char)
    if statement := "".join(current).strip():
        statements.append(statement)
    return statements


def _parse_flow_statement(builder: _Builder, statement: str, line: int) -> None:
    keyword = statement.split(maxsplit=1)[0].casefold()
    if keyword in {"class", "classdef", "click", "linkstyle", "style"}:
        builder.warning(line, DiagnosticCode.UNSUPPORTED_FLOW_STATEMENT)
        return
    if keyword in {"end", "subgraph"}:
        builder.warning(line, DiagnosticCode.UNSUPPORTED_FLOW_STATEMENT)
        return
    if direction := re.fullmatch(r"direction\s+(TB|TD|BT|LR|RL)", statement, re.IGNORECASE):
        builder.direction = _diagram_direction(direction.group(1))
        return
    first = _parse_node_ref(statement)
    if first is None:
        builder.error(line, DiagnosticCode.EXPECTED_NODE_OR_EDGE)
        return
    node_id, label, shape, explicit, position = first
    if builder.node(node_id, label, line, shape=shape, explicit=explicit) is None:
        return
    found_edge = False
    while not builder.has_fatal_error and position < len(statement):
        while position < len(statement) and statement[position].isspace():
            position += 1
        if position == len(statement):
            break
        operator = _parse_flow_operator(statement, position)
        if operator is None:
            builder.error(line, DiagnosticCode.MALFORMED_FLOW_EDGE)
            return
        position = operator.position
        while position < len(statement) and statement[position].isspace():
            position += 1
        target = _parse_node_ref(statement, position)
        if target is None:
            builder.error(line, DiagnosticCode.MISSING_EDGE_TARGET)
            return
        target_id, target_label, target_shape, target_explicit, position = target
        if builder.node(target_id, target_label, line, shape=target_shape, explicit=target_explicit) is None:
            return
        edge_source, edge_target = (target_id, node_id) if operator.reverse else (node_id, target_id)
        builder.edge(
            DiagramEdge(
                edge_source,
                edge_target,
                operator.label,
                operator.style,
                operator.directed,
                source_marker=operator.source_marker,
                target_marker=operator.target_marker,
                line=line,
            )
        )
        node_id = target_id
        found_edge = True
    if not found_edge and not explicit:
        builder.node(node_id, None, line, explicit=True)


def _parse_flow(lines: list[tuple[int, str]], header_index: int, header: re.Match[str]) -> DiagramIR:
    direction = _diagram_direction(header.group(2) or "TB")
    builder = _Builder(DiagramKind.FLOWCHART, direction)
    subgraph_depth = 0
    for line, raw in lines[header_index + 1 :]:
        if raw.startswith("%%"):
            if raw.startswith("%%{"):
                builder.warning(line, DiagnosticCode.UNSUPPORTED_DIRECTIVE)
            continue
        for statement in _split_statements(raw):
            keyword = statement.split(maxsplit=1)[0].casefold()
            if keyword == "subgraph":
                subgraph_depth += 1
                builder.warning(line, DiagnosticCode.UNSUPPORTED_FLOW_STATEMENT)
                continue
            if keyword == "end":
                subgraph_depth = max(0, subgraph_depth - 1)
                builder.warning(line, DiagnosticCode.UNSUPPORTED_FLOW_STATEMENT)
                continue
            if subgraph_depth and keyword == "direction":
                builder.warning(line, DiagnosticCode.UNSUPPORTED_FLOW_STATEMENT)
                continue
            _parse_flow_statement(builder, statement, line)
            if builder.has_fatal_error:
                return builder.finish()
    return builder.finish()


def _relation_edge(match: re.Match[str], line: int) -> DiagramEdge:
    left, left_mult, operator, right_mult, right, label = match.groups()
    source = left
    target = right
    source_marker = ""
    target_marker = ""
    directed = operator not in {"--", "o--", "--o", "*--", "--*"}
    style = EdgeStyle.DOTTED if "." in operator else EdgeStyle.SOLID
    if operator in {"<|--", "<|..", "<..", "<--"}:
        source, target = right, left
        left_mult, right_mult = right_mult, left_mult
        target_marker = "△" if operator in {"<|--", "<|.."} else "▶"
    elif operator in {"--|>", "..|>"}:
        target_marker = "△"
    elif operator in {"..>", "-->", "--|>"}:
        target_marker = "▶"
    elif operator == "*--":
        source_marker = "◆"
    elif operator == "--*":
        target_marker = "◆"
    elif operator == "o--":
        source_marker = "◇"
    elif operator == "--o":
        target_marker = "◇"
    return DiagramEdge(
        source,
        target,
        _clean_label(label or ""),
        style,
        directed,
        source_marker,
        target_marker,
        left_mult or "",
        right_mult or "",
        line,
    )


def _class_declaration(raw: str) -> tuple[str, str | None, str | None, bool] | None:
    declaration = re.fullmatch(rf"class\s+({_ID_PATTERN})(?:~([^~]+)~)?(.*)", raw)
    if declaration is None:
        return None
    node_id, generic, remainder = declaration.groups()
    remainder = remainder.strip()
    opens = remainder.endswith("{")
    if opens:
        remainder = remainder[:-1].rstrip()
    annotation: str | None = None
    if annotation_match := re.search(r"\s*<<([^<>]+)>>\s*$", remainder):
        annotation = annotation_match.group(1).strip()
        remainder = remainder[: annotation_match.start()].rstrip()
    label: str | None = None
    if remainder:
        if bracket_label := re.fullmatch(r'\[\s*"([^"]*)"\s*\]', remainder):
            label = bracket_label.group(1)
        elif remainder.casefold().startswith("as ") and remainder[3:].strip():
            label = remainder[3:].strip()
        else:
            return None
    if generic:
        label = f"{label or node_id}<{_clean_label(generic)}>"
    return node_id, label, annotation, opens


def _parse_class(lines: list[tuple[int, str]], header_index: int) -> DiagramIR:
    builder = _Builder(DiagramKind.CLASS, Direction.TOP_DOWN)
    active_class: str | None = None
    namespace_depth = 0
    for line, raw in lines[header_index + 1 :]:
        if raw.startswith("%%"):
            continue
        if direction := re.fullmatch(r"direction\s+(TB|TD|BT|LR|RL)", raw, re.IGNORECASE):
            builder.direction = _diagram_direction(direction.group(1))
            continue
        keyword = raw.split(maxsplit=1)[0].casefold()
        if keyword in {"callback", "classdef", "click", "cssclass", "link", "style"}:
            builder.warning(line, DiagnosticCode.UNSUPPORTED_CLASS_STATEMENT)
            continue
        if active_class is None and re.fullmatch(r"namespace\s+.+\{", raw, re.IGNORECASE):
            namespace_depth += 1
            builder.warning(line, DiagnosticCode.UNSUPPORTED_CLASS_STATEMENT)
            continue
        if raw == "}":
            if active_class is not None:
                active_class = None
            elif namespace_depth:
                namespace_depth -= 1
                builder.warning(line, DiagnosticCode.UNSUPPORTED_CLASS_STATEMENT)
            else:
                builder.error(line, DiagnosticCode.UNEXPECTED_CLASS_TERMINATOR)
            continue
        if active_class is not None:
            if "{" in raw:
                builder.error(line, DiagnosticCode.NESTED_CLASS_BODY)
            elif annotation := re.fullmatch(r"<<([^<>]+)>>", raw):
                builder.annotate(active_class, annotation.group(1), line)
            else:
                builder.add_member(active_class, raw, line)
            continue
        if note := re.fullmatch(rf'note\s+for\s+({_ID_PATTERN})\s+"([^"]*)"', raw, re.IGNORECASE):
            builder.add_note(note.group(1), note.group(2), line)
            continue
        if re.fullmatch(r'note\s+"[^"]*"', raw, re.IGNORECASE):
            builder.warning(line, DiagnosticCode.UNSUPPORTED_CLASS_STATEMENT)
            continue
        if relation := _CLASS_RELATION_RE.fullmatch(raw):
            edge = _relation_edge(relation, line)
            builder.node(edge.source, None, line)
            builder.node(edge.target, None, line)
            builder.edge(edge)
            continue
        if declaration := _class_declaration(raw):
            node_id, label, annotation, opens = declaration
            builder.node(node_id, _clean_label(label) if label else None, line, explicit=True)
            if annotation:
                builder.annotate(node_id, annotation, line)
            if opens:
                active_class = node_id
            continue
        if annotation := re.fullmatch(rf"<<([^<>]+)>>\s+({_ID_PATTERN})", raw):
            builder.annotate(annotation.group(2), annotation.group(1), line)
            continue
        if member := re.fullmatch(rf"({_ID_PATTERN})\s*:\s*(.+)", raw):
            builder.add_member(member.group(1), member.group(2).strip(), line)
            continue
        builder.error(line, DiagnosticCode.UNSUPPORTED_CLASS_STATEMENT)
    if active_class is not None:
        builder.error(lines[-1][0], DiagnosticCode.UNCLOSED_CLASS_BODY, node_id=active_class)
    if namespace_depth:
        builder.error(lines[-1][0], DiagnosticCode.UNSUPPORTED_CLASS_STATEMENT)
    return builder.finish()


def _parse_er_entity_ref(raw: str) -> tuple[str, str | None] | None:
    match = _ER_ENTITY_REF_RE.fullmatch(raw.strip())
    if match is None:
        return None
    name = _clean_label(match.group("name"))
    alias = match.group("alias")
    if not name:
        return None
    return name, _clean_label(alias) if alias is not None else (name if raw.lstrip().startswith('"') else None)


def _er_cardinality(raw: str) -> str:
    normalized = re.sub(r"\s+", " ", raw.strip().casefold())
    if normalized in {"||", "only one", "1"}:
        return "1"
    if normalized in {"|o", "o|", "one or zero", "zero or one"}:
        return "0..1"
    if normalized in {"}|", "|{", "one or more", "one or many", "many(1)", "1+"}:
        return "1..*"
    return "0..*"


def _parse_er_relation(
    raw: str, line: int
) -> tuple[DiagramEdge, tuple[str, str | None], tuple[str, str | None]] | None:
    match = _ER_SYMBOLIC_RELATION_RE.fullmatch(raw)
    if match is not None:
        left_raw, left_cardinality, operator, right_cardinality, right_raw, label = match.groups()
        style = EdgeStyle.DOTTED if operator == ".." else EdgeStyle.SOLID
    else:
        match = _ER_WORD_RELATION_RE.fullmatch(raw)
        if match is None:
            return None
        left_raw, left_cardinality, operator, right_cardinality, right_raw, label = match.groups()
        style = EdgeStyle.DOTTED if operator.casefold().startswith("optionally") else EdgeStyle.SOLID
    left = _parse_er_entity_ref(left_raw)
    right = _parse_er_entity_ref(right_raw)
    if left is None or right is None:
        return None
    edge = DiagramEdge(
        left[0],
        right[0],
        _clean_label(label),
        style,
        False,
        source_label=_er_cardinality(left_cardinality),
        target_label=_er_cardinality(right_cardinality),
        line=line,
    )
    return edge, left, right


def _parse_er_attribute(raw: str) -> str | None:
    comment = ""
    if comment_match := re.search(r'\s+"([^"\r\n]*)"\s*$', raw):
        comment = _clean_label(comment_match.group(1))
        raw = raw[: comment_match.start()].rstrip()
    parts = raw.split(maxsplit=2)
    if len(parts) < 2:
        return None
    attribute_type, name = parts[:2]
    if re.fullmatch(r"[^\W\d][\w()\[\],-]*\??", attribute_type) is None:
        return None
    if re.fullmatch(r"\*?[^\W\d][\w()\[\]-]*", name) is None:
        return None
    keys: tuple[str, ...] = ()
    if len(parts) == 3:
        candidates = tuple(value.strip().upper() for value in parts[2].split(","))
        if not candidates or any(value not in {"PK", "FK", "UK"} for value in candidates):
            return None
        keys = candidates
    rendered = f"{attribute_type} {name}"
    if keys:
        rendered += f" [{','.join(keys)}]"
    if comment:
        rendered += f" — {comment}"
    return _clean_label(rendered)


def _parse_er(lines: list[tuple[int, str]], header_index: int) -> DiagramIR:
    builder = _Builder(DiagramKind.ER, Direction.TOP_DOWN)
    active_entity: str | None = None
    subgraph_depth = 0
    for line, raw in lines[header_index + 1 :]:
        if raw.startswith("%%"):
            if raw.startswith("%%{"):
                builder.warning(line, DiagnosticCode.UNSUPPORTED_DIRECTIVE)
            continue
        if active_entity is not None:
            if raw == "}":
                active_entity = None
                continue
            attribute = _parse_er_attribute(raw)
            if attribute is None:
                code = (
                    DiagnosticCode.NESTED_ER_BODY
                    if "{" in raw.partition('"')[0]
                    else DiagnosticCode.UNSUPPORTED_ER_STATEMENT
                )
                builder.error(line, code)
            else:
                builder.add_er_attribute(active_entity, attribute, line)
            continue
        if direction := re.fullmatch(r"direction\s+(TB|TD|BT|LR|RL)", raw, re.IGNORECASE):
            builder.direction = _diagram_direction(direction.group(1))
            continue
        keyword = raw.split(maxsplit=1)[0].casefold()
        if keyword in {"class", "classdef", "style"}:
            builder.warning(line, DiagnosticCode.UNSUPPORTED_ER_STATEMENT)
            continue
        if keyword == "subgraph":
            subgraph_depth += 1
            builder.warning(line, DiagnosticCode.UNSUPPORTED_ER_STATEMENT)
            continue
        if keyword == "end" and subgraph_depth:
            subgraph_depth -= 1
            builder.warning(line, DiagnosticCode.UNSUPPORTED_ER_STATEMENT)
            continue
        if raw == "}":
            builder.error(line, DiagnosticCode.UNEXPECTED_ER_TERMINATOR)
            continue
        if relation := _parse_er_relation(raw, line):
            edge, left, right = relation
            builder.node(left[0], left[1], line, shape=NodeShape.ENTITY, explicit=left[1] is not None)
            builder.node(right[0], right[1], line, shape=NodeShape.ENTITY, explicit=right[1] is not None)
            builder.edge(edge)
            continue
        if declaration := re.fullmatch(rf"({_ER_ENTITY_TOKEN_PATTERN})\s*\{{", raw):
            entity = _parse_er_entity_ref(declaration.group(1))
            if entity is None:
                builder.error(line, DiagnosticCode.UNSUPPORTED_ER_STATEMENT)
                continue
            builder.node(entity[0], entity[1], line, shape=NodeShape.ENTITY, explicit=True)
            active_entity = entity[0]
            continue
        entity = _parse_er_entity_ref(raw)
        if entity is not None:
            builder.node(entity[0], entity[1], line, shape=NodeShape.ENTITY, explicit=True)
            continue
        builder.error(line, DiagnosticCode.UNSUPPORTED_ER_STATEMENT)
    if active_entity is not None:
        builder.error(lines[-1][0], DiagnosticCode.UNCLOSED_ER_BODY, node_id=active_entity)
    if subgraph_depth:
        builder.error(lines[-1][0], DiagnosticCode.UNSUPPORTED_ER_STATEMENT)
    return builder.finish()


def _parse_state(lines: list[tuple[int, str]], header_index: int) -> DiagramIR:
    builder = _Builder(DiagramKind.STATE, Direction.TOP_DOWN)
    pseudo_counter = 0
    note_target: str | None = None
    note_lines: list[str] = []
    for line, raw in lines[header_index + 1 :]:
        if note_target is not None:
            if raw.casefold() == "end note":
                builder.add_note(note_target, " ".join(note_lines), line)
                note_target = None
                note_lines = []
            else:
                note_lines.append(raw)
            continue
        if raw.startswith("%%"):
            continue
        if direction := re.fullmatch(r"direction\s+(TB|TD|BT|LR|RL)", raw, re.IGNORECASE):
            builder.direction = _diagram_direction(direction.group(1))
            continue
        keyword = raw.split(maxsplit=1)[0].casefold()
        if keyword in {"class", "classdef", "style"}:
            builder.warning(line, DiagnosticCode.UNSUPPORTED_STATE_STATEMENT)
            continue
        if "{" in raw or raw == "}":
            builder.error(line, DiagnosticCode.NESTED_STATE)
            continue
        if note := re.fullmatch(rf"note\s+(?:left|right)\s+of\s+({_ID_PATTERN})\s*:\s*(.+)", raw, re.IGNORECASE):
            builder.add_note(note.group(1), note.group(2), line)
            continue
        if note := re.fullmatch(rf"note\s+(?:left|right)\s+of\s+({_ID_PATTERN})", raw, re.IGNORECASE):
            note_target = note.group(1)
            note_lines = []
            builder.node(note_target, None, line)
            continue
        if relation := _STATE_RELATION_RE.fullmatch(raw):
            source, target, label = relation.groups()
            if source == "[*]":
                pseudo_counter += 1
                source = _state_pseudo_id("start", pseudo_counter)
                builder.node(source, "●", line, shape=NodeShape.PSEUDO_START, explicit=True)
            else:
                builder.node(source, None, line)
            if target == "[*]":
                pseudo_counter += 1
                target = _state_pseudo_id("end", pseudo_counter)
                builder.node(target, "◎", line, shape=NodeShape.PSEUDO_END, explicit=True)
            else:
                builder.node(target, None, line)
            builder.edge(DiagramEdge(source, target, _clean_label(label or ""), line=line))
            continue
        if special := re.fullmatch(rf"state\s+({_ID_PATTERN})\s+<<(choice|fork|join)>>", raw, re.IGNORECASE):
            shape = NodeShape.DECISION if special.group(2).casefold() == "choice" else NodeShape.FORK_JOIN
            builder.node(special.group(1), None, line, shape=shape, explicit=True)
            continue
        if declaration := re.fullmatch(rf'state\s+"([^"]+)"\s+as\s+({_ID_PATTERN})', raw):
            label, node_id = declaration.groups()
            builder.node(node_id, label, line, explicit=True)
            continue
        if declaration := re.fullmatch(rf"state\s+({_ID_PATTERN})(?:\s*:\s*(.+))?", raw):
            node_id, label = declaration.groups()
            builder.node(node_id, _clean_label(label) if label else None, line, explicit=True)
            continue
        if description := re.fullmatch(rf"({_ID_PATTERN})\s*:\s*(.+)", raw):
            builder.node(description.group(1), _clean_label(description.group(2)), line, explicit=True)
            continue
        builder.error(line, DiagnosticCode.UNSUPPORTED_STATE_STATEMENT)
    if note_target is not None:
        builder.error(lines[-1][0], DiagnosticCode.UNSUPPORTED_STATE_STATEMENT)
    return builder.finish()


def _sequence_config_value(config: str, key: str) -> str | None:
    match = re.search(rf'["\']?{key}["\']?\s*:\s*["\']([^"\']*)["\']', config, re.IGNORECASE)
    return match.group(1) if match is not None else None


def _sequence_participant(raw: str) -> tuple[str, str, str | None, str | None] | None:
    participant = re.fullmatch(
        rf"(participant|actor)\s+({_ID_PATTERN})(?:@\{{(.*?)\}})?(?:\s+as\s+(.+))?",
        raw,
        re.IGNORECASE,
    )
    if participant is None:
        return None
    declaration, node_id, config, external_alias = participant.groups()
    participant_type = _sequence_config_value(config, "type") if config else None
    inline_alias = _sequence_config_value(config, "alias") if config else None
    return declaration.casefold(), node_id, external_alias or inline_alias, participant_type


def _add_sequence_participant(
    builder: _Builder,
    declaration: str,
    node_id: str,
    label: str | None,
    participant_type: str | None,
    line: int,
) -> None:
    normalized_type = participant_type.casefold() if participant_type else ""
    if normalized_type in {"database", "queue"}:
        shape = NodeShape.CYLINDER
    elif normalized_type == "collections":
        shape = NodeShape.SUBROUTINE
    else:
        shape = NodeShape.ACTOR if declaration == "actor" else NodeShape.RECTANGLE
    builder.node(node_id, _clean_label(label) if label else None, line, shape=shape, explicit=True)
    if normalized_type:
        builder.annotate(node_id, normalized_type, line)
        if normalized_type not in {"boundary", "collections", "control", "database", "entity", "queue"}:
            builder.warning(line, DiagnosticCode.UNSUPPORTED_SEQUENCE_STATEMENT)


def _parse_sequence(lines: list[tuple[int, str]], header_index: int) -> DiagramIR:
    builder = _Builder(DiagramKind.SEQUENCE, Direction.LEFT_RIGHT)
    autonumber = False
    message_number = Decimal(1)
    message_increment = Decimal(1)
    for line, raw in lines[header_index + 1 :]:
        if raw.startswith("%%"):
            continue
        if numbering := re.fullmatch(
            r"autonumber(?:\s+(\d{1,9}(?:\.\d{1,2})?)(?:\s+(\d{1,9}(?:\.\d{1,2})?))?)?",
            raw,
            re.IGNORECASE,
        ):
            autonumber = True
            if numbering.group(1):
                message_number = Decimal(numbering.group(1))
                message_increment = Decimal(numbering.group(2) or 1)
            continue
        keyword = raw.split(maxsplit=1)[0].casefold()
        if keyword in {"alt", "and", "box", "break", "critical", "else", "end", "loop", "opt", "option", "par", "rect"}:
            builder.warning(line, DiagnosticCode.UNSUPPORTED_SEQUENCE_FRAGMENT)
            continue
        if note := re.fullmatch(rf"note\s+(?:left|right)\s+of\s+({_ID_PATTERN})\s*:\s*(.+)", raw, re.IGNORECASE):
            builder.add_note(note.group(1), note.group(2), line)
            builder.warning(line, DiagnosticCode.UNSUPPORTED_SEQUENCE_STATEMENT)
            continue
        if note := re.fullmatch(
            rf"note\s+over\s+({_ID_PATTERN})(?:\s*,\s*({_ID_PATTERN}))?\s*:\s*(.+)", raw, re.IGNORECASE
        ):
            first, second, text = note.groups()
            builder.add_note(first, text, line)
            if second and second != first:
                builder.add_note(second, text, line)
            builder.warning(line, DiagnosticCode.UNSUPPORTED_SEQUENCE_STATEMENT)
            continue
        if keyword in {"link", "links"}:
            builder.warning(line, DiagnosticCode.UNSUPPORTED_SEQUENCE_STATEMENT)
            continue
        if activation := re.fullmatch(rf"(?:activate|deactivate)\s+({_ID_PATTERN})", raw, re.IGNORECASE):
            builder.node(activation.group(1), None, line)
            builder.warning(line, DiagnosticCode.UNSUPPORTED_SEQUENCE_STATEMENT)
            continue
        if raw.casefold().startswith("create "):
            participant = _sequence_participant(raw[7:].strip())
            if participant is None:
                builder.error(line, DiagnosticCode.UNSUPPORTED_SEQUENCE_STATEMENT)
            else:
                _add_sequence_participant(builder, *participant, line)
                builder.warning(line, DiagnosticCode.UNSUPPORTED_SEQUENCE_STATEMENT)
            continue
        if destroyed := re.fullmatch(rf"destroy\s+({_ID_PATTERN})", raw, re.IGNORECASE):
            builder.node(destroyed.group(1), None, line)
            builder.warning(line, DiagnosticCode.UNSUPPORTED_SEQUENCE_STATEMENT)
            continue
        if participant := _sequence_participant(raw):
            _add_sequence_participant(builder, *participant, line)
            continue
        if message := _SEQUENCE_MESSAGE_RE.fullmatch(raw):
            source, operator, target, label = message.groups()
            builder.node(source, None, line)
            builder.node(target, None, line)
            marker = "x" if operator.endswith("x") else ""
            clean_label = _clean_label(label or "")
            if autonumber:
                number = _decimal_text(message_number)
                clean_label = f"{number}. {clean_label}" if clean_label else f"{number}."
                message_number += message_increment
            builder.edge(
                DiagramEdge(
                    source,
                    target,
                    clean_label,
                    EdgeStyle.DOTTED if operator.startswith("--") else EdgeStyle.SOLID,
                    True,
                    target_marker=marker,
                    line=line,
                )
            )
            continue
        builder.error(line, DiagnosticCode.UNSUPPORTED_SEQUENCE_STATEMENT)
    return builder.finish()
