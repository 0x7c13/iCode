# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cell-measured formula boxes with explicit baselines and bounded composition."""

from __future__ import annotations

from dataclasses import dataclass, replace

from rich.cells import cell_len

from .parser import MathError, Node
from .symbols import FONT_ALPHABETS, SUBSCRIPT, SUPERSCRIPT

MAX_WIDTH = 1024
MAX_HEIGHT = 128
MAX_AREA = 32768
RELATIONS = frozenset("=:<>≤≥≠≈≡∼≃≅∝≪≫∈∉∋⊂⊃⊆⊇→←↔⇒⇐⇔↦⟶⟹⟸⟺⊥∥")  # noqa: RUF001 - mathematical operators
BINARY = frozenset("+-±∓×÷·∗∘∪∩∖∧∨⊕⊗⊙†")  # noqa: RUF001 - mathematical operators
DELIMITER_PAIRS = {"(": ")", "[": "]", "{": "}", "|": "|", "‖": "‖", "⟨": "⟩", "⌊": "⌋", "⌈": "⌉"}
DELIMITER_GLYPHS = frozenset(DELIMITER_PAIRS) | frozenset(DELIMITER_PAIRS.values())


def _boundary_kind(node: Node, *, first: bool) -> str:
    """Classify a visible edge, looking through transparent notation wrappers."""
    if node.kind == "row":
        for child in node.children if first else reversed(node.children):
            kind = _boundary_kind(child, first=first)
            if kind != "empty":
                return "ordinary" if kind == "binary" else kind
        return "empty"
    if node.kind == "font":
        kind = _boundary_kind(node.children[0], first=first)
        return "ordinary" if kind == "binary" else kind
    if _binary_atom(node):
        return "binary"
    if node.kind in {"boxed", "over", "under"}:
        return _boundary_kind(node.children[0], first=first)
    if node.kind == "scripts":
        base = node.children[0]
        return "operator" if _script_atom(base) and _boundary_kind(base, first=first) == "operator" else "ordinary"
    if node.kind == "delimited":
        mark = node.text.split("\n")[0 if first else 1]
        return ("open" if first else "close") if mark else _boundary_kind(node.children[0], first=first)
    if node.kind in {"function", "operator"}:
        return "operator"
    if node.kind == "glue":
        return "space"
    if node.kind in {"open", "close", "relation", "punctuation"}:
        return node.kind
    if node.kind == "literal":
        if not node.text:
            return "empty"
        if node.text[0 if first else -1].isspace():
            return "space"
        if node.text in {"(", "[", "{", "⟨", "⌊", "⌈"}:
            return "open"
        if node.text in {")", "]", "}", "⟩", "⌋", "⌉"}:
            return "close"
        if node.text in {"!", "?"}:
            return "postfix"
        if node.text in RELATIONS:
            return "relation"
        if node.text in BINARY:
            return "binary"
        if node.text in {",", ";", "."}:
            return "punctuation"
    return "ordinary"


def _is_spacing(node: Node) -> bool:
    return node.kind == "glue" or (node.kind == "literal" and not node.text.strip())


def _is_relation(node: Node) -> bool:
    return node.kind == "relation" or (node.kind == "literal" and node.text in RELATIONS)


def _row_children(node: Node) -> tuple[Node, ...]:
    """Resolve neutral bars locally without changing explicit relation bars."""
    children: list[Node] = []
    previous = "open"
    bars: list[str] = []
    for child in node.children:
        if _boundary_kind(child, first=True) == "empty":
            continue
        base = child.children[0] if child.kind == "scripts" else child
        if base.kind == "literal" and base.text in {"|", "‖"}:
            opening = previous in {"open", "relation", "binary", "punctuation", "operator"}
            kind = "open" if opening or (base.text not in bars and child.kind != "scripts") else "close"
            base = replace(base, kind=kind)
            child = replace(child, children=(base, *child.children[1:])) if child.kind == "scripts" else base
        if base.text in {"|", "‖"}:
            if base.kind == "open":
                bars.append(base.text)
            elif base.kind == "close" and bars and bars[-1] == base.text:
                bars.pop()
        children.append(child)
        if not _is_spacing(child):
            previous = _boundary_kind(child, first=False)
    return tuple(children)


def _relation_edges(children: tuple[Node, ...], index: int, *, leading: bool, trailing: bool) -> tuple[bool, bool]:
    """Transparent groups inherit only the spaces their neighbours require."""
    left = (
        leading if not index else _boundary_kind(children[index - 1], first=False) not in {"open", "relation", "space"}
    )
    right = (
        trailing
        if index + 1 == len(children)
        else _boundary_kind(children[index + 1], first=True) not in {"close", "relation", "space"}
    )
    return left, right


def _spaced_operator(children: tuple[Node, ...], index: int, *, in_script: bool = False) -> bool:
    """TeX scripts omit implicit operator gaps; explicit space nodes survive."""
    node = children[index]
    if in_script:
        return False
    if _is_relation(node):
        return True
    if node.kind != "literal" or node.text not in BINARY:
        return False
    return _binary_position(children, index)


def _binary_position(children: tuple[Node, ...], index: int) -> bool:
    previous = next(
        (child for child in reversed(children[:index]) if not _is_spacing(child)),
        None,
    )
    return previous is not None and _boundary_kind(previous, first=False) not in (
        "operator",
        "relation",
        "binary",
        "open",
        "punctuation",
    )


def _binary_atom(node: Node) -> bool:
    # Annotations inherit their base's class; explicit groups remain ordinary.
    if node.kind not in {"over", "under"}:
        return False
    node = node.children[0]
    while node.kind in {"font", "over", "under"} or (node.kind == "row" and len(node.children) == 1):
        node = node.children[0]
    return node.kind == "literal" and node.text in BINARY


def _spaced_binary(children: tuple[Node, ...], index: int, *, in_script: bool) -> bool:
    """Decorated binary atoms inherit their enclosing row's unary/binary role."""
    return not in_script and _binary_atom(children[index]) and _binary_position(children, index)


def _operator_text(
    children: tuple[Node, ...], index: int, *, leading_relation_space: bool, trailing_relation_space: bool
) -> str:
    """Adjacent relations form one symbol run; only its outer edges get glue."""
    node = children[index]
    left = right = " "
    if _is_relation(node):
        before, after = _relation_edges(
            children, index, leading=leading_relation_space, trailing=trailing_relation_space
        )
        left, right = " " if before else "", " " if after else ""
    return left + node.text + right


def _row_gap(children: tuple[Node, ...], index: int, *, in_script: bool, spatial: bool = False) -> str:
    """Add one shared boundary gap; explicit glue and operator padding own theirs."""
    if not index:
        return ""
    previous, current = children[index - 1 : index + 1]
    left = _boundary_kind(previous, first=False)
    right = _boundary_kind(current, first=True)
    if (
        left == "space"
        or right == "space"
        or any(
            _spaced_operator(children, i, in_script=in_script) or _spaced_binary(children, i, in_script=in_script)
            for i in (index - 1, index)
        )
    ):
        return ""
    if (right == "operator" and left in {"ordinary", "close", "postfix", "operator"}) or (
        left == "operator" and right in {"ordinary", "operator", "binary"}
    ):
        return " "
    if not in_script and (
        previous.kind == "punctuation" or (previous.kind == "literal" and previous.text in {",", ";"})
    ):
        return " "
    if spatial and _linear_fraction(current) and previous.kind == "literal" and previous.text in {"+", "-", "±", "∓"}:
        # Even in scripts, a sign must remain distinct from the fraction bar.
        return " "
    return ""


def _font_text(text: str, name: str) -> str:
    return text.translate(str.maketrans(FONT_ALPHABETS[name]))


def _visibly_delimited(node: Node) -> bool:
    left, _, right = node.text.partition("\n")
    return node.kind == "delimited" and left in DELIMITER_PAIRS and DELIMITER_PAIRS[left] == right


def _right_only_delimited(node: Node) -> bool:
    if node.kind in {"font", "boxed"} or (node.kind == "row" and len(node.children) == 1):
        return _right_only_delimited(node.children[0])
    return node.kind == "delimited" and node.text.startswith("\n") and len(node.text) > 1


def _script_atom(node: Node) -> bool:
    if node.kind == "row":
        if len(node.children) == 1:
            return _script_atom(node.children[0])
        return bool(node.children) and (
            all(child.kind == "literal" and child.text.isdecimal() for child in node.children)
            or all(child.kind in {"operator", "function"} for child in node.children)
        )
    if node.kind in {"font", "boxed"}:
        return _script_atom(node.children[0])
    return (
        (
            node.kind in {"literal", "open", "close", "relation", "punctuation"}
            and (len(node.text) == 1 or node.text.isdecimal())
        )
        or node.kind in {"operator", "function", "root", "accent", "binomial"}
        # Scripts attach to the visible right delimiter, including evaluation bars.
        or (node.kind == "delimited" and bool(node.text.split("\n")[1]))
    )


def _enclosure_children(children: tuple[Node, ...]) -> tuple[Node, ...]:
    """Invisible groups cannot hide a delimiter from their enclosing row."""
    result: list[Node] = []
    for child in children:
        if child.kind in {"row", "font", "boxed"}:
            result.extend(_enclosure_children(child.children))
        elif child.kind == "delimited":
            left, right = child.text.split("\n")
            if left:
                result.append(Node("open", left))
            result.extend(_enclosure_children(child.children))
            if right:
                result.append(Node("close", right))
        else:
            result.append(child)
    return tuple(result)


def _delimiter_free(node: Node) -> bool:
    return not any(char in DELIMITER_GLYPHS for char in node.text) and all(
        _delimiter_free(child) for child in node.children
    )


def _enclosed_row(children: tuple[Node, ...]) -> bool:
    """A raw delimiter pair must enclose the entire operand, without closing early."""
    children = _enclosure_children(children)
    expected: list[str] = []
    for index, child in enumerate(children):
        # Scripts have their own Unicode glyphs or explicit ^{...}/_{...} boundary.
        delimiter = child.children[0] if child.kind == "scripts" else child
        mark = delimiter.text if delimiter.kind in {"literal", "open", "close"} else ""
        if mark in DELIMITER_PAIRS and delimiter.kind != "close":
            if child.kind == "scripts":
                return False
            expected.append(DELIMITER_PAIRS[mark])
        elif mark in DELIMITER_PAIRS.values():
            if not expected or expected.pop() != mark:
                return False
            if not expected:
                return index == len(children) - 1
        elif not expected or not _delimiter_free(delimiter):
            return False
    return False


def _fraction_atom(node: Node) -> bool:
    """Recognize an already grouped operand without inferring structure from glyphs."""
    if node.kind in {"font", "boxed"}:
        return _fraction_atom(node.children[0])
    if node.kind == "row":
        children = tuple(child for child in _row_children(node) if not _is_spacing(child))
        while len(children) > 1 and children[-1].kind == "literal" and children[-1].text == "!":
            children = children[:-1]
        if len(children) == 1:
            return _fraction_atom(children[0])
        return bool(children) and (
            all(child.kind == "literal" and child.text.isdecimal() for child in children)
            or all(child.kind in {"operator", "function"} and _fraction_atom(child) for child in children)
            or _enclosed_row(children)
        )
    if node.kind == "scripts":
        base = node.children[0]
        if _right_only_delimited(base):
            return False
        # A compound base gains parentheses in inline(); an atomic base does not.
        return (
            _fraction_atom(base) if _script_atom(base) else _enclosed_row((Node("open", "("), base, Node("close", ")")))
        )
    if node.kind == "delimited":
        return _visibly_delimited(node) and _enclosed_row((node,))
    if node.kind in {"root", "accent", "binomial"}:
        return True
    if node.kind in {"function", "operator"} and node.text.replace(" ", "").isalnum():
        return True
    return _script_atom(node) and (len(node.text) == 1 or node.text.isdecimal())


def _fraction_operand(node: Node, *, in_script: bool) -> str:
    text = inline(node, in_script=in_script).strip()
    return text if _fraction_atom(node) else "(" + text + ")"


def _needs_linear_group(children: tuple[Node, ...], index: int, *, fraction: bool) -> bool:
    """Fractions bind tightly; removed boxes may contain any precedence level."""
    for neighbours, boundary, delimiter_kind in (
        (reversed(children[:index]), {"(", "[", "{", "⟨", "⌊", "⌈"}, "open"),
        (iter(children[index + 1 :]), {")", "]", "}", "⟩", "⌋", "⌉"} | ({"/"} if fraction else set()), "close"),
    ):
        boundary |= {",", ";"}
        if fraction:
            boundary |= BINARY | {"*"}
        neighbour = next((child for child in neighbours if not _is_spacing(child)), None)
        while neighbour is not None and neighbour.kind == "scripts":
            neighbour = neighbour.children[0]
        if neighbour is not None and not (
            _is_relation(neighbour)
            or neighbour.kind == "punctuation"
            or neighbour.kind == delimiter_kind
            or (neighbour.kind in {"literal", "open", "close"} and neighbour.text in boundary)
        ):
            return True
    return False


def _linear_fraction(node: Node) -> bool:
    """Font and TeX grouping wrappers do not remove a fraction's precedence."""
    if node.kind == "row":
        children = tuple(child for child in node.children if not _is_spacing(child))
        return len(children) == 1 and _linear_fraction(children[0])
    if node.kind in {"font", "boxed"} or (node.kind == "delimited" and node.text == "\n"):
        return _linear_fraction(node.children[0])
    return node.kind == "fraction"


def _unboxed_group(node: Node) -> bool:
    """A removed box may still need parentheses next to an implicit factor."""
    if (
        node.kind == "font"
        or (node.kind == "row" and len(node.children) == 1)
        or (node.kind == "delimited" and node.text == "\n")
    ):
        return _unboxed_group(node.children[0])
    return node.kind == "boxed" and (not _script_atom(node.children[0]) or _right_only_delimited(node.children[0]))


@dataclass(frozen=True, slots=True)
class Box:
    """Padded rows and a baseline measured in terminal cells, never code points."""

    rows: tuple[str, ...]
    width: int
    baseline: int

    @property
    def height(self) -> int:
        return len(self.rows)


def box(rows: tuple[str, ...], baseline: int = 0) -> Box:
    width = max((cell_len(row) for row in rows), default=0)
    if width > MAX_WIDTH or len(rows) > MAX_HEIGHT or width * len(rows) > MAX_AREA:
        raise MathError("formula canvas limit")
    if not rows or not 0 <= baseline < len(rows):
        raise MathError("invalid baseline")
    return Box(tuple(row + " " * (width - cell_len(row)) for row in rows), width, baseline)


def literal(text: str) -> Box:
    return box((text,))


def center(text: str, width: int) -> str:
    padding = width - cell_len(text)
    return " " * (padding // 2) + text + " " * (padding - padding // 2)


def horizontal(parts: tuple[Box, ...]) -> Box:
    if not parts:
        return literal("")
    baseline = max(part.baseline for part in parts)
    height = baseline + max(part.height - part.baseline for part in parts)
    width = sum(part.width for part in parts)
    if width > MAX_WIDTH or height > MAX_HEIGHT or width * height > MAX_AREA:
        raise MathError("formula canvas limit")
    rows = tuple(
        "".join(
            part.rows[y - baseline + part.baseline]
            if 0 <= y - baseline + part.baseline < part.height
            else " " * part.width
            for part in parts
        )
        for y in range(height)
    )
    return box(rows, baseline)


def stacked(base: Box, upper: Box | None = None, lower: Box | None = None) -> Box:
    width = max(base.width, upper.width if upper else 0, lower.width if lower else 0)
    rows = tuple(center(row, width) for part in (upper, base, lower) if part for row in part.rows)
    return box(rows, base.baseline + (upper.height if upper else 0))


def delimit(body: Box, left: str, right: str) -> Box:
    """Use ASCII structural pieces for tall brackets, avoiding font join glyphs."""
    if body.height == 1:
        return horizontal((literal(left), body, literal(right)))

    def side(mark: str, y: int) -> str:
        if not mark:
            return ""
        if mark in {"(", ")"}:
            if y == 0:
                return "/" if mark == "(" else "\\"
            if y == body.height - 1:
                return "\\" if mark == "(" else "/"
            return "|"
        if mark in {"[", "]"}:
            return mark if y in {0, body.height - 1} else "|"
        if mark in {"{", "}"}:
            return mark if y == body.baseline else "|"
        return mark

    return box(
        tuple(side(left, y) + " " + row + " " + side(right, y) for y, row in enumerate(body.rows)), body.baseline
    )


def _script_text(node: Node, table: dict[str, str], marker: str) -> str:
    value = inline(node, in_script=True)
    if not value:
        return ""
    if all(char in table for char in value):
        return "".join(table[char] for char in value)
    return marker + "{" + value + "}"


def inline(
    node: Node, *, in_script: bool = False, leading_relation_space: bool = False, trailing_relation_space: bool = False
) -> str:
    """Linear notation preserves grouping even where Unicode lacks a glyph."""
    kind = node.kind
    if kind in {"literal", "function", "operator", "glue", "open", "close", "relation", "punctuation"}:
        return node.text
    if kind == "row":
        children = _row_children(node)
        parts: list[str] = []
        for index, child in enumerate(children):
            before, after = _relation_edges(
                children, index, leading=leading_relation_space, trailing=trailing_relation_space
            )
            gap = _row_gap(children, index, in_script=in_script)
            if _spaced_operator(children, index, in_script=in_script):
                value = _operator_text(
                    children,
                    index,
                    leading_relation_space=leading_relation_space,
                    trailing_relation_space=trailing_relation_space,
                )
            elif (_linear_fraction(child) and _needs_linear_group(children, index, fraction=True)) or (
                _unboxed_group(child) and _needs_linear_group(children, index, fraction=False)
            ):
                value = "(" + inline(child, in_script=in_script) + ")"
            else:
                value = inline(child, in_script=in_script, leading_relation_space=before, trailing_relation_space=after)
                if _spaced_binary(children, index, in_script=in_script):
                    value = " " + value + " "
            parts.append(gap + value)
        return "".join(parts)
    if kind == "fraction":
        numerator, denominator = (_fraction_operand(child, in_script=in_script) for child in node.children)
        return f"{numerator}/{denominator}"
    if kind == "binomial":
        upper, lower = (inline(child, in_script=in_script) for child in node.children)
        return f"binom({upper}, {lower})"
    if kind == "root":
        body = inline(node.children[0], in_script=in_script)
        index = inline(node.children[1], in_script=True)
        return f"root[{index}]({body})" if index else f"√({body})"
    if kind == "scripts":
        base, lower, upper = node.children
        value = (
            base.text
            if base.kind in {"function", "operator"}
            else inline(
                base,
                in_script=in_script,
                leading_relation_space=leading_relation_space,
                trailing_relation_space=trailing_relation_space,
            )
        )
        if value.strip() and not _script_atom(base):
            value = "(" + value + ")"
        return value + _script_text(lower, SUBSCRIPT, "_") + _script_text(upper, SUPERSCRIPT, "^")
    if kind == "delimited":
        left, right = node.text.split("\n")
        return (
            left
            + inline(
                node.children[0],
                in_script=in_script,
                leading_relation_space=leading_relation_space if not left else False,
                trailing_relation_space=trailing_relation_space if not right else False,
            )
            + right
        )
    if kind == "font":
        return _font_text(
            inline(
                node.children[0],
                in_script=in_script,
                leading_relation_space=leading_relation_space,
                trailing_relation_space=trailing_relation_space,
            ),
            node.text,
        )
    if kind == "boxed":
        return inline(node.children[0], in_script=in_script)
    if kind == "accent":
        return f"{node.text}({inline(node.children[0], in_script=in_script)})"
    if kind in {"over", "under"}:
        base, annotation = node.children
        value = inline(
            base,
            in_script=in_script,
            leading_relation_space=leading_relation_space,
            trailing_relation_space=trailing_relation_space,
        )
        before = value[: len(value) - len(value.lstrip())]
        after = value[len(value.rstrip()) :]
        core = value.strip()
        if core and not _script_atom(base):
            core = "(" + core + ")"
        table, marker = (SUPERSCRIPT, "^") if kind == "over" else (SUBSCRIPT, "_")
        return before + core + _script_text(annotation, table, marker) + after
    if kind == "matrix":
        rows = "; ".join(", ".join(inline(cell, in_script=in_script) for cell in row.children) for row in node.children)
        return f"{node.text}[{rows}]"
    raise MathError("unknown notation node")


def layout(
    node: Node, *, in_script: bool = False, leading_relation_space: bool = False, trailing_relation_space: bool = False
) -> Box:
    """Lay out one immutable syntax tree, keeping every child in source order."""
    kind = node.kind
    if kind in {"literal", "operator", "function", "glue", "open", "close", "relation", "punctuation"}:
        return literal(node.text)
    if kind == "row":
        children = _row_children(node)
        parts: list[Box] = []
        for i, child in enumerate(children):
            before, after = _relation_edges(
                children, i, leading=leading_relation_space, trailing=trailing_relation_space
            )
            if gap := _row_gap(children, i, in_script=in_script, spatial=True):
                parts.append(literal(gap))
            decorated_binary = _spaced_binary(children, i, in_script=in_script)
            if decorated_binary:
                parts.append(literal(" "))
            parts.append(
                literal(
                    _operator_text(
                        children,
                        i,
                        leading_relation_space=leading_relation_space,
                        trailing_relation_space=trailing_relation_space,
                    )
                )
                if _spaced_operator(children, i, in_script=in_script)
                else layout(child, in_script=in_script, leading_relation_space=before, trailing_relation_space=after)
            )
            if decorated_binary:
                parts.append(literal(" "))
        return horizontal(tuple(parts))
    if kind in {"fraction", "binomial"}:
        numerator, denominator = (layout(child, in_script=in_script) for child in node.children)
        width = max(numerator.width, denominator.width) + 2
        rows = tuple(center(row, width) for row in numerator.rows)
        rows += (("-" * width if kind == "fraction" else " " * width),)
        rows += tuple(center(row, width) for row in denominator.rows)
        result = box(rows, numerator.height)
        return delimit(result, "(", ")") if kind == "binomial" else result
    if kind == "root":
        body = layout(node.children[0], in_script=in_script)
        index = layout(node.children[1], in_script=True)
        rows = (
            " " + "-" * body.width,
            *(("√" if y == body.baseline else "|") + row for y, row in enumerate(body.rows)),
        )
        result = box(rows, body.baseline + 1)
        if index.width:
            index = box(
                index.rows + (" " * index.width,) * max(1, result.baseline), index.height + max(1, result.baseline) - 1
            )
            result = horizontal((index, result))
        return result
    if kind == "scripts":
        base_node = node.children[0]
        base = layout(
            base_node,
            in_script=in_script,
            leading_relation_space=leading_relation_space,
            trailing_relation_space=trailing_relation_space,
        )
        lower, upper = (layout(child, in_script=True) for child in node.children[1:])
        if base_node.kind == "operator" and (
            base_node.limits == "above" or (base_node.limits == "display" and not in_script)
        ):
            return stacked(base, upper if upper.width else None, lower if lower.width else None)
        if not lower.width and not upper.width:
            return base
        if base.height == 1 and all(
            all(char in alphabet for char in inline(script, in_script=True))
            for script, alphabet in ((node.children[1], SUBSCRIPT), (node.children[2], SUPERSCRIPT))
        ):
            return literal(
                inline(
                    node,
                    in_script=in_script,
                    leading_relation_space=leading_relation_space,
                    trailing_relation_space=trailing_relation_space,
                )
            )
        # Lift/lower scripts outside the base's full box so a nested fraction
        # cannot overwrite its numerator or denominator.
        script_width = max(lower.width, upper.width)
        upper_rows = upper.rows if upper.width else ()
        lower_rows = lower.rows if lower.width else ()
        scripts = box(
            tuple(row + " " * (script_width - cell_len(row)) for row in upper_rows)
            + (" " * script_width,) * base.height
            + tuple(row + " " * (script_width - cell_len(row)) for row in lower_rows),
            len(upper_rows) + base.baseline,
        )
        return horizontal((base, scripts))
    if kind == "delimited":
        left, right = node.text.split("\n")
        return delimit(
            layout(
                node.children[0],
                in_script=in_script,
                leading_relation_space=leading_relation_space if not left else False,
                trailing_relation_space=trailing_relation_space if not right else False,
            ),
            left,
            right,
        )
    if kind == "font":
        body = layout(
            node.children[0],
            in_script=in_script,
            leading_relation_space=leading_relation_space,
            trailing_relation_space=trailing_relation_space,
        )
        return box(tuple(_font_text(row, node.text) for row in body.rows), body.baseline)
    if kind == "boxed":
        body = layout(node.children[0], in_script=in_script)
        border = "+" + "-" * (body.width + 2) + "+"
        return box((border, *("| " + row + " |" for row in body.rows), border), body.baseline + 1)
    if kind == "accent":
        body = layout(node.children[0], in_script=in_script)
        mark = {
            "overline": "-",
            "bar": "-",
            "underline": "-",
            "hat": "^",
            "widehat": "^",
            "tilde": "~",
            "widetilde": "~",
            "vec": "→",
            "dot": ".",
            "ddot": "..",
        }[node.text]
        accent = literal(mark * body.width if mark == "-" else mark)
        return stacked(body, lower=accent) if node.text == "underline" else stacked(body, upper=accent)
    if kind in {"over", "under"}:
        base = layout(
            node.children[0],
            in_script=in_script,
            leading_relation_space=leading_relation_space,
            trailing_relation_space=trailing_relation_space,
        )
        annotation = layout(node.children[1], in_script=True)
        return stacked(base, lower=annotation) if kind == "under" else stacked(base, upper=annotation)
    if kind == "matrix":
        grid = [
            [
                layout(
                    cell, in_script=in_script, leading_relation_space=node.text in {"aligned", "split"} and bool(i % 2)
                )
                for i, cell in enumerate(row.children)
            ]
            for row in node.children
        ]
        widths = [max(row[i].width for row in grid) for i in range(len(grid[0]))]
        rows: list[str] = []
        for row in grid:
            padded = []
            for i, cell in enumerate(row):
                # Aligned environments alternate right/left alignment around &.
                if node.text in {"aligned", "split"}:
                    padding = widths[i] - cell.width
                    values = tuple(
                        (" " * padding + value) if i % 2 == 0 else (value + " " * padding) for value in cell.rows
                    )
                else:
                    values = tuple(center(value, widths[i]) for value in cell.rows)
                padded.append(box(values, cell.baseline))
                if i < len(row) - 1 and (node.text not in {"aligned", "split"} or i % 2):
                    padded.append(literal("  "))
            rows.extend(horizontal(tuple(padded)).rows)
        result = box(tuple(rows), len(rows) // 2)
        left, right = {
            "pmatrix": ("(", ")"),
            "bmatrix": ("[", "]"),
            "Bmatrix": ("{", "}"),
            "vmatrix": ("|", "|"),
            "Vmatrix": ("‖", "‖"),
            "cases": ("{", ""),
        }.get(node.text, ("", ""))
        return delimit(result, left, right) if left or right else result
    raise MathError("unknown notation node")
