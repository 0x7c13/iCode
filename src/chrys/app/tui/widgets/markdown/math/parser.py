# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bounded, notation-preserving TeX subset parser; never evaluates expressions.

Unknown commands and malformed groups reject the entire formula. The caller
retains its exact source instead of displaying a partially interpreted result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .symbols import FONTS, FUNCTIONS, INTEGRAL_OPERATORS, LARGE_OPERATORS, SYMBOLS, TEXT_COMMANDS

MAX_SOURCE = 8192
MAX_NODES = 2048
MAX_DEPTH = 48
MAX_MATRIX_CELLS = 256
DELIMITER_SIZES = frozenset(size + suffix for size in ("big", "Big", "bigg", "Bigg") for suffix in ("", "l", "r", "m"))

type LimitPlacement = Literal["display", "above", "side"]


class MathError(ValueError):
    """The source cannot be represented faithfully within the supported subset."""


@dataclass(frozen=True, slots=True)
class Node:
    """A notation node, independent of Markdown and terminal geometry."""

    kind: str
    text: str = ""
    children: tuple[Node, ...] = ()
    limits: LimitPlacement = "side"


class Parser:
    """Recursive descent with explicit source, recursion and allocation bounds."""

    def __init__(self, source: str) -> None:
        if not source.strip() or len(source) > MAX_SOURCE:
            raise MathError("empty or oversized formula")
        if any((ord(c) < 32 and c not in "\n\t\r") or 0x7F <= ord(c) < 0xA0 for c in source):
            raise MathError("control character")
        self.source = source
        self.pos = 0
        self.nodes = 0
        self.depth = 0

    def node(
        self, kind: str, text: str = "", children: tuple[Node, ...] = (), *, limits: LimitPlacement = "side"
    ) -> Node:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise MathError("too many nodes")
        return Node(kind, text, children, limits)

    def skip_space(self) -> None:
        while self.pos < len(self.source) and self.source[self.pos].isspace():
            self.pos += 1

    def starts(self, text: str) -> bool:
        self.skip_space()
        return self.at(text)

    def at(self, text: str) -> bool:
        if not self.source.startswith(text, self.pos):
            return False
        end = self.pos + len(text)
        return not (
            text.startswith("\\")
            and text[-1:].isalpha()
            and end < len(self.source)
            and self.source[end].isascii()
            and self.source[end].isalpha()
        )

    def consume(self, text: str) -> None:
        if not self.starts(text):
            raise MathError(f"expected {text}")
        self.pos += len(text)

    def sequence(self, stops: tuple[str, ...] = ()) -> Node:
        children: list[Node] = []
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise MathError("nesting limit")
        while True:
            self.skip_space()
            if self.pos == len(self.source) or any(self.at(stop) for stop in stops):
                break
            base = self.atom()
            lower: Node | None = None
            upper: Node | None = None
            if self.starts(r"\limits") or self.starts(r"\nolimits"):
                command = self.command()
                if base.kind != "operator":
                    raise MathError("limits on a non-operator")
                base = self.node("operator", base.text, limits="above" if command == "limits" else "side")
            while self.starts("_") or self.starts("^"):
                marker = self.source[self.pos]
                self.pos += 1
                value = self.argument()
                if marker == "_":
                    if lower is not None:
                        raise MathError("duplicate subscript")
                    lower = value
                else:
                    if upper is not None:
                        raise MathError("duplicate superscript")
                    upper = value
            if lower is not None or upper is not None:
                empty = self.node("literal")
                base = self.node("scripts", children=(base, lower or empty, upper or empty))
            children.append(base)
        self.depth -= 1
        return self.node("row", children=tuple(children))

    def argument(self) -> Node:
        self.skip_space()
        if self.starts("{"):
            self.pos += 1
            result = self.sequence(("}",))
            self.consume("}")
            return result
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise MathError("nesting limit")
        result = self.atom()
        self.depth -= 1
        return result

    def raw_group(self) -> str:
        self.consume("{")
        start = self.pos
        level = 1
        while self.pos < len(self.source):
            char = self.source[self.pos]
            if char == "\\":
                self.pos += 2
                continue
            if char == "{":
                level += 1
            elif char == "}":
                level -= 1
                if not level:
                    result = self.source[start : self.pos]
                    self.pos += 1
                    return result
            self.pos += 1
        raise MathError("unclosed text group")

    def command(self) -> str:
        self.consume("\\")
        match = re.match(r"[A-Za-z]+|.", self.source[self.pos :])
        if match is None:
            raise MathError("incomplete command")
        self.pos += len(match[0])
        return match[0]

    def delimiter(self) -> str:
        self.skip_space()
        if self.starts("\\"):
            name = self.command()
            if name not in SYMBOLS:
                raise MathError("unknown delimiter")
            value = SYMBOLS[name]
        else:
            value = self.source[self.pos : self.pos + 1]
            self.pos += 1
        if value not in {"(", ")", "[", "]", "{", "}", "|", "‖", "⟨", "⟩", "⌊", "⌋", "⌈", "⌉", "."}:
            raise MathError("invalid delimiter")
        return "" if value == "." else value

    def environment(self) -> Node:
        name = self.raw_group()
        if name not in {
            "matrix",
            "pmatrix",
            "bmatrix",
            "Bmatrix",
            "vmatrix",
            "Vmatrix",
            "smallmatrix",
            "cases",
            "aligned",
            "gathered",
            "split",
        }:
            raise MathError("unsupported environment")
        rows: list[Node] = []
        cells: list[Node] = []
        count = 0
        while True:
            cell = self.sequence(("&", r"\\", r"\end"))
            count += 1
            if count > MAX_MATRIX_CELLS:
                raise MathError("oversized matrix")
            cells.append(cell)
            if self.starts("&"):
                self.pos += 1
                continue
            rows.append(self.node("row", children=tuple(cells)))
            cells = []
            if self.starts(r"\\"):
                self.pos += 2
                if not self.starts(r"\end"):
                    continue
            self.consume(r"\end")
            if self.raw_group() != name:
                raise MathError("mismatched environment")
            break
        columns = len(rows[0].children)
        if any(len(row.children) != columns for row in rows):
            raise MathError("ragged matrix")
        if name == "cases" and columns != 2:
            raise MathError("cases needs two columns")
        return self.node("matrix", name, tuple(rows))

    def atom(self) -> Node:
        self.skip_space()
        if self.pos >= len(self.source):
            raise MathError("missing argument")
        char = self.source[self.pos]
        if char == "{":
            return self.argument()
        if char in "}^_&$%#" or char == "\n":
            raise MathError("unexpected math punctuation")
        if char != "\\":
            self.pos += 1
            return self.node("literal", char)
        command = self.command()
        if command in DELIMITER_SIZES:
            delimiter = self.delimiter()
            kind = {"l": "open", "r": "close", "m": "relation"}.get(command[-1], "literal")
            return self.node(kind if delimiter else "literal", delimiter)
        if command == "mid":
            # The same glyph is an ordinary delimiter when written as | or \vert.
            return self.node("relation", "|")
        if command == "colon":
            return self.node("punctuation", ":")
        if command in {"lvert", "lVert", "rvert", "rVert"}:
            return self.node("open" if command.startswith("l") else "close", SYMBOLS[command])
        if command == "not":
            target = self.argument()
            negated = {"=": "≠", "∈": "∉"}.get(target.text) if target.kind == "literal" else None
            if negated is None:
                raise MathError("unsupported negated symbol")
            return self.node("literal", negated)
        if command == "boxed":
            return self.node("boxed", children=(self.argument(),))
        if command in {"frac", "dfrac", "tfrac", "binom", "dbinom", "tbinom"}:
            return self.node(
                "binomial" if "binom" in command else "fraction", children=(self.argument(), self.argument())
            )
        if command == "sqrt":
            index = self.node("literal")
            if self.starts("["):
                self.pos += 1
                index = self.sequence(("]",))
                self.consume("]")
            return self.node("root", children=(self.argument(), index))
        if command == "left":
            left = self.delimiter()
            body = self.sequence((r"\right",))
            self.consume(r"\right")
            return self.node("delimited", left + "\n" + self.delimiter(), (body,))
        if command == "begin":
            return self.environment()
        if command in TEXT_COMMANDS:
            raw = self.raw_group()
            # Only escaped literal punctuation is valid in a text box. Retain
            # unsupported nested formatting via the formula's source fallback.
            unescaped = re.sub(r"\\[{}%_$&# ]", "", raw)
            if any(c in unescaped for c in "\\{}"):
                raise MathError("unsupported text command")
            text = re.sub(r"\\([{}%_$&# ])", r"\1", raw)
            result = self.node("function" if command == "operatorname" else "literal", re.sub(r"\s+", " ", text))
            return result
        if command in FONTS:
            child = self.argument()
            if command in {"mathrm", "mathit", "mathsf", "mathtt"}:
                return child
            return self.node("font", command, (child,))
        if command in {"overline", "underline", "bar", "hat", "widehat", "tilde", "widetilde", "vec", "dot", "ddot"}:
            return self.node("accent", command, (self.argument(),))
        if command in {"overset", "underset", "stackrel"}:
            annotation, base = self.argument(), self.argument()
            if base.kind != "row":
                base = self.node("row", children=(base,))
            return self.node("under" if command == "underset" else "over", children=(base, annotation))
        if command in {"displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle"}:
            return self.node("literal")
        if command == "pmod":
            return self.node(
                "delimited", "(\n)", (self.node("row", children=(self.node("literal", "mod "), self.argument())),)
            )
        if command in {"bmod", "mod"}:
            return self.node("literal", " mod ")
        if command == "!":
            # A terminal has no fractional cells: retain a zero-width boundary
            # that suppresses automatic glue instead of shifting painted cells.
            return self.node("glue")
        if command in SYMBOLS or command in FUNCTIONS:
            kind = (
                "operator"
                if command in LARGE_OPERATORS | INTEGRAL_OPERATORS
                else ("function" if command in FUNCTIONS or command in {"Re", "Im"} else "literal")
            )
            return self.node(
                kind, SYMBOLS.get(command, command), limits="display" if command in LARGE_OPERATORS else "side"
            )
        raise MathError(f"unsupported command: {command}")


def parse_math(source: str) -> Node:
    """Parse one complete expression or raise a bounded, expected MathError."""
    parser = Parser(source)
    result = parser.sequence()
    if parser.pos != len(source) or not result.children:
        raise MathError("incomplete expression")
    return result
