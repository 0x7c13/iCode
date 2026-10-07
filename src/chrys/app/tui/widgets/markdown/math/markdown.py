# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Math delimiters before CommonMark escaping, with conservative dollar rules."""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from typing import TYPE_CHECKING

from .parser import MAX_SOURCE

if TYPE_CHECKING:
    from markdown_it import MarkdownIt
    from markdown_it.rules_block import StateBlock
    from markdown_it.rules_core import StateCore
    from markdown_it.rules_inline import StateInline
    from markdown_it.utils import EnvType

MATH_ENABLED = "chrys_math_enabled"
_MATH_INDEXES = "chrys_math_delimiter_indexes"


def _index_delimiters(source: str) -> dict[str, list[int]]:
    """Index each source once; repeated incomplete openers never rescan its tail."""
    positions: dict[str, list[int]] = {"$": [], "$$": [], r"\)": [], r"\]": []}
    backslashes = 0
    previous_dollar = -2
    for position, character in enumerate(source):
        if character == "\\":
            backslashes += 1
            continue
        if character in ")]" and backslashes % 2:
            positions["\\" + character].append(position - 1)
        elif character == "$" and backslashes % 2 == 0:
            positions["$"].append(position)
            if previous_dollar == position - 1:
                positions["$$"].append(previous_dollar)
            previous_dollar = position
        backslashes = 0
    return positions


def _closing(source: str, marker: str, start: int, env: EnvType) -> int:
    indexes: dict[str, dict[str, list[int]]] = env.setdefault(_MATH_INDEXES, {})
    if source not in indexes:
        indexes[source] = _index_delimiters(source)
    positions = indexes[source][marker]
    index = bisect_left(positions, start)
    return positions[index] if index < len(positions) else -1


def _clear_delimiter_indexes(state: StateCore) -> None:
    """Parsing caches must not retain old transcripts in a caller-owned env."""
    state.env.pop(_MATH_INDEXES, None)


def dollar_is_math(body: str) -> bool:
    """Prefer a false negative over changing prices, paths or shell variables."""
    if not body or body != body.strip() or "\n" in body or "`" in body:
        return False
    if body[-1] in "+-*/=<>" or body.startswith(("?", "#", "@", "!")):
        return False
    if len(body) == 1 and body.isalpha():
        return True
    if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", body):
        return len(body) == 1 or re.fullmatch(r"[A-Za-z]_(?:[A-Za-z]|\d+)", body) is not None
    if re.match(r"[A-Za-z_][A-Za-z_0-9]{1,}(?:/|\.|:|\[)", body):
        return False
    if re.fullmatch(r"\{[A-Za-z_][A-Za-z_0-9]*\}.*", body):
        return False
    if re.search(r"\b[A-Za-z]{2,}\b", re.sub(r"\\[A-Za-z]+", "", body)):
        # Explicit TeX commands provide intent; bare prose does not.
        return "\\" in body or re.search(r"[=^_{}]", body) is not None
    return bool(
        re.search(r"\\(?:[A-Za-z]+|[{}|])|[=+*/^_<>-]", body)
        or body.isdecimal()
        or re.fullmatch(r"[A-Za-z]\([A-Za-z0-9_, ]+\)", body)
    )


def _prose_without_links(body: str) -> str:
    """Simplify links for detection without rescanning failed destinations."""
    closings: dict[int, int] = {}
    parentheses: list[int] = []
    for index, character in enumerate(body):
        if character == "(":
            parentheses.append(index)
        elif character == ")" and parentheses:
            closings[parentheses.pop()] = index + 1
        elif character == "\n":
            parentheses.clear()
    parts: list[str] = []
    consumed = 0
    for opening in re.finditer(r"\[([^\[\]\n]+)\]\(", body):
        if opening.start() < consumed:
            continue
        closing = closings.get(opening.end() - 1)
        if closing is None:
            continue
        parts.extend((body[consumed : opening.start()], opening.group(1)))
        consumed = closing
    parts.append(body[consumed:])
    return "".join(parts)


def bracket_is_math(body: str) -> bool:
    """Explicit TeX delimiters accept notation but preserve ordinary escapes."""
    word = body
    if "\\" not in body:
        if "`" in body:
            return False
        # Formatting wrappers must not hide ordinary prose from the heuristic.
        # Only the detection copy is simplified; CommonMark renders the source.
        prose = _prose_without_links(body)
        word = prose
        for _ in range(4):
            unwrapped = re.sub(r"([*_~]{1,3})([A-Za-z][A-Za-z0-9]*)\1", r"\2", prose)
            # A marker within a lone word may be multiplication, as in AB*CD*EF.
            unwrapped_word = re.sub(r"(?<![A-Za-z0-9])([*_~]{1,3})([A-Za-z][A-Za-z0-9]*[.,:;!?]*)\1", r"\2", word)
            if unwrapped == prose and unwrapped_word == word:
                break
            prose, word = unwrapped, unwrapped_word
        if re.search(r"(?<![A-Za-z])[A-Za-z]{2,}[.,:;!?]?\s+[A-Za-z]{2,}", prose):
            return False
    return bool(body) and not body.isdecimal() and re.fullmatch(r"[A-Za-z]{3,}(?:\s+[A-Za-z]+)*[.,:;!?]*", word) is None


def _emphasis_opens_before_math(state: StateInline, start: int) -> bool:
    """Use CommonMark's flanking rules for a marker run before a formula."""
    source = state.src
    if not start or source[start - 1] not in "*_":
        return False
    marker = source[start - 1]
    opening = start - 1
    while opening and source[opening - 1] == marker:
        opening -= 1
    backslashes = 0
    preceding = opening - 1
    while preceding >= 0 and source[preceding] == "\\":
        backslashes += 1
        preceding -= 1
    if backslashes % 2:
        return False
    return state.scanDelims(opening, marker == "*").can_open


def _emphasis_wraps_math(state: StateInline, start: int, finish: int) -> bool:
    """Recognize adjacent emphasis without treating bare shell suffixes as markup."""
    return (
        _emphasis_opens_before_math(state, start)
        and finish < state.posMax
        and state.src[finish] == state.src[start - 1]
        and state.scanDelims(finish, state.src[finish] == "*").can_close
    )


def math_inline(state: StateInline, silent: bool) -> bool:
    start = state.pos
    source = state.src
    bracket = source.startswith(r"\(", start)
    display_bracket = source.startswith(r"\[", start)
    display_dollar = source.startswith("$$", start)
    dollar = source.startswith("$", start) and not display_dollar
    if not (bracket or display_bracket or dollar or display_dollar):
        return False
    enabled = bool(state.md.options.get(MATH_ENABLED))
    if (dollar or display_dollar) and not enabled:
        return False
    if bracket:
        opening, closing = r"\(", r"\)"
    elif display_bracket:
        opening, closing = r"\[", r"\]"
    else:
        opening = closing = "$$" if display_dollar else "$"
    end = _closing(source, closing, start + len(opening), state.env)
    if end < 0 or end + len(closing) > state.posMax:
        # Let CommonMark process an ordinary escape and the following inline
        # Markdown independently while an incomplete delimiter is streaming.
        return False
    finish = end + len(closing)
    body = source[start + len(opening) : end]
    candidate = body if dollar else body.strip()
    valid = bracket_is_math(candidate) if bracket or display_bracket else dollar_is_math(candidate)
    if dollar or display_dollar:
        emphasis = _emphasis_wraps_math(state, start, finish)
        notation = (
            re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", candidate) is None
            and re.search(r"\\(?:[A-Za-z]+|[{}|])|[=+*/^_<>-]", candidate) is not None
        )
        if start and (
            (source[start - 1].isascii() and source[start - 1].isalnum())
            or source[start - 1] in "$\\"
            or (source[start - 1] == "_" and not (emphasis or (notation and _emphasis_opens_before_math(state, start))))
        ):
            valid = False
        if finish < len(source):
            following = source[finish]
            # A real formula may end before emphasis or sentence punctuation.
            # Bare identifiers still keep ambiguous shell suffixes as text.
            if (
                (following.isascii() and following.isalnum())
                or following in "${#@("
                or (following == "_" and not emphasis)
                or (following in "*?!" and not (emphasis or notation))
            ):
                valid = False
    if enabled and not valid:
        # Ordinary escapes, prose, prices and shell variables belong to the
        # CommonMark rules; rejecting a candidate must not consume its contents.
        return False
    if not silent:
        compile_candidate = enabled and len(candidate) <= MAX_SOURCE
        token = state.push("math_inline" if compile_candidate else "math_literal", "", 0)
        token.content = candidate if compile_candidate else source[start:finish]
        token.meta["math_source"] = source[start:finish]
    state.pos = finish
    return True


def math_block(state: StateBlock, start_line: int, end_line: int, silent: bool) -> bool:
    if state.sCount[start_line] - state.blkIndent >= 4:
        return False
    start = state.bMarks[start_line] + state.tShift[start_line]
    first = state.src[start : state.eMarks[start_line]]
    if first.startswith("$$"):
        opening, closing = "$$", "$$"
    elif first.startswith(r"\["):
        opening, closing = r"\[", r"\]"
    else:
        return False
    closing_position = _closing(state.src, closing, start + len(opening), state.env)
    if closing_position < 0:
        return False
    closing_line = bisect_right(state.eMarks, closing_position)
    if closing_line >= end_line:
        return False
    # Each matching interval is scanned once, stopping at structural boundaries.
    # Compilation remains bounded, while larger closed formulas retain source.
    lines: list[str] = []
    for last in range(start_line, closing_line + 1):
        if last > start_line and state.sCount[last] < state.blkIndent and not state.isEmpty(last):
            return False
        line = state.getLines(last, last + 1, state.blkIndent, False).rstrip("\n")
        if last > start_line and (not line.strip() or re.match(r"\s*(?:`{3,}|~{3,})", line)):
            return False
        if last == start_line:
            line = line.lstrip()
        elif opening == r"\[" and line.lstrip().startswith(opening):
            # Nested display openers cannot close this block; stopping here
            # keeps repeated incomplete streamed formulas linear to scan.
            return False
        lines.append(line)
        if last != closing_line:
            continue
        end = _closing(line, closing, len(opening) if last == start_line else 0, state.env)
        if end < 0:
            return False
        if line[end + len(closing) :].strip():
            return False
        if opening == r"\[" and last == start_line:
            body = line[len(opening) : end].strip()
            if not bracket_is_math(body):
                return False
        if silent:
            return True
        token = state.push("math_block", "math", 0)
        token.block = True
        token.map = [start_line, last + 1]
        token.meta["math_source"] = "\n".join(lines).strip()
        lines[-1] = line[:end]
        token.content = "\n".join(lines)[len(opening) :].strip()
        state.line = last + 1
        return True
    return False


def enable_math(parser: MarkdownIt) -> None:
    """Install rules only on the assistant/document parser."""
    parser.options[MATH_ENABLED] = True
    parser.inline.ruler.before("escape", "chrys_math_inline", math_inline)
    parser.block.ruler.before(
        "fence", "chrys_math_block", math_block, {"alt": ["paragraph", "reference", "blockquote", "list"]}
    )
    parser.core.ruler.push("chrys_math_cleanup", _clear_delimiter_indexes)


def disable_math(parser: MarkdownIt) -> None:
    """User messages preserve their literal mathematical spelling."""
    parser.options[MATH_ENABLED] = False
    parser.block.ruler.disable("chrys_math_block")
