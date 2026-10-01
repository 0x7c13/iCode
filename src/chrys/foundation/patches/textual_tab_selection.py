# Copyright (c) 2021 Will McGugan
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Contains code adapted from Textual (MIT License; see NOTICE).

"""Patch: keep Textual selection offsets stable across tab expansion.

Problem
-------
Textual renders ``\\t`` by expanding it to spaces, but selection hit-testing
returns offsets from the rendered text. Downstream selection extraction still
uses those offsets against the original source text, so selecting text after a
tab drifts to the wrong character.

Solution
--------
When ``Content`` expands tabs for rendering, carry a small rendered-offset to
source-offset map through the rendered segments. The compositor uses that map
when converting mouse cells back to selection offsets. Runtime monkey patches
cover Textual modules imported before Chrys' site-package patcher runs.
"""

from __future__ import annotations

import logging
from itertools import pairwise
from typing import Any

from rich.segment import Segment
from rich.style import Style as RichStyle

from chrys.foundation.patches.patcher import FilePatch, register

_RUNTIME_PATCH_MARKER = "_chrys_tab_selection_offsets"
# The runtime monkey patch reproduces Textual 8.2.7's compositor and content
# render path. The project pins Textual to an exact version, so a version
# equality check is the reliable signal that the patched bodies still match.
# Bump this together with the pin in pyproject.toml on every Textual upgrade.
#
# This deliberately avoids inspecting the live function source: the site-package
# patcher rewrites _compositor.py after import, which shifts line numbers and
# leaves the in-memory code object's co_firstlineno stale, so inspect.getsource
# can no longer be trusted to return the right function body.
_RUNTIME_PATCH_TEXTUAL_VERSION = "8.2.7"
logger = logging.getLogger(__name__)


def _runtime_compositor_patch_compatible() -> bool:
    """Return true when the loaded Textual matches the version the patch targets."""
    try:
        import textual
    except ImportError:
        return False
    return getattr(textual, "__version__", None) == _RUNTIME_PATCH_TEXTUAL_VERSION


_CONTENT_HELPER_OLD = '''\
def _strip_control_codes(
    text: str, _translate_table: dict[int, None] = _CONTROL_STRIP_TRANSLATE
) -> str:
    """Remove control codes from text.

    Args:
        text (str): A string possibly contain control codes.

    Returns:
        str: String with control codes removed.
    """
    return text.translate(_translate_table)


@rich.repr.auto
class Span(NamedTuple):'''

_CONTENT_HELPER_NEW = '''\
def _strip_control_codes(
    text: str, _translate_table: dict[int, None] = _CONTROL_STRIP_TRANSLATE
) -> str:
    """Remove control codes from text.

    Args:
        text (str): A string possibly contain control codes.

    Returns:
        str: String with control codes removed.
    """
    return text.translate(_translate_table)


def _expand_tabs_with_offsets(
    line: "Content", tab_size: int
) -> tuple["Content", tuple[int, ...] | None]:
    """Expand tabs and map rendered character offsets back to source offsets."""
    if "\\t" not in line.plain:
        return line, None

    if not line._spans:
        # Preserve Textual's unstyled Content.expand_tabs behavior: it delegates
        # to str.expandtabs(), whose tab stops are character-position based.
        source_offsets: list[int] = []
        source_position = 0
        cell_position = 0
        for character in line.plain:
            if character == "\\t":
                spaces = 0 if tab_size <= 0 else tab_size - cell_position % tab_size
                source_offsets.extend([source_position] * spaces)
                cell_position += spaces
            else:
                source_offsets.append(source_position)
                cell_position += 1
            source_position += 1
        source_offsets.append(source_position)
        return Content(line.plain.expandtabs(tab_size)), tuple(source_offsets)

    new_text: list[Content] = []
    source_offsets: list[int] = []
    append = new_text.append
    rendered_position = 0
    source_position = 0

    for part in line.split("\\t", include_separator=True):
        part_text = part.plain
        if part_text.endswith("\\t"):
            text_before_tab = part_text[:-1]
            source_offsets.extend(
                range(source_position, source_position + len(text_before_tab))
            )
            tab_source_offset = source_position + len(text_before_tab)
            rendered_position += len(text_before_tab)
            spaces = 0 if tab_size <= 0 else tab_size - rendered_position % tab_size
            if spaces:
                part = Content(part._text[:-1] + " ", part._spans)
                if spaces > 1:
                    part = part.extend_style(spaces - 1)
            else:
                part = Content(
                    part._text[:-1],
                    [
                        Span(span.start, min(span.end, len(text_before_tab)), span.style)
                        for span in part._spans
                        if min(span.end, len(text_before_tab)) > span.start
                    ],
                )
            source_offsets.extend([tab_source_offset] * spaces)
            rendered_position += spaces
        else:
            source_offsets.extend(
                range(source_position, source_position + len(part_text))
            )
            rendered_position += len(part_text)
        source_position += len(part_text)
        append(part)

    source_offsets.append(source_position)
    return EMPTY_CONTENT.join(new_text), tuple(source_offsets)


def _pad_source_offsets(
    source_offsets: tuple[int, ...], left: int, right: int
) -> tuple[int, ...]:
    """Pad source offsets for rendered line padding."""
    if not source_offsets:
        return source_offsets
    return (
        *([source_offsets[0]] * left),
        *source_offsets,
        *([source_offsets[-1]] * right),
    )


@rich.repr.auto
class Span(NamedTuple):'''


_CONTENT_HELPER_PREVIOUS_STYLED_TAB_FRAGMENT = """\
    new_text: list[Content] = []
    source_offsets: list[int] = []
    append = new_text.append
    cell_position = 0
    source_position = 0

    for part in line.split("\\t", include_separator=True):
        part_text = part.plain
        if part_text.endswith("\\t"):
            text_before_tab = part_text[:-1]
            source_offsets.extend(
                range(source_position, source_position + len(text_before_tab))
            )
            tab_source_offset = source_position + len(text_before_tab)
            part = Content(part._text[:-1] + " ", part._spans, part._cell_length)
            cell_position += part.cell_length
            spaces = 1
            tab_remainder = cell_position % tab_size
            if tab_remainder:
                extra_spaces = tab_size - tab_remainder
                part = part.extend_style(extra_spaces)
                cell_position += extra_spaces
                spaces += extra_spaces
            source_offsets.extend([tab_source_offset] * spaces)
        else:
            source_offsets.extend(
                range(source_position, source_position + len(part_text))
            )
            cell_position += part.cell_length
        source_position += len(part_text)
        append(part)"""


_CONTENT_HELPER_STYLED_TAB_FRAGMENT_NEW = """\
    new_text: list[Content] = []
    source_offsets: list[int] = []
    append = new_text.append
    rendered_position = 0
    source_position = 0

    for part in line.split("\\t", include_separator=True):
        part_text = part.plain
        if part_text.endswith("\\t"):
            text_before_tab = part_text[:-1]
            source_offsets.extend(
                range(source_position, source_position + len(text_before_tab))
            )
            tab_source_offset = source_position + len(text_before_tab)
            rendered_position += len(text_before_tab)
            spaces = 0 if tab_size <= 0 else tab_size - rendered_position % tab_size
            if spaces:
                part = Content(part._text[:-1] + " ", part._spans)
                if spaces > 1:
                    part = part.extend_style(spaces - 1)
            else:
                part = Content(
                    part._text[:-1],
                    [
                        Span(span.start, min(span.end, len(text_before_tab)), span.style)
                        for span in part._spans
                        if min(span.end, len(text_before_tab)) > span.start
                    ],
                )
            source_offsets.extend([tab_source_offset] * spaces)
            rendered_position += spaces
        else:
            source_offsets.extend(
                range(source_position, source_position + len(part_text))
            )
            rendered_position += len(part_text)
        source_position += len(part_text)
        append(part)"""


_CONTENT_WRAP_AND_FORMAT_OLD = '''\
    def _wrap_and_format(
        self,
        width: int,
        align: TextAlign = "left",
        overflow: TextOverflow = "fold",
        no_wrap: bool = False,
        line_pad: int = 0,
        tab_size: int = 8,
        selection: Selection | None = None,
        selection_style: Style | None = None,
        post_style: Style | None = None,
        get_style: Callable[[str | Style], Style] = Style.parse,
    ) -> list[_FormattedLine]:
        """Wraps the text and applies formatting.

        Args:
            width: Desired width.
            align: Text alignment.
            overflow: Overflow method.
            no_wrap: Disabled wrapping.
            tab_size: Cell with of tabs.
            selection: Selection information or `None` if no selection.
            selection_style: Selection style, or `None` if no selection.

        Returns:
            List of formatted lines.
        """
        output_lines: list[_FormattedLine] = []

        if selection is not None:
            get_span = selection.get_span
        else:

            def get_span(y: int) -> tuple[int, int] | None:
                return None

        for y, line in enumerate(self.split(allow_blank=True)):
            if post_style is not None:
                line = line.stylize(post_style)

            if selection_style is not None and (span := get_span(y)) is not None:
                start, end = span
                if end == -1:
                    end = len(line.plain)
                line = line.stylize(selection_style, start, end)

            line = line.expand_tabs(tab_size)

            if no_wrap:
                if overflow == "fold":
                    cuts = list(range(0, line.cell_length, width))[1:]
                    new_lines = [
                        _FormattedLine(get_style, line, width, y=y, align=align)
                        for line in line.divide(cuts)
                    ]
                else:
                    line = line.truncate(width, ellipsis=overflow == "ellipsis")
                    content_line = _FormattedLine(
                        get_style, line, width, y=y, align=align
                    )
                    new_lines = [content_line]
            else:
                content_line = _FormattedLine(get_style, line, width, y=y, align=align)
                offsets = divide_line(
                    line.plain, width - line_pad * 2, fold=overflow == "fold"
                )
                divided_lines = content_line.content.divide(offsets)
                ellipsis = overflow == "ellipsis"
                divided_lines = [
                    (
                        line.truncate(width, ellipsis=ellipsis)
                        if last
                        else line.rstrip().truncate(width, ellipsis=ellipsis)
                    )
                    for last, line in loop_last(divided_lines)
                ]

                new_lines = [
                    _FormattedLine(
                        get_style,
                        content.rstrip_end(width).pad(line_pad, line_pad),
                        width,
                        offset,
                        y,
                        align=align,
                    )
                    for content, offset in zip(divided_lines, [0, *offsets])
                ]
                new_lines[-1].line_end = True

            output_lines.extend(new_lines)

        return output_lines'''

_CONTENT_WRAP_AND_FORMAT_NEW = '''\
    def _wrap_and_format(
        self,
        width: int,
        align: TextAlign = "left",
        overflow: TextOverflow = "fold",
        no_wrap: bool = False,
        line_pad: int = 0,
        tab_size: int = 8,
        selection: Selection | None = None,
        selection_style: Style | None = None,
        post_style: Style | None = None,
        get_style: Callable[[str | Style], Style] = Style.parse,
    ) -> list[_FormattedLine]:
        """Wraps the text and applies formatting.

        Args:
            width: Desired width.
            align: Text alignment.
            overflow: Overflow method.
            no_wrap: Disabled wrapping.
            tab_size: Cell with of tabs.
            selection: Selection information or `None` if no selection.
            selection_style: Selection style, or `None` if no selection.

        Returns:
            List of formatted lines.
        """
        output_lines: list[_FormattedLine] = []

        if selection is not None:
            get_span = selection.get_span
        else:

            def get_span(y: int) -> tuple[int, int] | None:
                return None

        for y, line in enumerate(self.split(allow_blank=True)):
            if post_style is not None:
                line = line.stylize(post_style)

            if selection_style is not None and (span := get_span(y)) is not None:
                start, end = span
                if end == -1:
                    end = len(line.plain)
                line = line.stylize(selection_style, start, end)

            line, source_offsets = _expand_tabs_with_offsets(line, tab_size)

            if no_wrap:
                if overflow == "fold":
                    cuts = list(range(0, line.cell_length, width))[1:]
                    divided_lines = line.divide(cuts)
                    divide_offsets = [0, *cuts, len(line.plain)]
                    divided_source_offsets = (
                        [
                            source_offsets[start : end + 1]
                            for start, end in zip(divide_offsets, divide_offsets[1:], strict=False)
                        ]
                        if source_offsets is not None
                        else [None] * len(divided_lines)
                    )
                    new_lines = [
                        _FormattedLine(
                            get_style,
                            divided_line,
                            width,
                            y=y,
                            align=align,
                            source_offsets=offset_source_offsets,
                        )
                        for divided_line, offset_source_offsets in zip(
                            divided_lines, divided_source_offsets, strict=True
                        )
                    ]
                else:
                    line = line.truncate(width, ellipsis=overflow == "ellipsis")
                    content_line = _FormattedLine(
                        get_style,
                        line,
                        width,
                        y=y,
                        align=align,
                        source_offsets=source_offsets,
                    )
                    new_lines = [content_line]
            else:
                content_line = _FormattedLine(
                    get_style,
                    line,
                    width,
                    y=y,
                    align=align,
                    source_offsets=source_offsets,
                )
                offsets = divide_line(
                    line.plain, width - line_pad * 2, fold=overflow == "fold"
                )
                divided_lines = content_line.content.divide(offsets)
                divide_offsets = [0, *offsets, len(line.plain)]
                divided_source_offsets = (
                    [
                        source_offsets[start : end + 1]
                        for start, end in zip(divide_offsets, divide_offsets[1:], strict=False)
                    ]
                    if source_offsets is not None
                    else [None] * len(divided_lines)
                )
                ellipsis = overflow == "ellipsis"
                divided_lines = [
                    (
                        line.truncate(width, ellipsis=ellipsis)
                        if last
                        else line.rstrip().truncate(width, ellipsis=ellipsis)
                    )
                    for last, line in loop_last(divided_lines)
                ]

                new_lines = []
                for content, offset, offset_source_offsets in zip(
                    divided_lines,
                    [0, *offsets],
                    divided_source_offsets,
                    strict=True,
                ):
                    content = content.rstrip_end(width)
                    if offset_source_offsets is not None:
                        offset_source_offsets = offset_source_offsets[
                            : len(content.plain) + 1
                        ]
                        offset_source_offsets = _pad_source_offsets(
                            offset_source_offsets, line_pad, line_pad
                        )
                    new_lines.append(
                        _FormattedLine(
                            get_style,
                            content.pad(line_pad, line_pad),
                            width,
                            offset,
                            y,
                            align=align,
                            source_offsets=offset_source_offsets,
                        )
                    )
                new_lines[-1].line_end = True

            output_lines.extend(new_lines)

        return output_lines'''


_FORMATTED_LINE_OLD = '''\
class _FormattedLine:
    """A line of content with additional formatting information.

    This class is used internally within Content, and you are unlikely to need it an an app.
    """

    def __init__(
        self,
        get_style: Callable[[str | Style], Style],
        content: Content,
        width: int,
        x: int = 0,
        y: int = 0,
        align: TextAlign = "left",
        line_end: bool = False,
        link_style: Style | None = None,
    ) -> None:
        self.get_style = get_style
        self.content = content
        self.width = width
        self.x = x
        self.y = y
        self.align = align
        self.line_end = line_end
        self.link_style = link_style

    @property
    def plain(self) -> str:
        return self.content.plain

    def to_strip(self, style: Style) -> tuple[list[Segment], int]:
        _Segment = Segment
        align = self.align
        width = self.width
        pad_left = pad_right = 0
        content = self.content
        x = self.x
        y = self.y
        get_style = self.get_style

        if align in ("start", "left") or (align == "justify" and self.line_end):
            pass

        elif align == "center":
            excess_space = width - self.content.cell_length
            pad_left = excess_space // 2
            pad_right = excess_space - pad_left

        elif align in ("end", "right"):
            pad_left = width - self.content.cell_length

        elif align == "justify":
            words = content.split(" ", include_separator=False)
            words_size = sum(cell_len(word.plain.rstrip(" ")) for word in words)
            num_spaces = len(words) - 1
            spaces = [1] * num_spaces
            index = 0
            if spaces:
                while words_size + num_spaces < width:
                    spaces[len(spaces) - index - 1] += 1
                    num_spaces += 1
                    index = (index + 1) % len(spaces)

            segments: list[Segment] = []
            add_segment = segments.append
            x = self.x
            for index, word in enumerate(words):
                for text, text_style in word.render(
                    style, end="", parse_style=get_style
                ):
                    add_segment(
                        _Segment(
                            text, (style + text_style).rich_style_with_offset(x, y)
                        )
                    )
                    x += len(text) + 1
                if index < len(spaces) and (pad := spaces[index]):
                    add_segment(_Segment(" " * pad, (style + text_style).rich_style))

            return segments, width

        segments = (
            [Segment(" " * pad_left, style.background_style.rich_style)]
            if pad_left
            else []
        )
        add_segment = segments.append

        for text, text_style in content.render(style, end="", parse_style=get_style):
            add_segment(
                _Segment(text, (style + text_style).rich_style_with_offset(x, y))
            )
            x += len(text)

        if pad_right:
            segments.append(
                _Segment(" " * pad_right, style.background_style.rich_style)
            )

        return (segments, content.cell_length + pad_left + pad_right)

    def _apply_link_style(
        self, link_style: RichStyle, segments: list[Segment]
    ) -> list[Segment]:
        _Segment = Segment
        segments = [
            _Segment(
                text,
                (
                    style
                    if style._meta is None
                    else (style + link_style if "@click" in style.meta else style)
                ),
                control,
            )
            for text, style, control in segments
            if style is not None
        ]
        return segments'''

_FORMATTED_LINE_NEW = '''\
class _FormattedLine:
    """A line of content with additional formatting information.

    This class is used internally within Content, and you are unlikely to need it an an app.
    """

    def __init__(
        self,
        get_style: Callable[[str | Style], Style],
        content: Content,
        width: int,
        x: int = 0,
        y: int = 0,
        align: TextAlign = "left",
        line_end: bool = False,
        link_style: Style | None = None,
        source_offsets: tuple[int, ...] | None = None,
    ) -> None:
        self.get_style = get_style
        self.content = content
        self.width = width
        self.x = x
        self.y = y
        self.align = align
        self.line_end = line_end
        self.link_style = link_style
        self.source_offsets = source_offsets

    @property
    def plain(self) -> str:
        return self.content.plain

    def to_strip(self, style: Style) -> tuple[list[Segment], int]:
        _Segment = Segment
        align = self.align
        width = self.width
        pad_left = pad_right = 0
        content = self.content
        x = self.x
        y = self.y
        get_style = self.get_style

        if align in ("start", "left") or (align == "justify" and self.line_end):
            pass

        elif align == "center":
            excess_space = width - self.content.cell_length
            pad_left = excess_space // 2
            pad_right = excess_space - pad_left

        elif align in ("end", "right"):
            pad_left = width - self.content.cell_length

        elif align == "justify":
            # Justification synthesizes word-gap cells. Chrys does not route raw
            # tabbed selectable content through justified widgets, so keep the
            # upstream rendered-offset behavior here. If this path ever needs
            # source maps, the synthetic gaps and len(text) + 1 advance below
            # need explicit map entries.
            words = content.split(" ", include_separator=False)
            words_size = sum(cell_len(word.plain.rstrip(" ")) for word in words)
            num_spaces = len(words) - 1
            spaces = [1] * num_spaces
            index = 0
            if spaces:
                while words_size + num_spaces < width:
                    spaces[len(spaces) - index - 1] += 1
                    num_spaces += 1
                    index = (index + 1) % len(spaces)

            segments: list[Segment] = []
            add_segment = segments.append
            x = self.x
            for index, word in enumerate(words):
                for text, text_style in word.render(
                    style, end="", parse_style=get_style
                ):
                    add_segment(
                        _Segment(
                            text, (style + text_style).rich_style_with_offset(x, y)
                        )
                    )
                    x += len(text) + 1
                if index < len(spaces) and (pad := spaces[index]):
                    add_segment(_Segment(" " * pad, (style + text_style).rich_style))

            return segments, width

        segments = (
            [Segment(" " * pad_left, style.background_style.rich_style)]
            if pad_left
            else []
        )
        add_segment = segments.append
        source_offsets = self.source_offsets
        content_offset = 0

        for text, text_style in content.render(style, end="", parse_style=get_style):
            offset_map: tuple[int, ...] | None = None
            if source_offsets is not None:
                end_offset = content_offset + len(text) + 1
                offset_map = source_offsets[content_offset:end_offset]
                if len(offset_map) != len(text) + 1:
                    offset_map = None
            add_segment(
                _Segment(
                    text,
                    (style + text_style).rich_style_with_offset(
                        x,
                        y,
                        offset_map=offset_map,
                    ),
                )
            )
            x += len(text)
            content_offset += len(text)

        if pad_right:
            segments.append(
                _Segment(" " * pad_right, style.background_style.rich_style)
            )

        return (segments, content.cell_length + pad_left + pad_right)

    def _apply_link_style(
        self, link_style: RichStyle, segments: list[Segment]
    ) -> list[Segment]:
        _Segment = Segment
        segments = [
            _Segment(
                text,
                (
                    style
                    if style._meta is None
                    else (style + link_style if "@click" in style.meta else style)
                ),
                control,
            )
            for text, style, control in segments
            if style is not None
        ]
        return segments'''


_STYLE_RICH_STYLE_WITH_OFFSET_OLD = '''\
    def rich_style_with_offset(self, x: int, y: int) -> RichStyle:
        """Get a Rich style with the given offset included in meta.

        This is used in text selection.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            A Rich Style object.
        """
        (
            background,
            foreground,
            bold,
            dim,
            italic,
            underline,
            underline2,
            reverse,
            strike,
            blink,
            link,
            _meta,
        ) = _get_simple_attributes(self)
        color = None if foreground is None else background + foreground
        return RichStyle(
            color=None if color is None else color.rich_color,
            bgcolor=None if background is None else background.rich_color,
            bold=bold,
            dim=dim,
            italic=italic,
            underline=underline,
            underline2=underline2,
            reverse=reverse,
            strike=strike,
            blink=blink,
            link=link,
            meta={**self.meta, "offset": (x, y)},
        )'''

_STYLE_RICH_STYLE_WITH_OFFSET_NEW = '''\
    def rich_style_with_offset(
        self, x: int, y: int, *, offset_map: Iterable[int] | None = None
    ) -> RichStyle:
        """Get a Rich style with the given offset included in meta.

        This is used in text selection.

        Args:
            x: X coordinate.
            y: Y coordinate.
            offset_map: Optional mapping from rendered character offsets to
                source offsets.

        Returns:
            A Rich Style object.
        """
        (
            background,
            foreground,
            bold,
            dim,
            italic,
            underline,
            underline2,
            reverse,
            strike,
            blink,
            link,
            _meta,
        ) = _get_simple_attributes(self)
        color = None if foreground is None else background + foreground
        meta: dict[str, Any] = {**self.meta, "offset": (x, y)}
        if offset_map is not None:
            meta["offset_map"] = tuple(offset_map)
        return RichStyle(
            color=None if color is None else color.rich_color,
            bgcolor=None if background is None else background.rich_color,
            bold=bold,
            dim=dim,
            italic=italic,
            underline=underline,
            underline2=underline2,
            reverse=reverse,
            strike=strike,
            blink=blink,
            link=link,
            meta=meta,
        )'''


_COMPOSITOR_GET_WIDGET_AND_OFFSET_AT_OLD = '''\
    def get_widget_and_offset_at(
        self, x: int, y: int
    ) -> tuple[Widget | None, Offset | None]:
        """Get the Style at the given cell, the offset within the content.

        Args:
            x: X position within the Layout.
            y: Y position within the Layout.

        Returns:
            A tuple of the widget at (x, y) and the offset within the widget.
        """
        try:
            widget, region = self.get_widget_at(x, y)
        except errors.NoWidget:
            return None, None
        if widget not in self.visible_widgets:
            return None, None

        if y >= widget.content_region.bottom:
            x, y = widget.content_region.bottom_right_inclusive

        gutter_left, gutter_right = widget.gutter.top_left
        x -= region.x + gutter_left
        y -= region.y + gutter_right

        if y < 0:
            return None, None

        if x < 0:
            return widget, Offset(0, y)

        visible_screen_stack.set(widget.app._background_screens)
        line = widget.render_line(y)

        end = 0
        start = 0
        offset_y: int | None = None
        offset_x = 0
        offset_x2 = 0

        from rich.cells import get_character_cell_size

        offset: Offset | None = None
        for segment in line:
            end += segment.cell_length
            style = segment.style
            if style is not None and style._meta is not None:
                meta = style.meta
                if "offset" in meta:
                    offset_x, offset_y = meta["offset"]
                    if offset_y is None:
                        continue
                    offset_x2 = offset_x + len(segment.text)

                    if x < end and x >= start:
                        segment_cell_length = 0
                        cell_cut = x - start
                        segment_offset = 0
                        for character in segment.text:
                            if segment_cell_length >= cell_cut:
                                break
                            segment_cell_length += get_character_cell_size(character)
                            segment_offset += 1

                        offset = Offset(offset_x + segment_offset, offset_y)
                        break
            start = end

        if offset is None and offset_y is not None:
            offset = Offset(offset_x2, offset_y)
        return widget, offset'''

_COMPOSITOR_GET_WIDGET_AND_OFFSET_AT_NEW = '''\
    def get_widget_and_offset_at(
        self, x: int, y: int
    ) -> tuple[Widget | None, Offset | None]:
        """Get the Style at the given cell, the offset within the content.

        Args:
            x: X position within the Layout.
            y: Y position within the Layout.

        Returns:
            A tuple of the widget at (x, y) and the offset within the widget.
        """
        try:
            widget, region = self.get_widget_at(x, y)
        except errors.NoWidget:
            return None, None
        if widget not in self.visible_widgets:
            return None, None

        if y >= widget.content_region.bottom:
            x, y = widget.content_region.bottom_right_inclusive

        gutter_left, gutter_right = widget.gutter.top_left
        x -= region.x + gutter_left
        y -= region.y + gutter_right

        if y < 0:
            return None, None

        if x < 0:
            return widget, Offset(0, y)

        visible_screen_stack.set(widget.app._background_screens)
        line = widget.render_line(y)

        end = 0
        start = 0
        offset_y: int | None = None
        offset_x = 0
        offset_x2 = 0

        from rich.cells import get_character_cell_size

        offset: Offset | None = None
        for segment in line:
            end += segment.cell_length
            style = segment.style
            if style is not None and style._meta is not None:
                meta = style.meta
                if "offset" in meta:
                    offset_x, offset_y = meta["offset"]
                    if offset_y is None:
                        continue
                    offset_map = meta.get("offset_map")
                    if offset_map:
                        offset_x2 = offset_map[-1]
                    else:
                        offset_map = None
                        offset_x2 = offset_x + len(segment.text)

                    if x < end and x >= start:
                        segment_cell_length = 0
                        cell_cut = x - start
                        segment_offset = 0
                        for character in segment.text:
                            if segment_cell_length >= cell_cut:
                                break
                            segment_cell_length += get_character_cell_size(character)
                            segment_offset += 1
                        if offset_map is not None:
                            segment_offset = min(segment_offset, len(offset_map) - 1)
                            offset = Offset(offset_map[segment_offset], offset_y)
                            break
                        offset = Offset(offset_x + segment_offset, offset_y)
                        break
            start = end

        if offset is None and offset_y is not None:
            offset = Offset(offset_x2, offset_y)
        return widget, offset'''


_STRIP_APPLY_OFFSETS_OLD = '''\
    def apply_offsets(self, x: int, y: int) -> Strip:
        """Apply offsets used in text selection.

        Args:
            x: Offset on X axis (column).
            y: Offset on Y axis (row).

        Returns:
            New strip.
        """
        cache_key = (x, y)
        if (cached_strip := self._offsets_cache.get(cache_key)) is not None:
            return cached_strip
        segments = self._segments
        strip_segments: list[Segment] = []
        for segment in segments:
            text, style, _ = segment
            offset_style = Style.from_meta({"offset": (x, y)})
            strip_segments.append(
                Segment(text, style + offset_style if style else offset_style)
            )
            x += len(segment.text)
        strip = Strip(strip_segments, self._cell_length)
        strip._render_cache = self._render_cache
        self._offsets_cache[cache_key] = strip
        return strip'''

_STRIP_APPLY_OFFSETS_NEW = '''\
    def apply_offsets(self, x: int, y: int) -> Strip:
        """Apply offsets used in text selection.

        Args:
            x: Offset on X axis (column).
            y: Offset on Y axis (row).

        Returns:
            New strip.
        """
        cache_key = (x, y)
        if (cached_strip := self._offsets_cache.get(cache_key)) is not None:
            return cached_strip
        segments = self._segments
        strip_segments: list[Segment] = []
        for segment in segments:
            text, style, _ = segment
            offset_meta: dict[str, object] = {"offset": (x, y)}
            if (
                style is not None
                and style._meta is not None
                and "offset_map" in style.meta
            ):
                # apply_offsets is used by line-API widgets that extract
                # selection text from rendered/cropped strips. Rebase any
                # source map to keep that rendered-offset contract.
                offset_meta["offset_map"] = tuple(range(x, x + len(text) + 1))
            offset_style = Style.from_meta(offset_meta)
            strip_segments.append(
                Segment(text, style + offset_style if style else offset_style)
            )
            x += len(segment.text)
        strip = Strip(strip_segments, self._cell_length)
        strip._render_cache = self._render_cache
        self._offsets_cache[cache_key] = strip
        return strip'''


def _expand_tabs_with_offsets(line: Any, tab_size: int) -> tuple[Any, tuple[int, ...] | None]:
    """Runtime copy of the patched Textual tab expansion helper."""
    if "\t" not in line.plain:
        return line, None

    from textual.content import EMPTY_CONTENT, Content

    if not line._spans:
        # Preserve Textual's unstyled Content.expand_tabs behavior: it delegates
        # to str.expandtabs(), whose tab stops are character-position based.
        source_offsets: list[int] = []
        source_position = 0
        cell_position = 0
        for character in line.plain:
            if character == "\t":
                spaces = 0 if tab_size <= 0 else tab_size - cell_position % tab_size
                source_offsets.extend([source_position] * spaces)
                cell_position += spaces
            else:
                source_offsets.append(source_position)
                cell_position += 1
            source_position += 1
        source_offsets.append(source_position)
        return Content(line.plain.expandtabs(tab_size)), tuple(source_offsets)

    new_text: list[Any] = []
    source_offsets: list[int] = []
    append = new_text.append
    rendered_position = 0
    source_position = 0

    for part in line.split("\t", include_separator=True):
        part_text = part.plain
        if part_text.endswith("\t"):
            text_before_tab = part_text[:-1]
            source_offsets.extend(range(source_position, source_position + len(text_before_tab)))
            tab_source_offset = source_position + len(text_before_tab)
            rendered_position += len(text_before_tab)
            spaces = 0 if tab_size <= 0 else tab_size - rendered_position % tab_size
            if spaces:
                part = Content(part._text[:-1] + " ", part._spans)
                if spaces > 1:
                    part = part.extend_style(spaces - 1)
            else:
                from textual.content import Span

                part = Content(
                    part._text[:-1],
                    [
                        Span(span.start, min(span.end, len(text_before_tab)), span.style)
                        for span in part._spans
                        if min(span.end, len(text_before_tab)) > span.start
                    ],
                )
            source_offsets.extend([tab_source_offset] * spaces)
            rendered_position += spaces
        else:
            source_offsets.extend(range(source_position, source_position + len(part_text)))
            rendered_position += len(part_text)
        source_position += len(part_text)
        append(part)

    source_offsets.append(source_position)
    return EMPTY_CONTENT.join(new_text), tuple(source_offsets)


def _pad_source_offsets(source_offsets: tuple[int, ...], left: int, right: int) -> tuple[int, ...]:
    """Pad source offsets for rendered line padding."""
    if not source_offsets:
        return source_offsets
    return (
        *([source_offsets[0]] * left),
        *source_offsets,
        *([source_offsets[-1]] * right),
    )


def apply_runtime_patch() -> None:
    """Patch Textual's loaded selection/rendering classes in the current process."""
    try:
        import textual._compositor as compositor_mod
        import textual.content as content_mod
        import textual.strip as strip_mod
        import textual.style as style_mod
    except ImportError:
        return

    # Check compatibility before dereferencing any of Textual's private APIs;
    # a future version may rename or remove them entirely.
    if not _runtime_compositor_patch_compatible():
        logger.warning(
            "Skipping Textual tab-selection runtime patch: loaded Textual is not the pinned %s that the patch targets.",
            _RUNTIME_PATCH_TEXTUAL_VERSION,
        )
        return

    # Runtime reapplication is cheap; only skip when every patched target has
    # our marker so a partial external monkey patch can still be repaired.
    patch_targets = (
        getattr(content_mod, "_expand_tabs_with_offsets", None),
        content_mod._FormattedLine.__init__,
        content_mod._FormattedLine.to_strip,
        content_mod.Content._wrap_and_format,
        style_mod.Style.rich_style_with_offset,
        strip_mod.Strip.apply_offsets,
        compositor_mod.Compositor.get_widget_and_offset_at,
    )
    if all(getattr(target, _RUNTIME_PATCH_MARKER, False) for target in patch_targets):
        return

    def rich_style_with_offset(self: Any, x: int, y: int, *, offset_map: Any = None) -> RichStyle:
        (
            background,
            foreground,
            bold,
            dim,
            italic,
            underline,
            underline2,
            reverse,
            strike,
            blink,
            link,
            _meta,
        ) = style_mod._get_simple_attributes(self)
        color = None if foreground is None else background + foreground
        meta = {**self.meta, "offset": (x, y)}
        if offset_map is not None:
            meta["offset_map"] = tuple(offset_map)
        return RichStyle(
            color=None if color is None else color.rich_color,
            bgcolor=None if background is None else background.rich_color,
            bold=bold,
            dim=dim,
            italic=italic,
            underline=underline,
            underline2=underline2,
            reverse=reverse,
            strike=strike,
            blink=blink,
            link=link,
            meta=meta,
        )

    def formatted_line_init(
        self: Any,
        get_style: Any,
        content: Any,
        width: int,
        x: int = 0,
        y: int = 0,
        align: str = "left",
        line_end: bool = False,
        link_style: Any = None,
        source_offsets: tuple[int, ...] | None = None,
    ) -> None:
        self.get_style = get_style
        self.content = content
        self.width = width
        self.x = x
        self.y = y
        self.align = align
        self.line_end = line_end
        self.link_style = link_style
        self.source_offsets = source_offsets

    def formatted_line_to_strip(self: Any, style: Any) -> tuple[list[Segment], int]:
        _Segment = Segment
        align = self.align
        width = self.width
        pad_left = pad_right = 0
        content = self.content
        x = self.x
        y = self.y
        get_style = self.get_style

        if align in ("start", "left") or (align == "justify" and self.line_end):
            pass

        elif align == "center":
            excess_space = width - self.content.cell_length
            pad_left = excess_space // 2
            pad_right = excess_space - pad_left

        elif align in ("end", "right"):
            pad_left = width - self.content.cell_length

        elif align == "justify":
            # Justification synthesizes word-gap cells. Chrys does not route raw
            # tabbed selectable content through justified widgets, so keep the
            # upstream rendered-offset behavior here. If this path ever needs
            # source maps, the synthetic gaps and len(text) + 1 advance below
            # need explicit map entries.
            words = content.split(" ", include_separator=False)
            words_size = sum(content_mod.cell_len(word.plain.rstrip(" ")) for word in words)
            num_spaces = len(words) - 1
            spaces = [1] * num_spaces
            index = 0
            if spaces:
                while words_size + num_spaces < width:
                    spaces[len(spaces) - index - 1] += 1
                    num_spaces += 1
                    index = (index + 1) % len(spaces)

            segments: list[Segment] = []
            add_segment = segments.append
            x = self.x
            for index, word in enumerate(words):
                for text, text_style in word.render(style, end="", parse_style=get_style):
                    add_segment(_Segment(text, (style + text_style).rich_style_with_offset(x, y)))
                    x += len(text) + 1
                if index < len(spaces) and (pad := spaces[index]):
                    add_segment(_Segment(" " * pad, (style + text_style).rich_style))

            return segments, width

        segments = [_Segment(" " * pad_left, style.background_style.rich_style)] if pad_left else []
        add_segment = segments.append
        source_offsets = self.source_offsets
        content_offset = 0

        for text, text_style in content.render(style, end="", parse_style=get_style):
            offset_map = None
            if source_offsets is not None:
                end_offset = content_offset + len(text) + 1
                offset_map = source_offsets[content_offset:end_offset]
                if len(offset_map) != len(text) + 1:
                    offset_map = None
            add_segment(
                _Segment(
                    text,
                    (style + text_style).rich_style_with_offset(x, y, offset_map=offset_map),
                )
            )
            x += len(text)
            content_offset += len(text)

        if pad_right:
            segments.append(_Segment(" " * pad_right, style.background_style.rich_style))

        return (segments, content.cell_length + pad_left + pad_right)

    def wrap_and_format(
        self: Any,
        width: int,
        align: str = "left",
        overflow: str = "fold",
        no_wrap: bool = False,
        line_pad: int = 0,
        tab_size: int = 8,
        selection: Any = None,
        selection_style: Any = None,
        post_style: Any = None,
        get_style: Any = style_mod.Style.parse,
    ) -> list[Any]:
        output_lines: list[Any] = []

        if selection is not None:
            get_span = selection.get_span
        else:

            def get_span(y: int) -> tuple[int, int] | None:
                return None

        for y, line in enumerate(self.split(allow_blank=True)):
            if post_style is not None:
                line = line.stylize(post_style)

            if selection_style is not None and (span := get_span(y)) is not None:
                start, end = span
                if end == -1:
                    end = len(line.plain)
                line = line.stylize(selection_style, start, end)

            line, source_offsets = _expand_tabs_with_offsets(line, tab_size)

            if no_wrap:
                if overflow == "fold":
                    cuts = list(range(0, line.cell_length, width))[1:]
                    divided_lines = line.divide(cuts)
                    divide_offsets = [0, *cuts, len(line.plain)]
                    divided_source_offsets = (
                        [source_offsets[start : end + 1] for start, end in pairwise(divide_offsets)]
                        if source_offsets is not None
                        else [None] * len(divided_lines)
                    )
                    new_lines = [
                        content_mod._FormattedLine(
                            get_style,
                            divided_line,
                            width,
                            y=y,
                            align=align,
                            source_offsets=offset_source_offsets,
                        )
                        for divided_line, offset_source_offsets in zip(
                            divided_lines,
                            divided_source_offsets,
                            strict=True,
                        )
                    ]
                else:
                    line = line.truncate(width, ellipsis=overflow == "ellipsis")
                    content_line = content_mod._FormattedLine(
                        get_style,
                        line,
                        width,
                        y=y,
                        align=align,
                        source_offsets=source_offsets,
                    )
                    new_lines = [content_line]
            else:
                content_line = content_mod._FormattedLine(
                    get_style,
                    line,
                    width,
                    y=y,
                    align=align,
                    source_offsets=source_offsets,
                )
                offsets = content_mod.divide_line(line.plain, width - line_pad * 2, fold=overflow == "fold")
                divided_lines = content_line.content.divide(offsets)
                divide_offsets = [0, *offsets, len(line.plain)]
                divided_source_offsets = (
                    [source_offsets[start : end + 1] for start, end in pairwise(divide_offsets)]
                    if source_offsets is not None
                    else [None] * len(divided_lines)
                )
                ellipsis = overflow == "ellipsis"
                divided_lines = [
                    (
                        line.truncate(width, ellipsis=ellipsis)
                        if last
                        else line.rstrip().truncate(width, ellipsis=ellipsis)
                    )
                    for last, line in content_mod.loop_last(divided_lines)
                ]

                new_lines = []
                for content, offset, offset_source_offsets in zip(
                    divided_lines,
                    [0, *offsets],
                    divided_source_offsets,
                    strict=True,
                ):
                    content = content.rstrip_end(width)
                    if offset_source_offsets is not None:
                        offset_source_offsets = offset_source_offsets[: len(content.plain) + 1]
                        offset_source_offsets = _pad_source_offsets(offset_source_offsets, line_pad, line_pad)
                    new_lines.append(
                        content_mod._FormattedLine(
                            get_style,
                            content.pad(line_pad, line_pad),
                            width,
                            offset,
                            y,
                            align=align,
                            source_offsets=offset_source_offsets,
                        )
                    )
                new_lines[-1].line_end = True

            output_lines.extend(new_lines)

        return output_lines

    def apply_offsets(self: Any, x: int, y: int) -> Any:
        cache_key = (x, y)
        if (cached_strip := self._offsets_cache.get(cache_key)) is not None:
            return cached_strip
        segments = self._segments
        strip_segments: list[Segment] = []
        for segment in segments:
            text, style, _ = segment
            offset_meta: dict[str, object] = {"offset": (x, y)}
            if style is not None and style._meta is not None and "offset_map" in style.meta:
                # apply_offsets is used by line-API widgets that extract selection
                # text from rendered/cropped strips. Rebase any source map to keep
                # that rendered-offset contract.
                offset_meta["offset_map"] = tuple(range(x, x + len(text) + 1))
            offset_style = strip_mod.Style.from_meta(offset_meta)
            strip_segments.append(Segment(text, style + offset_style if style else offset_style))
            x += len(segment.text)
        strip = strip_mod.Strip(strip_segments, self._cell_length)
        strip._render_cache = self._render_cache
        self._offsets_cache[cache_key] = strip
        return strip

    def get_widget_and_offset_at(self: Any, x: int, y: int) -> tuple[Any | None, Any | None]:
        try:
            widget, region = self.get_widget_at(x, y)
        except compositor_mod.errors.NoWidget:
            return None, None
        if widget not in self.visible_widgets:
            return None, None

        if y >= widget.content_region.bottom:
            x, y = widget.content_region.bottom_right_inclusive

        gutter_left, gutter_right = widget.gutter.top_left
        x -= region.x + gutter_left
        y -= region.y + gutter_right

        if y < 0:
            return None, None

        if x < 0:
            return widget, compositor_mod.Offset(0, y)

        compositor_mod.visible_screen_stack.set(widget.app._background_screens)
        line = widget.render_line(y)

        end = 0
        start = 0
        offset_y: int | None = None
        offset_x = 0
        offset_x2 = 0

        from rich.cells import get_character_cell_size

        offset = None
        for segment in line:
            end += segment.cell_length
            style = segment.style
            if style is not None and style._meta is not None:
                meta = style.meta
                if "offset" in meta:
                    offset_x, offset_y = meta["offset"]
                    if offset_y is None:
                        continue
                    offset_map = meta.get("offset_map")
                    if offset_map:
                        offset_x2 = offset_map[-1]
                    else:
                        offset_map = None
                        offset_x2 = offset_x + len(segment.text)

                    if x < end and x >= start:
                        segment_cell_length = 0
                        cell_cut = x - start
                        segment_offset = 0
                        for character in segment.text:
                            if segment_cell_length >= cell_cut:
                                break
                            segment_cell_length += get_character_cell_size(character)
                            segment_offset += 1
                        if offset_map is not None:
                            segment_offset = min(segment_offset, len(offset_map) - 1)
                            offset = compositor_mod.Offset(offset_map[segment_offset], offset_y)
                            break
                        offset = compositor_mod.Offset(offset_x + segment_offset, offset_y)
                        break
            start = end

        if offset is None and offset_y is not None:
            offset = compositor_mod.Offset(offset_x2, offset_y)
        return widget, offset

    patched_targets = (
        rich_style_with_offset,
        _expand_tabs_with_offsets,
        formatted_line_init,
        formatted_line_to_strip,
        wrap_and_format,
        apply_offsets,
        get_widget_and_offset_at,
    )
    for target in patched_targets:
        setattr(target, _RUNTIME_PATCH_MARKER, True)

    style_mod.Style.rich_style_with_offset = rich_style_with_offset
    content_mod._expand_tabs_with_offsets = _expand_tabs_with_offsets
    content_mod._FormattedLine.__init__ = formatted_line_init
    content_mod._FormattedLine.to_strip = formatted_line_to_strip
    content_mod.Content._wrap_and_format = wrap_and_format
    strip_mod.Strip.apply_offsets = apply_offsets
    compositor_mod.Compositor.get_widget_and_offset_at = get_widget_and_offset_at
    setattr(content_mod.Content, _RUNTIME_PATCH_MARKER, True)


register(
    FilePatch(
        package="textual",
        module_file="content.py",
        old_fragment=_CONTENT_HELPER_OLD,
        new_fragment=_CONTENT_HELPER_NEW,
        description="Add tab expansion source-offset mapping helper",
        equivalent_fragments=(_CONTENT_HELPER_PREVIOUS_STYLED_TAB_FRAGMENT,),
    ),
    FilePatch(
        package="textual",
        module_file="content.py",
        old_fragment=_CONTENT_HELPER_PREVIOUS_STYLED_TAB_FRAGMENT,
        new_fragment=_CONTENT_HELPER_STYLED_TAB_FRAGMENT_NEW,
        description="Upgrade styled tab expansion source-offset mapping",
    ),
    FilePatch(
        package="textual",
        module_file="content.py",
        old_fragment=_CONTENT_WRAP_AND_FORMAT_OLD,
        new_fragment=_CONTENT_WRAP_AND_FORMAT_NEW,
        description="Preserve tab source offsets through Textual Content wrapping",
    ),
    FilePatch(
        package="textual",
        module_file="content.py",
        old_fragment=_FORMATTED_LINE_OLD,
        new_fragment=_FORMATTED_LINE_NEW,
        description="Emit Textual tab source offset maps from formatted lines",
    ),
    FilePatch(
        package="textual",
        module_file="style.py",
        old_fragment=_STYLE_RICH_STYLE_WITH_OFFSET_OLD,
        new_fragment=_STYLE_RICH_STYLE_WITH_OFFSET_NEW,
        description="Accept source offset maps in Textual selection styles",
    ),
    FilePatch(
        package="textual",
        module_file="_compositor.py",
        old_fragment=_COMPOSITOR_GET_WIDGET_AND_OFFSET_AT_OLD,
        new_fragment=_COMPOSITOR_GET_WIDGET_AND_OFFSET_AT_NEW,
        description="Map Textual hit-test cells through tab source offsets",
    ),
    FilePatch(
        package="textual",
        module_file="strip.py",
        old_fragment=_STRIP_APPLY_OFFSETS_OLD,
        new_fragment=_STRIP_APPLY_OFFSETS_NEW,
        description="Rebase Textual offset maps when applying strip offsets",
    ),
)
