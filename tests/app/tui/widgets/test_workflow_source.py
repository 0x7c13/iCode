# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The workflow source view replays its rendering per width, identical to a fresh ``Syntax``."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal

import pytest
from rich.console import Console, ConsoleDimensions, ConsoleOptions
from rich.syntax import Syntax
from rich.text import Text

from chrys.app.tui.util.source_text import sanitize_source_text
from chrys.app.tui.widgets.workflow.source import WorkflowSourceSyntax

if TYPE_CHECKING:
    from rich.segment import Segment

_SOURCE = (
    "# 工作流: a wide-character comment\n"
    "from chrys.workflows import workflow\n"
    "\n"
    "\n"
    "@workflow\n"
    "def run(ctx):\n"
    "\tvalue = '\x1b[31mred\x9b'  # control characters\r\n"
    f"    return ctx.ask({'x' * 150!r})\n"
    "# no trailing newline"
).encode()

type _ColorSystem = Literal["standard", "256", "truecolor", "windows"] | None

# A value for every ConsoleOptions field that differs from the base options. A
# field Rich adds must be reviewed here: if Syntax reads it, the render key must too.
_OPTION_VARIANTS: dict[str, Any] = {
    "size": ConsoleDimensions(31, 7),
    "legacy_windows": True,
    "min_width": 3,
    "max_width": 47,
    "is_terminal": False,
    "encoding": "ascii",
    "max_height": 4,
    "justify": "right",
    "overflow": "ellipsis",
    "no_wrap": True,
    "highlight": True,
    "markup": False,
    "height": 5,
}


def _console(color_system: _ColorSystem) -> Console:
    return Console(width=80, height=25, color_system=color_system, force_terminal=True, _environ={})


def _plain(source: bytes) -> Syntax:
    code = sanitize_source_text(source.decode("utf-8", errors="replace"))
    return Syntax(code, "python", line_numbers=True, word_wrap=False)


def _render(console: Console, renderable: Syntax, options: ConsoleOptions) -> list[Segment]:
    return list(console.render(renderable, options))


@pytest.mark.parametrize("color_system", ["standard", "256", "truecolor", "windows", None])
def test_replayed_rendering_matches_a_fresh_syntax(color_system: _ColorSystem) -> None:
    console = _console(color_system)
    view = WorkflowSourceSyntax(_SOURCE)
    assert view.code == _plain(_SOURCE).code and view.source == _SOURCE

    for width in (1, 12, 40, 138, 139, 200):
        base = console.options.update_width(width)
        for options in (base, base.update(height=3), base.update(no_wrap=True)):
            expected = _render(console, _plain(_SOURCE), options)
            assert _render(console, view, options) == expected
            assert _render(console, view, options) == expected


def test_render_key_covers_every_option_and_color_system_syntax_reads() -> None:
    assert {field.name for field in dataclasses.fields(ConsoleOptions)} == set(_OPTION_VARIANTS)
    view = WorkflowSourceSyntax(_SOURCE)
    console = _console("truecolor")
    base = console.options.update_width(60)

    for name, value in _OPTION_VARIANTS.items():
        # Render the base options first, so a variant the key does not tell apart replays them.
        assert _render(console, view, base) == _render(console, _plain(_SOURCE), base)
        variant = dataclasses.replace(base, **{name: value})
        assert _render(console, view, variant) == _render(console, _plain(_SOURCE), variant), name

    color_systems: tuple[_ColorSystem, ...] = ("standard", "256", "windows", None)
    for color_system in color_systems:
        assert _render(console, view, base) == _render(console, _plain(_SOURCE), base)
        other = _console(color_system)
        assert _render(other, view, base) == _render(other, _plain(_SOURCE), base), color_system


def test_source_is_lexed_once_per_recent_width(monkeypatch: pytest.MonkeyPatch) -> None:
    lexed: list[str] = []
    highlight = Syntax.highlight

    def counting_highlight(self: Syntax, code: str, line_range: tuple[int | None, int | None] | None = None) -> Text:
        lexed.append(code)
        return highlight(self, code, line_range)

    monkeypatch.setattr(Syntax, "highlight", counting_highlight)
    console = _console("truecolor")
    view = WorkflowSourceSyntax(_SOURCE)

    # A layout pass alternates between the widths with and without the scrollbar.
    for width in (139, 138, 139, 138, 139):
        _render(console, view, console.options.update_width(width))
    assert len(lexed) == 2
    # A third width replaces the least recently rendered one.
    for width, total in ((100, 3), (139, 3), (138, 4)):
        _render(console, view, console.options.update_width(width))
        assert len(lexed) == total, width
