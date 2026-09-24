# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the scrollable terminal diagram dialog."""

from __future__ import annotations

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.containers import VerticalGroup
from textual.geometry import Size
from textual.scroll_view import ScrollView
from textual.widgets import Static

from chrys.app.tui.screens.dialogs.diagram import DiagramDialog
from chrys.app.tui.widgets.markdown.diagram.model import CompiledDiagram, DiagramKind
from chrys.app.tui.widgets.markdown.diagram.viewer import DiagramViewport


class _HostApp(App[None]):
    def compose(self) -> ComposeResult:
        yield Static("host", markup=False)


def _large_diagram() -> CompiledDiagram:
    rows = tuple(f"row {index:02} " + str(index % 10) * 90 for index in range(60))
    return CompiledDiagram(
        source="flowchart LR\n    A[界面] --> B[Worker]",
        kind=DiagramKind.FLOWCHART,
        width=97,
        height=len(rows),
        rows=rows,
    )


@pytest.mark.asyncio
async def test_dialog_autofocuses_natural_two_axis_viewport() -> None:
    diagram = _large_diagram()
    dialog = DiagramDialog(diagram)

    async with _HostApp().run_test(size=(52, 16)) as pilot:
        await pilot.app.push_screen(dialog)
        await pilot.pause()

        viewport = dialog.query_one(DiagramViewport)
        assert viewport.parent is dialog.query_one("#diagram-container")
        assert list(dialog.query(ScrollView)) == [viewport]
        assert pilot.app.focused is viewport
        assert viewport.virtual_size == Size(diagram.width, diagram.height)
        assert viewport.show_horizontal_scrollbar is True
        assert viewport.show_vertical_scrollbar is True

        container = dialog.query_one("#diagram-container", VerticalGroup)
        assert container.border_subtitle is not None
        assert container.border_subtitle == "Press Space to switch source/rendered view · Press c to copy"
        assert not dialog.query("#diagram-hint")


@pytest.mark.asyncio
async def test_viewport_renders_cell_crop_at_both_scroll_offsets() -> None:
    rows = ("界0123456789abcdefghijklmnop", "second row", *(f"line {index}" for index in range(30)))
    diagram = CompiledDiagram(
        source="flowchart LR\n    A[界] --> B",
        kind=DiagramKind.FLOWCHART,
        width=80,
        height=len(rows),
        rows=rows,
    )
    dialog = DiagramDialog(diagram)

    async with _HostApp().run_test(size=(24, 10)) as pilot:
        await pilot.app.push_screen(dialog)
        await pilot.pause()
        viewport = dialog.query_one(DiagramViewport)

        viewport.scroll_to(x=1, y=0, animate=False, force=True, immediate=True)
        assert viewport.render_line(0).text.startswith(" 012345")

        viewport.scroll_to(x=0, y=1, animate=False, force=True, immediate=True)
        assert viewport.render_line(0).text.startswith("second row")


@pytest.mark.asyncio
async def test_native_scrollbar_drag_moves_diagram_in_both_axes() -> None:
    dialog = DiagramDialog(_large_diagram())

    async with _HostApp().run_test(size=(52, 16)) as pilot:
        await pilot.app.push_screen(dialog)
        await pilot.pause()
        viewport = dialog.query_one(DiagramViewport)

        x = viewport.size.width - 1
        await pilot._post_mouse_events([events.MouseDown], viewport, offset=(x, 1), button=1)
        await pilot._post_mouse_events([events.MouseMove], viewport, offset=(x, viewport.size.height - 2), button=1)
        await pilot._post_mouse_events([events.MouseUp], viewport, offset=(x, viewport.size.height - 2), button=1)
        assert viewport.scroll_offset.y > 0

        y = viewport.size.height - 1
        await pilot._post_mouse_events([events.MouseDown], viewport, offset=(1, y), button=1)
        await pilot._post_mouse_events([events.MouseMove], viewport, offset=(viewport.size.width - 2, y), button=1)
        await pilot._post_mouse_events([events.MouseUp], viewport, offset=(viewport.size.width - 2, y), button=1)
        assert viewport.scroll_offset.x > 0


@pytest.mark.asyncio
async def test_copy_source_and_close_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    diagram = _large_diagram()
    copied: list[str] = []
    monkeypatch.setattr(
        "chrys.app.tui.screens.dialogs.diagram.copy_text_to_clipboards",
        lambda _app, source: copied.append(source),
    )
    dialog = DiagramDialog(diagram)

    async with _HostApp().run_test(size=(52, 16)) as pilot:
        await pilot.app.push_screen(dialog)
        await pilot.pause()
        viewport = dialog.query_one(DiagramViewport)

        viewport.scroll_to(x=10, y=10, animate=False, force=True, immediate=True)
        await pilot.press("space")
        await pilot.pause()
        assert viewport.diagram is not diagram
        assert viewport.diagram.source == diagram.source
        assert viewport.diagram.rows == (" flowchart LR", "     A[界面] --> B[Worker]")
        assert viewport.virtual_size == Size(27, 2)
        for row in range(viewport.diagram.height):
            padded_row = viewport.diagram.crop_plain_row(row, 0, viewport.diagram.width)
            assert padded_row.startswith(" ")
            assert padded_row.endswith(" ")
        assert viewport.scroll_offset == (0, 0)

        await pilot.press("space")
        await pilot.pause()
        assert viewport.diagram is diagram
        assert viewport.virtual_size == Size(diagram.width, diagram.height)

        await pilot.press("c")
        assert copied == [diagram.source]

        await pilot.press("q")
        await pilot.pause()
        assert dialog not in pilot.app.screen_stack
