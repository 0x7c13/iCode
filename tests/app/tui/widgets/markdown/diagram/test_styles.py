# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cell styles survive wide glyphs, viewport cropping and geometry-preserving updates."""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from rich.style import Style
from textual.app import App, ComposeResult
from textual.geometry import Offset

from chrys.app.tui.widgets.markdown.diagram.canvas import CellStyleSpan, TerminalCanvas
from chrys.app.tui.widgets.markdown.diagram.layout import compile_ir, compile_ir_with_geometry
from chrys.app.tui.widgets.markdown.diagram.model import DiagramEdge, DiagramIR, DiagramKind, DiagramNode, Direction
from chrys.app.tui.widgets.markdown.diagram.viewer import DiagramViewport
from tests.support.tui_helpers import resize_when_settled
from tests.support.waiting import wait_for


def test_style_plane_uses_leading_cell_and_preserves_wide_glyphs() -> None:
    canvas = TerminalCanvas()
    canvas.draw_text(0, 0, "A界e\u0301Z")
    rows = canvas.rows(6, 1)
    canvas.style_span(0, 1, 3, Style(color="green"))
    canvas.style_span(0, 2, 4, Style(bold=True))
    segments = canvas.render_line(0, 5)
    assert "".join(segment.text for segment in segments) == "A界e\u0301Z"
    wide = next(segment for segment in segments if segment.text == "界")
    assert wide.style == Style(color="green")
    assert canvas.rows(6, 1) == rows
    cropped = canvas.render_line(0, 3, x=2)
    assert "".join(segment.text for segment in cropped) == " e\u0301Z"
    assert cropped[0].style == Style(color="green")
    assert cell_len("".join(segment.text for segment in cropped)) == 3


@pytest.mark.parametrize("row", ["A   ", "    "], ids=["trailing-blanks", "blank-row"])
@pytest.mark.parametrize("start", [0, 1], ids=["whole-row", "blank-crop"])
def test_style_plane_preserves_spans_on_trimmed_blanks(row: str, start: int) -> None:
    canvas = TerminalCanvas()
    canvas.draw_text(0, 0, row)
    base = Style(color="white", bgcolor="blue")
    red = base + Style(bgcolor="red")
    bold = Style(bold=True)
    canvas.style_span(0, 0, 4, Style(bgcolor="red"))
    canvas.style_span(0, 2, 5, bold)

    assert canvas.rows(6, 1) == (row.rstrip(),)
    segments = canvas.render_line(0, 6 - start, x=start, style=base)

    assert "".join(segment.text for segment in segments) == row.ljust(6)[start:]
    assert [segment.style for segment in segments for _ in segment.text] == [
        red,
        red,
        red + bold,
        red + bold,
        base + bold,
        base,
    ][start:]


@pytest.mark.parametrize("direction", list(Direction))
def test_geometry_compile_preserves_existing_mermaid_contract(direction: Direction) -> None:
    ir = DiagramIR(
        DiagramKind.FLOWCHART,
        direction,
        (DiagramNode("a", "界", min_width=24), DiagramNode("b", "Next")),
        (DiagramEdge("a", "b"),),
        title="Example",
    )
    compiled, geometry = compile_ir_with_geometry("source", ir)
    assert compiled == compile_ir("source", ir)
    assert set(geometry) == {"a", "b"}
    assert geometry["a"].width >= 24
    for box in geometry.values():
        assert box.x >= 0 and box.y >= 0
        assert box.x + box.width <= compiled.width
        assert box.y + box.height <= compiled.height
        assert box.node.label in "\n".join(compiled.rows[box.y : box.y + box.height])


async def test_update_styles_preserves_scroll_canvas_geometry_through_resize() -> None:
    ir = DiagramIR(
        DiagramKind.FLOWCHART,
        Direction.TOP_DOWN,
        tuple(DiagramNode(str(i), f"Wide 界 node {i}" + " x" * 20) for i in range(12)),
        tuple(DiagramEdge(str(i), str(i + 1)) for i in range(11)),
    )
    compiled, geometry = compile_ir_with_geometry("", ir)
    original_geometry = dict(geometry)

    class Harness(App):
        def compose(self) -> ComposeResult:
            yield DiagramViewport(compiled)

    app = Harness()
    async with app.run_test(size=(30, 10)) as pilot:
        viewport = app.query_one(DiagramViewport)
        viewport.scroll_to(x=4, y=3, animate=False, immediate=True, force=True)
        await wait_for(lambda: viewport.scroll_offset.x == 4 and viewport.scroll_offset.y == 3, pilot=pilot)
        scroll = viewport.scroll_offset
        viewport.update_styles(
            {y: (CellStyleSpan(0, compiled.width, Style(color="red")),) for y in range(compiled.height)}
        )
        assert viewport.diagram is compiled
        assert viewport.scroll_offset == scroll
        assert geometry == original_geometry
        strip = viewport.render_line(0)
        assert strip.cell_length == viewport.scrollable_content_region.width
        assert any(segment.style and segment.style.color == Style(color="red").color for segment in strip)
        await resize_when_settled(pilot, 35, 12)
        assert viewport.diagram.rows is compiled.rows
        assert viewport.scroll_offset == scroll
        assert viewport.render_line(0).cell_length == viewport.scrollable_content_region.width


async def test_centered_viewport_keeps_canvas_coordinates_when_resized() -> None:
    ir = DiagramIR(DiagramKind.FLOWCHART, Direction.TOP_DOWN, (DiagramNode("a", "Centered"),), ())
    compiled = compile_ir("", ir)

    class Harness(App):
        def compose(self) -> ComposeResult:
            yield DiagramViewport(compiled, center_diagram=True)

    app = Harness()
    async with app.run_test(size=(50, 20)) as pilot:
        viewport = app.query_one(DiagramViewport)
        await wait_for(lambda: viewport.diagram_origin.x > 0 and viewport.diagram_origin.y > 0, pilot=pilot)
        origin = viewport.diagram_origin
        assert viewport.render_line(origin.y).text == (" " * origin.x + compiled.rows[0]).ljust(
            viewport.scrollable_content_region.width
        )
        assert not viewport.render_line(origin.y - 1).text.strip()
        await pilot.resize_terminal(8, 3)
        await wait_for(lambda: viewport.diagram_origin == Offset(0, 0), pilot=pilot)
        assert viewport.render_line(0).text == compiled.crop_plain_row(0, 0, viewport.scrollable_content_region.width)
