# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Cell-exact invariant: the scroll fast path repaints everything that changed.

ChatPanel schedules no cleanup repaint after manual scrolling (the old
debounced full ``refresh()`` was removed once this invariant held).  This
test is the tripwire that keeps that removal safe: for every partial
compositor update produced while scrolling a populated transcript, every
terminal cell whose ground-truth content changed must be covered by an
emitted ``ChopsUpdate`` span with a real strip.  A miss here is exactly the
class of artifact the old workaround papered over (stale rows in
margins/blank bands).

Run this when bumping Textual (upgrade playbook) or when a new transcript
widget changes arrange/render behavior.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from textual._compositor import ChopsUpdate, Compositor
from textual.app import App, ComposeResult

try:
    # Added by Chrys' CJK compositor site-package patch (applied to the venv by
    # an app run's ``patches.apply_all()``).  Stock Textual — e.g. a fresh CI
    # install — has no merged-chop sentinel: its chops only hold ``Strip | None``.
    from textual._compositor import _MERGED_STRIP
except ImportError:
    _MERGED_STRIP = object()

from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.service.tools.kinds import KIND_FILESYSTEM_READ, KIND_SEARCH, KIND_SHELL
from tests.support.tui_helpers import _simulate_chat_panel_user_scroll_y

WIDTH, HEIGHT = 100, 30
TURNS = 4

_CJK_PARAGRAPH = (
    "中文段落:滚动残留检测用的宽字符文本,混排 English words、标点符号,"
    "以及跨行换行的长句子——用来覆盖 compositor 的宽字符切分路径。"
)


class FastPathApp(App):
    def compose(self) -> ComposeResult:
        yield ChatPanel()


def _agent_markdown(turn: int) -> str:
    lines = [f"## Turn {turn} summary", ""]
    for index in range(12):
        lines.append(f"- item {turn}.{index}: some explanatory prose that wraps at narrow widths")
    lines += ["", _CJK_PARAGRAPH, "", "```python", *(f"value_{i} = {i}  # turn {turn}" for i in range(8)), "```"]
    return "\n".join(lines)


async def _populate(panel: ChatPanel) -> None:
    panel.set_tool_kinds({"read_file": KIND_FILESYSTEM_READ, "grep": KIND_SEARCH, "shell": KIND_SHELL})
    for turn in range(TURNS):
        await panel.add_user_message(f"Request {turn}: 检查模块 {turn} 并总结。")
        await panel.add_agent_message(f"Turn {turn}: inspecting files.", is_final=False)
        for tool, kind in (("read_file", KIND_FILESYSTEM_READ), ("grep", KIND_SEARCH), ("shell", KIND_SHELL)):
            call_id = f"{tool}-{turn}"
            await panel.add_tool_start(call_id, tool, kind, args={"target": f"src/module_{turn}.py"})
            await panel.add_tool_result(call_id, tool, "\n".join(f"{tool} output line {i}" for i in range(6)), 12)
        await panel.add_agent_message(_agent_markdown(turn), is_final=True)


def _strips_to_cells(strips: list, width: int, height: int) -> list[list[tuple[str, str]]]:
    """Expand strips to a width-by-height grid of (glyph, style) cells."""
    grid: list[list[tuple[str, str]]] = []
    for strip in strips:
        row: list[tuple[str, str]] = []
        for segment in strip:
            style = str(segment.style) if segment.style is not None else ""
            for ch in segment.text:
                row.append((ch, style))
                if cell_len(ch) == 2:
                    row.append(("\x00wide", style))
        while len(row) < width:
            row.append((" ", "<pad>"))
        grid.append(row[:width])
    while len(grid) < height:
        grid.append([(" ", "<pad>")] * width)
    return grid


@pytest.mark.asyncio
async def test_scroll_fast_path_repaints_every_changed_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    async with FastPathApp().run_test(size=(WIDTH, HEIGHT)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await _populate(panel)
        await pilot.pause()
        await pilot.pause()

        compositor = pilot.app.screen._compositor
        prev_grid = _strips_to_cells(compositor.render_strips(), WIDTH, HEIGHT)
        partials = 0
        stale_reports: list[str] = []

        orig_partial = Compositor.render_partial_update
        orig_full = Compositor.render_full_update

        def patched_partial(self: Compositor):
            nonlocal prev_grid, partials
            update = orig_partial(self)
            curr = _strips_to_cells(self.render_strips(), WIDTH, HEIGHT)
            prev = prev_grid
            prev_grid = curr
            changed = [(x, y) for y in range(HEIGHT) for x in range(WIDTH) if prev[y][x] != curr[y][x]]
            if update is None:
                if changed:
                    stale_reports.append(f"update=None but {len(changed)} cells changed: {changed[:6]}")
                return update
            assert isinstance(update, ChopsUpdate)
            partials += 1
            written: set[tuple[int, int]] = set()
            for y, x1, x2 in update.spans:
                for x, strip in update.chops[y].items():
                    if strip is None or strip is _MERGED_STRIP:
                        continue
                    for cx in range(max(x, x1), min(x + strip.cell_length, x2)):
                        written.add((cx, y))
            stale = [(x, y) for x, y in changed if (x, y) not in written]
            if stale:
                samples = ", ".join(f"({x},{y}) {prev[y][x]!r}->{curr[y][x]!r}" for x, y in stale[:6])
                stale_reports.append(f"{len(stale)} stale cells: {samples}")
            return update

        def patched_full(self: Compositor, simplify: bool = False):
            nonlocal prev_grid
            update = orig_full(self, simplify=simplify)
            prev_grid = _strips_to_cells(self.render_strips(), WIDTH, HEIGHT)
            return update

        monkeypatch.setattr(Compositor, "render_partial_update", patched_partial)
        monkeypatch.setattr(Compositor, "render_full_update", patched_full)

        max_y = panel.max_scroll_y
        assert max_y > HEIGHT * 2, "transcript too short to exercise two-page scrolling"

        y = float(max_y)
        targets: list[float] = []
        for delta in (-3, -3, -3, 3, -6, -3, 3, -3):  # wheel-ish steps with flips
            y = max(0.0, min(float(max_y), y + delta))
            targets.append(y)
        targets += [max_y * 0.75, max_y * 0.5, max_y * 0.55, max_y * 0.2]  # drag-ish jumps
        targets += [max_y - HEIGHT, max_y - 2 * HEIGHT + 4]  # page jumps
        targets += [0.0, float(max_y)]  # extremes

        for target in targets:
            _simulate_chat_panel_user_scroll_y(panel, target)
            await pilot.pause()
            await pilot.pause()

        assert not stale_reports, "\n".join(stale_reports)
        assert partials >= 8, f"only {partials} partial updates exercised"
