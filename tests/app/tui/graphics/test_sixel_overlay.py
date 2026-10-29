# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Chat images drawn as Sixel over the half-blocks of each frame Textual writes."""

from __future__ import annotations

import logging
import re
import threading
from typing import TYPE_CHECKING

import pytest
from rich.cells import cell_len
from textual.containers import Horizontal, VerticalScroll
from textual.geometry import Region
from textual.screen import ModalScreen
from textual.widgets import Static

from chrys.app.tui.graphics import manager, payloads
from chrys.app.tui.graphics.detect import GraphicsProtocol
from chrys.app.tui.widgets.chat.image_view import TerminalImages
from tests.support.sixel_decoder import decode_sixel
from tests.support.terminal_images import TerminalImagesApp, image_previews
from tests.support.waiting import wait_for, wait_until

if TYPE_CHECKING:
    from collections.abc import Callable

_DRAW = re.compile(r"\x1b\[(\d+);(\d+)H(\x1bP0;1;0q.*?\x1b\\)", re.DOTALL)
_TOKEN = re.compile(r"\x1bP.*?\x1b\\|\x1b\[([0-9;?<>=]*)([@-~])|\x1b.|(\n)|([^\x1b\n]+)", re.DOTALL)
_RASTER_WIDTH = re.compile(r'"1;1;(\d+);')


def _draws(frame: str) -> list[tuple[int, int]]:
    """Each Sixel row a frame draws, as its screen (row, column)."""
    return [(int(row) - 1, int(column) - 1) for row, column, _ in _DRAW.findall(frame)]


def _printed_cells(frame: str) -> set[tuple[int, int]]:
    """The cells a frame prints text into, as (row, column); Sixel and other escapes print none."""
    cells: set[tuple[int, int]] = set()
    y = x = 0
    for match in _TOKEN.finditer(frame):
        params, final, newline, text = match.groups()
        if final == "H":
            row, _, column = params.partition(";")
            y, x = int(row or 1) - 1, int(column or 1) - 1
        elif newline:
            y += 1
        elif text:
            for char in text:
                width = cell_len(char)
                cells.update((y, x + offset) for offset in range(width))
                x += width
    return cells


def _on_screen(frames: list[str]) -> set[tuple[int, int]]:
    """Replay ``frames`` as Windows Terminal would: printing into a cell erases the image there.

    A frame's text comes before the Sixel appended to it. Fails on a row drawn again while all of it
    is still on screen: that is output for nothing.
    """
    shown: dict[tuple[int, int], int] = {}
    for frame in frames:
        printed = _printed_cells(frame)
        shown = {
            (y, x): columns
            for (y, x), columns in shown.items()
            if not any((y, column) in printed for column in range(x, x + columns))
        }
        for row, column, sixel in _DRAW.findall(frame):
            draw = (int(row) - 1, int(column) - 1)
            assert draw not in shown, f"{draw} drawn again with nothing printed over it"
            (width,) = _RASTER_WIDTH.findall(sixel)
            shown[draw] = int(width) // 10
    return set(shown)


def _drawn_since(app: TerminalImagesApp, first_frame: int) -> list[tuple[int, int]]:
    return [draw for frame in app.frames[first_frame:] for draw in _draws(frame)]


def _image_rows(view: TerminalImages) -> set[tuple[int, int]]:
    content = view.content_region
    return {(content.y + row, content.x) for row in range(view.content_size.height)}


async def _all_drawn(app: TerminalImagesApp, view: TerminalImages, pilot: object, first_frame: int = 0) -> None:
    def drawn() -> bool:
        rows = _image_rows(view)
        return bool(rows) and rows <= set(_drawn_since(app, first_frame))

    await wait_for(drawn, pilot=pilot, description="every image row was drawn")


async def test_each_row_is_drawn_where_the_half_blocks_are_and_only_when_erased() -> None:
    app = TerminalImagesApp(GraphicsProtocol.SIXEL)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.mount(Static("above"))
        view = TerminalImages(image_previews((40, 20)))
        await app.mount(view)
        await _all_drawn(app, view, pilot)

        assert _on_screen(app.frames) == {(row, 0) for row in range(1, 11)}
        first = next(_DRAW.finditer("".join(app.frames)))
        assert decode_sixel(first[3]).raster == (1, 1, 400, 20)


async def test_images_side_by_side_are_each_drawn_at_their_own_column() -> None:
    app = TerminalImagesApp(GraphicsProtocol.SIXEL)
    async with app.run_test(size=(80, 24)) as pilot:
        view = TerminalImages(image_previews((20, 20), (20, 20)))
        await app.mount(view)
        expected = {(row, column) for row in range(10) for column in (0, 22)}
        await wait_for(lambda: expected <= set(_drawn_since(app, 0)), pilot=pilot, description="both images drawn")

        assert [box.x for _slot, box in view.image_cells(view.content_size.width)] == [0, 22]
        assert _on_screen(app.frames) == expected


async def test_a_repaint_of_one_row_draws_that_row_again() -> None:
    app = TerminalImagesApp(GraphicsProtocol.SIXEL)
    async with app.run_test(size=(80, 24)) as pilot:
        view = TerminalImages(image_previews((40, 20)))
        await app.mount(view)
        await _all_drawn(app, view, pilot)
        mark = len(app.frames)

        view.refresh(Region(3, 4, 2, 1))
        await wait_for(lambda: _drawn_since(app, mark), pilot=pilot, description="the repainted row was drawn")

        assert _drawn_since(app, mark) == [(4, 0)]


async def test_a_widget_beside_the_image_repainting_draws_nothing_again() -> None:
    """Windows Terminal erases an image only in the cells text is printed into."""
    app = TerminalImagesApp(GraphicsProtocol.SIXEL)
    async with app.run_test(size=(80, 24)) as pilot:
        view = TerminalImages(image_previews((40, 20)))
        view.styles.width = 50
        side = Static("tick 0")
        side.styles.height = 10
        await app.mount(Horizontal(view, side))
        await _all_drawn(app, view, pilot)
        mark = len(app.frames)

        for tick in range(1, 4):
            text = f"tick {tick}"
            side.update(text)
            await wait_for(lambda text=text: text in "".join(app.frames[mark:]), pilot=pilot, description="ticked")
        assert not await wait_until(lambda: _drawn_since(app, mark), pilot=pilot, timeout=0.5)
        assert _on_screen(app.frames) == _image_rows(view)


async def test_rows_no_longer_wanted_stop_being_encoded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tall image takes seconds to encode: unmounting it frees the job for the next image."""
    started, release = threading.Event(), threading.Event()
    answers: list[bool] = []

    def held_sixel_rows(
        data_uri: str, *, columns: int, rows: int, pixel_height: int, wanted: Callable[[], bool] | None
    ) -> tuple[str, ...]:
        if not started.is_set():
            started.set()
            release.wait(timeout=10)
            answers.append(wanted())
        return payloads.sixel_rows(data_uri, columns=columns, rows=rows, pixel_height=pixel_height, wanted=wanted)

    monkeypatch.setattr(manager, "sixel_rows", held_sixel_rows)
    app = TerminalImagesApp(GraphicsProtocol.SIXEL)
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            tall = TerminalImages(image_previews((40, 200)))
            await app.mount(tall)
            await wait_for(started.is_set, pilot=pilot, description="the tall image is being encoded")
            slot = tall.slots[0]
            await tall.remove()
            release.set()

            after = TerminalImages(image_previews((40, 20)))
            await app.mount(after)
            await _all_drawn(app, after, pilot)
            assert answers == [False]
            assert (slot.sixel, slot.failed) == ((), None)
    finally:
        release.set()


async def test_a_moving_image_is_drawn_again_once_it_settles() -> None:
    app = TerminalImagesApp(GraphicsProtocol.SIXEL)
    async with app.run_test(size=(80, 24)) as pilot:
        view = TerminalImages(image_previews((40, 20)))
        scroll = VerticalScroll(Static("\n".join(["above"] * 4)), view, Static("\n".join(["below"] * 40)))
        await app.mount(scroll)
        await _all_drawn(app, view, pilot)
        mark = len(app.frames)

        scroll.scroll_to(y=2, animate=False, immediate=True)
        await wait_for(lambda: len(app.frames) > mark, pilot=pilot, description="the scroll was painted")
        assert _draws(app.frames[mark]) == []
        await _all_drawn(app, view, pilot, mark)

        assert set(_drawn_since(app, mark)) == {(row, 0) for row in range(2, 12)}
        assert _on_screen(app.frames) == {(row, 0) for row in range(2, 12)}


async def test_rows_under_a_widget_above_them_are_left_as_half_blocks() -> None:
    app = TerminalImagesApp(GraphicsProtocol.SIXEL)
    async with app.run_test(size=(80, 24)) as pilot:
        view = TerminalImages(image_previews((40, 20)))
        cover = Static("toast")
        cover.styles.layer = "above"
        cover.styles.dock = "top"
        cover.styles.width = 10
        cover.styles.offset = (20, 3)
        await app.mount(view, cover)
        await wait_for(lambda: view.slots[0].sixel, pilot=pilot, description="the rows were encoded")
        cover_region = app.screen.find_widget(cover).region
        expected = {(y, x) for y, x in _image_rows(view) if not cover_region.overlaps(Region(x, y, 40, 1))}
        await wait_for(lambda: expected <= set(_drawn_since(app, 0)), pilot=pilot, description="uncovered rows drawn")

        assert len(expected) == 9
        assert set(_drawn_since(app, 0)) == expected


async def test_rows_a_scroll_container_clips_are_not_drawn() -> None:
    app = TerminalImagesApp(GraphicsProtocol.SIXEL)
    async with app.run_test(size=(80, 24)) as pilot:
        view = TerminalImages(image_previews((20, 20)))
        clip = VerticalScroll(view)
        clip.styles.height = 4
        clip.styles.width = 30
        await app.mount(clip)
        in_view = {(row, 0) for row in range(4)}
        await wait_for(lambda: in_view <= set(_drawn_since(app, 0)), pilot=pilot, description="rows in view drawn")
        assert set(_drawn_since(app, 0)) == in_view

        scrolled = len(app.frames)
        clip.scroll_to(y=3, animate=False, immediate=True)
        await wait_for(
            lambda: in_view <= set(_drawn_since(app, scrolled)), pilot=pilot, description="rows scrolled in drawn"
        )
        assert set(_drawn_since(app, scrolled)) == in_view


async def test_a_row_on_the_last_screen_line_is_not_drawn() -> None:
    app = TerminalImagesApp(GraphicsProtocol.SIXEL)
    async with app.run_test(size=(80, 10)) as pilot:
        view = TerminalImages(image_previews((20, 20)))
        await app.mount(view)
        above_last = {(row, 0) for row in range(9)}
        await wait_for(lambda: above_last <= set(_drawn_since(app, 0)), pilot=pilot, description="rows drawn")

        assert view.content_region.bottom == 10
        assert set(_drawn_since(app, 0)) == above_last


async def test_a_frame_draws_at_most_its_budget_and_the_rest_follows(monkeypatch: pytest.MonkeyPatch) -> None:
    app = TerminalImagesApp(GraphicsProtocol.SIXEL)
    async with app.run_test(size=(80, 24)) as pilot:
        view = TerminalImages(image_previews((40, 20)))
        await app.mount(view)
        await wait_for(lambda: view.slots[0].sixel, pilot=pilot, description="the rows were encoded")
        monkeypatch.setattr(manager, "FRAME_BUDGET_CHARS", 3 * len(view.slots[0].sixel[0]))
        mark = len(app.frames)

        view.refresh()
        await _all_drawn(app, view, pilot, mark)

        per_frame = [len(_draws(frame)) for frame in app.frames[mark:] if _draws(frame)]
        assert max(per_frame) == 3
        assert sum(per_frame) == 10


async def test_nothing_is_drawn_while_a_modal_covers_the_screen() -> None:
    app = TerminalImagesApp(GraphicsProtocol.SIXEL)
    async with app.run_test(size=(80, 24)) as pilot:
        view = TerminalImages(image_previews((40, 20)))
        await app.mount(view)
        await _all_drawn(app, view, pilot)

        await app.push_screen(_Modal())
        await wait_for(lambda: isinstance(app.screen, _Modal) and app.screen.is_mounted, pilot=pilot)
        mark = len(app.frames)
        app.screen.refresh()
        await wait_for(lambda: len(app.frames) > mark, pilot=pilot, description="the modal was painted")
        assert _drawn_since(app, mark) == []

        app.pop_screen()
        await _all_drawn(app, view, pilot, mark)
        assert _on_screen(app.frames) == _image_rows(view)


async def test_a_failing_overlay_still_writes_the_frame(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    app = TerminalImagesApp(GraphicsProtocol.SIXEL)
    async with app.run_test(size=(80, 24)) as pilot:
        await app.mount(Static("text"))
        assert app.graphics is not None
        overlay = app.graphics.overlay
        assert overlay is not None

        def fail(_update: object) -> str:
            raise RuntimeError("boom")

        monkeypatch.setattr(overlay, "frame_suffix", fail)
        mark = len(app.frames)
        with caplog.at_level(logging.ERROR, logger=manager.__name__):
            app.screen.refresh()
            await wait_for(lambda: len(app.frames) > mark, pilot=pilot, description="a frame was written")

        assert "text" in app.frames[-1]
        assert "Could not draw terminal images" in caplog.text


class _Modal(ModalScreen[None]):
    def compose(self):
        yield Static("modal")
