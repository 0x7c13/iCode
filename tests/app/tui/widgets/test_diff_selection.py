# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Selecting code in a diff that is scrolled sideways, and in lines of wide characters."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.selection import Selection
from textual.strip import Strip

from chrys.app.tui.widgets.diff_view import CodeColumn, DiffView
from tests.support.waiting import wait_for

# Four characters over eight cells, then one cell each.
_LINE = "你好世界" + "abcdefghij" * 12
_TAIL = "abcdefghij" * 12
# Characters that are not a cell each: a variation selector that widens the character before it,
# an emoji sequence of five characters over two cells, and combining marks over no cell at all.
_CLUSTER_LINES = ("☺\ufe0fABC" + _TAIL, "👨\u200d👩\u200d👧xyz" + _TAIL, "e\u0301\u0302xyz" + _TAIL)


class _LongLineDiffApp(App):
    def __init__(self, *lines: str) -> None:
        super().__init__()
        self._lines = lines or (_LINE, "second")

    def compose(self) -> ComposeResult:
        diff = DiffView("notes.txt", "notes.txt", "", "".join(f"{line}\n" for line in self._lines))
        diff.split = False
        yield diff


def _first_offset(strip: Strip) -> tuple[int, int]:
    for segment in strip:
        if segment.style is not None and "offset" in segment.style.meta:
            return segment.style.meta["offset"]
    raise AssertionError("the strip carries no offsets")


def _text_by_background(strip: Strip) -> dict[str, str]:
    text: dict[str, str] = {}
    for segment in strip:
        assert segment.style is not None
        assert segment.style.bgcolor is not None
        text[segment.style.bgcolor.name] = text.get(segment.style.bgcolor.name, "") + segment.text
    return text


async def _scroll_to(pilot, code: CodeColumn, x: int) -> None:
    code.scroll_to(x=x, animate=False)
    await wait_for(lambda: code.scroll_offset.x == x, pilot=pilot, description=f"the code column scrolled to x={x}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scroll_x", "first_character"),
    [
        (0, 0),
        # Half of "你" is cut off; the blank that stands in for it still counts as that character.
        (1, 0),
        (2, 1),
        (8, 4),
        (10, 6),
        (30, 26),
    ],
)
async def test_offsets_name_the_character_under_each_cell_when_scrolled_sideways(
    scroll_x: int, first_character: int
) -> None:
    """Textual turns a pointer position into a place in the text through the offsets of a strip.

    They count characters of the whole line. A strip that started counting at the left edge of
    the viewport would select text ``scroll_x`` cells to the left of the pointer.
    """
    async with _LongLineDiffApp().run_test(size=(60, 10)) as pilot:
        await pilot.pause()
        code = pilot.app.query_one(CodeColumn)
        assert code.max_scroll_x >= 30

        await _scroll_to(pilot, code, scroll_x)

        assert _first_offset(code.render_line(0)) == (first_character, 0)
        # A character to the cell, and the scroll position is the character: also past the end of
        # the text, where Textual clamps to it as it does for a line that is not scrolled.
        assert _first_offset(code.render_line(1)) == (scroll_x, 1)


@pytest.mark.asyncio
async def test_pointer_names_the_character_under_it_past_an_emoji_scrolled_out_of_view() -> None:
    """``☺`` and its variation selector are two characters over two cells, counted one cell each
    they are one: the character after them would be taken for the one after that."""
    async with _LongLineDiffApp(*_CLUSTER_LINES).run_test(size=(60, 10)) as pilot:
        await pilot.pause()
        code = pilot.app.query_one(CodeColumn)
        await _scroll_to(pilot, code, 2)
        assert code.render_line(0).text.startswith("ABC")

        widget, offset = pilot.app.screen.get_widget_and_offset_at(code.region.x, code.region.y)

        assert widget is code
        assert offset is not None
        assert _CLUSTER_LINES[0][offset.x] == "A"


@pytest.mark.asyncio
@pytest.mark.parametrize("scroll_x", range(7))
async def test_offsets_agree_with_wherever_the_crop_cut_the_line(scroll_x: int) -> None:
    """Cutting through sequences and marks, the crop keeps or drops characters by rules of its own."""
    async with _LongLineDiffApp(*_CLUSTER_LINES).run_test(size=(60, 10)) as pilot:
        await pilot.pause()
        code = pilot.app.query_one(CodeColumn)
        await _scroll_to(pilot, code, scroll_x)

        for index, line in enumerate(_CLUSTER_LINES):
            strip = code.render_line(index)
            first_character, _y = _first_offset(strip)
            # From the second character on: the first may be the blank that half a wide one became.
            assert strip.text[1:20] == line[first_character + 1 : first_character + 20], (index, strip.text[:20])


@pytest.mark.asyncio
async def test_selection_is_drawn_over_the_selected_characters_when_scrolled_sideways() -> None:
    async with _LongLineDiffApp().run_test(size=(60, 10)) as pilot:
        await pilot.pause()
        code = pilot.app.query_one(CodeColumn)
        await _scroll_to(pilot, code, 10)
        line_background = _text_by_background(code.render_line(0))
        assert len(line_background) == 1

        pilot.app.screen.selections = {code: Selection(Offset(12, 0), Offset(15, 0))}
        await pilot.pause()
        drawn = _text_by_background(code.render_line(0))

        assert _LINE[12:15] == "ija"
        assert [text for background, text in drawn.items() if background not in line_background] == ["ija"]
        assert pilot.app.screen.get_selected_text() == "ija"


@pytest.mark.asyncio
async def test_rows_outside_a_selection_stay_cached_while_it_lasts() -> None:
    async with _LongLineDiffApp().run_test(size=(60, 10)) as pilot:
        await pilot.pause()
        code = pilot.app.query_one(CodeColumn)
        unselected = code.render_line(1)

        pilot.app.screen.selections = {code: Selection(Offset(0, 0), Offset(3, 0))}
        await pilot.pause()

        assert code.render_line(1) is unselected
        assert code.render_line(0) is not code.render_line(0)

        pilot.app.screen.clear_selection()
        await pilot.pause()

        assert _text_by_background(code.render_line(0)).keys() == _text_by_background(unselected).keys()
