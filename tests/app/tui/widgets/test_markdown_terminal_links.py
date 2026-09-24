# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Markdown terminal hyperlinks survive wrapping, repaint and platform output paths."""

from __future__ import annotations

import asyncio
import re
from unittest.mock import patch

import pytest
from rich.console import RenderableType
from rich.control import Control
from textual import app as textual_app
from textual._cells import cell_len
from textual._compositor import ChopsUpdate, CompositorUpdate
from textual.app import App
from textual.drivers.headless_driver import HeadlessDriver
from textual.geometry import Region
from textual.screen import Screen
from textual.selection import SELECT_ALL
from textual.strip import Strip

from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from tests.support.waiting import wait_for, with_wait_deadline

_URL = "https://example.com/" + "long-segment/" * 7 + "My%20Report?q=a%2Fb#section"
_LABEL = "中文链接需要跨越多行显示完整目标\uff0c并保留百分号编码和查询参数"
_OSC8 = re.compile(r"\x1b\]8;[^;]*;([^\x1b]*)\x1b\\")


def _targets(output: str) -> list[str]:
    return [target for target in _OSC8.findall(output) if target]


def _clickable_rows(strips: list[Strip]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for y, strip in enumerate(strips):
        x = 0
        for segment in strip:
            if segment.style and "@click" in segment.style.meta and segment.text.strip():
                cells.append((x, y))
                break
            x += cell_len(segment.text)
    return cells


@pytest.mark.parametrize("size", [2048, 2049])
@pytest.mark.parametrize("kind", ["link", "image", "linked-image", "table"])
async def test_osc8_limit_keeps_full_ordinary_click_target(size: int, kind: str) -> None:
    prefix, suffix = "https://example.com/", "?q=a%2Fb#tail"
    url = prefix + "a" * (size - len(prefix) - len(suffix)) + suffix
    label = f"![image]({_URL})" if kind == "linked-image" else "click"
    source = f"{'!' if kind == 'image' else ''}[{label}]({url})"
    if kind == "table":
        source = f"| value |\n| --- |\n| {source} |"
    app = App()
    async with app.run_test() as pilot:
        markdown = VirtualizedMarkdown(source)
        markdown.styles.padding = 0
        await app.mount(markdown)
        await wait_for(lambda: markdown.region.width > 0 and bool(markdown._blocks), pilot=pilot)
        output = app.screen._compositor.render_full_update().render_segments(app.console)
        assert set(_targets(output)) == ({url} if size <= 2048 else set())
        with patch.object(app, "open_url", autospec=True) as opened:
            strips = markdown.render_lines(Region(0, 0, markdown.size.width, markdown._total_lines))
            assert await pilot.click(markdown, offset=_clickable_rows(strips)[0])
            await wait_for(lambda: opened.call_count == 1, pilot=pilot)
            opened.assert_called_once_with(url)


async def test_only_explicit_links_and_http_prose_emit_terminal_targets() -> None:
    app = App()
    source = (
        "README.md script.sh foo.rs example.com me@example.com\n\n"
        "`https://example.com/inline`\n\n"
        "```text\nhttps://example.com/fence\n```\n\n"
        "https://example.com/allowed\uff0c后续正文"
    )
    async with app.run_test(size=(80, 24)) as pilot:
        markdown = VirtualizedMarkdown(source)
        await app.mount(markdown)
        await wait_for(lambda: markdown.region.width > 0 and bool(markdown._blocks), pilot=pilot)
        output = app.screen._compositor.render_full_update().render_segments(app.console)
        assert set(_targets(output)) == {"https://example.com/allowed"}


@with_wait_deadline(30)
async def test_table_links_wrap_resize_and_click_without_linking_padding_or_neighbors() -> None:
    other_url = "https://example.com/other?q=%E4%B8%AD%E6%96%87#end"
    source = (
        f"| [标题链接]({_URL}) | 文件 |\n| --- | --- |\n"
        f"| [{_LABEL}]({_URL})尾文 | README.md |\n"
        f"| [**{'中文👩‍💻重复' * 4}**]({other_url}) | `https://example.com/code` |"
    )
    app = App()
    async with app.run_test(size=(42, 60)) as pilot:
        markdown = VirtualizedMarkdown(source)
        markdown.styles.padding = 0
        await app.mount(markdown)
        await wait_for(lambda: markdown.region.width == 42 and bool(markdown._blocks), pilot=pilot)
        for width in (42, 32):
            if app.size.width != width:
                await pilot.resize_terminal(width, 60)
            await wait_for(lambda width=width: markdown._width_at_last_layout == width, pilot=pilot)
            strips = markdown.render_lines(Region(0, 0, width, markdown._total_lines))
            clickable: dict[str, list[tuple[int, int]]] = {_URL: [], other_url: []}
            for y, strip in enumerate(strips):
                assert cell_len(strip.text) == width
                x = 0
                for segment in strip:
                    if segment.style and segment.style.link:
                        target = segment.style.link
                        assert target in clickable
                        assert segment.style.meta["@click"] == f"link({target!r})"
                        assert segment.text.strip()
                        assert not any(char in segment.text for char in "│─尾")
                        clickable[target].append((x, y))
                    elif segment.style:
                        assert "@click" not in segment.style.meta
                    x += cell_len(segment.text)
            output = app.screen._compositor.render_full_update().render_segments(app.console)
            assert set(_targets(output)) == {_URL, other_url}
            with patch.object(app, "open_url", autospec=True) as opened:
                expected: list[tuple[str]] = []
                for target, offsets in clickable.items():
                    assert len({y for _x, y in offsets}) >= 2
                    for offset in (offsets[0], offsets[-1]):
                        assert await pilot.click(markdown, offset=offset)
                        expected.append((target,))
                        count = len(expected)
                        await wait_for(lambda count=count: opened.call_count == count, pilot=pilot)
                assert [call.args for call in opened.call_args_list] == expected


@pytest.mark.parametrize(
    "source",
    [
        f"{_URL}\uff0c后续正文",
        f"[{_LABEL}]({_URL})",
        f"[**{_LABEL}**]({_URL})",
        f"![{_LABEL}]({_URL})",
    ],
    ids=["bare-cjk-boundary", "explicit", "nested-style", "image"],
)
@with_wait_deadline(30)
async def test_every_wrapped_row_has_complete_terminal_and_click_target(source: str) -> None:
    app = App()
    async with app.run_test(size=(32, 30)) as pilot:
        markdown = VirtualizedMarkdown(source)
        markdown.styles.padding = 0
        await app.mount(markdown)
        await wait_for(lambda: markdown.region.width == 32 and bool(markdown._blocks), pilot=pilot)
        strips = markdown.render_lines(Region(0, 0, 32, markdown._total_lines))
        cells = _clickable_rows(strips)
        assert len(cells) >= 2
        for _x, y in cells:
            assert _targets(strips[y].render(app.console))
            assert set(_targets(strips[y].render(app.console))) == {_URL}
        for strip in strips:
            for segment in strip:
                if segment.style and "@click" in segment.style.meta:
                    assert segment.style.link == _URL
                elif segment.style:
                    assert segment.style.link is None
        output = app.screen._compositor.render_full_update().render_segments(app.console)
        assert set(_targets(output)) == {_URL}
        with patch.object(app, "open_url", autospec=True) as opened:
            for count, offset in enumerate(cells, 1):
                assert await pilot.click(markdown, offset=offset)
                await wait_for(lambda count=count: opened.call_count >= count, pilot=pilot)
            assert [call.args for call in opened.call_args_list] == [(_URL,)] * len(cells)


@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("open_links", [False, True])
@pytest.mark.parametrize("table", [False, True])
async def test_terminal_links_respect_open_links_and_keep_click_events(
    image: bool, open_links: bool, table: bool
) -> None:
    class EventApp(App):
        def __init__(self) -> None:
            super().__init__()
            self.clicked: list[str] = []

        def on_virtualized_markdown_link_clicked(self, event: VirtualizedMarkdown.LinkClicked) -> None:
            self.clicked.append(event.raw_href)

    app = EventApp()
    async with app.run_test() as pilot:
        source = f"{'!' if image else ''}[click]({_URL})"
        if table:
            source = f"| value |\n| --- |\n| {source} |"
        markdown = VirtualizedMarkdown(source, open_links=open_links)
        markdown.styles.padding = 0
        await app.mount(markdown)
        await wait_for(lambda: markdown.region.width > 0 and bool(markdown._blocks), pilot=pilot)
        output = app.screen._compositor.render_full_update().render_segments(app.console)
        assert bool(_targets(output)) is open_links
        if open_links:
            assert set(_targets(output)) == {_URL}
        with patch.object(app, "open_url", autospec=True) as opened:
            strips = markdown.render_lines(Region(0, 0, markdown.size.width, markdown._total_lines))
            assert await pilot.click(markdown, offset=_clickable_rows(strips)[0])
            await wait_for(lambda: bool(app.clicked), pilot=pilot)
            assert app.clicked == [_URL]
            assert [call.args for call in opened.call_args_list] == ([(_URL,)] if open_links else [])


@pytest.mark.parametrize("href", ["./image.png", "#anchor", "vscode://file/tmp/a", "https://example.com/%0D%0A"])
async def test_image_targets_are_filtered_before_terminal_output(href: str) -> None:
    app = App()
    async with app.run_test() as pilot:
        markdown = VirtualizedMarkdown(f"![image]({href})")
        await app.mount(markdown)
        await wait_for(lambda: markdown.region.width > 0 and bool(markdown._blocks), pilot=pilot)
        output = app.screen._compositor.render_full_update().render_segments(app.console)
        assert not _targets(output)


@pytest.mark.parametrize("outer_href", ["./notes.md", "vscode://file/tmp/a", "https://example.com/%0D%0A", _URL])
async def test_linked_image_uses_outer_click_policy(outer_href: str) -> None:
    """A rejected outer link must clear the nested image's otherwise allowed OSC 8 target."""
    app = App()
    async with app.run_test() as pilot:
        markdown = VirtualizedMarkdown(f"[![image](https://example.com/image.png)]({outer_href})")
        markdown.styles.padding = 0
        await app.mount(markdown)
        await wait_for(lambda: markdown.region.width > 0 and bool(markdown._blocks), pilot=pilot)
        expected = {_URL} if outer_href == _URL else set()
        output = app.screen._compositor.render_full_update().render_segments(app.console)
        assert set(_targets(output)) == expected
        with patch.object(app, "open_url", autospec=True) as opened:
            assert await pilot.click(markdown, offset=(0, 0))
            # Pilot dispatches the click; the widget's message pump owns LinkClicked handling.
            handled = asyncio.Event()
            assert markdown.call_later(handled.set)
            await wait_for(handled.is_set, pilot=pilot)
            assert [call.args for call in opened.call_args_list] == ([(_URL,)] if expected else [])


@with_wait_deadline(30)
async def test_resize_crop_hover_and_copy_keep_complete_link_targets() -> None:
    app = App()
    async with app.run_test(size=(32, 30)) as pilot:
        markdown = VirtualizedMarkdown(_URL)
        markdown.styles.padding = 0
        await app.mount(markdown)
        await wait_for(lambda: markdown.region.width == 32 and bool(markdown._blocks), pilot=pilot)
        for width in (32, 21, 43):
            if app.size.width != width:
                await pilot.resize_terminal(width, 30)
            await wait_for(lambda width=width: markdown._width_at_last_layout == width, pilot=pilot)
            strips = markdown.render_lines(Region(0, 0, width, markdown._total_lines))
            cells = _clickable_rows(strips)
            assert len(cells) >= 2
            assert await pilot.hover(markdown, offset=cells[1])
            await wait_for(lambda: markdown.hover_style.link == _URL, pilot=pilot)
            # Clip away both horizontal edges and the first visual row.
            crop = Region(1, 1, width - 2, len(cells) - 1)
            for strip in markdown.render_lines(crop):
                assert set(_targets(strip.render(app.console))) == {_URL}
            # Bare URL display decodes the space; copying keeps that existing display behavior.
            assert markdown.get_selection(SELECT_ALL) == (_URL.replace("%20", " "), "\n")


@with_wait_deadline(30)
async def test_stream_update_replaces_terminal_destination_without_stale_links() -> None:
    app = App()
    async with app.run_test(size=(32, 30)) as pilot:
        markdown = VirtualizedMarkdown()
        await app.mount(markdown)
        await markdown.append(f"[{_LABEL}]")
        await wait_for(lambda: markdown.region.width == 32 and bool(markdown._blocks), pilot=pilot)
        assert not _targets(app.screen._compositor.render_full_update().render_segments(app.console))
        await markdown.append(f"({_URL})")
        assert set(_targets(app.screen._compositor.render_full_update().render_segments(app.console))) == {_URL}
        await markdown.update("Plain text after the link is removed.")
        assert not _targets(app.screen._compositor.render_full_update().render_segments(app.console))


@with_wait_deadline(30)
async def test_scrolling_repaints_complete_terminal_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    app = App()
    async with app.run_test(size=(32, 12)) as pilot:
        markdown = VirtualizedMarkdown(_URL)
        markdown.styles.padding = 0
        markdown.styles.height = 4
        await app.mount(markdown)
        await wait_for(lambda: markdown.max_scroll_y > 0, pilot=pilot)
        painted = asyncio.Event()
        assert app.screen.call_after_refresh(painted.set)
        await wait_for(painted.is_set, pilot=pilot)
        partial_frames: list[str] = []
        original_display = app._display

        def record_display(screen: Screen, renderable: RenderableType | None) -> None:
            if isinstance(renderable, ChopsUpdate):
                partial_frames.append(renderable.render_segments(app.console))
            original_display(screen, renderable)

        monkeypatch.setattr(app, "_display", record_display)
        target_y = markdown.max_scroll_y
        markdown.scroll_to(y=target_y, animate=False)
        await wait_for(
            lambda: markdown.scroll_y == target_y and any(_targets(frame) for frame in partial_frames), pilot=pilot
        )
        assert {target for frame in partial_frames for target in _targets(frame)} == {_URL}


@pytest.mark.parametrize("windows_output", [False, True], ids=["posix", "windows-chunks"])
async def test_platform_display_preserves_osc8_across_write_boundaries(
    monkeypatch: pytest.MonkeyPatch, windows_output: bool
) -> None:
    """Exercise both pinned Textual output branches without opening a native terminal."""

    class CaptureDriver(HeadlessDriver):
        def __init__(self, app: App) -> None:
            super().__init__(app)
            self.writes: list[str] = []

        @property
        def is_headless(self) -> bool:
            return False

        def write(self, data: str) -> None:
            self.writes.append(data)

    # Repeating a permitted destination across wrapped rows still splits OSC 8
    # at Windows' 8192-character write boundary. Each URI stays below 2048 bytes.
    url = "https://example.com/?value=" + "a" * 1900
    app = App()
    async with app.run_test(size=(32, 12)) as pilot:
        markdown = VirtualizedMarkdown(f"[{_LABEL * 3}]({url})")
        await app.mount(markdown)
        await wait_for(lambda: markdown.region.width == 32 and bool(markdown._blocks), pilot=pilot)
        update = app.screen._compositor.render_full_update()
        assert isinstance(update, CompositorUpdate)
        expected = update.render_segments(app.console)
        expected += Control.move_to(*app.screen.outer_size.clamp_offset(app.cursor_position)).segment.text
        assert set(_targets(expected)) == {url}
        assert any(match.start() < 8192 < match.end() for match in _OSC8.finditer(expected))
        driver = CaptureDriver(app)
        # Keep the real App._display serializer and only replace its terminal sink.
        # Restore before yielding so headless lifecycle ownership remains unchanged.
        with monkeypatch.context() as output_patch:
            output_patch.setattr(app, "_driver", driver)
            output_patch.setattr(textual_app, "WINDOWS", windows_output)
            app._display(app.screen, update)
        assert "".join(driver.writes) == expected
        if windows_output:
            assert len(driver.writes) > 1
            assert all(len(chunk) <= 8192 for chunk in driver.writes)
        else:
            assert driver.writes == [expected]
