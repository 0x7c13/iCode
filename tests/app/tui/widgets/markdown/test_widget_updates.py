# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""VirtualizedMarkdown updates coalesce into one parse of the newest source and leave its state."""

from __future__ import annotations

import asyncio
import threading

from textual.app import App
from textual.await_complete import AwaitComplete
from textual.pilot import Pilot

from chrys.app.tui.widgets.markdown.blocks import MarkdownBlock
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from tests.support.waiting import wait_for

_PARSE_HOLD_TIMEOUT = 10.0


class _RecordingMarkdown(VirtualizedMarkdown):
    """Records every source handed to the executor parse; can hold the first parse open."""

    def __init__(self) -> None:
        super().__init__("")
        self.parsed: list[str] = []
        self.hold_first_parse = False
        self.first_parse_started = threading.Event()
        self.release_first_parse = threading.Event()

    def _build_blocks(self, markdown: str) -> list[MarkdownBlock]:
        self.parsed.append(markdown)
        if self.hold_first_parse and not self.first_parse_started.is_set():
            self.first_parse_started.set()
            self.release_first_parse.wait(_PARSE_HOLD_TIMEOUT)
        return super()._build_blocks(markdown)


class _ContentsApp(App[None]):
    """Collects the table-of-contents announcements its markdown widget posts."""

    def __init__(self) -> None:
        super().__init__()
        self.contents_updates: list[VirtualizedMarkdown.TableOfContentsUpdated] = []

    def on_virtualized_markdown_table_of_contents_updated(
        self, message: VirtualizedMarkdown.TableOfContentsUpdated
    ) -> None:
        self.contents_updates.append(message)


async def _mounted(app: App[None], pilot: Pilot[None]) -> _RecordingMarkdown:
    widget = _RecordingMarkdown()
    await app.mount(widget)
    await wait_for(lambda: widget.region.width > 0, pilot=pilot, description="markdown widget laid out")
    await pilot.pause()
    widget.parsed.clear()
    return widget


def _installed_text(widget: VirtualizedMarkdown) -> str:
    return "\n".join(block.content.plain for block in widget._blocks)


async def test_overlapping_updates_resolve_anchors_in_the_newest_document() -> None:
    app = App()
    async with app.run_test():
        widget = VirtualizedMarkdown("")
        await app.mount(widget)
        await wait_for(lambda: widget.region.width > 0, description="markdown widget laid out")
        # Neither parse has started when the second update arrives, so both
        # awaitables share one parse of the newest source.
        first = widget.update("# Getting started\n\nold\n")
        second = widget.update("# 开始使用\n\nnew\n")
        await first
        await second

        assert [title for _, title, _ in widget.table_of_contents] == ["开始使用"]
        assert widget.goto_anchor("开始使用")
        assert not widget.goto_anchor("getting-started")


async def test_a_burst_of_cumulative_updates_in_one_loop_step_parses_once() -> None:
    app = App()
    async with app.run_test() as pilot:
        widget = await _mounted(app, pilot)
        lines = [f"Streamed line {index} with `code` and **emphasis**.\n\n" for index in range(150)]
        cumulative = ["".join(lines[: count + 1]) for count in range(len(lines))]

        # One synchronous burst, as the main turn replays a buffered answer line by line.
        awaitables = [widget.update(text) for text in cumulative]
        await awaitables[0]

        assert widget.parsed == [cumulative[-1]]
        assert widget.source == cumulative[-1]
        assert _installed_text(widget).endswith("Streamed line 149 with code and emphasis.")
        await asyncio.gather(*(awaitable() for awaitable in awaitables))
        assert widget.parsed == [cumulative[-1]]


async def test_an_update_during_a_parse_waits_for_one_more_parse_of_the_newest_source() -> None:
    app = App()
    async with app.run_test() as pilot:
        widget = await _mounted(app, pilot)
        widget.hold_first_parse = True
        try:
            first = widget.update("first\n")
            await wait_for(widget.first_parse_started.is_set, pilot=pilot, description="first parse started")
            second = widget.update("second\n")
            third = widget.update("third\n")
            assert not first.is_done
        finally:
            widget.release_first_parse.set()
        await first
        await second
        await third

        # The running parse had already read its source; the two later updates share the next one.
        assert widget.parsed == ["first\n", "third\n"]
        assert _installed_text(widget) == "third"


async def test_every_awaitable_resolves_after_its_own_text_or_newer_is_installed() -> None:
    app = App()
    async with app.run_test() as pilot:
        widget = await _mounted(app, pilot)
        widget.hold_first_parse = True
        seen: dict[str, str] = {}

        async def installed_after(awaitable: AwaitComplete, label: str) -> None:
            await awaitable
            seen[label] = _installed_text(widget)

        try:
            first = asyncio.create_task(installed_after(widget.update("one\n"), "one"))
            await wait_for(widget.first_parse_started.is_set, pilot=pilot, description="first parse started")
            # Joins no running parse: the append and the later update share the next one.
            second = asyncio.create_task(installed_after(widget.append("two\n"), "two"))
            third = asyncio.create_task(installed_after(widget.update("three\n"), "three"))
        finally:
            widget.release_first_parse.set()
        await asyncio.gather(first, second, third)

        assert widget.parsed == ["one\n", "three\n"]
        assert seen["one"] in {"one", "three"}
        assert seen["two"] == "three"
        assert seen["three"] == "three"


async def test_cancelling_one_awaiter_leaves_the_shared_parse_to_the_others() -> None:
    app = App()
    async with app.run_test() as pilot:
        widget = await _mounted(app, pilot)
        first = widget.update("# Kept\n\nbody\n")
        second = widget.update("# Kept\n\nnewest body\n")

        async def await_first() -> None:
            await first

        waiter = asyncio.create_task(await_first())
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await second

        assert waiter.cancelled()
        assert widget.parsed == ["# Kept\n\nnewest body\n"]
        assert _installed_text(widget) == "Kept\nnewest body"


async def test_a_coalesced_group_announces_its_contents_once_when_any_update_joined() -> None:
    app = _ContentsApp()
    async with app.run_test() as pilot:
        widget = await _mounted(app, pilot)
        app.contents_updates.clear()

        # Heading-free appends announce nothing, as each append() on its own would not.
        for index in range(20):
            widget.append(f"fragment {index}\n\n")
        await widget.append("last fragment\n")
        await pilot.pause()
        assert app.contents_updates == []
        assert len(widget.parsed) == 1

        # An update() joining heading-free appends still announces, once for the group.
        widget.parsed.clear()
        widget.append("more\n\n")
        widget.update("replaced\n\n")
        await widget.append("tail\n")
        await pilot.pause()
        assert widget.parsed == ["replaced\n\ntail\n"]
        assert [message.table_of_contents for message in app.contents_updates] == [[]]

        app.contents_updates.clear()
        for index in range(20):
            widget.update(f"# Heading {index}\n")
        await widget.update("# Final heading\n")
        await pilot.pause()
        assert [message.table_of_contents for message in app.contents_updates] == [
            [(1, "Final heading", widget._blocks[0].block_id)]
        ]
