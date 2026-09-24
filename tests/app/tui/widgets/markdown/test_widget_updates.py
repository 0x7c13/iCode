# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Overlapping VirtualizedMarkdown updates leave the newest document's state."""

from __future__ import annotations

from textual.app import App

from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from tests.support.waiting import wait_for


async def test_overlapping_updates_resolve_anchors_in_the_newest_document() -> None:
    app = App()
    async with app.run_test():
        widget = VirtualizedMarkdown("")
        await app.mount(widget)
        await wait_for(lambda: widget.region.width > 0, description="markdown widget laid out")
        # Each update starts at once: the first parses under the lock while the
        # second has already been called and queues behind it.
        first = widget.update("# Getting started\n\nold\n")
        second = widget.update("# 开始使用\n\nnew\n")
        await first
        await second

        assert [title for _, title, _ in widget.table_of_contents] == ["开始使用"]
        assert widget.goto_anchor("开始使用")
        assert not widget.goto_anchor("getting-started")
