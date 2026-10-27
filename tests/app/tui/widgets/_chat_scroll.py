# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Helpers for chat-panel scroll tests: a transcript that overflows the view, and the layouts it goes through."""

from __future__ import annotations

import pytest
from textual.geometry import Size

from chrys.app.tui.widgets.chat.panel import ChatPanel
from tests.support.pilot_barrier import screen_is_settled
from tests.support.waiting import wait_for


async def fill_eight_turns(cp: ChatPanel, pilot: object) -> None:
    """Add eight user/agent turns so an 80x20 panel overflows, then let layout settle.

    A reply laid out before its text parsed shrinks as the text shows, and the view pins to the bottom a refresh
    after that layout: a scroll made before then would be taken back.
    """
    body = "\n".join(f"line {i}" for i in range(8))
    for i in range(8):
        await cp.add_user_message(f"message {i}")
        await cp.add_agent_message(body)
    await wait_for(lambda: screen_is_settled(cp.app, cp.screen), pilot=pilot, description="the turns laid out")


async def scrolled_to_bottom(cp: ChatPanel, pilot: object) -> None:
    """Fill the panel with eight turns and settle it at the exact bottom."""
    await fill_eight_turns(cp, pilot)
    cp.scroll_end(immediate=True, animate=False)
    await pilot.pause()


def record_layouts(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, float]]:
    """Record the arranged virtual height and the scroll offset after each chat panel layout."""
    layouts: list[tuple[int, float]] = []
    size_updated = ChatPanel._size_updated

    def record(panel: ChatPanel, size: Size, virtual_size: Size, container_size: Size, layout: bool = True) -> bool:
        changed = size_updated(panel, size, virtual_size, container_size, layout)
        layouts.append((virtual_size.height, panel.scroll_y))
        return changed

    monkeypatch.setattr(ChatPanel, "_size_updated", record)
    return layouts
