# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Thinking body release uses the same young/frozen removal contract as tool groups."""

from __future__ import annotations

import pytest

from chrys.app.tui.support import gc_freeze
from chrys.app.tui.support.gc_freeze import GcReclaimRequested
from chrys.app.tui.widgets.chat.messages import ThinkingMessage
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.markdown.widget import VirtualizedMarkdown
from tests.support.tui_helpers import GcMessageChatPanelApp
from tests.support.waiting import wait_for


@pytest.mark.parametrize("frozen", [False, True], ids=["young", "frozen"])
async def test_fold_releases_body_with_epoch_aware_reclaim(monkeypatch: pytest.MonkeyPatch, frozen: bool) -> None:
    app = GcMessageChatPanelApp(show_thinking=lambda: True)
    async with app.run_test() as pilot:
        panel = app.query_one(ChatPanel)
        await panel.add_user_message("prompt")
        await panel.add_thinking_delta("r1", "reasoning", "# Evidence\n\nLook at the file.")
        panel.finish_thinking_segment("r1")
        message = panel.query_one(ThinkingMessage)
        await wait_for(
            lambda: message.query_one(VirtualizedMarkdown).shown_source == message.text,
            pilot=pilot,
            description="thinking parse ready",
        )
        body = message.query_one(VirtualizedMarkdown)
        epoch = gc_freeze.current_gc_freeze_epoch()
        if frozen:
            monkeypatch.setattr(gc_freeze, "_freeze_epoch", epoch + 1)
        message.collapsed = True
        await wait_for(
            lambda: any(isinstance(event, GcReclaimRequested) for event in app.gc_messages),
            pilot=pilot,
            description="thinking reclaim request",
        )
        reclaim = next(event for event in app.gc_messages if isinstance(event, GcReclaimRequested))
        assert reclaim.prompt is False
        assert not message.query(VirtualizedMarkdown)
        assert not body.is_attached
        if frozen:
            assert reclaim.removed is None
        else:
            assert reclaim.removed is not None
            assert reclaim.removed.built_at_epoch == epoch
            assert body in [node() for node in reclaim.removed.nodes]
        message.collapsed = False
        await wait_for(lambda: bool(message.query(VirtualizedMarkdown)), pilot=pilot, description="thinking rebuilt")
        assert message.query_one(VirtualizedMarkdown) is not body
        await wait_for(
            lambda: message.query_one(VirtualizedMarkdown).shown_source == message.text,
            pilot=pilot,
            description="rebuilt thinking text ready",
        )


async def test_fast_collapse_expand_keeps_one_body_and_no_automatic_completion_fold() -> None:
    async with GcMessageChatPanelApp(show_thinking=lambda: True).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_user_message("prompt")
        await panel.add_thinking_delta("r", "reasoning", "Reasoning.")
        panel.finish_thinking_segment("r")
        await panel.add_agent_message("Done.")
        message = panel.query_one(ThinkingMessage)
        assert not message.collapsed
        message.collapsed = True
        message.collapsed = False
        message.collapsed = True
        message.collapsed = False
        await wait_for(
            lambda: len(message.query(VirtualizedMarkdown)) == 1,
            pilot=pilot,
            description="one final body after rapid toggles",
        )
        assert not message.collapsed
