# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the agent picker modal."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import OptionList, Static

from chrys.app.tui.screens.agents.picker import _AGENTS, AgentsScreen
from chrys.foundation.i18n import Localizer
from chrys.foundation.i18n.formatting import format_message
from chrys.service.profiles.agents.registry import AgentProfileRegistry


def test_agent_picker_title_keeps_english_and_localizes_chinese() -> None:
    assert format_message(_AGENTS.bind()) == "Agents"
    assert Localizer("zh-Hans").render(_AGENTS.bind()) == "智能体"


@pytest.mark.asyncio
async def test_agent_picker_duplicate_queued_selection_events_complete_once() -> None:
    class _Harness(App):
        def __init__(self) -> None:
            super().__init__()
            self.agent_screen: AgentsScreen | None = None
            self.results: list[str | None] = []

        def compose(self) -> ComposeResult:
            yield Static("base")

        def on_mount(self) -> None:
            registry = AgentProfileRegistry()
            registry.load_builtins()
            self.agent_screen = AgentsScreen(registry, current_profile="Code")
            self.push_screen(self.agent_screen, self.results.append)

    async with _Harness().run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.agent_screen
        assert screen is not None
        option_list = screen.query_one(OptionList)
        option_id = "QA"
        option = option_list.get_option(option_id)
        option_index = option_list.get_option_index(option_id)

        option_list.post_message(OptionList.OptionSelected(option_list, option, option_index))
        option_list.post_message(OptionList.OptionSelected(option_list, option, option_index))
        await pilot.pause()

    assert pilot.app.results == ["QA"]
