# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Simplified-Chinese coverage for agent and model configuration chrome."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Label, OptionList, Static, TabbedContent

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.agents.config import AgentsConfigScreen
from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog
from chrys.app.tui.screens.models.screen import ModelConfigScreen
from chrys.app.tui.widgets import EnhancedInput as Input
from chrys.foundation.config.settings import Settings
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import AgentProfile
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.waiting import wait_until


class _LocalizedHost(App[None]):
    def __init__(self) -> None:
        self.locale_controller = LocaleController(Settings(locale="zh-Hans"))
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Static("host")


def _agent_profile(name: str, display_name: str) -> AgentProfile:
    return AgentProfile(
        name=name,
        display_name=display_name,
        description=f"{display_name} description",
        instructions="Follow the user's instructions.",
    )


@pytest.mark.asyncio
async def test_agent_config_tabs_buttons_and_panel_placeholder_render_chinese() -> None:
    registry = AgentProfileRegistry()
    registry.register(_agent_profile("primary", "Primary"))
    app = _LocalizedHost()

    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Primary")
        await app.push_screen(screen)
        await pilot.pause()

        assert screen.query_one("#ac-tabs", TabbedContent).get_tab("basic").label.plain == "基本"
        assert screen.query_one("#ac-new", Button).label.plain == "新建"
        assert screen.query_one("#bc-display-name", Input).placeholder == "用户友好的名称"


@pytest.mark.asyncio
async def test_agent_config_delete_confirmation_renders_chinese_title_and_action() -> None:
    registry = AgentProfileRegistry()
    registry.register(_agent_profile("primary", "Primary"))
    registry.register(_agent_profile("fallback", "Fallback"))
    app = _LocalizedHost()

    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Primary")
        await app.push_screen(screen)
        await pilot.pause()

        screen.query_one("#ac-delete", Button).press()
        await pilot.pause()

        dialog = app.screen
        assert isinstance(dialog, ConfirmDialog)
        # The buttons are composed by the nested DialogButtonRow, which mounts
        # a refresh after the dialog itself; poll rather than assume one pause.
        assert await wait_until(lambda: bool(dialog.query("#confirm-yes")), pilot=pilot), "confirm button never mounted"
        assert str(dialog.query_one("#confirm-container").border_title) == "删除智能体"
        assert dialog.query_one("#confirm-yes", Button).label.plain == "删除"


@pytest.mark.asyncio
async def test_model_config_sidebar_and_provider_label_keep_brand_name() -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-a", name="Model A", provider="openai", model_id="gpt-test")
    registry.register(profile)
    app = _LocalizedHost()

    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        option = screen.query_one("#mc-list", OptionList).get_option(profile.id)
        assert option.prompt.plain == "Model A"
        assert screen.query_one("#mc-model-label", Label).render().plain.endswith("OpenAI 模型")
