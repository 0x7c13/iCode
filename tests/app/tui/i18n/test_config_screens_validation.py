# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Simplified-Chinese coverage for configuration validation prose."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from chrys.app.tui.i18n import LocaleController, render_str
from chrys.app.tui.screens.agents.agent_draft_store import AgentDraftStore
from chrys.app.tui.screens.agents.config import AgentsConfigScreen
from chrys.app.tui.screens.agents.panels.basic import BasicConfigPanel
from chrys.app.tui.screens.models.screen import ModelConfigScreen, _validate_chat_option_float
from chrys.app.tui.widgets import EnhancedInput as Input
from chrys.foundation.config.settings import Settings
from chrys.foundation.i18n import Localizer, MessageRef
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import AgentProfile
from chrys.service.profiles.models.options import ProtectedChatOptionsWarning
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile


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
        description=f"{name} description",
        instructions=f"{name} instructions",
    )


@dataclass
class _Draft:
    key: str
    original_name: str | None
    profile: AgentProfile
    dirty: bool


def _zh_renderer(reference: MessageRef) -> str:
    return render_str(Localizer("zh-Hans"), reference)


@pytest.mark.asyncio
async def test_agent_panel_validation_renders_chinese() -> None:
    registry = AgentProfileRegistry()
    registry.register(_agent_profile("primary", "Primary"))
    app = _LocalizedHost()

    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Primary")
        await app.push_screen(screen)
        await pilot.pause()

        panel = screen.query_one(BasicConfigPanel)
        panel.query_one("#bc-name", Input).value = ""

        assert "配置文件名称为必填项。" in panel.validate()


def test_agent_draft_store_uses_injected_chinese_renderer_and_defaults_to_english() -> None:
    profile = _agent_profile("primary", "")
    store = AgentDraftStore([_Draft("draft:1", "primary", profile, True)], ["primary"])

    assert "primary：显示名称为必填项。" in store.validate(render_message=_zh_renderer)  # noqa: RUF001
    assert "primary: display name is required." in store.validate()


def test_model_chat_option_clause_validation_renders_chinese_with_row_and_key() -> None:
    errors = _validate_chat_option_float(
        3,
        "temperature",
        "hot",
        render_message=_zh_renderer,
    )

    assert errors == ["Chat 选项第 3 行：'temperature' 必须是 0.0 到 2.0 之间的 JSON 数字。"]  # noqa: RUF001


@pytest.mark.asyncio
async def test_mcp_card_invalid_command_read_during_switch_renders_chinese() -> None:
    from textual.widgets import TextArea

    from chrys.app.tui.screens.agents.panels.mcp import MCPConnectionCard
    from chrys.service.profiles.agents.schema import MCPServerConfig

    seed = MCPServerConfig(name="local", transport="stdio", command="uvx", args=["server"])
    app = _LocalizedHost()

    async with app.run_test(size=(120, 40)) as pilot:
        card = MCPConnectionCard(seed, index=0)
        await app.mount(card)
        await pilot.pause()

        card.query_one("#mcp-cmd-0", TextArea).text = 'foo "bar'

        with pytest.raises(ValueError, match="命令行无效："):  # noqa: RUF001
            card.get_config()


def test_protected_chat_options_notification_localizes_without_changing_legacy_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = ("conversation_id", "prompt")
    legacy = ProtectedChatOptionsWarning(keys=keys)
    assert legacy.message == (
        "Protected chat option key(s) were stripped: conversation_id, prompt. "
        "These keys can no longer be configured in profile YAML because they bypass context admission. "
        "Reusable prompts and profile-pinned continuation IDs are unavailable there; store: true only enables "
        "automatically managed Responses continuation."
    )

    registry = ModelProfileRegistry()
    profile = ModelProfile(
        id="model-a",
        name="Model A",
        model_id="gpt-test",
        chat_options='{"conversation_id": "abc", "prompt": "hello"}',
    )
    screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
    notify = MagicMock()
    monkeypatch.setattr(screen, "_render_toast", _zh_renderer)
    monkeypatch.setattr(screen, "notify", notify)

    screen._notify_unsupported_options(profile)

    assert notify.call_args.args[0] == (
        "已移除受保护的 Chat 选项键：conversation_id, prompt。这些键不能再在配置文件 YAML 中设置，因为它们会绕过上下文准入。"  # noqa: RUF001
        "该处不支持可复用提示词和配置文件固定的续接 ID；store: true 仅启用自动管理的 Responses 续接。"  # noqa: RUF001
    )
