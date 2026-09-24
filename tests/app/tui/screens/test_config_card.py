# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for shared agent-config card chrome."""

from __future__ import annotations

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Label

from chrys.app.tui.screens.agents.panels.config_card import ConfigCard
from chrys.app.tui.screens.agents.panels.mcp import MCPConnectionCard
from chrys.app.tui.screens.agents.panels.subagents import SubAgentCard


class _ProbeCard(ConfigCard):
    _delete_button_prefix = "probe-delete"

    def compose(self) -> ComposeResult:
        yield from self.compose_header("Probe", row_class="probe-header", title_class="probe-title")


class _ConfigCardApp(App):
    def __init__(self) -> None:
        super().__init__()
        self.removed_indexes: list[int] = []
        self.pressed_ids: list[str] = []

    def compose(self) -> ComposeResult:
        yield _ProbeCard(index=7)

    @on(ConfigCard.Removed)
    def _on_removed(self, event: ConfigCard.Removed) -> None:
        self.removed_indexes.append(event.index)

    @on(Button.Pressed)
    def _on_button_pressed(self, event: Button.Pressed) -> None:
        self.pressed_ids.append(event.button.id or "")


class _MultiConfigCardApp(App):
    def compose(self) -> ComposeResult:
        yield _ProbeCard(index=0)
        yield _ProbeCard(index=1)


def test_large_config_cards_use_shared_base() -> None:
    assert issubclass(MCPConnectionCard, ConfigCard)
    assert issubclass(SubAgentCard, ConfigCard)


@pytest.mark.asyncio
async def test_config_card_delete_button_posts_removed_message() -> None:
    async with _ConfigCardApp().run_test() as pilot:
        pilot.app.query_one("#probe-delete-7", Button).press()
        await pilot.pause()

        assert pilot.app.removed_indexes == [7]
        assert pilot.app.pressed_ids == ["probe-delete-7"]


@pytest.mark.asyncio
async def test_config_card_delete_button_ids_include_card_index() -> None:
    async with _MultiConfigCardApp().run_test() as pilot:
        buttons = list(pilot.app.query(Button))

        assert [button.id for button in buttons] == ["probe-delete-0", "probe-delete-1"]


@pytest.mark.asyncio
async def test_config_card_default_css_applies_to_card_node_and_delete_glyph() -> None:
    async with _ConfigCardApp().run_test() as pilot:
        await pilot.pause()

        card = pilot.app.query_one(_ProbeCard)
        title = card.query_one(".config-card-title", Label)
        delete_button = card.query_one("#probe-delete-7", Button)

        assert card.styles.padding.left == 2
        assert card.styles.padding.top == 1
        assert card.styles.background.a > 0
        assert str(title.styles.width) == "1fr"
        assert str(title.styles.text_style) == "bold"
        assert str(delete_button.styles.min_width) == "3"
        assert not delete_button.styles.border
        assert delete_button.styles.background.a == 0
