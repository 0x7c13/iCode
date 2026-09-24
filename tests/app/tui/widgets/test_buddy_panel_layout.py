# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The buddy sits in the middle of a tall sidebar and at the top of a short one."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from chrys.app.features.buddy.model import Rarity, Species
from chrys.app.tui.util.visibility import is_widget_shown
from chrys.app.tui.widgets.sidebar import buddy as buddy_module
from chrys.app.tui.widgets.sidebar.buddy import BuddyPanel
from tests.support.buddies import a_buddy, turns_to_finish
from tests.support.waiting import wait_for


class _BuddyApp(App[None]):
    def compose(self) -> ComposeResult:
        yield BuddyPanel()


def _block_height(panel: BuddyPanel) -> int:
    """Rows from the portrait's top to the last detail's bottom margin: what the panel aligns."""
    sprite = panel.query_one("#buddy-sprite", Static)
    info = panel.query_one("#buddy-info", Static)
    return info.region.bottom + info.styles.margin.bottom - sprite.region.y


def _top_gap(panel: BuddyPanel) -> int:
    return panel.query_one("#buddy-sprite", Static).region.y - panel.content_region.y


@pytest.fixture
def lucas(monkeypatch: pytest.MonkeyPatch) -> None:
    buddy = a_buddy(rarity=Rarity.SSR, species=Species.RABBIT, shiny=True, name="Lucas", turns=turns_to_finish(1))
    monkeypatch.setattr(buddy_module, "current_buddy", lambda: buddy)


@pytest.mark.usefixtures("lucas")
async def test_a_tall_sidebar_centers_the_buddy_and_its_details() -> None:
    async with _BuddyApp().run_test(size=(28, 40)) as pilot:
        panel = pilot.app.query_one(BuddyPanel)
        sprite = panel.query_one("#buddy-sprite", Static)
        await wait_for(lambda: is_widget_shown(sprite), pilot=pilot)

        # The panel first lays the block out at the top, then aligns it.
        await wait_for(lambda: _top_gap(panel) > 0, pilot=pilot, description="the buddy block moved down")
        assert _top_gap(panel) == (panel.content_region.height - _block_height(panel)) // 2
        assert _block_height(panel) < panel.content_region.height
        assert panel.max_scroll_y == 0


@pytest.mark.usefixtures("lucas")
async def test_a_short_sidebar_keeps_the_buddy_at_the_top_and_scrolls() -> None:
    async with _BuddyApp().run_test(size=(24, 15)) as pilot:
        panel = pilot.app.query_one(BuddyPanel)
        sprite = panel.query_one("#buddy-sprite", Static)
        await wait_for(lambda: is_widget_shown(sprite), pilot=pilot)

        await wait_for(lambda: _block_height(panel) > panel.content_region.height, pilot=pilot, description="overflow")
        assert _top_gap(panel) == 0
        assert panel.max_scroll_y > 0
