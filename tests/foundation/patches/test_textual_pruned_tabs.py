# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the patch that lets a tab bar being removed ignore activation."""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Tab, TabbedContent, TabPane, Tabs

from chrys.foundation.patches.textual_pruned_tabs import apply_runtime_patch

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.widget import Widget


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield Container(id="host")


class _Panes(Container):
    def compose(self) -> ComposeResult:
        with TabbedContent(initial="two"):
            yield TabPane("One", id="one")
            yield TabPane("Two", id="two")


def _tabs() -> Widget:
    return Tabs(Tab("One", id="one"), Tab("Two", id="two"), active="two")


_BARS: dict[str, Callable[[], Widget]] = {"tabs": _tabs, "tabbed-content": _Panes}


async def _mount_then_drop(bar: Callable[[], Widget], turns: int, *, exit_app: bool) -> None:
    """Mount a bar that opens on a later tab, and drop it ``turns`` loop turns later."""
    app = _Host()
    async with app.run_test() as pilot:
        host = app.query_one("#host", Container)
        host.mount(bar())
        # The turns pick how far the bar's nested mount has got when it is removed.
        for _ in range(turns):
            await asyncio.sleep(0)
        if not exit_app:
            await host.remove_children()
            await pilot.pause()
            assert not host.children and app.is_running
    # Leaving run_test re-raises what took the App down, such as "No Tab with id ...".


async def test_upstream_crashes_on_a_bar_removed_before_its_tabs_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins that the patch is still needed; drop the patch once a Textual upgrade makes this fail."""
    apply_runtime_patch()
    monkeypatch.setattr(Tabs, "validate_active", inspect.unwrap(Tabs.validate_active))

    with pytest.raises(ValueError, match="No Tab with id 'two'"):
        await _mount_then_drop(_tabs, 0, exit_app=False)


@pytest.mark.parametrize("turns", range(6))
@pytest.mark.parametrize("exit_app", [False, True], ids=["removed", "app-exit"])
@pytest.mark.parametrize("bar", list(_BARS))
async def test_a_bar_dropped_before_its_tabs_mount_leaves_the_app_running(bar: str, exit_app: bool, turns: int) -> None:
    apply_runtime_patch()

    await _mount_then_drop(_BARS[bar], turns, exit_app=exit_app)


async def test_a_live_bar_still_rejects_an_unknown_tab() -> None:
    apply_runtime_patch()
    app = _Host()
    async with app.run_test():
        tabs = Tabs(Tab("One", id="one"), Tab("Two", id="two"), active="two")
        await app.query_one("#host", Container).mount(tabs)
        assert tabs.active == "two"

        with pytest.raises(ValueError, match="No Tab with id 'missing'"):
            tabs.active = "missing"

        tabs.active = "one"
        assert tabs.active_tab is not None and tabs.active_tab.id == "one"


def test_runtime_patch_is_idempotent() -> None:
    apply_runtime_patch()
    patched = Tabs.validate_active

    apply_runtime_patch()

    assert Tabs.validate_active is patched
    assert inspect.unwrap(Tabs.validate_active) is not patched
