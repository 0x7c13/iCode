# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Save-time prompt-caching reminder: a Claude profile on the Anthropic protocol that never asks for caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Input, Static

from chrys.app.tui.screens.dialogs.prompt_cache import PromptCacheDialog
from chrys.app.tui.widgets import Select
from chrys.service.profiles.models.schema import ModelProfile
from tests.app.tui.screens._model_config_support import (
    _capture_notifications,
    _wait_for_kv_rows,
    open_model_config,
    single_profile_registry,
)
from tests.support.pilot_barrier import screen_is_settled
from tests.support.tui_helpers import click_when_settled, rich_plain
from tests.support.waiting import wait_for

pytestmark = pytest.mark.usefixtures("isolated_model_config_dir", "clear_model_profile_env")

_CACHE_CONTROL = {"type": "ephemeral"}


def _saved_options(profile: ModelProfile | None) -> dict[str, Any]:
    assert profile is not None
    return json.loads(profile.chat_options) if profile.chat_options else {}


async def _switch_to_claude(screen: Any, pilot: Any) -> None:
    screen.query_one("#mc-provider", Select).value = "anthropic"
    screen.query_one("#mc-model", Input).value = "claude-sonnet-5-5"
    await pilot.pause()


async def _press_save_and_wait_for_dialog(screen: Any, pilot: Any) -> PromptCacheDialog:
    screen.query_one("#mc-save", Button).press()
    await wait_for(
        lambda: isinstance(pilot.app.screen, PromptCacheDialog) and pilot.app.screen.is_mounted,
        pilot=pilot,
        description="prompt-caching reminder shown",
    )
    dialog = pilot.app.screen
    assert isinstance(dialog, PromptCacheDialog)
    return dialog


@pytest.mark.parametrize(
    ("chat_options", "expected"),
    [
        pytest.param("", {"extra_body": {"cache_control": _CACHE_CONTROL}}, id="blank-row"),
        pytest.param(
            json.dumps({"temperature": 0.2}),
            {"temperature": 0.2, "extra_body": {"cache_control": _CACHE_CONTROL}},
            id="new-row",
        ),
        pytest.param(
            json.dumps({"extra_body": {"top_k": 5}}),
            {"extra_body": {"top_k": 5, "cache_control": _CACHE_CONTROL}},
            id="merge-extra-body",
        ),
    ],
)
async def test_add_and_save_merges_cache_control_into_extra_body(
    chat_options: str,
    expected: dict[str, Any],
    tmp_path: Path,
) -> None:
    registry, _profile = single_profile_registry(provider="openai", chat_options=chat_options)

    async with open_model_config(registry, global_default_profile_id="model-a") as (screen, pilot):
        await _wait_for_kv_rows(screen.query_one("#mc-options-list"), pilot, 1)
        await _switch_to_claude(screen, pilot)
        captured = _capture_notifications(screen)
        dialog = await _press_save_and_wait_for_dialog(screen, pilot)
        message = rich_plain(dialog.query_one("#prompt-cache-message", Static).render())
        assert '"cache_control": {"type": "ephemeral"}' in message

        dialog.query_one("#prompt-cache-add", Button).press()
        await wait_for(
            lambda: ("information", "Model profile saved") in captured,
            pilot=pilot,
            description="profile saved after Add and Save",
        )

        saved = registry.get("model-a")
        assert pilot.app.screen is screen

    assert _saved_options(saved) == expected
    assert saved is not None
    assert saved.provider == "anthropic"
    assert (tmp_path / "models" / "model-a.yaml").is_file()


async def test_save_as_is_keeps_options_unchanged(tmp_path: Path) -> None:
    registry, _profile = single_profile_registry(provider="openai", chat_options=json.dumps({"temperature": 0.2}))

    async with open_model_config(registry, global_default_profile_id="model-a") as (screen, pilot):
        await _switch_to_claude(screen, pilot)
        captured = _capture_notifications(screen)
        dialog = await _press_save_and_wait_for_dialog(screen, pilot)

        dialog.query_one("#prompt-cache-save", Button).press()
        await wait_for(
            lambda: ("information", "Model profile saved") in captured,
            pilot=pilot,
            description="profile saved as is",
        )
        saved = registry.get("model-a")

    assert _saved_options(saved) == {"temperature": 0.2}
    assert saved is not None
    assert saved.model_id == "claude-sonnet-5-5"
    assert (tmp_path / "models" / "model-a.yaml").is_file()


@pytest.mark.parametrize(
    ("size", "scrolls"),
    [((120, 40), False), ((80, 15), True), ((70, 12), True)],
    ids=["120x40", "80x15", "70x12"],
)
async def test_reminder_text_scrolls_only_when_the_terminal_is_short(size: tuple[int, int], scrolls: bool) -> None:
    registry, _profile = single_profile_registry(provider="openai")

    async with open_model_config(registry, size=size, global_default_profile_id="model-a") as (screen, pilot):
        await _switch_to_claude(screen, pilot)
        dialog = await _press_save_and_wait_for_dialog(screen, pilot)
        await wait_for(lambda: screen_is_settled(pilot.app, dialog), pilot=pilot, description="dialog laid out")
        container = dialog.query_one("#prompt-cache-container").region
        scroll = dialog.query_one("#prompt-cache-inner", VerticalScroll)
        scroll_region = scroll.region
        max_scroll_y = scroll.max_scroll_y
        message_height = dialog.query_one("#prompt-cache-message").outer_size.height
        buttons = dialog.query_one("#prompt-cache-buttons").region

    # Text that doesn't fit scrolls inside the dialog instead of being clipped by it.
    assert (max_scroll_y > 0) is scrolls
    assert scroll_region.height + max_scroll_y == message_height + 2
    assert container.contains_region(scroll_region)
    assert container.contains_region(buttons)


@pytest.mark.parametrize("dismiss_by", ["escape", "backdrop-click"])
async def test_dismissing_the_reminder_returns_to_the_form_without_saving(dismiss_by: str, tmp_path: Path) -> None:
    registry, profile = single_profile_registry(provider="openai")

    async with open_model_config(registry, global_default_profile_id="model-a") as (screen, pilot):
        await _switch_to_claude(screen, pilot)
        captured = _capture_notifications(screen)
        dialog = await _press_save_and_wait_for_dialog(screen, pilot)

        if dismiss_by == "escape":
            await pilot.press("escape")
        else:
            await click_when_settled(pilot, dialog, offset=(0, 0))
        await wait_for(lambda: pilot.app.screen is screen, pilot=pilot, description="back on the form")
        model_value = screen.query_one("#mc-model", Input).value
        # Dismissing re-arms Save: the next press asks again.
        await _press_save_and_wait_for_dialog(screen, pilot)

    assert captured == []
    assert registry.get("model-a") is profile
    assert model_value == "claude-sonnet-5-5"
    assert not (tmp_path / "models" / "model-a.yaml").exists()


async def test_second_save_press_before_the_reminder_mounts_is_ignored(tmp_path: Path) -> None:
    registry, _profile = single_profile_registry(provider="openai")

    async with open_model_config(registry, global_default_profile_id="model-a") as (screen, pilot):
        await _switch_to_claude(screen, pilot)
        captured = _capture_notifications(screen)
        screen.query_one("#mc-save", Button).press()
        dialog = await _press_save_and_wait_for_dialog(screen, pilot)

        dialog.query_one("#prompt-cache-save", Button).press()
        await wait_for(
            lambda: ("information", "Model profile saved") in captured,
            pilot=pilot,
            description="profile saved as is",
        )
        stacked_reminders = [shown for shown in pilot.app.screen_stack if isinstance(shown, PromptCacheDialog)]
        top = pilot.app.screen

    assert stacked_reminders == []
    assert top is screen
    assert captured.count(("information", "Model profile saved")) == 1
    assert (tmp_path / "models" / "model-a.yaml").is_file()


@pytest.mark.parametrize(
    ("overrides", "edit_model"),
    [
        pytest.param(
            {"provider": "anthropic", "model_id": "claude-sonnet-5-5"},
            None,
            id="already-a-saved-claude-profile",
        ),
        pytest.param(
            {"provider": "openai", "chat_options": json.dumps({"cache_control": _CACHE_CONTROL})},
            "claude-sonnet-5-5",
            id="top-level-cache-control",
        ),
        pytest.param(
            {"provider": "openai", "chat_options": json.dumps({"extra_body": {"cache_control": _CACHE_CONTROL}})},
            "claude-sonnet-5-5",
            id="extra-body-cache-control",
        ),
        pytest.param({"provider": "openai"}, "kimi-k2", id="not-a-claude-model"),
    ],
)
async def test_save_goes_straight_through_without_the_reminder(
    overrides: dict[str, Any],
    edit_model: str | None,
    tmp_path: Path,
) -> None:
    registry, _profile = single_profile_registry(**overrides)

    async with open_model_config(registry, global_default_profile_id="model-a") as (screen, pilot):
        screen.query_one("#mc-provider", Select).value = "anthropic"
        if edit_model is not None:
            screen.query_one("#mc-model", Input).value = edit_model
        screen.query_one("#mc-name", Input).value = "Renamed"
        await pilot.pause()
        captured = _capture_notifications(screen)
        screen.query_one("#mc-save", Button).press()
        await wait_for(
            lambda: ("information", "Model profile saved") in captured,
            pilot=pilot,
            description="profile saved without the reminder",
        )
        saw_dialog = isinstance(pilot.app.screen, PromptCacheDialog)

    assert not saw_dialog
    saved = registry.get("model-a")
    assert saved is not None
    assert saved.name == "Renamed"
    assert (tmp_path / "models" / "model-a.yaml").is_file()


async def test_hollow_new_profile_counts_as_not_yet_claude(tmp_path: Path) -> None:
    """A never-filled profile (blank model) gets the reminder on its first complete save."""
    registry, _profile = single_profile_registry(provider="anthropic", model_id="")

    async with open_model_config(registry, global_default_profile_id="model-a") as (screen, pilot):
        await _switch_to_claude(screen, pilot)
        dialog = await _press_save_and_wait_for_dialog(screen, pilot)
        dialog.query_one("#prompt-cache-save", Button).press()
        await wait_for(lambda: pilot.app.screen is screen, pilot=pilot, description="dialog closed")
        await wait_for(
            (tmp_path / "models" / "model-a.yaml").is_file,
            pilot=pilot,
            description="profile file written",
        )
