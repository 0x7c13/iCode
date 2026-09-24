# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for main-screen action availability rules."""

from __future__ import annotations

from chrys.app.tui.screens.main.action_availability import check_main_action


def _check(
    action: str,
    *,
    fullscreen_terminal: bool = False,
    shell_mode: bool = False,
    chat_foreground: bool = True,
    suggestions_active: bool = False,
    agent_running: bool = False,
) -> bool | None:
    return check_main_action(
        action,
        fullscreen_terminal=fullscreen_terminal,
        shell_mode=shell_mode,
        chat_foreground=chat_foreground,
        suggestions_active=suggestions_active,
        agent_running=agent_running,
    )


def test_fullscreen_terminal_disables_every_action() -> None:
    actions = ["toggle_sidebar", "quit", "chat_page_up", "interrupt", "sessions", "pick_theme", "other"]

    assert all(_check(action, fullscreen_terminal=True) is False for action in actions)


def test_shell_mode_allows_only_sidebar_and_quit() -> None:
    assert _check("toggle_sidebar", shell_mode=True) is True
    assert _check("quit", shell_mode=True) is True
    assert _check("interrupt", shell_mode=True, agent_running=True) is False


def test_chat_scroll_actions_require_unobscured_chat() -> None:
    for action in ("chat_page_up", "chat_page_down", "chat_scroll_bottom"):
        assert _check(action) is True
        assert _check(action, chat_foreground=False) is False
        assert _check(action, suggestions_active=True) is False


def test_prompt_history_requires_visible_chat_but_allows_running_agent() -> None:
    assert _check("prompt_history") is True
    assert _check("prompt_history", agent_running=True) is True
    assert _check("prompt_history", chat_foreground=False) is False
    assert _check("prompt_history", shell_mode=True) is False


def test_interrupt_requires_running_agent() -> None:
    assert _check("interrupt", agent_running=True) is True
    assert _check("interrupt", agent_running=False) is False


def test_sessions_require_idle_agent() -> None:
    assert _check("sessions", agent_running=False) is True
    assert _check("sessions", agent_running=True) is False


def test_theme_picker_requires_idle_agent() -> None:
    assert _check("pick_theme", agent_running=False) is True
    assert _check("pick_theme", agent_running=True) is False


def test_settings_require_idle_agent() -> None:
    assert _check("settings", agent_running=False) is True
    assert _check("settings", agent_running=True) is False
