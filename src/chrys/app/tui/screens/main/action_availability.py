# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Action availability rules for the main screen."""

from __future__ import annotations


def check_main_action(
    action: str,
    *,
    fullscreen_terminal: bool,
    shell_mode: bool,
    chat_foreground: bool,
    suggestions_active: bool,
    agent_running: bool,
) -> bool | None:
    """Return whether a Textual action should be available."""
    if fullscreen_terminal:
        return False
    if shell_mode and action not in ("toggle_sidebar", "quit"):
        return False
    if action in {"chat_page_up", "chat_page_down", "chat_scroll_bottom"}:
        return chat_foreground and not suggestions_active
    if action == "prompt_history":
        return chat_foreground
    if action == "interrupt":
        return agent_running
    if action in {"sessions", "pick_theme", "settings"}:
        return not agent_running
    return True
