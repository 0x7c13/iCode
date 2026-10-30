# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Pinned message ids for prompt shortcut expansion and its session-switch card."""

from __future__ import annotations

PROMPT_MESSAGE_IDS: frozenset[str | tuple[str, str]] = frozenset(
    {
        "tui.prompt_card.agent",
        "tui.prompt_card.agent_not_found",
        "tui.prompt_card.busy",
        "tui.prompt_card.busy_commands",
        "tui.prompt_card.cancelled",
        ("tui.prompt_card.command_count", "tui.prompt_card.command_count#plural"),
        "tui.prompt_card.commands",
        "tui.prompt_card.dont_switch",
        "tui.prompt_card.model",
        "tui.prompt_card.model_locked",
        "tui.prompt_card.model_not_found",
        "tui.prompt_card.note",
        "tui.prompt_card.remove",
        "tui.prompt_card.running",
        "tui.prompt_card.loading",
        "tui.prompt_shortcuts.loading",
        "tui.prompt_shortcuts.loading_tab",
        "tui.prompt_card.send_failed",
        "tui.prompt_card.session_changed",
        "tui.prompt_card.shell_start_failed",
        "tui.prompt_card.shell_stopped",
        "tui.prompt_card.shell_timeout",
        "tui.prompt_card.switch_failed",
        "tui.prompt_card.title",
        "tui.prompt_shortcuts.hint.argument",
        "tui.prompt_shortcuts.hint.arguments",
        "tui.suggestions.prompt_shortcuts",
    }
)
