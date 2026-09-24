# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Custom Textual widgets for the Chrys TUI."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ASK_USER_INPUT_MAX_HEIGHT": "ask_user_controls",
    "ASK_USER_INPUT_MIN_HEIGHT": "ask_user_controls",
    "ASK_USER_INTERACTIVE_CLASS": "ask_user_controls",
    "AskUserActiveQuestionChanged": "ask_user_prompt",
    "AskUserInlineRequested": "ask_user_controls",
    "AskUserOptionChosen": "ask_user_controls",
    "AskUserOptions": "ask_user_controls",
    "AskUserPrompt": "ask_user_prompt",
    "AskUserQuestionPane": "ask_user_prompt",
    "AskUserResponseFooter": "ask_user_controls",
    "AskUserResponseResized": "ask_user_controls",
    "AskUserReviewPane": "ask_user_prompt",
    "AskUserSubmitted": "ask_user_controls",
    "AskUserTabbedContent": "ask_user_prompt",
    "CHECKED_MARKER": "checkbox",
    "ChrysLoadingIndicator": "loading",
    "Checkbox": "checkbox",
    "ConfigActionButton": "buttons",
    "ConfigAddButton": "buttons",
    "DialogButtonRow": "dialog_buttons",
    "DialogButtonSpec": "dialog_buttons",
    "EnhancedInput": "input",
    "EnhancedTextArea": "text_area",
    "HATCH_GLYPH": "hatch",
    "hatch_style": "hatch",
    "HatchedEmptyState": "hatch",
    "PromptDraft": "ask_user_prompt",
    "Select": "select",
    "StableAutoHeightScroll": "dialog_scroll",
    "UNCHECKED_MARKER": "checkbox",
    "normalize_selection_rich_style": "selection",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily resolve curated leaf widgets without importing feature packages."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{_EXPORTS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the default module attributes plus facade exports."""
    return sorted({*globals(), *__all__})
