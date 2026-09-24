# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared semantic messages for Buddy toast presentation."""

from chrys.foundation.i18n import msg

BUDDY_TITLE = msg("tui.buddy.title", fallback="Buddy")

_TILTS_HEAD = msg("tui.buddy.thinking.tilts_head", fallback="💭 {name} tilts its head...")
_CHOOSES_WORDS = msg("tui.buddy.thinking.chooses_words", fallback="💭 {name} is choosing its words...")
_GLANCES = msg("tui.buddy.thinking.glances", fallback="💭 {name} looks at you, then at the screen...")
_ABOUT_TO_SPEAK = msg("tui.buddy.thinking.about_to_speak", fallback="💭 {name} is about to say something...")

# Shown while an answer to a pet is being written; one is picked at random.
THINKING_LINES = (_TILTS_HEAD, _CHOOSES_WORDS, _GLANCES, _ABOUT_TO_SPEAK)

__all__ = ["BUDDY_TITLE", "THINKING_LINES"]
