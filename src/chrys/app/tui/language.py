# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shared requested-locale choices and localized TUI copy."""

from __future__ import annotations

from chrys.foundation.config.settings import DEFAULT_LOCALE
from chrys.foundation.i18n import SUPPORTED_LOCALES, MessageDef, msg
from chrys.foundation.i18n.locale import ENGLISH_LOCALE, SIMPLIFIED_CHINESE_LOCALE

LANGUAGE_PICKER_TITLE = msg(
    "tui.language_picker.title",
    fallback="Language",
)
LANGUAGE_UNKNOWN_LOCALE = msg(
    "tui.language_command.unknown_locale",
    fallback="Unknown /language locale: {locale}",
)

_LANGUAGE_SYSTEM = msg(
    "tui.language_picker.system",
    fallback="Follow System",
)
_LANGUAGE_ENGLISH = msg(
    "tui.language_picker.english",
    fallback="English",
)
_LANGUAGE_SIMPLIFIED_CHINESE = msg(
    "tui.language_picker.simplified_chinese",
    fallback="简体中文",
)

_LANGUAGE_LABELS: dict[str, MessageDef] = {
    DEFAULT_LOCALE: _LANGUAGE_SYSTEM,
    ENGLISH_LOCALE: _LANGUAGE_ENGLISH,
    SIMPLIFIED_CHINESE_LOCALE: _LANGUAGE_SIMPLIFIED_CHINESE,
}


def language_label(locale: str) -> MessageDef | None:
    """Return the localized display-label definition for *locale*.

    ``None`` for locales without a label (e.g. custom docs languages); callers
    fall back to the raw locale string.
    """
    return _LANGUAGE_LABELS.get(locale)


LANGUAGE_OPTIONS = tuple(
    (requested_locale, _LANGUAGE_LABELS[requested_locale]) for requested_locale in (DEFAULT_LOCALE, *SUPPORTED_LOCALES)
)

__all__ = [
    "LANGUAGE_OPTIONS",
    "LANGUAGE_PICKER_TITLE",
    "LANGUAGE_UNKNOWN_LOCALE",
    "language_label",
]
