# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Supported locale normalization and side-effect-free system detection."""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Any, cast

from chrys.foundation.platform import get_platform

logger = logging.getLogger(__name__)

ENGLISH_LOCALE = "en"
SIMPLIFIED_CHINESE_LOCALE = "zh-Hans"
SUPPORTED_LOCALES = (ENGLISH_LOCALE, SIMPLIFIED_CHINESE_LOCALE)

_ENGLISH_PREFIXES = ("en-", "english-")
_SIMPLIFIED_CHINESE_PREFIXES = ("zh-cn-", "zh-hans-", "zh-sg-", "zh-chs-")
_WINDOWS_LOCALE_NAME_MAX_LENGTH = 85


def normalize_locale(value: str) -> str:
    """Resolve one requested locale to the supported effective locale.

    ``system`` consults the operating system without mutating the process C
    locale. Unsupported system languages quietly use English; malformed or
    unsupported explicit settings emit one content-free diagnostic.
    """
    if type(value) is not str:
        _warn_unsupported()
        return ENGLISH_LOCALE

    requested = value.strip()
    if requested.casefold() == "system":
        system_locale = _system_locale_name()
        if system_locale is None:
            logger.warning("System locale unavailable; using English.")
            return ENGLISH_LOCALE
        return (
            SIMPLIFIED_CHINESE_LOCALE
            if _normalize_known_locale(system_locale) == SIMPLIFIED_CHINESE_LOCALE
            else ENGLISH_LOCALE
        )

    normalized = _normalize_known_locale(requested)
    if normalized is not None:
        return normalized
    _warn_unsupported()
    return ENGLISH_LOCALE


def _normalize_known_locale(value: str) -> str | None:
    normalized = value.strip().casefold().replace("_", "-")
    normalized = normalized.split("@", maxsplit=1)[0].split(".", maxsplit=1)[0]
    normalized = "-".join(normalized.split())

    if normalized in {"c", "posix", "en", "english"} or normalized.startswith(_ENGLISH_PREFIXES):
        return ENGLISH_LOCALE
    if normalized in {"zh", "zh-cn", "zh-hans", "zh-sg", "zh-chs", "chinese-(simplified)-china"}:
        return SIMPLIFIED_CHINESE_LOCALE
    if normalized.startswith(_SIMPLIFIED_CHINESE_PREFIXES):
        return SIMPLIFIED_CHINESE_LOCALE
    return None


def _system_locale_name() -> str | None:
    if get_platform().is_windows:
        return _windows_locale_name()
    return os.environ.get("LC_ALL") or os.environ.get("LC_MESSAGES") or os.environ.get("LANG") or None


def _windows_locale_name() -> str | None:
    """Read the Windows user-default locale through ``GetUserDefaultLocaleName``."""
    try:
        windows_ctypes = cast(Any, ctypes)
        kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
        get_user_default_locale_name = kernel32.GetUserDefaultLocaleName
        get_user_default_locale_name.argtypes = [ctypes.POINTER(ctypes.c_wchar), ctypes.c_int]
        get_user_default_locale_name.restype = ctypes.c_int
        buffer = ctypes.create_unicode_buffer(_WINDOWS_LOCALE_NAME_MAX_LENGTH)
        if get_user_default_locale_name(buffer, len(buffer)) == 0:
            return None
        return buffer.value or None
    except OSError, AttributeError, TypeError:
        return None


def _warn_unsupported() -> None:
    logger.warning("Unsupported locale setting; using English.")
