# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for supported locale normalization and system detection."""

from __future__ import annotations

import locale as stdlib_locale
import logging
from types import SimpleNamespace

import pytest

from chrys.foundation.i18n import locale as locale_module


@pytest.fixture(autouse=True)
def _english_system_locale() -> None:
    """Keep the real system detection: it is what this module tests, and the suite otherwise replaces it."""


@pytest.mark.parametrize(
    "value",
    [
        "en",
        "EN",
        "en-US",
        "en_US",
        "en-GB",
        "en_GB.UTF-8",
        "English_United States.1252",
    ],
)
def test_common_english_spellings_normalize_to_en(value: str) -> None:
    assert locale_module.normalize_locale(value) == "en"


@pytest.mark.parametrize(
    "value",
    [
        "zh",
        "zh-CN",
        "zh_CN",
        "zh-Hans",
        "zh_Hans_CN.UTF-8",
        "Chinese (Simplified)_China",
    ],
)
def test_supported_simplified_chinese_spellings_normalize_to_zh_hans(value: str) -> None:
    assert locale_module.normalize_locale(value) == "zh-Hans"


@pytest.mark.parametrize("value", ["", "fr-FR", "zh-TW", "not a locale", None])
def test_unsupported_or_invalid_values_fall_back_with_one_diagnostic(
    value: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=locale_module.__name__):
        assert locale_module.normalize_locale(value) == "en"  # ty: ignore[invalid-argument-type]

    assert [record.message for record in caplog.records] == ["Unsupported locale setting; using English."]
    if value:
        assert str(value) not in caplog.text


def test_system_uses_posix_locale_environment_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = SimpleNamespace(is_windows=False)
    monkeypatch.setattr(locale_module, "get_platform", lambda: platform)
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setenv("LC_MESSAGES", "zh_CN.UTF-8")
    monkeypatch.setenv("LC_ALL", "en_GB.UTF-8")

    assert locale_module.normalize_locale("system") == "en"

    monkeypatch.delenv("LC_ALL")
    assert locale_module.normalize_locale("system") == "zh-Hans"

    monkeypatch.delenv("LC_MESSAGES")
    assert locale_module.normalize_locale("system") == "en"


@pytest.mark.parametrize(
    ("system_value", "expected"),
    [
        ("zh_CN.UTF-8", "zh-Hans"),
        ("zh-Hans", "zh-Hans"),
        ("Chinese (Simplified)_China", "zh-Hans"),
        ("zh_TW", "en"),
        ("en_US", "en"),
    ],
)
def test_system_windows_uses_windows_locale_api(
    system_value: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = SimpleNamespace(is_windows=True)
    calls = 0

    def windows_locale_name() -> str:
        nonlocal calls
        calls += 1
        return system_value

    monkeypatch.setattr(locale_module, "get_platform", lambda: platform)
    monkeypatch.setattr(locale_module, "_windows_locale_name", windows_locale_name)

    assert locale_module.normalize_locale("system") == expected
    assert calls == 1


def test_system_detection_never_mutates_or_uses_deprecated_process_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = SimpleNamespace(is_windows=False)
    monkeypatch.setattr(locale_module, "get_platform", lambda: platform)
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("process-global locale API must not be called")

    monkeypatch.setattr(stdlib_locale, "setlocale", forbidden)
    monkeypatch.setattr(stdlib_locale, "getdefaultlocale", forbidden, raising=False)

    assert locale_module.normalize_locale("system") == "zh-Hans"


def test_system_detection_calls_foundation_platform_abstraction(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def platform_info() -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(is_windows=False)

    monkeypatch.setattr(locale_module, "get_platform", platform_info)
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    monkeypatch.setenv("LANG", "en_US.UTF-8")

    assert locale_module.normalize_locale("system") == "en"
    assert calls == 1


def test_unavailable_system_locale_falls_back_with_one_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(locale_module, "get_platform", lambda: SimpleNamespace(is_windows=False))
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    monkeypatch.delenv("LANG", raising=False)

    with caplog.at_level(logging.WARNING, logger=locale_module.__name__):
        assert locale_module.normalize_locale("system") == "en"

    assert [record.message for record in caplog.records] == ["System locale unavailable; using English."]
