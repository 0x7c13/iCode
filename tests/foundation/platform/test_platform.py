# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for platform detection module."""

from __future__ import annotations

import pytest

from chrys.foundation.platform import PlatformInfo, ShellInfo, detect_platform, get_platform


def test_detect_platform_returns_platform_info() -> None:
    info = detect_platform()
    assert isinstance(info, PlatformInfo)
    assert info.os_name in ("windows", "macos", "linux")
    assert info.arch  # non-empty


def test_platform_os_booleans() -> None:
    info = detect_platform()
    # Exactly one OS boolean should be True
    os_flags = [info.is_windows, info.is_macos, info.is_linux]
    assert sum(os_flags) == 1


def test_platform_arch_booleans() -> None:
    info = detect_platform()
    # At least one arch boolean should be True on common hardware
    arch_flags = [info.is_arm64, info.is_amd64, info.is_x86]
    assert sum(arch_flags) >= 1


def test_platform_shell_info() -> None:
    info = detect_platform()
    assert isinstance(info.shell, ShellInfo)
    assert info.shell.name
    assert info.shell.path
    assert info.shell.args


def test_get_platform_is_cached() -> None:
    a = get_platform()
    b = get_platform()
    assert a is b  # Same object due to @functools.cache


@pytest.mark.real_config_dir
def test_platform_config_dir_exists() -> None:
    # Asserts where the *real* config dir lives, so it opts out of the autouse
    # redirect that points every other test at a temp directory.
    info = detect_platform()
    # config_dir should be a Path (may not exist yet, but should be valid)
    assert info.config_dir.name == "chrys" or info.config_dir.name.endswith("chrys")
