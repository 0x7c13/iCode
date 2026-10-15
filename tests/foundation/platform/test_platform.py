# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for platform detection module."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from chrys.foundation.platform import PlatformInfo, ShellInfo, conventional_config_dir, detect_platform, get_platform
from chrys.foundation.platform.files import atomic_write_owner_only_bytes
from tests.support.symlinks import junction_or_skip, symlink_or_skip


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


@pytest.mark.parametrize("linked", ["home", "home by a junction", "config folder"])
def test_the_config_dir_is_the_real_folder_behind_a_link(
    linked: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Files there are opened by helpers that refuse a link anywhere on the path,
    so detection follows the links once; the name the user knows stays at hand.

    On Windows "home" is AppData, which is moved to another drive with a
    junction as often as with a symlink.
    """
    from tests import conftest as harness

    name = "chrys" if sys.platform == "win32" else ".chrys"
    home = tmp_path / "home"
    if linked.startswith("home"):
        (tmp_path / "real-home").mkdir()
        if linked == "home by a junction":
            junction_or_skip(home, tmp_path / "real-home")
        else:
            symlink_or_skip(home, tmp_path / "real-home", target_is_directory=True)
        real = tmp_path / "real-home" / name
    else:
        home.mkdir()
        real = tmp_path / "dotfiles" / "chrys"
        real.mkdir(parents=True)
        symlink_or_skip(home / name, real, target_is_directory=True)
    monkeypatch.setenv("APPDATA" if sys.platform == "win32" else "HOME", str(home))

    # Real detection: the suite's own pins the config dir to a temp folder.
    info = harness._original_detect_platform()

    assert info.config_dir == Path(os.path.realpath(real))
    assert info.config_dir != home / name
    assert info.data_dir == info.config_dir
    assert conventional_config_dir() == home / name
    # What the links used to break: an owner-only file written there.
    atomic_write_owner_only_bytes(info.config_dir / "key", b"secret")
    assert (real / "key").read_bytes() == b"secret"


@pytest.mark.parametrize("linked", ["config folder", "home"])
def test_a_linked_folder_whose_disk_is_missing_keeps_its_name(
    linked: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Followed to the empty mount point, a new data folder would be created there
    and hidden once the disk comes back; by its name, creating it fails instead."""
    from tests import conftest as harness

    name = "chrys" if sys.platform == "win32" else ".chrys"
    home = tmp_path / "home"
    mount_point = tmp_path / "mnt" / "ext"
    mount_point.mkdir(parents=True)
    if linked == "home":
        symlink_or_skip(home, mount_point / "home", target_is_directory=True)
    else:
        home.mkdir()
        symlink_or_skip(home / name, mount_point / "chrys", target_is_directory=True)
    monkeypatch.setenv("APPDATA" if sys.platform == "win32" else "HOME", str(home))

    info = harness._original_detect_platform()

    assert info.config_dir == home / name
    # As session storage creates it on first use.
    with pytest.raises(OSError):
        info.config_dir.mkdir(parents=True, exist_ok=True)
    assert list(mount_point.iterdir()) == []
