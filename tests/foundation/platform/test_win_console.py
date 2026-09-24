# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for chrys.foundation.platform.win_console — preserve_console_mode."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from chrys.foundation.platform import win_console
from chrys.foundation.platform.win_console import preserve_console_mode


def test_noop_on_non_windows() -> None:
    """On non-Windows platforms the context manager does nothing."""
    if sys.platform == "win32":
        pytest.skip("non-Windows behavior check")

    entered = False
    with preserve_console_mode():
        entered = True
    assert entered


def test_snapshot_and_restore_calls_kernel32() -> None:
    """On Windows, GetConsoleMode is called on enter and SetConsoleMode on exit."""
    fake_get_calls: list[int] = []
    fake_set_calls: list[tuple[int, int]] = []

    def fake_get(handle_id: int) -> int | None:
        fake_get_calls.append(handle_id)
        # Pretend the input handle has mouse flags enabled, output has VT.
        return {win_console._STD_INPUT_HANDLE: 0x1F8, win_console._STD_OUTPUT_HANDLE: 0x0007}.get(handle_id)

    def fake_set(handle_id: int, mode: int) -> None:
        fake_set_calls.append((handle_id, mode))

    with (
        patch.object(win_console.sys, "platform", "win32"),
        patch.object(win_console, "_get_console_mode", side_effect=fake_get),
        patch.object(win_console, "_set_console_mode", side_effect=fake_set),
        preserve_console_mode(),
    ):
        pass

    assert fake_get_calls == [win_console._STD_INPUT_HANDLE, win_console._STD_OUTPUT_HANDLE]
    assert (win_console._STD_INPUT_HANDLE, 0x1F8) in fake_set_calls
    assert (win_console._STD_OUTPUT_HANDLE, 0x0007) in fake_set_calls


def test_restore_skipped_when_snapshot_unavailable() -> None:
    """If GetConsoleMode fails (None), no restore is attempted."""
    fake_set_calls: list[tuple[int, int]] = []

    def fake_set(handle_id: int, mode: int) -> None:
        fake_set_calls.append((handle_id, mode))

    with (
        patch.object(win_console.sys, "platform", "win32"),
        patch.object(win_console, "_get_console_mode", return_value=None),
        patch.object(win_console, "_set_console_mode", side_effect=fake_set),
        preserve_console_mode(),
    ):
        pass

    assert fake_set_calls == []


def test_restore_happens_on_exception() -> None:
    """Console mode is restored even if the guarded block raises."""
    fake_set_calls: list[tuple[int, int]] = []

    def fake_set(handle_id: int, mode: int) -> None:
        fake_set_calls.append((handle_id, mode))

    with (
        patch.object(win_console.sys, "platform", "win32"),
        patch.object(win_console, "_get_console_mode", return_value=0x123),
        patch.object(win_console, "_set_console_mode", side_effect=fake_set),
        pytest.raises(RuntimeError, match="boom"),
        preserve_console_mode(),
    ):
        raise RuntimeError("boom")

    # Both handles restored to the snapshotted value.
    assert (win_console._STD_INPUT_HANDLE, 0x123) in fake_set_calls
    assert (win_console._STD_OUTPUT_HANDLE, 0x123) in fake_set_calls


def test_get_console_mode_non_windows_returns_none() -> None:
    """Helper returns None on non-Windows without touching ctypes."""
    if sys.platform == "win32":
        pytest.skip("non-Windows branch")

    assert win_console._get_console_mode(win_console._STD_INPUT_HANDLE) is None


def test_get_console_mode_handles_ctypes_errors() -> None:
    """Helper swallows OSError / AttributeError from the ctypes layer."""
    fake_ctypes = MagicMock()
    fake_ctypes.windll.kernel32.GetStdHandle.side_effect = OSError("no console")
    with (
        patch.object(win_console.sys, "platform", "win32"),
        patch.dict(sys.modules, {"ctypes": fake_ctypes}),
    ):
        assert win_console._get_console_mode(win_console._STD_INPUT_HANDLE) is None
