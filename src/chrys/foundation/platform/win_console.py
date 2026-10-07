# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Windows console-mode preservation.

When the chrys TUI spawns a child process on Windows (e.g. the agent
runs ``uv run chrys`` via PowerShell through ``ShellTools``), the child
can mutate the mode flags of any console input/output handle it has
access to.  Even after PR #146 (``CREATE_NEW_CONSOLE + SW_HIDE``), the
child still inherits the parent's standard handles unless ``stdin`` /
``stdout`` / ``stderr`` are explicitly redirected — so a grandchild TUI
calling ``SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), …)`` can wipe
``ENABLE_MOUSE_INPUT`` / ``ENABLE_WINDOW_INPUT`` on the outer TUI and
leave mouse and keyboard events broken after it exits.

:func:`preserve_console_mode` snapshots the outer console mode on entry
and restores it on exit, protecting the TUI regardless of what the
child (or its descendants) do.  On non-Windows platforms it is a
no-op.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from typing import Any

from chrys.foundation.platform.c_api import declare_functions

_STD_INPUT_HANDLE = -10
_STD_OUTPUT_HANDLE = -11


def _load_kernel32() -> Any:
    """Load ``kernel32.dll`` with correct argtypes / restype for 64-bit safety.

    Without explicit ``restype`` on ``GetStdHandle``, ctypes defaults to
    ``c_int`` and truncates the 64-bit ``HANDLE`` return value, silently
    breaking ``GetConsoleMode`` / ``SetConsoleMode``.  Returns ``None`` on
    non-Windows platforms or if the DLL cannot be loaded.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        declare_functions(
            kernel32,
            {
                "GetStdHandle": (wintypes.HANDLE, [wintypes.DWORD]),
                "GetConsoleMode": (wintypes.BOOL, [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]),
                "SetConsoleMode": (wintypes.BOOL, [wintypes.HANDLE, wintypes.DWORD]),
            },
        )
        return kernel32
    except OSError, AttributeError:
        return None


def _get_console_mode(handle_id: int) -> int | None:
    """Return the current console mode for a standard handle, or ``None``.

    ``None`` is returned if the handle is not a console (e.g. redirected
    to a pipe or file), if the Windows API call fails, or if called on
    a non-Windows platform.
    """
    kernel32 = _load_kernel32()
    if kernel32 is None:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        handle = kernel32.GetStdHandle(handle_id)
        # ``INVALID_HANDLE_VALUE`` is (HANDLE)-1 — as an unsigned pointer
        # on 64-bit that is ``0xFFFFFFFFFFFFFFFF``.  ``GetStdHandle`` can
        # also return ``NULL`` when there is no associated standard
        # handle (e.g. for a detached GUI process).
        if not handle or handle == wintypes.HANDLE(-1).value:
            return None
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return None
        return int(mode.value)
    except OSError, AttributeError:
        return None


def _set_console_mode(handle_id: int, mode: int) -> None:
    """Restore the console mode for a standard handle. Best-effort."""
    kernel32 = _load_kernel32()
    if kernel32 is None:
        return
    try:
        from ctypes import wintypes

        handle = kernel32.GetStdHandle(handle_id)
        if not handle or handle == wintypes.HANDLE(-1).value:
            return
        kernel32.SetConsoleMode(handle, mode)
    except OSError, AttributeError:
        pass


@contextlib.contextmanager
def preserve_console_mode() -> Iterator[None]:
    """Snapshot and restore Windows console mode around a block.

    Wrap any subprocess that may spawn a Windows console TUI (shell tool,
    skill-script runner) so the outer chrys TUI's mouse / keyboard mode
    flags survive regardless of what the child does to them.

    No-op on non-Windows platforms.
    """
    if sys.platform != "win32":
        yield
        return

    saved_in = _get_console_mode(_STD_INPUT_HANDLE)
    saved_out = _get_console_mode(_STD_OUTPUT_HANDLE)
    try:
        yield
    finally:
        if saved_in is not None:
            _set_console_mode(_STD_INPUT_HANDLE, saved_in)
        if saved_out is not None:
            _set_console_mode(_STD_OUTPUT_HANDLE, saved_out)
