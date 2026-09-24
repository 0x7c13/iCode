# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Open a local directory in the desktop file manager."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from chrys.app.tui.terminal.launcher import can_access_local_desktop
from chrys.foundation.platform import get_platform

_MACOS_OPEN_PATH = Path("/usr/bin/open")


def can_open_in_file_manager(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the current frontend can display a desktop file manager."""
    if not can_access_local_desktop(env):
        return False
    platform = get_platform()
    if platform.is_macos:
        return _MACOS_OPEN_PATH.is_file()
    if platform.is_windows:
        return callable(getattr(os, "startfile", None))
    return shutil.which("xdg-open") is not None


def open_in_file_manager(path: Path) -> None:
    """Reveal *path* in the platform file manager.

    Raises :class:`OSError` when no opener could be started; the caller owns
    the user-facing message. Windows failures are normalised into that contract
    because the shell hand-off can report one of its own outside it.
    """
    platform = get_platform()
    if platform.is_macos:
        argv = [str(_MACOS_OPEN_PATH), str(path)]
    elif platform.is_windows:
        start_file = getattr(os, "startfile", None)
        if not callable(start_file):
            raise OSError("os.startfile is unavailable")
        try:
            start_file(str(path))
        except NotImplementedError as error:
            # CPython delay-loads shell32's ShellExecuteW until the first
            # startfile call and reports a failed resolution as
            # NotImplementedError -- a RuntimeError, so it would escape the
            # OSError this function promises and crash the caller's handler.
            raise OSError(str(error)) from error
        return
    else:
        opener = shutil.which("xdg-open")
        if opener is None:
            raise FileNotFoundError("xdg-open")
        argv = [opener, str(path)]
    subprocess.Popen(  # noqa: S603
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
