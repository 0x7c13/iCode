# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The program that runs a hook or skill script, chosen by the script's extension."""

from __future__ import annotations

import errno
import os
import shutil
import sys
from pathlib import Path

from chrys.foundation.platform import get_platform, is_wsl_bash, windows_program_files_dirs
from chrys.foundation.platform.runtime_paths import path_matches, which_excluding_runtime

_INTERPRETERS: dict[str, list[str]] = {
    ".js": ["node"],
    ".mjs": ["node"],
    ".ts": ["npx", "tsx"],
    ".rb": ["ruby"],
    ".pl": ["perl"],
}
_SHELL_SCRIPT_EXTS = frozenset({".sh", ".bash", ".zsh"})

_NO_WINDOWS_BASH = (
    "No bash was found to run this script: install Git for Windows, or put a bash on PATH. "
    "WSL's bash is not used, since the script's Windows path does not exist inside WSL."
)


def build_script_command(script_path: Path) -> list[str]:
    """The argv that starts *script_path*: the program its extension calls for, then the script.

    Raises ``FileNotFoundError`` when a ``.sh`` script on Windows has no bash to run it.
    """
    suffix = script_path.suffix.lower()
    if suffix == ".py":
        return [*_find_python_runner(), str(script_path)]
    if suffix == ".ps1":
        return [_find_powershell(), "-NoProfile", "-File", str(script_path)]
    if suffix in _SHELL_SCRIPT_EXTS:
        return [_find_bash(), str(script_path)]
    interpreter = _INTERPRETERS.get(suffix)
    if interpreter is not None:
        return [*interpreter, str(script_path)]
    return [*_find_python_runner(), str(script_path)]


def _find_python_runner() -> list[str]:
    """Find a Python script runner: ``uv run`` first (supports PEP 723 inline script metadata), system python next, chrys' own as fallback."""
    runner = _find_python_runner_on_path(fallback_to_runtime=False)
    if runner is not None:
        return runner
    runner = _find_python_runner_on_path(fallback_to_runtime=True)
    if runner is not None:
        return runner
    return [sys.executable]


def _find_python_runner_on_path(*, fallback_to_runtime: bool) -> list[str] | None:
    uv = which_excluding_runtime("uv", fallback_to_runtime=fallback_to_runtime)
    if uv:
        return [uv, "run"]
    for name in ("python3", "python"):
        path = which_excluding_runtime(name, fallback_to_runtime=fallback_to_runtime)
        if path:
            return [path]
    return None


def _find_powershell() -> str:
    """Find PowerShell: ``pwsh`` (v7+) first, ``powershell`` (v5.1) as fallback."""
    for name in ("pwsh", "powershell"):
        path = shutil.which(name)
        if path:
            return path
    return "pwsh"  # let it fail with a clear FileNotFoundError


def _find_bash() -> str:
    """Find a bash-compatible shell: ``bash``, then ``sh``, from PATH, then Git Bash.

    On Windows, ``sh``/``bash`` aren't on PATH by default, and the one that is may be WSL's, which is passed over.
    Git Bash (``Git\\bin\\bash.exe``) is looked for in the Program Files directories the environment names (so
    non-``C:`` Windows installs work too). Finding none is an error there: a bare ``bash`` would start WSL's, which
    Windows looks for before PATH.
    """
    windows = get_platform().is_windows
    for name in ("bash", "sh"):
        if windows:
            found = next((path for path in path_matches(name) if not is_wsl_bash(path)), "")
        else:
            found = shutil.which(name) or ""
        if found:
            return found
    for base in windows_program_files_dirs():
        git_bash = os.path.join(base, "Git", "bin", "bash.exe")
        if os.path.isfile(git_bash):
            return git_bash
    if windows:
        raise FileNotFoundError(errno.ENOENT, _NO_WINDOWS_BASH)
    return "bash"  # let it fail with a clear FileNotFoundError
