# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Extended tests for platform detection — shell version, clipboard helpers, architecture."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from chrys.foundation.platform import (
    PlatformInfo,
    ShellInfo,
    _detect_arch,
    _detect_unix_shell,
    _detect_windows_shell,
    _get_shell_version,
    clipboard_paste,
    detect_platform,
)

IS_UNIX = sys.platform != "win32"


# ──────────────── ShellInfo ─────────────────────────────────────────────


def test_shell_info_frozen() -> None:
    s = ShellInfo(name="bash", path="/bin/bash", args=["-c"], version="5.2")
    assert s.name == "bash"
    with pytest.raises(AttributeError):
        s.name = "zsh"  # type: ignore[misc]


def test_shell_info_default_version() -> None:
    s = ShellInfo(name="sh", path="/bin/sh", args=["-c"])
    assert s.version == ""


# ──────────────── PlatformInfo properties ───────────────────────────────


def test_platform_info_os_properties() -> None:
    shell = ShellInfo(name="bash", path="/bin/bash", args=["-c"])
    from pathlib import Path

    p = PlatformInfo(
        os_name="macos", os_version="15.4.1", arch="arm64", shell=shell, config_dir=Path("/tmp"), data_dir=Path("/tmp")
    )
    assert p.is_macos is True
    assert p.is_windows is False
    assert p.is_linux is False


def test_platform_info_arch_properties() -> None:
    shell = ShellInfo(name="bash", path="/bin/bash", args=["-c"])
    from pathlib import Path

    p = PlatformInfo(
        os_name="linux", os_version="22.04", arch="amd64", shell=shell, config_dir=Path("/tmp"), data_dir=Path("/tmp")
    )
    assert p.is_amd64 is True
    assert p.is_arm64 is False
    assert p.is_x86 is False


def test_platform_info_extra_shells_default() -> None:
    shell = ShellInfo(name="bash", path="/bin/bash", args=["-c"])
    from pathlib import Path

    p = PlatformInfo(
        os_name="linux", os_version="22.04", arch="amd64", shell=shell, config_dir=Path("/tmp"), data_dir=Path("/tmp")
    )
    assert p.extra_shells == ()


# ──────────────── _detect_arch ──────────────────────────────────────────


def test_detect_arch_returns_string() -> None:
    arch = _detect_arch()
    assert isinstance(arch, str)
    assert len(arch) > 0


# ──────────────── _get_shell_version ────────────────────────────────────


@pytest.mark.skipif(not IS_UNIX, reason="Unix shell only")
def test_get_shell_version_bash() -> None:
    """Should return a version string for bash."""
    import shutil

    bash = shutil.which("bash")
    if bash:
        version = _get_shell_version(bash, "bash")
        assert version == "" or version[0].isdigit()


def test_get_shell_version_nonexistent() -> None:
    """Non-existent shell should return empty string."""
    assert _get_shell_version("/nonexistent/shell", "bash") == ""


def test_get_shell_version_cmd() -> None:
    """cmd is explicitly skipped (no --version flag)."""
    assert _get_shell_version("cmd", "cmd") == ""


def _capture_subprocess_runs(monkeypatch: pytest.MonkeyPatch, stdout: str) -> list[dict[str, object]]:
    """Replace ``subprocess.run`` with a recorder returning *stdout* successfully.

    The platform probes import ``subprocess`` inside the function body, so
    the module attribute is the seam.
    """
    calls: list[dict[str, object]] = []

    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": list(argv), **kwargs})
        return subprocess.CompletedProcess(argv, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    return calls


@pytest.mark.parametrize(
    ("name", "expected_argv_tail"), [("bash", ["--version"]), ("pwsh", ["-NoProfile", "-Command"])]
)
def test_get_shell_version_probe_never_inherits_the_process_stdin(
    monkeypatch: pytest.MonkeyPatch, name: str, expected_argv_tail: list[str]
) -> None:
    """The version probe runs at startup — under ACP before the protocol
    reader even starts — so the child must get ``stdin=DEVNULL``, never the
    inherited (protocol) stdin."""
    calls = _capture_subprocess_runs(monkeypatch, stdout="7.4.0\n")

    assert _get_shell_version(f"/opt/{name}", name) == "7.4.0"
    assert len(calls) == 1
    assert calls[0]["argv"][: 1 + len(expected_argv_tail)] == [f"/opt/{name}", *expected_argv_tail]
    assert calls[0]["stdin"] is subprocess.DEVNULL


def test_detect_git_bash_version_probes_never_inherit_the_process_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both startup probes of the Git for Windows install — ``bash --version``
    and the sibling ``git.exe --version`` — hand their child ``stdin=DEVNULL``."""
    import chrys.foundation.platform as platform_mod

    bash_path = r"C:\Program Files\Git\bin\bash.exe"
    git_path = platform_mod.os.path.join(platform_mod.os.path.dirname(bash_path), "git.exe")
    monkeypatch.setattr(platform_mod, "windows_program_files_dirs", lambda: ())
    monkeypatch.setattr(platform_mod.shutil, "which", lambda name: bash_path if name == "bash" else None)
    monkeypatch.setattr(platform_mod.os.path, "isfile", lambda path: path in {bash_path, git_path})
    calls = _capture_subprocess_runs(monkeypatch, stdout="git version 2.45.0\n")

    shell = platform_mod._detect_git_bash()

    assert shell is not None
    assert shell.path == bash_path
    probed = sorted(tuple(call["argv"]) for call in calls)
    assert probed == sorted([(bash_path, "--version"), (git_path, "--version")])
    assert all(call["stdin"] is subprocess.DEVNULL for call in calls)


def test_detect_unix_shell_uses_absolute_fallback_for_stale_shell_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale ``$SHELL`` should not be reused when bash exists at ``/bin/bash``."""
    import chrys.foundation.platform as platform_mod

    executable_paths = {"/bin/bash"}

    monkeypatch.setenv("SHELL", "/usr/bin/bash")
    monkeypatch.setattr(platform_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(platform_mod.os.path, "isfile", lambda path: path in executable_paths)
    monkeypatch.setattr(
        platform_mod.os,
        "access",
        lambda path, mode: path in executable_paths and mode == platform_mod.os.X_OK,
    )
    monkeypatch.setattr(platform_mod, "_get_shell_version", lambda _path, _name: "")

    shell = _detect_unix_shell()

    assert shell.name == "bash"
    assert shell.path == "/bin/bash"
    assert shell.args == ["-c"]


def test_detect_unix_shell_requires_executable_shell_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing but non-executable ``$SHELL`` should fall back to PATH lookup."""
    import chrys.foundation.platform as platform_mod

    shell_env = "/tmp/not-executable/bash"
    resolved_bash = "/nix/store/bash/bin/bash"
    file_paths = {shell_env, resolved_bash}
    executable_paths = {resolved_bash}

    monkeypatch.setenv("SHELL", shell_env)
    monkeypatch.setattr(platform_mod.shutil, "which", lambda name: resolved_bash if name == "bash" else None)
    monkeypatch.setattr(platform_mod.os.path, "isfile", lambda path: path in file_paths)
    monkeypatch.setattr(
        platform_mod.os,
        "access",
        lambda path, mode: path in executable_paths and mode == platform_mod.os.X_OK,
    )
    monkeypatch.setattr(platform_mod, "_get_shell_version", lambda _path, _name: "")

    shell = _detect_unix_shell()

    assert shell.name == "bash"
    assert shell.path == resolved_bash
    assert shell.args == ["-c"]


def test_detect_windows_shell_names_pwsh_when_pwsh_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """PowerShell 7 should be exposed as the distinct ``pwsh`` shell/tool."""
    import chrys.foundation.platform as platform_mod

    paths = {
        "pwsh": r"C:\Program Files\PowerShell\7\pwsh.exe",
        "powershell": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
    }

    def which(name: str) -> str | None:
        return paths.get(name)

    def shell_version(_path: str, name: str) -> str:
        return f"{name}-version"

    monkeypatch.setattr(platform_mod.shutil, "which", which)
    monkeypatch.setattr(platform_mod, "_get_shell_version", shell_version)

    shell = _detect_windows_shell()

    assert shell.name == "pwsh"
    assert shell.path == paths["pwsh"]
    assert shell.args == ["-NoProfile", "-Command"]
    assert shell.version == "pwsh-version"


def test_detect_windows_shell_names_windows_powershell_5_as_powershell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows PowerShell 5 remains exposed as ``powershell``."""
    import chrys.foundation.platform as platform_mod

    path = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"

    def which(name: str) -> str | None:
        return path if name == "powershell" else None

    def shell_version(_path: str, name: str) -> str:
        return f"{name}-version"

    monkeypatch.setattr(platform_mod.shutil, "which", which)
    monkeypatch.setattr(platform_mod, "_get_shell_version", shell_version)

    shell = _detect_windows_shell()

    assert shell.name == "powershell"
    assert shell.path == path
    assert shell.args == ["-NoProfile", "-Command"]
    assert shell.version == "powershell-version"


# ──────────────── clipboard_paste ──────────────────────────────────────


def test_clipboard_paste_decodes_subprocess_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clipboard subprocess output should not rely on locale-default decode."""
    from pathlib import Path

    import chrys.foundation.platform as platform_mod

    platform = PlatformInfo(
        os_name="macos",
        os_version="15",
        arch="arm64",
        shell=ShellInfo(name="zsh", path="/bin/zsh", args=["-c"]),
        config_dir=Path("/tmp"),
        data_dir=Path("/tmp"),
    )

    monkeypatch.setattr(platform_mod, "get_platform", lambda: platform)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=b"clipboard \xff text"),
    )

    assert clipboard_paste() == "clipboard � text"


def test_detect_platform_config_dir_is_path() -> None:
    from pathlib import Path

    info = detect_platform()
    assert isinstance(info.config_dir, Path)
    assert isinstance(info.data_dir, Path)


def test_detect_platform_data_dir_equals_config_dir() -> None:
    info = detect_platform()
    assert info.data_dir == info.config_dir
