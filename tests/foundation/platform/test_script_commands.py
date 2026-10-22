# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Which program runs a hook or skill script: Python, and bash, including what Windows puts in their place."""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import pytest

import chrys.foundation.platform as platform_pkg
from chrys.foundation.platform import get_platform, runtime_paths, script_commands
from tests.support.app_execution_aliases import STORE_PYTHON, STORE_PYTHON_STUB


def _exe(path: Path) -> Path:
    if sys.platform == "win32" and not path.suffix:
        return path.with_name(path.name + ".exe")
    return path


def _make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    path.chmod(0o755)
    return path


def _patch_runtime(monkeypatch: pytest.MonkeyPatch, *, python: Path, prefix: Path, scripts: Path) -> None:
    monkeypatch.setattr(runtime_paths.sys, "executable", str(python))
    monkeypatch.setattr(runtime_paths.sys, "prefix", str(prefix))
    monkeypatch.setattr(runtime_paths.sys, "exec_prefix", str(prefix))
    monkeypatch.setattr(
        runtime_paths.sysconfig,
        "get_path",
        lambda name: str(scripts) if name == "scripts" else "",
    )


def _fake_windows(monkeypatch: pytest.MonkeyPatch, *, aliases: dict[Path, bytes]) -> None:
    """Windows lookups on any host, *aliases* being the App Execution Aliases there, with their reparse data."""
    data = {str(path): reparse for path, reparse in aliases.items()}
    monkeypatch.setattr(runtime_paths.sys, "platform", "win32")
    monkeypatch.setattr(runtime_paths, "is_app_execution_alias", lambda path: path in data)
    monkeypatch.setattr(platform_pkg, "is_app_execution_alias", lambda path: path in data)
    monkeypatch.setattr(runtime_paths, "_read_reparse_point", lambda path: data.get(path, b""))
    windows = dataclasses.replace(get_platform(), os_name="windows")
    monkeypatch.setattr(script_commands, "get_platform", lambda: windows)
    monkeypatch.setenv("PATHEXT", ".EXE")
    for name in ("PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "SYSTEMDRIVE"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


def test_find_python_runner_prefers_system_uv_when_frozen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_bin = tmp_path / "runtime" / "bin"
    system_bin = tmp_path / "system" / "bin"
    _make_executable(_exe(runtime_bin / "uv"))
    system_uv = _make_executable(_exe(system_bin / "uv"))
    _patch_runtime(
        monkeypatch,
        python=_exe(runtime_bin / "python"),
        prefix=tmp_path / "runtime",
        scripts=runtime_bin,
    )
    if sys.platform == "win32":
        monkeypatch.setenv("PATHEXT", ".exe")
    monkeypatch.setenv("PYAPP", "1")
    monkeypatch.setenv("PATH", os.pathsep.join([str(runtime_bin), str(system_bin)]))

    assert script_commands._find_python_runner() == [str(system_uv), "run"]


def test_find_python_runner_prefers_system_python_over_runtime_uv_when_frozen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_bin = tmp_path / "runtime" / "bin"
    system_bin = tmp_path / "system" / "bin"
    _make_executable(_exe(runtime_bin / "uv"))
    system_python = _make_executable(_exe(system_bin / "python"))
    _patch_runtime(
        monkeypatch,
        python=_exe(runtime_bin / "python"),
        prefix=tmp_path / "runtime",
        scripts=runtime_bin,
    )
    if sys.platform == "win32":
        monkeypatch.setenv("PATHEXT", ".exe")
    monkeypatch.setenv("PYAPP", "1")
    monkeypatch.setenv("PATH", os.pathsep.join([str(runtime_bin), str(system_bin)]))

    assert script_commands._find_python_runner() == [str(system_python)]


@pytest.mark.parametrize("frozen", [False, True], ids=["ordinary", "frozen"])
def test_the_store_placeholder_python3_gives_way_to_an_installed_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen: bool
) -> None:
    # A python.org install has no python3.exe, and Windows' placeholders come first on PATH.
    apps = tmp_path / "WindowsApps"
    python3_stub = _make_executable(apps / "python3.exe")
    python_stub = _make_executable(apps / "python.exe")
    installed = _make_executable(tmp_path / "Python313" / "python.exe")
    runtime_bin = tmp_path / "runtime" / "Scripts"
    _patch_runtime(
        monkeypatch, python=_make_executable(runtime_bin / "python.exe"), prefix=runtime_bin, scripts=runtime_bin
    )
    _fake_windows(monkeypatch, aliases={python3_stub: STORE_PYTHON_STUB, python_stub: STORE_PYTHON_STUB})
    path = [str(apps), str(installed.parent)]
    if frozen:
        # A packaged run's own interpreter may come first on PATH; it is the last resort.
        monkeypatch.setenv("PYAPP", "1")
        path.insert(0, str(runtime_bin))
    monkeypatch.setenv("PATH", os.pathsep.join(path))

    assert script_commands.build_script_command(Path("hook.py")) == [str(installed), "hook.py"]


def test_with_only_the_store_placeholders_on_path_chrys_runs_its_own_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    apps = tmp_path / "WindowsApps"
    python3_stub = _make_executable(apps / "python3.exe")
    python_stub = _make_executable(apps / "python.exe")
    _fake_windows(monkeypatch, aliases={python3_stub: STORE_PYTHON_STUB, python_stub: STORE_PYTHON_STUB})
    monkeypatch.setenv("PATH", str(apps))

    assert script_commands._find_python_runner() == [sys.executable]


@pytest.mark.parametrize("reparse", [STORE_PYTHON, b""], ids=["store_python", "unreadable"])
def test_an_alias_that_is_not_the_placeholder_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reparse: bytes
) -> None:
    apps = tmp_path / "WindowsApps"
    python3 = _make_executable(apps / "python3.exe")
    _fake_windows(monkeypatch, aliases={python3: reparse})
    monkeypatch.setenv("PATH", str(apps))

    assert script_commands._find_python_runner() == [str(python3)]


def test_wsl_bash_gives_way_to_a_later_bash_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    system_root = tmp_path / "Windows"
    wsl_launcher = _make_executable(system_root / "System32" / "bash.exe")
    store_wsl = _make_executable(tmp_path / "WindowsApps" / "bash.exe")
    msys_bash = _make_executable(tmp_path / "msys64" / "usr" / "bin" / "bash.exe")
    _fake_windows(monkeypatch, aliases={store_wsl: b""})
    monkeypatch.setenv("SYSTEMROOT", str(system_root))
    monkeypatch.setenv("PATH", os.pathsep.join(str(path.parent) for path in (wsl_launcher, store_wsl, msys_bash)))

    assert script_commands.build_script_command(Path("hook.sh")) == [str(msys_bash), "hook.sh"]


def test_sh_runs_a_script_when_the_only_bash_is_wsl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    system_root = tmp_path / "Windows"
    wsl_launcher = _make_executable(system_root / "System32" / "bash.exe")
    sh = _make_executable(tmp_path / "tools" / "sh.exe")
    _fake_windows(monkeypatch, aliases={})
    monkeypatch.setenv("SYSTEMROOT", str(system_root))
    monkeypatch.setenv("PATH", os.pathsep.join([str(wsl_launcher.parent), str(sh.parent)]))

    assert script_commands.build_script_command(Path("hook.sh")) == [str(sh), "hook.sh"]


def test_git_bash_runs_a_script_when_the_only_bash_on_path_is_wsl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_root = tmp_path / "Windows"
    wsl_launcher = _make_executable(system_root / "System32" / "bash.exe")
    git_bash = _make_executable(tmp_path / "Program Files" / "Git" / "bin" / "bash.exe")
    _fake_windows(monkeypatch, aliases={})
    monkeypatch.setenv("SYSTEMROOT", str(system_root))
    monkeypatch.setenv("PROGRAMFILES", str(tmp_path / "Program Files"))
    monkeypatch.setenv("PATH", str(wsl_launcher.parent))

    assert script_commands.build_script_command(Path("hook.sh")) == [str(git_bash), "hook.sh"]


def test_a_script_with_only_wsl_bash_on_windows_names_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bare "bash" would start WSL's again: Windows looks in System32 before PATH.
    system_root = tmp_path / "Windows"
    wsl_launcher = _make_executable(system_root / "System32" / "bash.exe")
    _fake_windows(monkeypatch, aliases={})
    monkeypatch.setenv("SYSTEMROOT", str(system_root))
    monkeypatch.setenv("PATH", str(wsl_launcher.parent))

    with pytest.raises(FileNotFoundError, match="install Git for Windows") as raised:
        script_commands.build_script_command(Path("hook.sh"))
    assert "WSL's bash is not used" in str(raised.value)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX PATH lookup")
def test_on_posix_an_empty_path_entry_still_finds_the_shell_in_the_current_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_executable(tmp_path / "sh")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", os.pathsep)

    assert script_commands.build_script_command(Path("hook.sh")) == ["sh", "hook.sh"]
