# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Platform dispatch for the desktop file-manager opener."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from chrys.app.tui.support import file_manager


@pytest.fixture
def popen_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict[str, object]]]:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def record(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((argv, kwargs))
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(file_manager.subprocess, "Popen", record)
    return calls


def _fake_platform(monkeypatch: pytest.MonkeyPatch, *, macos: bool = False, windows: bool = False) -> None:
    monkeypatch.setattr(file_manager, "get_platform", lambda: SimpleNamespace(is_macos=macos, is_windows=windows))


def test_capability_requires_a_local_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_platform(monkeypatch, windows=True)
    monkeypatch.setattr(file_manager.os, "startfile", lambda _path: None, raising=False)
    monkeypatch.setattr(file_manager, "can_access_local_desktop", lambda _env=None: False)

    assert not file_manager.can_open_in_file_manager()


def test_capability_checks_each_platform_opener(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(file_manager, "can_access_local_desktop", lambda _env=None: True)

    opener = tmp_path / "open"
    opener.touch()
    monkeypatch.setattr(file_manager, "_MACOS_OPEN_PATH", opener)
    _fake_platform(monkeypatch, macos=True)
    assert file_manager.can_open_in_file_manager()
    opener.unlink()
    assert not file_manager.can_open_in_file_manager()

    _fake_platform(monkeypatch, windows=True)
    monkeypatch.setattr(file_manager.os, "startfile", lambda _path: None, raising=False)
    assert file_manager.can_open_in_file_manager()
    monkeypatch.delattr(file_manager.os, "startfile", raising=False)
    assert not file_manager.can_open_in_file_manager()

    _fake_platform(monkeypatch)
    monkeypatch.setattr(file_manager.shutil, "which", lambda _name: "/usr/bin/xdg-open")
    assert file_manager.can_open_in_file_manager()
    monkeypatch.setattr(file_manager.shutil, "which", lambda _name: None)
    assert not file_manager.can_open_in_file_manager()


def test_macos_uses_the_system_open_binary_with_detached_stdio(
    monkeypatch: pytest.MonkeyPatch, popen_calls: list[tuple[list[str], dict[str, object]]]
) -> None:
    _fake_platform(monkeypatch, macos=True)
    probe = Path("/probe")

    assert Path("/usr/bin/open") == file_manager._MACOS_OPEN_PATH
    file_manager.open_in_file_manager(probe)

    argv, kwargs = popen_calls[0]
    # str(probe) rather than a literal: pathlib renders the separator per
    # the host platform, and these dispatch tests run on every host.
    assert argv == [str(file_manager._MACOS_OPEN_PATH), str(probe)]
    # The opener must not inherit our stdio: an inherited pipe is the ACP
    # stall class, and inherited output would scribble over the TUI.
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert "creationflags" not in kwargs


def test_windows_uses_shell_startfile_without_spawning_a_subprocess(
    monkeypatch: pytest.MonkeyPatch, popen_calls: list[tuple[list[str], dict[str, object]]]
) -> None:
    _fake_platform(monkeypatch, windows=True)
    startfile_calls: list[str] = []
    monkeypatch.setattr(file_manager.os, "startfile", startfile_calls.append, raising=False)
    probe = Path("/probe")

    file_manager.open_in_file_manager(probe)

    assert startfile_calls == [str(probe)]
    assert popen_calls == []


def test_windows_startfile_not_implemented_is_normalised_to_os_error(
    monkeypatch: pytest.MonkeyPatch, popen_calls: list[tuple[list[str], dict[str, object]]]
) -> None:
    _fake_platform(monkeypatch, windows=True)

    def refuse(_path: str) -> None:
        # What CPython raises when its delay-loaded ShellExecuteW never resolves.
        raise NotImplementedError("startfile not available on this platform")

    monkeypatch.setattr(file_manager.os, "startfile", refuse, raising=False)

    # NotImplementedError is a RuntimeError, so an unconverted one would slip
    # past the caller's ``except OSError`` and take the TUI down with it.
    with pytest.raises(OSError, match="startfile not available") as caught:
        file_manager.open_in_file_manager(Path("/probe"))
    assert isinstance(caught.value.__cause__, NotImplementedError)
    assert popen_calls == []


def test_linux_resolves_xdg_open_and_raises_when_it_is_absent(
    monkeypatch: pytest.MonkeyPatch, popen_calls: list[tuple[list[str], dict[str, object]]]
) -> None:
    _fake_platform(monkeypatch)
    monkeypatch.setattr(file_manager.shutil, "which", lambda name: f"/usr/bin/{name}")
    probe = Path("/probe")

    file_manager.open_in_file_manager(probe)

    argv, _kwargs = popen_calls[0]
    assert argv == ["/usr/bin/xdg-open", str(probe)]

    monkeypatch.setattr(file_manager.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError):
        file_manager.open_in_file_manager(probe)
    assert len(popen_calls) == 1
