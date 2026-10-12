# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

import pytest

from chrys import DISTRIBUTION_NAME
from chrys.app import install_flavor
from chrys.app.install_flavor import InstallFlavor, detect_install_flavor, uninstall_command


class _FakeDistribution:
    def __init__(self, direct_url: str | None) -> None:
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return self._direct_url


def _environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    receipt: str | None = None,
    venv: bool = True,
    pyapp: str | None = None,
    direct_url: str | None = None,
    executable: Path | None = None,
) -> None:
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    if receipt is not None:
        (prefix / receipt).write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base") if venv else str(prefix))
    monkeypatch.setattr(sys, "executable", str(executable or prefix / "bin" / "python3"))
    for name in ("PYAPP_INSTALL_DIR_CHRYS", "PYAPP_INSTALL_DIR_ICODE-TUI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYAPP", "")
    if pyapp is None:
        monkeypatch.delenv("PYAPP")
    else:
        monkeypatch.setenv("PYAPP", pyapp)

    def distribution(name: str) -> _FakeDistribution:
        assert name == DISTRIBUTION_NAME
        return _FakeDistribution(direct_url)

    monkeypatch.setattr(install_flavor.metadata, "distribution", distribution)


@pytest.mark.parametrize(
    ("receipt", "expected"),
    [("uv-receipt.toml", InstallFlavor.UV), ("pipx_metadata.json", InstallFlavor.PIPX), (None, InstallFlavor.PIP)],
    ids=["uv", "pipx", "pip"],
)
def test_installers_are_told_apart_by_their_receipts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, receipt: str | None, expected: InstallFlavor
) -> None:
    _environment(monkeypatch, tmp_path, receipt=receipt)

    assert detect_install_flavor() is expected


def _unpacked_interpreter(tmp_path: Path) -> Path:
    """Where PyApp unpacks an offline package's Python: <data>/pyapp/chrys/<dist>/<version>/python."""
    return tmp_path / "data" / "pyapp" / "chrys" / "dist-id" / "0.30.0" / "python" / "bin" / "python3"


def test_an_offline_package_runs_the_interpreter_pyapp_unpacked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _environment(
        monkeypatch, tmp_path, venv=False, pyapp=str(tmp_path / "icode"), executable=_unpacked_interpreter(tmp_path)
    )

    assert detect_install_flavor() is InstallFlavor.OFFLINE


def test_an_offline_package_unpacked_into_a_folder_of_the_users_choice_is_still_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _environment(monkeypatch, tmp_path, venv=False, pyapp=str(tmp_path / "icode"))
    monkeypatch.setenv("PYAPP_INSTALL_DIR_CHRYS", str(tmp_path / "prefix"))

    assert detect_install_flavor() is InstallFlavor.OFFLINE


@pytest.mark.parametrize(
    ("receipt", "venv", "chosen_folder"),
    [
        ("uv-receipt.toml", True, False),
        (None, True, False),
        (None, False, False),
        (None, False, True),
    ],
    ids=[
        "uv-in-offline-shell",
        "pip-venv-in-offline-shell",
        "system-pip-in-offline-shell",
        "system-pip-in-shell-of-offline-package-in-chosen-folder",
    ],
)
def test_pyapp_inherited_from_an_offline_shell_does_not_make_another_install_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, receipt: str | None, venv: bool, chosen_folder: bool
) -> None:
    # PyApp exports PYAPP, and the install folder the user chose for it, to every child, so an
    # `icode` started from a shell inside an offline install sees them as well.
    _environment(monkeypatch, tmp_path, receipt=receipt, venv=venv, pyapp=str(tmp_path / "icode"))
    if chosen_folder:
        monkeypatch.setenv("PYAPP_INSTALL_DIR_CHRYS", str(tmp_path / "offline-runtime"))

    assert detect_install_flavor() is not InstallFlavor.OFFLINE


@pytest.mark.parametrize(
    ("direct_url", "expected"),
    [
        (json.dumps({"url": "file:///src/icode", "dir_info": {"editable": True}}), InstallFlavor.SOURCE),
        (json.dumps({"url": "file:///dl/icode_tui.whl", "archive_info": {}}), InstallFlavor.UV),
        ("not json", InstallFlavor.UV),
        (json.dumps(["unexpected"]), InstallFlavor.UV),
        (None, InstallFlavor.UV),
    ],
    ids=["editable", "wheel-file", "unreadable", "not-an-object", "absent"],
)
def test_only_an_editable_install_counts_as_a_source_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, direct_url: str | None, expected: InstallFlavor
) -> None:
    _environment(monkeypatch, tmp_path, receipt="uv-receipt.toml", direct_url=direct_url)

    assert detect_install_flavor() is expected


def test_a_missing_distribution_is_not_a_source_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _environment(monkeypatch, tmp_path, receipt="uv-receipt.toml")

    def missing(_name: str) -> _FakeDistribution:
        raise metadata.PackageNotFoundError(DISTRIBUTION_NAME)

    monkeypatch.setattr(install_flavor.metadata, "distribution", missing)

    assert detect_install_flavor() is InstallFlavor.UV


def test_uninstall_commands_name_the_published_distribution() -> None:
    assert uninstall_command(InstallFlavor.UV) == "uv tool uninstall iCode-TUI"
    assert uninstall_command(InstallFlavor.PIPX) == "pipx uninstall iCode-TUI"
    assert uninstall_command(InstallFlavor.PIP) == "pip uninstall iCode-TUI"
    assert uninstall_command(InstallFlavor.OFFLINE) is None
    assert uninstall_command(InstallFlavor.SOURCE) is None
