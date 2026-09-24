# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Display paths accept aliased roots without changing entry identity."""

from __future__ import annotations

import ntpath
import os
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from chrys.foundation.platform.paths import display_relpath


def test_display_relpath_preserves_alias_and_deleted_or_symlink_entry_names(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    link = real / "file-link"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    try:
        alias.symlink_to(real, target_is_directory=True)
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks unavailable")
    assert display_relpath(str(real / "deleted.txt"), str(alias)) == "deleted.txt"
    assert display_relpath(str(alias / "deleted.txt"), str(alias)) == "deleted.txt"
    assert display_relpath(str(link), str(alias)) == "file-link"
    assert display_relpath(str(outside), str(alias)) == os.path.relpath(outside, alias)


@pytest.mark.parametrize("external", [False, True])
def test_display_relpath_accepts_drive_alias_to_unc_root(monkeypatch: pytest.MonkeyPatch, external: bool) -> None:
    cwd = r"Z:\project"
    physical_cwd = r"\\server\share\project"
    path = r"\\other\share\file.txt" if external else physical_cwd + r"\file.txt"
    # Exercise Windows path rules without requiring a mapped network drive.
    with monkeypatch.context() as patch:
        patch.setattr(os.path, "relpath", create_autospec(os.path.relpath, side_effect=ntpath.relpath))
        realpath = create_autospec(os.path.realpath, return_value=physical_cwd)
        patch.setattr(os.path, "realpath", realpath)
        result = display_relpath(path, cwd)
    assert result == (path if external else "file.txt")
    realpath.assert_called_once_with(cwd)


def test_display_relpath_keeps_external_label_when_root_cannot_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = str(tmp_path / "cwd")
    path = str(tmp_path / "outside")
    expected = os.path.relpath(path, cwd)
    monkeypatch.setattr(os.path, "realpath", create_autospec(os.path.realpath, side_effect=OSError("unavailable")))
    assert display_relpath(path, cwd) == expected
