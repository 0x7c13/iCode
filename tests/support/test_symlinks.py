# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Capability skips must never absorb broken fixtures or unrelated I/O errors."""

from __future__ import annotations

import errno
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.support.symlinks import symlink_or_skip


@pytest.mark.parametrize("directory", [False, True])
def test_forwards_real_symlink_arguments(tmp_path: Path, directory: bool) -> None:
    link, target = tmp_path / "link", tmp_path / "target"
    with patch.object(Path, "symlink_to", autospec=True) as create:
        symlink_or_skip(link, target, target_is_directory=directory)
    create.assert_called_once_with(link, target, target_is_directory=directory)


@pytest.mark.parametrize(
    "error", [NotImplementedError(), OSError(errno.ENOSYS, "unsupported"), OSError(errno.ENOTSUP, "unsupported")]
)
def test_unsupported_capability_is_reported_as_skip(
    tmp_path: Path, error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CI", raising=False)
    with patch.object(Path, "symlink_to", autospec=True, side_effect=error), pytest.raises(pytest.skip.Exception):
        symlink_or_skip(tmp_path / "link", tmp_path / "target")


def test_missing_windows_privilege_is_reported_as_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    error = OSError("privilege not held")
    error.winerror = 1314
    with patch.object(Path, "symlink_to", autospec=True, side_effect=error), pytest.raises(pytest.skip.Exception):
        symlink_or_skip(tmp_path / "link", tmp_path / "target")


@pytest.mark.parametrize(
    "error",
    [
        NotImplementedError(),
        OSError(errno.ENOSYS, "unsupported"),
        OSError(errno.ENOTSUP, "unsupported"),
        OSError("privilege not held"),
    ],
)
def test_ci_never_skips_missing_symlink_capability(
    tmp_path: Path, error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CI", "true")
    if isinstance(error, OSError) and error.errno is None:
        error.winerror = 1314
    with (
        patch.object(Path, "symlink_to", autospec=True, side_effect=error),
        pytest.raises(type(error)),
    ):
        symlink_or_skip(tmp_path / "link", tmp_path / "target")


@pytest.mark.parametrize("code", [errno.ENOENT, errno.EEXIST, errno.EACCES, errno.EIO])
def test_unexpected_filesystem_failures_propagate(tmp_path: Path, code: int) -> None:
    error = OSError(code, "broken fixture")
    with patch.object(Path, "symlink_to", autospec=True, side_effect=error), pytest.raises(OSError) as failure:
        symlink_or_skip(tmp_path / "link", tmp_path / "target")
    assert failure.value is error
