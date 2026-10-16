# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Key publication when neither links nor exclusive rename are available."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from chrys.foundation.platform import files, key_files
from chrys.foundation.platform.files import SecureFileError
from chrys.foundation.trajectory.keys import TRAJECTORY_KEY_FILE_NAME, load_or_create_fingerprint_key
from tests.support.secure_files import plant_owner_only_bytes
from tests.support.symlinks import symlink_or_skip


@pytest.fixture
def no_atomic_file_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        key_files, "publish_file_no_replace", Mock(side_effect=OSError(errno.ENOTSUP, "No atomic file publish"))
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX hard-link and exclusive-rename path")
def test_directory_fallback_after_both_filesystem_operations_refuse(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(os, "link", Mock(side_effect=OSError(errno.ENOTSUP, "No hard links")))
    monkeypatch.setattr(
        files, "_rename_file_no_replace", Mock(side_effect=OSError(errno.ENOTSUP, "No exclusive rename"))
    )

    key = load_or_create_fingerprint_key(tmp_path)

    assert (tmp_path / TRAJECTORY_KEY_FILE_NAME / "key").read_bytes() == key
    assert load_or_create_fingerprint_key(tmp_path) == key
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("shape", ["file", "directory"])
def test_directory_publication_preserves_a_key_created_during_the_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_atomic_file_publish: None, shape: str
) -> None:
    source = tmp_path / "prepared"
    target = tmp_path / "installation.key"
    plant_owner_only_bytes(source, b"new" * 10 + b"!!")
    winner = b"w" * 32

    def racing_rename(old: Path, new: Path) -> None:
        if new == target:
            if shape == "directory":
                target.mkdir()
                plant_owner_only_bytes(target / "key", winner)
            else:
                plant_owner_only_bytes(target, winner)
        os.rename(old, new)

    monkeypatch.setattr(key_files, "os", SimpleNamespace(rename=racing_rename))

    with pytest.raises(FileExistsError):
        key_files.publish_owner_only_key(source, target)

    assert key_files.read_owner_only_key(target, max_bytes=33) == winner
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("after_publish", [False, True])
def test_interrupted_directory_publication_leaves_no_partial_final_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_atomic_file_publish: None, after_publish: bool
) -> None:
    target = tmp_path / TRAJECTORY_KEY_FILE_NAME
    prepared: list[bytes] = []

    def interrupted_rename(old: Path, new: Path) -> None:
        if new == target:
            prepared.append((old / "key").read_bytes())
            if after_publish:
                os.rename(old, new)
            raise KeyboardInterrupt
        os.rename(old, new)

    with monkeypatch.context() as patch:
        patch.setattr(key_files, "os", SimpleNamespace(rename=interrupted_rename))
        with pytest.raises(KeyboardInterrupt):
            load_or_create_fingerprint_key(tmp_path)

    assert target.exists() is after_publish
    assert list(tmp_path.glob("*.tmp")) == []
    key = load_or_create_fingerprint_key(tmp_path)
    assert len(key) == 32
    if after_publish:
        assert key == prepared[0]
    assert load_or_create_fingerprint_key(tmp_path) == key


@pytest.mark.parametrize("link", ["container", "payload"])
def test_directory_key_rejects_a_symlink(tmp_path: Path, link: str) -> None:
    container = tmp_path / "installation.key"
    outside = tmp_path / "outside"
    if link == "container":
        outside.mkdir()
        plant_owner_only_bytes(outside / "key", b"secret")
        symlink_or_skip(container, outside, target_is_directory=True)
    else:
        container.mkdir()
        plant_owner_only_bytes(outside, b"secret")
        symlink_or_skip(container / "key", outside)

    with pytest.raises(SecureFileError):
        key_files.read_owner_only_key(container, max_bytes=33)


@pytest.mark.parametrize("error", [errno.EIO, errno.EACCES, errno.ENOSPC])
def test_key_publication_does_not_mask_storage_errors(tmp_path: Path, monkeypatch, error: int) -> None:
    source = tmp_path / "prepared"
    plant_owner_only_bytes(source, b"s" * 32)
    monkeypatch.setattr(key_files, "publish_file_no_replace", Mock(side_effect=OSError(error, "Storage failed")))

    with pytest.raises(OSError) as raised:
        key_files.publish_owner_only_key(source, tmp_path / "installation.key")

    assert raised.value.errno == error
    assert sorted(path.name for path in tmp_path.iterdir()) == ["prepared"]
