# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Rollback probes retain endpoint identity without creating unreferenced blobs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from chrys.service.mutations.store import FileProbe, SnapshotPolicy, SnapshotStore
from chrys.service.mutations.types import SnapshotSkipReason


@pytest.mark.parametrize("limit", [0, 1024 * 1024])
def test_probe_hashes_in_chunks_without_creating_blob_directory(tmp_path: Path, limit: int) -> None:
    path = tmp_path / "source"
    data = b"text\n" * 40_000
    path.write_bytes(data)
    store = SnapshotStore(tmp_path / "session", policy=SnapshotPolicy(max_file_bytes=limit))
    assert store.probe(str(path)) == FileProbe(True, hashlib.sha256(data).hexdigest())
    assert not store.mutations_dir.exists()


def test_probe_rejects_oversized_file_without_opening_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"oversized")
    store = SnapshotStore(tmp_path / "session", policy=SnapshotPolicy(max_file_bytes=3))
    forbidden = create_autospec(Path.open, side_effect=AssertionError("must not open oversized content"))
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", forbidden)
        assert store.probe(str(path)) == FileProbe(True, skip_reason=SnapshotSkipReason.TOO_LARGE)
    forbidden.assert_not_called()
    assert not store.mutations_dir.exists()


@pytest.mark.parametrize("skip_binary", [True, False])
def test_probe_applies_binary_policy_and_preserves_endpoint_existence(tmp_path: Path, skip_binary: bool) -> None:
    path = tmp_path / "source"
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    path.write_bytes(data)
    store = SnapshotStore(tmp_path / "session", policy=SnapshotPolicy(skip_binary=skip_binary))
    expected = (
        FileProbe(True, skip_reason=SnapshotSkipReason.BINARY)
        if skip_binary
        else FileProbe(True, hashlib.sha256(data).hexdigest())
    )
    assert store.probe(str(path)) == expected
    assert store.probe(str(tmp_path / "missing")) == FileProbe(False)
    assert store.probe(str(tmp_path)) == FileProbe(True, skip_reason=SnapshotSkipReason.UNREADABLE)
    assert not store.mutations_dir.exists()
