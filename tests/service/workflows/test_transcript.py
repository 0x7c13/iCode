# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Optional node archives distinguish absent records from unreadable records."""

from __future__ import annotations

import errno
from collections.abc import Callable
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from chrys.foundation.platform.files import SecureFileError
from chrys.service.workflows import transcript as transcript_module
from chrys.service.workflows.store import node_value_path
from chrys.service.workflows.transcript import read_node_transcript, read_node_usage


@pytest.mark.parametrize("reader", [read_node_transcript, read_node_usage])
@pytest.mark.parametrize("parent_exists", [False, True])
def test_missing_node_archive_is_optional(
    tmp_path: Path, reader: Callable[[Path, str, int], object], parent_exists: bool
) -> None:
    if parent_exists:
        node_value_path(tmp_path, "node@iter#1", 1, "session").parent.mkdir()
    assert reader(tmp_path, "node@iter#1", 1) is None


@pytest.mark.parametrize("reader", [read_node_transcript, read_node_usage])
@pytest.mark.parametrize("error_number", [errno.ENOENT, errno.ENOTDIR, errno.EACCES, errno.EIO, None])
def test_only_missing_artifact_errors_are_suppressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: Callable[[Path, str, int], object],
    error_number: int | None,
) -> None:
    failure = SecureFileError(error_number, "archive read failed")
    monkeypatch.setattr(
        transcript_module,
        "read_json_object",
        create_autospec(transcript_module.read_json_object, side_effect=failure),
    )
    if error_number in (errno.ENOENT, errno.ENOTDIR):
        assert reader(tmp_path, "node@iter#1", 1) is None
    else:
        with pytest.raises(SecureFileError) as caught:
            reader(tmp_path, "node@iter#1", 1)
        assert caught.value is failure
