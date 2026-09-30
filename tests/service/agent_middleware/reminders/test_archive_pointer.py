# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for ``reminders.archive_pointer``: the pointer sentence and the restored record count."""

from __future__ import annotations

import pytest

from chrys.service.agent_middleware.reminders.archive_pointer import (
    _CATALOG_POINTER,
    ArchivePointerSource,
    _catalog_pointer_text,
)


def _source() -> ArchivePointerSource:
    return ArchivePointerSource(session_root=None, file_read_available=False, spill_quota=None, enabled=True)


@pytest.mark.parametrize("record_count", [1, 2, 10])
def test_built_pointer_is_the_shape_rendering_retargets(record_count: int) -> None:
    text = _catalog_pointer_text(record_count, "/root/sessions/abc/compactions/dropped/catalog.jsonl")

    assert _CATALOG_POINTER.fullmatch(text) is not None


@pytest.mark.parametrize("value", [0, 7, (1 << 63) - 1])
def test_restore_keeps_a_valid_count(value: int) -> None:
    source = _source()

    source.restore_record_count(value)

    assert source.restored_record_count == value
    assert source.record_count_state() == value


@pytest.mark.parametrize("value", [True, False, -1, 1 << 63, "3", 2.0])
def test_restore_drops_a_malformed_count(value: object) -> None:
    source = _source()
    source.restore_record_count(4)

    source.restore_record_count(value)

    assert source.restored_record_count is None
    assert source.record_count_state() is None
