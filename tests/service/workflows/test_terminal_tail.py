# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session summaries read the last complete workflow record without loading the log."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO
from unittest.mock import create_autospec
from uuid import uuid4

import pytest

from chrys.foundation.platform.files import atomic_write_owner_only_bytes
from chrys.foundation.trajectory.envelope import EventDraft, build_event, decode_event_line, encode_event_line
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.service.workflows import orphans
from chrys.service.workflows.layout import EVENTS_FILE
from chrys.service.workflows.store import RunRecord, RunTerminal


def _record(kind: str, *, sequence: int = 1, outcome: str = "completed", reason: str = "") -> bytes:
    return encode_event_line(
        build_event(
            EventDraft(event_type=kind, payload={"outcome": outcome, "reason": reason}),
            sequence=sequence,
            runtime_id=new_analytics_id(),
            coverage_id=new_analytics_id(),
            session_id=str(uuid4()),
            branch_id=new_analytics_id(),
        )
    )


@pytest.mark.parametrize("block_size", [1, 17, 4096])
@pytest.mark.parametrize("tail", [b"", b'{"torn": "\xe4\xb8', b"x" * 5000], ids=["clean", "utf8", "long"])
def test_terminal_survives_block_boundaries_runtime_footer_and_torn_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, block_size: int, tail: bytes
) -> None:
    monkeypatch.setattr(orphans, "_TAIL_BLOCK_BYTES", block_size)
    reason = "取消" * 100
    raw = (
        _record(RunRecord.NODE_STATE)
        + _record(RunRecord.RUN_FINISHED, sequence=2, outcome="cancelled", reason=reason)
        + _record("runtime.closed", sequence=3)
        + b"not json\n\xff\n\n"
        + tail
    )
    path = tmp_path / EVENTS_FILE
    atomic_write_owner_only_bytes(path, raw)
    assert orphans.read_run_terminal(tmp_path) == RunTerminal(
        outcome="cancelled", last_seq=2, reason=reason, finished_at=decode_event_line(raw.splitlines()[1]).occurred_at
    )
    assert path.read_bytes() == raw


@pytest.mark.parametrize("finished", [False, True])
def test_large_log_summary_reads_only_one_tail_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finished: bool
) -> None:
    monkeypatch.setattr(orphans, "MAX_EVENTS_BYTES", 128 * 1024)
    # Larger than the scan ceiling; an earlier terminal must not hide a later active record.
    raw = _record(RunRecord.RUN_FINISHED) * 4000
    kind = RunRecord.RUN_FINISHED if finished else RunRecord.NODE_STATE
    raw += _record(kind, sequence=4001) + _record("runtime.closed", sequence=4002)
    atomic_write_owner_only_bytes(tmp_path / EVENTS_FILE, raw)
    assert len(raw) > orphans.MAX_EVENTS_BYTES
    opener = orphans.secure_open_owner_verified_binary
    bytes_read = 0

    @contextmanager
    def track(path: Path) -> Iterator[BinaryIO]:
        with opener(path) as handle:
            reader = create_autospec(handle, spec_set=True)

            def read(size: int = -1) -> bytes:
                nonlocal bytes_read
                data = handle.read(size)
                bytes_read += len(data)
                return data

            reader.read.side_effect = read
            reader.seek.side_effect = handle.seek
            yield reader

    monkeypatch.setattr(orphans, "secure_open_owner_verified_binary", create_autospec(opener, side_effect=track))
    expected = (
        RunTerminal(outcome="completed", last_seq=4001, finished_at=decode_event_line(raw.splitlines()[-2]).occurred_at)
        if finished
        else None
    )
    assert orphans.read_run_terminal(tmp_path) == expected
    assert bytes_read == orphans._TAIL_BLOCK_BYTES


@pytest.mark.parametrize("raw", [b"", b"torn", b"\n\n", b"broken\n\xff\n"])
def test_log_without_complete_workflow_record_has_no_terminal(tmp_path: Path, raw: bytes) -> None:
    assert orphans.read_run_terminal(tmp_path) is None
    atomic_write_owner_only_bytes(tmp_path / EVENTS_FILE, raw)
    assert orphans.read_run_terminal(tmp_path) is None


def test_pathological_tail_scan_remains_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orphans, "MAX_EVENTS_BYTES", 128)
    atomic_write_owner_only_bytes(tmp_path / EVENTS_FILE, b"torn" * 100)
    with pytest.raises(ValueError, match="tail exceeds"):
        orphans.read_run_terminal(tmp_path)
