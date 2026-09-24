# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for the per-root session MRU index (store-level lookup tests live in test_latest_session_lookup.py)."""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import chrys.service.state.session_mru as session_mru_module
from chrys.foundation.util.lock import FileLock
from chrys.service.state.session_mru import (
    SESSION_MRU_FILE_NAME,
    SessionMruEntry,
    SessionMruIndex,
    coerce_utc,
)

_T0 = datetime(2026, 8, 16, 10, 0, tzinfo=UTC)


def _ts(minutes: int) -> datetime:
    return _T0 + timedelta(minutes=minutes)


def _ids(index: SessionMruIndex) -> list[str]:
    snapshot = index.load()
    assert snapshot is not None
    return [entry.session_id for entry in snapshot.sessions]


def test_load_missing_file_returns_none(tmp_path: Path) -> None:
    assert SessionMruIndex(tmp_path).load() is None


def test_load_corrupt_json_returns_none(tmp_path: Path) -> None:
    (tmp_path / SESSION_MRU_FILE_NAME).write_text("{not json", encoding="utf-8")
    assert SessionMruIndex(tmp_path).load() is None


def test_load_unknown_version_returns_none(tmp_path: Path) -> None:
    (tmp_path / SESSION_MRU_FILE_NAME).write_text(
        json.dumps({"version": 99, "complete": True, "sessions": []}), encoding="utf-8"
    )
    assert SessionMruIndex(tmp_path).load() is None


def _write_index(root: Path, sessions: list[object]) -> None:
    (root / SESSION_MRU_FILE_NAME).write_text(
        json.dumps({"version": 1, "complete": True, "sessions": sessions}), encoding="utf-8"
    )


def test_load_dedupes_duplicate_ids_newest_wins_and_treats_naive_as_utc(tmp_path: Path) -> None:
    _write_index(
        tmp_path,
        [
            {"session_id": "a", "last_updated_at": "2026-08-16T10:01:00Z"},
            {"session_id": "a", "last_updated_at": "2026-08-16T10:05:00Z"},
            {"session_id": "c", "last_updated_at": "2026-08-16T10:03:00"},
        ],
    )
    snapshot = SessionMruIndex(tmp_path).load()
    assert snapshot is not None
    assert snapshot.complete is True
    assert [(e.session_id, e.last_updated_at) for e in snapshot.sessions] == [("a", _ts(5)), ("c", _ts(3))]


def test_load_rejects_any_malformed_entry(tmp_path: Path) -> None:
    """A complete index that silently dropped an entry would hide that session for good."""
    good = {"session_id": "a", "last_updated_at": "2026-08-16T10:01:00Z"}
    for bad in (
        {"session_id": "", "last_updated_at": "2026-08-16T10:09:00Z"},
        {"session_id": "b", "last_updated_at": "not-a-date"},
        {"session_id": "b"},
        {"session_id": 7, "last_updated_at": "2026-08-16T10:09:00Z"},
        "junk",
    ):
        _write_index(tmp_path, [good, bad])
        assert SessionMruIndex(tmp_path).load() is None
    # record() then treats it like any corrupt file: recreated incomplete.
    SessionMruIndex(tmp_path).record("z", _ts(1))
    snapshot = SessionMruIndex(tmp_path).load()
    assert snapshot is not None and snapshot.complete is False
    assert [e.session_id for e in snapshot.sessions] == ["z"]


def test_record_creates_incomplete_index_and_orders_newest_first(tmp_path: Path) -> None:
    index = SessionMruIndex(tmp_path)
    index.record("old", _ts(1))
    index.record("new", _ts(2))
    index.record("mid-b", _ts(1))
    snapshot = index.load()
    assert snapshot is not None
    assert snapshot.complete is False
    assert [e.session_id for e in snapshot.sessions] == ["new", "mid-b", "old"]
    raw = json.loads((tmp_path / SESSION_MRU_FILE_NAME).read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["sessions"][0] == {"session_id": "new", "last_updated_at": "2026-08-16T10:02:00Z"}


def test_record_stale_timestamp_does_not_downgrade(tmp_path: Path) -> None:
    index = SessionMruIndex(tmp_path)
    index.record("a", _ts(5))
    index.record("b", _ts(3))
    index.record("a", _ts(1))
    snapshot = index.load()
    assert snapshot is not None
    assert snapshot.sessions[0] == SessionMruEntry("a", _ts(5))


def test_record_unchanged_entry_skips_write(tmp_path: Path) -> None:
    index = SessionMruIndex(tmp_path)
    index.record("a", _ts(5))
    before = (tmp_path / SESSION_MRU_FILE_NAME).read_bytes()
    index.record("a", _ts(5))
    assert (tmp_path / SESSION_MRU_FILE_NAME).read_bytes() == before


def test_record_trims_to_max_entries(tmp_path: Path) -> None:
    index = SessionMruIndex(tmp_path, max_entries=3)
    for i in range(6):
        index.record(f"s{i}", _ts(i))
    assert _ids(index) == ["s5", "s4", "s3"]


def test_record_recreates_corrupt_index_as_incomplete(tmp_path: Path) -> None:
    (tmp_path / SESSION_MRU_FILE_NAME).write_text("{not json", encoding="utf-8")
    index = SessionMruIndex(tmp_path)
    index.record("a", _ts(1))
    snapshot = index.load()
    assert snapshot is not None
    assert snapshot.complete is False
    assert [e.session_id for e in snapshot.sessions] == ["a"]


def test_remove_drops_entry_and_noops_when_absent(tmp_path: Path) -> None:
    index = SessionMruIndex(tmp_path)
    index.record("a", _ts(2))
    index.record("b", _ts(1))
    index.remove("a")
    assert _ids(index) == ["b"]
    before = (tmp_path / SESSION_MRU_FILE_NAME).read_bytes()
    index.remove("missing")
    assert (tmp_path / SESSION_MRU_FILE_NAME).read_bytes() == before


def test_rebuild_marks_complete_and_keeps_newer_concurrent_record(tmp_path: Path) -> None:
    index = SessionMruIndex(tmp_path)
    index.record("a", _ts(9))  # landed after the scan read "a" at _ts(2)
    index.rebuild([SessionMruEntry("a", _ts(2)), SessionMruEntry("b", _ts(4))])
    snapshot = index.load()
    assert snapshot is not None
    assert snapshot.complete is True
    assert list(snapshot.sessions) == [SessionMruEntry("a", _ts(9)), SessionMruEntry("b", _ts(4))]


def test_rebuild_from_missing_or_corrupt_file(tmp_path: Path) -> None:
    index = SessionMruIndex(tmp_path)
    index.rebuild([SessionMruEntry("a", _ts(1))])
    snapshot = index.load()
    assert snapshot is not None and snapshot.complete is True
    (tmp_path / SESSION_MRU_FILE_NAME).write_text("garbage", encoding="utf-8")
    index.rebuild([SessionMruEntry("b", _ts(2))])
    assert _ids(index) == ["b"]


def test_rebuild_naive_timestamps_are_treated_as_utc(tmp_path: Path) -> None:
    index = SessionMruIndex(tmp_path)
    index.rebuild([SessionMruEntry("a", datetime(2026, 8, 16, 10, 1))])  # noqa: DTZ001 - naive on purpose
    snapshot = index.load()
    assert snapshot is not None
    assert snapshot.sessions[0].last_updated_at == _ts(1)


def test_invalidate_removes_file_and_tolerates_missing(tmp_path: Path) -> None:
    index = SessionMruIndex(tmp_path)
    index.invalidate()
    index.record("a", _ts(1))
    assert index.path.exists()
    index.invalidate()
    assert not index.path.exists()
    assert index.load() is None


def test_write_leaves_no_temp_files(tmp_path: Path) -> None:
    index = SessionMruIndex(tmp_path)
    index.record("a", _ts(1))
    index.remove("a")
    assert sorted(p.name for p in tmp_path.iterdir()) == [".locks", SESSION_MRU_FILE_NAME]


def test_roots_are_isolated(tmp_path: Path) -> None:
    SessionMruIndex(tmp_path / "one").record("a", _ts(1))
    assert SessionMruIndex(tmp_path / "two").load() is None


def test_concurrent_records_do_not_lose_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Exercise serialization independently of the production latency budget.
    # Windows fsync under CI load can use that budget while a healthy writer
    # holds the lock; the bounded-timeout behavior has its own test below.
    monkeypatch.setattr(session_mru_module, "SESSION_MRU_LOCK_TIMEOUT_SECONDS", 15.0)
    index = SessionMruIndex(tmp_path, max_entries=100)
    errors: list[BaseException] = []
    round_start = threading.Barrier(4, timeout=15.0)

    def worker(offset: int) -> None:
        try:
            for i in range(10):
                # Keep every writer competing; an unfair thread lock must not
                # let one writer run all ten fsyncs before its peers get a turn.
                round_start.wait()
                index.record(f"s{offset}-{i}", _ts(offset * 10 + i))
        except BaseException as exc:  # pragma: no cover - surfaced via assertion
            errors.append(exc)
            round_start.abort()

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 30.0
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads)
    assert errors == []
    assert len(_ids(index)) == 40


def test_invalidate_waits_for_the_index_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A locked writer cannot re-materialize a stale index around the unlink."""
    index = SessionMruIndex(tmp_path)
    index.record("a", _ts(1))
    attempted = threading.Event()

    class SignallingLock(FileLock):
        def acquire(self) -> None:
            attempted.set()
            super().acquire()

    monkeypatch.setattr(index, "_lock", lambda: SignallingLock(index.lock_path, timeout=5.0))
    writer_lock = FileLock(index.lock_path, timeout=1.0)
    writer_lock.acquire()
    finished = threading.Event()
    try:
        worker = threading.Thread(target=lambda: (index.invalidate(), finished.set()))
        worker.start()
        assert attempted.wait(5.0)  # the unlink is at the lock...
        assert not finished.wait(0.2)  # ...and blocked behind the writer
        assert index.path.exists()
        # Writer completes its locked write-back, then releases.
        index._write(index._read() or index._new_index(complete=True), [SessionMruEntry("a", _ts(1))])
    finally:
        writer_lock.release()
    worker.join(timeout=5.0)
    assert finished.is_set()
    assert not index.path.exists()


def test_load_rejects_non_bool_complete_and_bad_horizon(tmp_path: Path) -> None:
    index = SessionMruIndex(tmp_path)
    for complete in ("false", 1, {}):
        index.path.write_text(json.dumps({"version": 1, "complete": complete, "sessions": []}), encoding="utf-8")
        assert index.load() is None
    index.path.write_text(
        json.dumps({"version": 1, "complete": True, "horizon": "not-a-date", "sessions": []}), encoding="utf-8"
    )
    assert index.load() is None
    index.path.write_text(json.dumps({"version": 1, "complete": True, "sessions": []}), encoding="utf-8")
    snapshot = index.load()
    assert snapshot is not None and snapshot.horizon is None


def test_trim_raises_horizon_and_remove_keeps_it(tmp_path: Path) -> None:
    index = SessionMruIndex(tmp_path, max_entries=2)
    index.record("a", _ts(1))
    index.record("b", _ts(2))
    snapshot = index.load()
    assert snapshot is not None and snapshot.horizon is None
    index.record("c", _ts(3))  # drops a
    snapshot = index.load()
    assert snapshot is not None and snapshot.horizon == _ts(1)
    index.record("d", _ts(4))  # drops b
    snapshot = index.load()
    assert snapshot is not None and snapshot.horizon == _ts(2)
    index.record("e", _ts(0))  # dropped immediately; horizon never lowers
    index.remove("d")
    snapshot = index.load()
    assert snapshot is not None
    assert [e.session_id for e in snapshot.sessions] == ["c"]
    assert snapshot.horizon == _ts(2)


def test_rebuild_keeps_horizon_and_returns_merged_entries(tmp_path: Path) -> None:
    index = SessionMruIndex(tmp_path, max_entries=2)
    for i in range(5):
        index.record(f"s{i}", _ts(i))
    snapshot = index.load()
    assert snapshot is not None and snapshot.horizon == _ts(2)
    index.record("late", _ts(9))  # landed during the scan below; drops s3 → horizon 3
    # Existing entries (concurrent records / not-yet-verified ghosts) are
    # kept and the horizon never drops: a concurrent record that was
    # trimmed straight away is remembered only there.
    ranked = index.rebuild([SessionMruEntry("s1", _ts(1)), SessionMruEntry("s0", _ts(0))])
    assert ranked == (
        SessionMruEntry("late", _ts(9)),
        SessionMruEntry("s4", _ts(4)),
        SessionMruEntry("s1", _ts(1)),
        SessionMruEntry("s0", _ts(0)),
    )  # the whole merged ranking, not just what the file keeps
    snapshot = index.load()
    assert snapshot is not None
    assert snapshot.sessions == ranked[:2]
    assert snapshot.horizon == _ts(3)
    assert index.rebuild([SessionMruEntry("only", _ts(5))]) == (
        SessionMruEntry("late", _ts(9)),
        SessionMruEntry("only", _ts(5)),
        SessionMruEntry("s4", _ts(4)),
    )
    snapshot = index.load()
    assert snapshot is not None and snapshot.horizon == _ts(4)  # s4 dropped now
    assert SessionMruIndex(tmp_path / "empty").rebuild([]) == ()


def test_writers_give_up_after_the_lock_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stuck peer must not stall index writers (or readers) indefinitely."""
    monkeypatch.setattr(session_mru_module, "SESSION_MRU_LOCK_TIMEOUT_SECONDS", 0.05)
    index = SessionMruIndex(tmp_path)
    index.record("a", _ts(1))
    holder = FileLock(index.lock_path)
    holder.acquire()
    try:
        assert index.load() is None
        for op in (
            lambda: index.record("b", _ts(2)),
            lambda: index.remove("a"),
            lambda: index.rebuild([SessionMruEntry("c", _ts(3))]),
        ):
            with pytest.raises(TimeoutError):
                op()
        index.invalidate()  # swallowed, logged
        assert index.path.exists()
    finally:
        holder.release()
    assert _ids(index) == ["a"]


def test_coerce_utc_clamps_offsets_that_leave_the_datetime_range() -> None:
    low = datetime.fromisoformat("0001-01-01T00:00:00+23:59")
    high = datetime.fromisoformat("9999-12-31T23:59:59-23:59")
    assert coerce_utc(low) == datetime.min.replace(tzinfo=UTC)
    assert coerce_utc(high) == datetime.max.replace(tzinfo=UTC)
    assert coerce_utc(datetime.fromisoformat("2026-08-16T10:00:00+02:00")) == datetime(2026, 8, 16, 8, tzinfo=UTC)
