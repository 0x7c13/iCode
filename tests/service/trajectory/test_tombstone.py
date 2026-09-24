# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Logical session deletion: physical removal when free, tombstone while a writer holds the lease."""

from __future__ import annotations

import os
import shutil
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chrys.foundation.platform import get_platform
from chrys.foundation.trajectory.envelope import EventDraft
from chrys.foundation.trajectory.event_types import EventType, RuntimeFinishReason
from chrys.foundation.trajectory.keys import ensure_owner_only_directory
from chrys.foundation.trajectory.lease import WriterLease
from chrys.service.trajectory import tombstone as tombstone_module
from chrys.service.trajectory.session import SessionTrajectory, trajectory_dir
from chrys.service.trajectory.tombstone import (
    INTENT_SUFFIX,
    TOMBSTONES_DIR_NAME,
    DeleteOutcome,
    delete_session_directory,
    pending_delete_intents,
    session_lease_path,
    sweep_tombstones,
    tombstones_dir,
    writer_lease_held,
)

SESSION_ID = "12345678-1234-1234-1234-123456789abc"

TrajectoryFactory = Callable[[Path], SessionTrajectory]


@pytest.fixture
async def live_trajectory(tmp_path: Path) -> AsyncIterator[TrajectoryFactory]:
    """Activate real recorders (so they really hold their lease) and close them at teardown."""
    created: list[SessionTrajectory] = []

    def _factory(session_dir: Path) -> SessionTrajectory:
        trajectory = SessionTrajectory(
            session_id=SESSION_ID,
            session_dir=session_dir,
            config_dir=tmp_path / "config",
        )
        created.append(trajectory)
        return trajectory

    yield _factory

    for trajectory in reversed(created):
        if not trajectory.is_closed:
            await trajectory.close(reason=RuntimeFinishReason.GRACEFUL_SHUTDOWN)


def _session_dir(sessions_root: Path, name: str = "sess-1") -> Path:
    session_dir = sessions_root / name
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    return session_dir


async def _activate(trajectory: SessionTrajectory) -> None:
    await trajectory.emit(EventDraft(event_type=EventType.TURN_STARTED, payload={"turn_number": 1}))


# ------------------------------------------------------------------- paths


def test_tombstones_dir_is_a_dot_prefixed_child_of_the_sessions_root(tmp_path: Path) -> None:
    assert tombstones_dir(tmp_path) == tmp_path / TOMBSTONES_DIR_NAME
    # Dot-prefixed so a session listing never picks it up as a session.
    assert TOMBSTONES_DIR_NAME.startswith(".")


def test_session_lease_path_lives_under_the_trajectory_directory(tmp_path: Path) -> None:
    assert session_lease_path(tmp_path).parent == trajectory_dir(tmp_path)


def test_writer_lease_held_is_false_without_a_lease_file(tmp_path: Path) -> None:
    assert writer_lease_held(tmp_path / "nowhere") is False


# ------------------------------------------------------------------ delete


def test_missing_directory_reports_missing(tmp_path: Path) -> None:
    result = delete_session_directory(tmp_path / "gone", sessions_root=tmp_path)
    assert result.outcome == DeleteOutcome.MISSING
    assert result.tombstone_path is None
    assert not tombstones_dir(tmp_path).exists()


def test_free_session_is_removed_physically(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    session_dir = _session_dir(sessions_root)

    result = delete_session_directory(session_dir, sessions_root=sessions_root)

    assert result.outcome == DeleteOutcome.REMOVED
    assert result.tombstone_path is None
    assert not session_dir.exists()
    # Nothing is deferred, so no graveyard is created at all.
    assert not tombstones_dir(sessions_root).exists()


def test_session_with_a_stale_lease_file_but_no_holder_is_removed(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    session_dir = _session_dir(sessions_root)
    lease_path = session_lease_path(session_dir)
    ensure_owner_only_directory(lease_path.parent)
    # The lease file a dead process left behind: present, owner-only, unheld.
    released = WriterLease.try_acquire(lease_path)
    assert released is not None
    released.release()
    assert lease_path.is_file()

    assert writer_lease_held(session_dir) is False
    assert delete_session_directory(session_dir, sessions_root=sessions_root).outcome == DeleteOutcome.REMOVED
    assert not session_dir.exists()


@pytest.mark.skipif(get_platform().is_windows, reason="POSIX permission bits")
def test_lease_file_that_cannot_be_opened_owner_only_fails_closed(tmp_path: Path) -> None:
    session_dir = _session_dir(tmp_path / "sessions")
    lease_path = session_lease_path(session_dir)
    ensure_owner_only_directory(lease_path.parent)
    lease_path.touch(mode=0o644)

    # A lock file the owner-only contract rejects is treated as held rather
    # than swept: deletion must never race a writer it cannot rule out.
    assert writer_lease_held(session_dir) is True


def _stale_lease_file(session_dir: Path) -> Path:
    """The lease file a dead process left behind: present, owner-only, unheld."""
    lease_path = session_lease_path(session_dir)
    ensure_owner_only_directory(lease_path.parent)
    released = WriterLease.try_acquire(lease_path)
    assert released is not None
    released.release()
    return lease_path


def _watch_removal(monkeypatch: pytest.MonkeyPatch, lease_path: Path, seen: list[bool]) -> None:
    """Record, at the moment of removal, whether a would-be writer is refused the lease."""

    def watching_rmtree(path: Path, **kwargs: Any) -> None:
        contender = WriterLease.try_acquire(lease_path)
        seen.append(contender is None)
        if contender is not None:
            contender.release()
        shutil.rmtree(path, **kwargs)

    monkeypatch.setattr(tombstone_module, "shutil", SimpleNamespace(rmtree=watching_rmtree))


def test_deletion_holds_the_lease_for_as_long_as_the_directory_takes_to_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking whether a writer exists leaves a gap; a writer starting in it writes into a grave."""
    sessions_root = tmp_path / "sessions"
    session_dir = _session_dir(sessions_root)
    lease_path = _stale_lease_file(session_dir)
    refused: list[bool] = []
    _watch_removal(monkeypatch, lease_path, refused)

    result = delete_session_directory(session_dir, sessions_root=sessions_root)

    assert result.outcome == DeleteOutcome.REMOVED
    assert refused == [True]


def test_deletion_takes_the_lease_a_session_never_had(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A session with no lock file is not one nobody can write: activation would create it."""
    sessions_root = tmp_path / "sessions"
    session_dir = _session_dir(sessions_root)
    lease_path = session_lease_path(session_dir)
    assert not lease_path.exists()
    refused: list[bool] = []
    _watch_removal(monkeypatch, lease_path, refused)

    result = delete_session_directory(session_dir, sessions_root=sessions_root)

    assert result.outcome == DeleteOutcome.REMOVED
    assert not session_dir.exists()
    # The lease was made for the removal and refused to everyone else while it ran.
    assert refused == [True]


def test_the_sweep_holds_the_lease_for_as_long_as_the_tombstone_takes_to_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    tombstone = tombstones_dir(sessions_root) / "sess-1-abandoned"
    tombstone.mkdir(parents=True)
    lease_path = _stale_lease_file(tombstone)
    refused: list[bool] = []
    _watch_removal(monkeypatch, lease_path, refused)

    assert sweep_tombstones(sessions_root) == 1
    assert refused == [True]


@pytest.mark.asyncio
async def test_held_lease_tombstones_instead_of_deleting(tmp_path: Path, live_trajectory: TrajectoryFactory) -> None:
    sessions_root = tmp_path / "sessions"
    session_dir = _session_dir(sessions_root)
    trajectory = live_trajectory(session_dir)
    await _activate(trajectory)
    assert trajectory.is_active
    assert writer_lease_held(session_dir) is True

    result = delete_session_directory(session_dir, sessions_root=sessions_root)

    # The caller never blocks on the live writer: the directory moves aside.
    assert result.outcome in (DeleteOutcome.TOMBSTONED, DeleteOutcome.INTENT_RECORDED)
    if result.outcome == DeleteOutcome.TOMBSTONED:
        assert result.tombstone_path is not None
        assert result.tombstone_path.parent == tombstones_dir(sessions_root)
        assert result.tombstone_path.name.startswith(session_dir.name)
        assert not session_dir.exists()
    else:
        # Windows can refuse the rename while a child handle is open.
        assert pending_delete_intents(sessions_root) == frozenset({session_dir.name})


@pytest.mark.asyncio
async def test_an_unrecordable_intent_is_reported_instead_of_claimed(
    tmp_path: Path, live_trajectory: TrajectoryFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing will ever sweep what no intent names, so the caller is told."""
    import chrys.service.trajectory.tombstone as tombstone_module

    sessions_root = tmp_path / "sessions"
    session_dir = _session_dir(sessions_root)
    trajectory = live_trajectory(session_dir)
    await _activate(trajectory)

    class _RefusesRename:
        """The module's own ``os`` name — patching the stdlib one is global."""

        def __getattr__(self, name: str) -> object:
            return getattr(os, name)

        def rename(self, *_args: object, **_kwargs: object) -> None:
            raise OSError("rename refused")

    def refuse_open(*_args: object, **_kwargs: object) -> int:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(tombstone_module, "os", _RefusesRename())
    monkeypatch.setattr(tombstone_module, "secure_open_owner_only", refuse_open)

    result = delete_session_directory(session_dir, sessions_root=sessions_root)

    assert result.outcome == DeleteOutcome.INTENT_FAILED
    assert pending_delete_intents(sessions_root) == frozenset()


@pytest.mark.asyncio
async def test_an_intent_whose_body_could_not_be_written_still_owns_the_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_trajectory: TrajectoryFactory
) -> None:
    """The sweep keys off the intent's name, so an empty one is still an owner."""
    sessions_root = tmp_path / "sessions"
    session_dir = _session_dir(sessions_root)
    trajectory = live_trajectory(session_dir)
    await _activate(trajectory)

    class _RefusesRenameAndWrite:
        """The module's own ``os`` name — patching the stdlib one is global."""

        def __getattr__(self, name: str) -> object:
            return getattr(os, name)

        def rename(self, *_args: object, **_kwargs: object) -> None:
            raise OSError("rename refused")

        def write(self, *_args: object, **_kwargs: object) -> int:
            raise OSError("no space left on device")

    monkeypatch.setattr(tombstone_module, "os", _RefusesRenameAndWrite())

    result = delete_session_directory(session_dir, sessions_root=sessions_root)

    assert result.outcome == DeleteOutcome.INTENT_RECORDED
    intent = tombstones_dir(sessions_root) / f"{session_dir.name}{INTENT_SUFFIX}"
    assert intent.is_file()
    assert intent.read_bytes() == b""
    assert pending_delete_intents(sessions_root) == frozenset({session_dir.name})

    monkeypatch.setattr(tombstone_module, "os", os)
    await trajectory.close(reason=RuntimeFinishReason.GRACEFUL_SHUTDOWN)

    assert sweep_tombstones(sessions_root) == 1
    assert not session_dir.exists()
    assert not intent.exists()


@pytest.mark.asyncio
async def test_sweep_skips_a_tombstone_whose_lease_is_still_held(
    tmp_path: Path, live_trajectory: TrajectoryFactory
) -> None:
    sessions_root = tmp_path / "sessions"
    session_dir = _session_dir(sessions_root)
    trajectory = live_trajectory(session_dir)
    await _activate(trajectory)
    result = delete_session_directory(session_dir, sessions_root=sessions_root)
    if result.outcome != DeleteOutcome.TOMBSTONED:
        pytest.skip("platform refused the tombstone rename; covered by the intent test")

    # The writer is still alive, so nothing may be pulled from under it.
    assert sweep_tombstones(sessions_root) == 0
    assert result.tombstone_path is not None
    assert result.tombstone_path.is_dir()

    await trajectory.close(reason=RuntimeFinishReason.GRACEFUL_SHUTDOWN)

    assert sweep_tombstones(sessions_root) == 1
    assert not result.tombstone_path.exists()
    # A second sweep has nothing left to do.
    assert sweep_tombstones(sessions_root) == 0


def test_sweep_removes_a_tombstone_that_never_had_a_lease(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    graveyard = tombstones_dir(sessions_root)
    ensure_owner_only_directory(graveyard)
    tombstone = graveyard / "sess-1-deadbeef"
    tombstone.mkdir()
    (tombstone / "session.json").write_text("{}", encoding="utf-8")

    assert sweep_tombstones(sessions_root) == 1
    assert not tombstone.exists()


def test_sweep_of_a_missing_root_is_zero(tmp_path: Path) -> None:
    assert sweep_tombstones(tmp_path / "nowhere") == 0
    assert pending_delete_intents(tmp_path / "nowhere") == frozenset()


def test_intent_file_survives_until_its_target_is_gone(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    session_dir = _session_dir(sessions_root)
    graveyard = tombstones_dir(sessions_root)
    ensure_owner_only_directory(graveyard)
    intent = graveyard / f"{session_dir.name}{INTENT_SUFFIX}"
    intent.write_text(session_dir.name, encoding="utf-8")

    assert pending_delete_intents(sessions_root) == frozenset({session_dir.name})

    # Sweeping consumes the intent and finishes the physical removal.
    assert sweep_tombstones(sessions_root) == 1
    assert not session_dir.exists()
    assert not intent.exists()
    assert pending_delete_intents(sessions_root) == frozenset()


def test_intent_without_a_target_is_still_consumed(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    graveyard = tombstones_dir(sessions_root)
    ensure_owner_only_directory(graveyard)
    intent = graveyard / f"sess-gone{INTENT_SUFFIX}"
    intent.write_text("sess-gone", encoding="utf-8")

    assert sweep_tombstones(sessions_root) == 1
    assert not intent.exists()


def test_intent_is_kept_while_its_target_lease_is_held(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    session_dir = _session_dir(sessions_root)
    lease_path = session_lease_path(session_dir)
    ensure_owner_only_directory(lease_path.parent)
    lease = WriterLease.try_acquire(lease_path)
    assert lease is not None
    try:
        graveyard = tombstones_dir(sessions_root)
        ensure_owner_only_directory(graveyard)
        intent = graveyard / f"{session_dir.name}{INTENT_SUFFIX}"
        intent.write_text(session_dir.name, encoding="utf-8")

        assert sweep_tombstones(sessions_root) == 0
        assert intent.exists()
        assert session_dir.exists()
    finally:
        lease.release()

    assert sweep_tombstones(sessions_root) == 1
    assert not session_dir.exists()


@pytest.mark.skipif(os.name == "nt", reason="tombstoning a directory whose lease is held is POSIX-only")
def test_a_tombstoned_lease_stops_blocking_a_session_recreated_at_its_path(tmp_path: Path) -> None:
    """A reset restarts on the same id, so the writer that moved into the
    tombstone must not keep the old path registered as taken."""
    sessions_root = tmp_path / "sessions"
    session_dir = _session_dir(sessions_root)
    lease_path = session_lease_path(session_dir)
    ensure_owner_only_directory(lease_path.parent)
    lease = WriterLease.try_acquire(lease_path)
    assert lease is not None
    try:
        result = delete_session_directory(session_dir, sessions_root=sessions_root)
        assert result.outcome == DeleteOutcome.TOMBSTONED
        assert result.tombstone_path is not None
        assert writer_lease_held(result.tombstone_path)

        # The restarted session writes into the very same directory again.
        ensure_owner_only_directory(lease_path.parent)
        reused = WriterLease.try_acquire(lease_path)
        assert reused is not None
        reused.release()
    finally:
        lease.release()


def test_sweep_ignores_stray_files_and_symlinked_entries(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    graveyard = tombstones_dir(sessions_root)
    ensure_owner_only_directory(graveyard)
    stray = graveyard / "notes.txt"
    stray.write_text("not a tombstone", encoding="utf-8")

    outside = tmp_path / "outside"
    outside.mkdir()
    if not get_platform().is_windows:
        link = graveyard / "linked"
        link.symlink_to(outside, target_is_directory=True)

    assert sweep_tombstones(sessions_root) == 0
    assert stray.exists()
    # A symlinked entry is never followed: the target survives untouched.
    assert outside.is_dir()
