# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the session-migration service (copy sessions between two roots)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

import chrys.service.state.session_migration as migration_module
from chrys.foundation.platform.files import atomic_write_owner_only_bytes, secure_open_owner_only_binary
from chrys.foundation.util.lock import FileLock
from chrys.service.state.session_migration import (
    MigrationItem,
    SessionMigrationError,
    plan_session_migration,
    run_session_migration,
)
from chrys.service.state.session_mru import SESSION_MRU_FILE_NAME
from chrys.service.state.store import (
    SESSION_FILE_NAME,
    session_active_lock_path,
    session_dir_candidates,
    session_write_lock_path,
)


def _make_session(root: Path, short_id: str, *, text: str = "hello") -> Path:
    session_dir = root / short_id
    session_dir.mkdir(parents=True)
    (session_dir / SESSION_FILE_NAME).write_text(json.dumps({"meta": {"session_id": short_id}, "text": text}))
    snapshots = session_dir / "snapshots"
    snapshots.mkdir()
    (snapshots / "turn_1.json").write_text("{}")
    return session_dir


def _snapshot_tree(root: Path) -> dict[str, bytes | None]:
    """Relative path -> file bytes (None for directories) for the whole tree."""
    tree: dict[str, bytes | None] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        tree[rel] = None if path.is_dir() else path.read_bytes()
    return tree


def _symlink_or_skip(link: Path, target: Path, *, directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")


def _partials(root: Path) -> list[Path]:
    return [p for p in root.iterdir() if ".partial-" in p.name]


def test_plan_and_run_copy_sessions_into_fresh_destination(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    first_session = _make_session(src, "aaaaaaaaaaaa")
    _make_session(src, "bbbbbbbbbbbb", text="second")
    source_image = first_session / "doc_converter" / "image-copied.png"
    atomic_write_owner_only_bytes(source_image, b"copied-image")
    (src / ".locks").mkdir()
    (src / ".locks" / "stray.write.lock").write_text("")
    (src / SESSION_MRU_FILE_NAME).write_text("{}")
    before = _snapshot_tree(src)

    plan = plan_session_migration(src, dst)
    assert [item.session_id for item in plan.items] == ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
    assert all(not item.already_present and not item.legacy_file for item in plan.items)
    assert plan.rejected == ()
    # Planning creates nothing.
    assert not dst.exists()

    with patch.object(
        migration_module,
        "reharden_document_image_artifacts",
        wraps=migration_module.reharden_document_image_artifacts,
    ) as reharden:
        report = run_session_migration(plan)

    assert report.copied == ("aaaaaaaaaaaa", "bbbbbbbbbbbb")
    assert report.skipped_present == report.skipped_active == report.skipped_busy == ()
    assert report.failed == ()
    assert reharden.call_count == 2
    assert (dst / ".locks").is_dir()
    for short_id in ("aaaaaaaaaaaa", "bbbbbbbbbbbb"):
        assert _snapshot_tree(dst / short_id) == _snapshot_tree(src / short_id)
    with secure_open_owner_only_binary(dst / "aaaaaaaaaaaa" / "doc_converter" / "image-copied.png") as copied:
        assert copied.read() == b"copied-image"
    # Root-level lock files and the MRU index are never carried over.
    assert not (dst / ".locks" / "stray.write.lock").exists()
    assert not (dst / SESSION_MRU_FILE_NAME).exists()
    assert _partials(dst) == []
    # Source is untouched apart from the lock files the copy took under .locks/.
    after = _snapshot_tree(src)
    assert {k: v for k, v in after.items() if not k.startswith(".locks/")} == {
        k: v for k, v in before.items() if not k.startswith(".locks/")
    }
    assert {k for k in after if k.startswith(".locks/")} == {
        ".locks/stray.write.lock",
        ".locks/aaaaaaaaaaaa.active.lock",
        ".locks/aaaaaaaaaaaa.write.lock",
        ".locks/bbbbbbbbbbbb.active.lock",
        ".locks/bbbbbbbbbbbb.write.lock",
    }
    assert session_dir_candidates(dst) == [dst / "aaaaaaaaaaaa", dst / "bbbbbbbbbbbb"]


def test_run_skips_already_present_destination_without_overwriting(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_session(src, "aaaaaaaaaaaa", text="from-source")
    _make_session(dst, "aaaaaaaaaaaa", text="already-there")
    _make_session(src, "bbbbbbbbbbbb")

    plan = plan_session_migration(src, dst)
    assert {item.session_id: item.already_present for item in plan.items} == {
        "aaaaaaaaaaaa": True,
        "bbbbbbbbbbbb": False,
    }

    report = run_session_migration(plan)

    assert report.skipped_present == ("aaaaaaaaaaaa",)
    assert report.copied == ("bbbbbbbbbbbb",)
    assert "already-there" in (dst / "aaaaaaaaaaaa" / SESSION_FILE_NAME).read_text()


def test_run_treats_destination_appearing_after_plan_as_present(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_session(src, "aaaaaaaaaaaa", text="from-source")
    plan = plan_session_migration(src, dst)
    assert plan.items[0].already_present is False

    _make_session(dst, "aaaaaaaaaaaa", text="raced-in")
    report = run_session_migration(plan)

    assert report.skipped_present == ("aaaaaaaaaaaa",)
    assert report.copied == ()
    assert "raced-in" in (dst / "aaaaaaaaaaaa" / SESSION_FILE_NAME).read_text()


def test_run_rechecks_the_destination_after_taking_its_write_lock(tmp_path: Path) -> None:
    """A second migration that lands the same session while this one waits for
    the destination lock must be left alone — even a legacy flat file, which
    ``os.replace`` would otherwise silently overwrite."""
    import threading

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "legacylegacy.json").write_text(json.dumps({"meta": {"session_id": "legacylegacy"}, "text": "mine"}))
    plan = plan_session_migration(src, dst)
    assert [item.legacy_file for item in plan.items] == [True]
    (dst / ".locks").mkdir(parents=True)
    holder = FileLock(session_write_lock_path(dst, "legacylegacy"), timeout=1.0)
    holder.acquire()

    def land_the_other_copy_then_release() -> None:
        (dst / "legacylegacy.json").write_text(json.dumps({"meta": {"session_id": "legacylegacy"}, "text": "theirs"}))
        holder.release()

    timer = threading.Timer(0.2, land_the_other_copy_then_release)
    timer.start()
    try:
        report = run_session_migration(plan)
    finally:
        timer.join()

    assert report.skipped_present == ("legacylegacy",)
    assert report.copied == ()
    assert "theirs" in (dst / "legacylegacy.json").read_text()


def test_run_skips_session_whose_active_lock_is_held(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_session(src, "aaaaaaaaaaaa")
    _make_session(src, "bbbbbbbbbbbb")
    (src / ".locks").mkdir()
    holder = FileLock(session_active_lock_path(src, "aaaaaaaaaaaa"), timeout=1.0)
    holder.acquire()
    try:
        report = run_session_migration(plan_session_migration(src, dst))
    finally:
        holder.release()

    assert report.skipped_active == ("aaaaaaaaaaaa",)
    assert report.copied == ("bbbbbbbbbbbb",)
    assert not (dst / "aaaaaaaaaaaa").exists()
    assert (dst / "bbbbbbbbbbbb" / SESSION_FILE_NAME).exists()


def test_run_holds_source_active_lock_while_copying(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_session(src, "aaaaaaaaaaaa")
    real_copytree = shutil.copytree
    contended: list[str] = []

    # copytree recurses through the module attribute, so nested calls land
    # here too (positionally); only probe the session-root call.
    def probing_copytree(source: object, *args: object, **kwargs: object) -> object:
        source_path = Path(str(source))
        if source_path.parent == src:
            probe = FileLock(session_active_lock_path(src, source_path.name), timeout=0)
            try:
                probe.acquire()
            except TimeoutError:
                contended.append(source_path.name)
            else:
                probe.release()
        return real_copytree(source, *args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", probing_copytree)
    report = run_session_migration(plan_session_migration(src, dst))

    assert report.copied == ("aaaaaaaaaaaa",)
    assert contended == ["aaaaaaaaaaaa"]
    # Released afterwards: the lock is free again.
    with FileLock(session_active_lock_path(src, "aaaaaaaaaaaa"), timeout=0):
        pass


@pytest.mark.parametrize("busy_root", ["source", "destination"])
def test_run_skips_session_whose_write_lock_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, busy_root: str
) -> None:
    monkeypatch.setattr(migration_module, "MIGRATION_WRITE_LOCK_TIMEOUT_SECONDS", 0.05)
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_session(src, "aaaaaaaaaaaa")
    _make_session(src, "bbbbbbbbbbbb")
    lock_root = src if busy_root == "source" else dst
    (lock_root / ".locks").mkdir(parents=True)
    holder = FileLock(session_write_lock_path(lock_root, "aaaaaaaaaaaa"), timeout=1.0)
    holder.acquire()
    try:
        report = run_session_migration(plan_session_migration(src, dst))
    finally:
        holder.release()

    assert report.skipped_busy == ("aaaaaaaaaaaa",)
    assert report.copied == ("bbbbbbbbbbbb",)
    assert not (dst / "aaaaaaaaaaaa").exists()


def test_failed_session_does_not_stop_others_and_leaves_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    for short_id in ("aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"):
        _make_session(src, short_id)
    real_copytree = shutil.copytree

    def failing_copytree(source: object, destination: object, *args: object, **kwargs: object) -> object:
        if Path(str(source)) == src / "bbbbbbbbbbbb":
            # Leave a half-written partial behind so cleanup has work to do.
            Path(str(destination)).mkdir(exist_ok=True)
            (Path(str(destination)) / SESSION_FILE_NAME).write_text("{}")
            raise OSError("boom")
        return real_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(shutil, "copytree", failing_copytree)
    report = run_session_migration(plan_session_migration(src, dst))

    assert report.copied == ("aaaaaaaaaaaa", "cccccccccccc")
    assert report.failed == ((src / "bbbbbbbbbbbb", "boom"),)
    assert not (dst / "bbbbbbbbbbbb").exists()
    assert _partials(dst) == []
    # Locks are released again after the failure.
    with FileLock(session_active_lock_path(src, "bbbbbbbbbbbb"), timeout=0):
        pass
    with FileLock(session_write_lock_path(dst, "bbbbbbbbbbbb"), timeout=0):
        pass


def test_unlockable_session_is_reported_and_does_not_stop_the_others(tmp_path: Path) -> None:
    """A lock path that cannot be opened fails that session only."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    for short_id in ("aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"):
        _make_session(src, short_id)
    # A directory where the source write lock file should be: ``os.open``
    # refuses it with an ``OSError`` that is not a timeout.
    session_write_lock_path(src, "bbbbbbbbbbbb").mkdir(parents=True)

    report = run_session_migration(plan_session_migration(src, dst))

    assert report.copied == ("aaaaaaaaaaaa", "cccccccccccc")
    assert [path for path, _reason in report.failed] == [src / "bbbbbbbbbbbb"]
    assert not (dst / "bbbbbbbbbbbb").exists()
    assert _partials(dst) == []
    # The active lock taken before the failure was released again.
    with FileLock(session_active_lock_path(src, "bbbbbbbbbbbb"), timeout=0):
        pass


def test_session_dir_copy_leaves_a_pre_existing_partial_dir_alone(tmp_path: Path) -> None:
    """The partial is created exclusively; a folder planted at the old predictable name is not removed."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_session(src, "aaaaaaaaaaaa")
    dst.mkdir()
    planted = dst / f".partial-aaaaaaaaaaaa-{os.getpid()}"
    planted.mkdir()
    (planted / "keep.txt").write_text("mine")

    report = run_session_migration(plan_session_migration(src, dst))

    assert report.copied == ("aaaaaaaaaaaa",)
    assert (dst / "aaaaaaaaaaaa" / SESSION_FILE_NAME).read_bytes() == (
        src / "aaaaaaaaaaaa" / SESSION_FILE_NAME
    ).read_bytes()
    assert (planted / "keep.txt").read_text() == "mine"
    assert [p.name for p in _partials(dst)] == [planted.name]


def test_a_session_dir_swapped_for_a_link_after_planning_is_not_followed(tmp_path: Path) -> None:
    """The plan saw a real folder; a link put in its place before the copy is refused."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_session(src, "aaaaaaaaaaaa")
    _make_session(src, "bbbbbbbbbbbb")
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / SESSION_FILE_NAME).write_text(json.dumps({"secret": True}))
    plan = plan_session_migration(src, dst)
    assert [item.session_id for item in plan.items] == ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]

    shutil.rmtree(src / "aaaaaaaaaaaa")
    _symlink_or_skip(src / "aaaaaaaaaaaa", victim, directory=True)
    report = run_session_migration(plan)

    assert report.copied == ("bbbbbbbbbbbb",)
    assert report.failed == ((src / "aaaaaaaaaaaa", "linked entry"),)
    assert not (dst / "aaaaaaaaaaaa").exists()
    assert _partials(dst) == []


def test_a_session_dir_swapped_for_a_junction_after_planning_is_not_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``lstat`` sees a junction as a plain directory; the run-time re-check asks by name too."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    session_dir = _make_session(src, "aaaaaaaaaaaa")
    _make_session(src, "bbbbbbbbbbbb")
    plan = plan_session_migration(src, dst)
    assert [item.session_id for item in plan.items] == ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]

    # "Replace" the planned folder with a junction: same real directory on
    # disk (junctions cannot be created on POSIX), reported as one from now on.
    planted = os.path.normcase(os.path.abspath(session_dir))
    real_isjunction = os.path.isjunction

    def fake_isjunction(candidate: object) -> bool:
        return os.path.normcase(os.path.abspath(candidate)) == planted or real_isjunction(candidate)  # type: ignore[arg-type]

    monkeypatch.setattr(os.path, "isjunction", fake_isjunction)
    report = run_session_migration(plan)

    assert report.copied == ("bbbbbbbbbbbb",)
    assert report.failed == ((src / "aaaaaaaaaaaa", "linked entry"),)
    assert not (dst / "aaaaaaaaaaaa").exists()


def test_a_legacy_file_swapped_for_a_link_after_planning_is_not_followed(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "legacylegacy.json").write_text(json.dumps({"meta": {"session_id": "legacylegacy"}}))
    victim = tmp_path / "victim.json"
    victim.write_text(json.dumps({"secret": True}))
    plan = plan_session_migration(src, dst)
    assert [item.session_id for item in plan.items] == ["legacylegacy"]

    (src / "legacylegacy.json").unlink()
    _symlink_or_skip(src / "legacylegacy.json", victim, directory=False)
    report = run_session_migration(plan)

    assert report.copied == ()
    assert report.failed == ((src / "legacylegacy.json", "linked entry"),)
    assert not (dst / "legacylegacy.json").exists()
    assert _partials(dst) == []


def test_a_legacy_file_open_refuses_a_link_even_without_the_lstat_check(tmp_path: Path) -> None:
    """The descriptor-level guard: opening the leaf never follows a link."""
    victim = tmp_path / "victim.json"
    victim.write_text("{}")
    link = tmp_path / "link.json"
    _symlink_or_skip(link, victim, directory=False)
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("O_NOFOLLOW unavailable on this platform")

    with pytest.raises(OSError):
        migration_module._open_source_file(link)


def test_leftover_partial_dirs_are_not_sessions(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _make_session(root, "aaaaaaaaaaaa")
    leftover = root / ".partial-bbbbbbbbbbbb-4242"
    leftover.mkdir()
    (leftover / SESSION_FILE_NAME).write_text("{}")

    assert session_dir_candidates(root) == [root / "aaaaaaaaaaaa"]
    plan = plan_session_migration(root, tmp_path / "dst")
    assert [item.session_id for item in plan.items] == ["aaaaaaaaaaaa"]


def test_legacy_flat_file_is_copied(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "legacylegacy.json").write_text(json.dumps({"meta": {"session_id": "legacylegacy"}}))
    (src / SESSION_MRU_FILE_NAME).write_text("{}")

    plan = plan_session_migration(src, dst)
    assert plan.items == (
        MigrationItem(
            session_id="legacylegacy",
            source=src / "legacylegacy.json",
            destination=dst / "legacylegacy.json",
            legacy_file=True,
            already_present=False,
        ),
    )

    report = run_session_migration(plan)

    assert report.copied == ("legacylegacy",)
    assert (dst / "legacylegacy.json").read_bytes() == (src / "legacylegacy.json").read_bytes()
    assert not (dst / SESSION_MRU_FILE_NAME).exists()
    assert _partials(dst) == []


def test_legacy_file_copy_never_follows_a_planted_partial_symlink(tmp_path: Path) -> None:
    """A predictable partial name planted as a symlink must not redirect the copy."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    payload = json.dumps({"meta": {"session_id": "legacylegacy"}})
    (src / "legacylegacy.json").write_text(payload)
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched")
    _symlink_or_skip(dst / f"legacylegacy.json.partial-{os.getpid()}", victim, directory=False)

    plan = plan_session_migration(src, dst)
    report = run_session_migration(plan)

    assert report.copied == ("legacylegacy",)
    assert victim.read_text() == "untouched"
    landed = dst / "legacylegacy.json"
    assert not landed.is_symlink() and landed.read_text() == payload
    assert landed.stat().st_mtime == (src / "legacylegacy.json").stat().st_mtime
    # Only the planted link is left behind; the copy's own partial is gone.
    assert [p.name for p in _partials(dst)] == [f"legacylegacy.json.partial-{os.getpid()}"]


def test_legacy_file_already_present_is_skipped(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "legacylegacy.json").write_text("source")
    (dst / "legacylegacy.json").write_text("existing")

    report = run_session_migration(plan_session_migration(src, dst))

    assert report.skipped_present == ("legacylegacy",)
    assert (dst / "legacylegacy.json").read_text() == "existing"


def test_plan_rejects_same_and_nested_roots(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    with pytest.raises(SessionMigrationError):
        plan_session_migration(root, root)
    with pytest.raises(SessionMigrationError):
        plan_session_migration(root, root / "child")
    with pytest.raises(SessionMigrationError):
        plan_session_migration(root / "child", root)
    with pytest.raises(SessionMigrationError):
        plan_session_migration(root, root / "a" / "b" / "c")
    # Nothing was created while validating.
    assert list(root.iterdir()) == []


def test_plan_rejects_same_and_nested_roots_spelled_in_another_case(tmp_path: Path) -> None:
    """On a case-insensitive volume the roots are compared by identity, not spelling."""
    root = tmp_path / "sessions"
    root.mkdir()
    alias = tmp_path / "SESSIONS"
    if not alias.is_dir() or not os.path.samefile(root, alias):
        pytest.skip("case-sensitive filesystem")
    with pytest.raises(SessionMigrationError, match="same directory"):
        plan_session_migration(root, alias)
    with pytest.raises(SessionMigrationError, match="inside source"):
        plan_session_migration(root, alias / "nested" / "deeper")
    with pytest.raises(SessionMigrationError, match="inside destination"):
        plan_session_migration(root / "child", alias)
    assert list(root.iterdir()) == []


def test_plan_rejects_same_root_through_symlink(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    link = tmp_path / "sessions-link"
    _symlink_or_skip(link, root, directory=True)
    with pytest.raises(SessionMigrationError):
        plan_session_migration(root, link)
    with pytest.raises(SessionMigrationError):
        plan_session_migration(link, root / "nested")


def test_plan_rejects_source_that_is_not_a_directory(tmp_path: Path) -> None:
    with pytest.raises(SessionMigrationError):
        plan_session_migration(tmp_path / "missing", tmp_path / "dst")
    file_source = tmp_path / "file"
    file_source.write_text("")
    with pytest.raises(SessionMigrationError):
        plan_session_migration(file_source, tmp_path / "dst")


def test_symlinked_session_root_is_rejected_not_copied(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _make_session(src, "aaaaaaaaaaaa")
    real = _make_session(tmp_path / "elsewhere", "linklinklink")
    _symlink_or_skip(src / "linklinklink", real, directory=True)

    plan = plan_session_migration(src, dst)
    assert [item.session_id for item in plan.items] == ["aaaaaaaaaaaa"]
    assert plan.rejected == ((src / "linklinklink", "linked entry"),)

    report = run_session_migration(plan)

    assert report.copied == ("aaaaaaaaaaaa",)
    assert report.failed == ((src / "linklinklink", "linked entry"),)
    assert not os.path.lexists(dst / "linklinklink")


def test_symlinked_legacy_file_is_rejected_not_copied(tmp_path: Path) -> None:
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    real = tmp_path / "elsewhere.json"
    real.write_text("{}")
    _symlink_or_skip(src / "linklinklink.json", real, directory=False)

    plan = plan_session_migration(src, dst)
    assert plan.items == ()
    assert plan.rejected == ((src / "linklinklink.json", "linked entry"),)

    report = run_session_migration(plan)
    assert report.failed == ((src / "linklinklink.json", "linked entry"),)
    assert not os.path.lexists(dst / "linklinklink.json")


def test_junctions_dropped_when_nested_and_rejected_at_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """NT directory junctions cannot be created on POSIX; simulate them via os.path.isjunction."""
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    session_dir = _make_session(src, "aaaaaaaaaaaa")
    nested_junction = session_dir / "compactions" / "planted"
    nested_junction.mkdir(parents=True)
    (nested_junction / "payload.txt").write_text("x")
    (session_dir / "compactions" / "real").mkdir()
    (session_dir / "compactions" / "real" / "keep.txt").write_text("y")
    root_junction = _make_session(src, "jjjjjjjjjjjj")
    planted_keys = {
        os.path.normcase(os.path.abspath(nested_junction)),
        os.path.normcase(os.path.abspath(root_junction)),
    }
    real_isjunction = os.path.isjunction

    def fake_isjunction(candidate: object) -> bool:
        return os.path.normcase(os.path.abspath(candidate)) in planted_keys or real_isjunction(candidate)  # type: ignore[arg-type]

    monkeypatch.setattr(os.path, "isjunction", fake_isjunction)

    plan = plan_session_migration(src, dst)
    assert [item.session_id for item in plan.items] == ["aaaaaaaaaaaa"]
    assert plan.rejected == ((root_junction, "linked entry"),)

    report = run_session_migration(plan)

    assert report.copied == ("aaaaaaaaaaaa",)
    assert report.failed == ((root_junction, "linked entry"),)
    assert not (dst / "jjjjjjjjjjjj").exists()
    assert not (dst / "aaaaaaaaaaaa" / "compactions" / "planted").exists()
    assert (dst / "aaaaaaaaaaaa" / "compactions" / "real" / "keep.txt").read_text() == "y"
