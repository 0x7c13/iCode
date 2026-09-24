# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Branch coverage for ``core/mutations/store.py`` and ``core/mutations/scanner.py``.

Focuses on error paths and small helpers not exercised by the high-level
``test_snapshots`` integration tests: OSError branches, symlinks, hash
short-circuits, directory-as-path refusal, ``save_blob`` / ``save_data_as_blob``
/ ``read_blob`` helpers, and scanner edge cases (max_depth, gitignore,
wildcard excludes, shallow directory walks, stat errors).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from chrys.service.mutations.scanner import (
    DEFAULT_EXCLUDES,
    FileStat,
    GitignoreFilter,
    WorkspaceScanner,
)
from chrys.service.mutations.store import SnapshotStore
from chrys.service.mutations.types import FileSnapshot, RestoreOutcome, SnapshotSkipReason
from tests.support.symlinks import symlink_or_skip

# ════════════════════════════ SnapshotStore ════════════════════════════


@pytest.fixture
def store(tmp_path: Path) -> SnapshotStore:
    return SnapshotStore(tmp_path)


def test_mutations_dir_property(store: SnapshotStore, tmp_path: Path) -> None:
    assert store.mutations_dir == tmp_path / "mutations"


def test_save_missing_file_returns_nonexistent_snapshot(store: SnapshotStore, tmp_path: Path) -> None:
    snap = store.save(str(tmp_path / "does_not_exist"), period_index=1)
    assert snap.existed is False
    assert snap.content_hash is None


def test_save_oserror_during_read_is_unrestorable(
    store: SnapshotStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable preexisting file must never get a deletion rollback target."""
    target = tmp_path / "unreadable.txt"
    target.write_text("hello")

    # Monkeypatch Path.read_bytes to raise
    original_read_bytes = Path.read_bytes

    def _boom(self: Path) -> bytes:
        if self == Path(os.path.normpath(os.path.abspath(str(target)))):
            raise OSError("permission denied")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _boom)

    snap = store.save(str(target), period_index=1)
    assert snap.existed is True
    assert snap.skip_reason is SnapshotSkipReason.UNREADABLE
    assert snap.restorable is False


# ── restore: delete-branch (existed=False) ────────────────────────────


def test_restore_delete_branch_removes_file(store: SnapshotStore, tmp_path: Path) -> None:
    target = tmp_path / "created_after_turn.txt"
    target.write_text("x")
    snap = FileSnapshot(path=str(target), period_index=1, existed=False)

    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.APPLIED
    assert not target.exists()


def test_restore_delete_branch_noop_when_absent(store: SnapshotStore, tmp_path: Path) -> None:
    """``existed=False`` and the file isn't on disk → NOOP."""
    snap = FileSnapshot(path=str(tmp_path / "never-existed.txt"), period_index=1, existed=False)
    assert store.restore(snap).outcome is RestoreOutcome.NOOP


def test_restore_delete_branch_refuses_directory(store: SnapshotStore, tmp_path: Path) -> None:
    """A directory sitting where a deleted file should be must not be rmtree'd."""
    victim = tmp_path / "sub"
    victim.mkdir()
    (victim / "precious.txt").write_text("do not delete me")

    snap = FileSnapshot(path=str(victim), period_index=1, existed=False)
    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.FAILED
    assert "refusing to remove directory" in (result.reason or "")
    # Directory and its contents are untouched.
    assert victim.is_dir()
    assert (victim / "precious.txt").read_text() == "do not delete me"


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation requires admin on Windows")
def test_restore_delete_branch_removes_symlink(store: SnapshotStore, tmp_path: Path) -> None:
    """Dangling symlinks are handled via unlink."""
    link = tmp_path / "dangling.link"
    link.symlink_to(tmp_path / "nope")  # dangling

    snap = FileSnapshot(path=str(link), period_index=1, existed=False)
    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.APPLIED
    assert not link.exists() and not link.is_symlink()


# Probe the actual capability so this safety guard also runs on Windows
# workers that can create links; do not blanket-skip an entire platform.
def test_restore_delete_removes_symlink_without_following(store: SnapshotStore, tmp_path: Path) -> None:
    """Delete-revert against a symlink removes the link, not its target."""
    target_path = tmp_path / "link_target.txt"
    target_path.write_text("keep-me", encoding="utf-8")
    symlink_path = tmp_path / "created.txt"
    symlink_or_skip(symlink_path, target_path)

    snap = FileSnapshot(path=str(symlink_path), period_index=1, existed=False)

    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.APPLIED
    assert not symlink_path.exists()
    assert not symlink_path.is_symlink()
    assert target_path.read_text(encoding="utf-8") == "keep-me"


def test_restore_delete_branch_unlink_oserror(
    store: SnapshotStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "locked.txt"
    target.write_text("x")

    def _boom(self: Path) -> None:
        raise OSError("locked")

    monkeypatch.setattr(Path, "unlink", _boom)

    snap = FileSnapshot(path=str(target), period_index=1, existed=False)
    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.FAILED
    assert "locked" in (result.reason or "")


@pytest.mark.skipif(sys.platform == "win32", reason="symlink needs admin on Windows")
def test_restore_delete_branch_symlink_unlink_oserror(
    store: SnapshotStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "target")

    def _boom(self: Path) -> None:
        raise OSError("busy")

    monkeypatch.setattr(Path, "unlink", _boom)

    snap = FileSnapshot(path=str(link), period_index=1, existed=False)
    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.FAILED
    assert "busy" in (result.reason or "")


# ── restore: write-branch (existed=True) ──────────────────────────────


def test_restore_write_branch_noop_when_hash_matches(store: SnapshotStore, tmp_path: Path) -> None:
    """Disk content already matches target → NOOP, no rewrite."""
    target = tmp_path / "same.txt"
    target.write_text("same content")

    # Snapshot the file
    snap = store.save(str(target), period_index=1)

    # Record mtime; ``restore`` must not touch the file.
    mtime_before = target.stat().st_mtime_ns

    # Restore — content is identical → NOOP
    assert store.restore(snap).outcome is RestoreOutcome.NOOP
    assert target.stat().st_mtime_ns == mtime_before
    assert target.read_text() == "same content"


def test_restore_write_branch_applied_overwrites(store: SnapshotStore, tmp_path: Path) -> None:
    """Modified file gets restored to the original content."""
    target = tmp_path / "changed.txt"
    target.write_text("original")
    snap = store.save(str(target), period_index=1)

    target.write_text("modified")  # simulate agent edit
    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.APPLIED
    assert target.read_text() == "original"


def test_restore_write_branch_creates_parent_dirs(store: SnapshotStore, tmp_path: Path) -> None:
    """If the original was deleted, parents are recreated on restore."""
    nested = tmp_path / "a" / "b" / "file.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("hello")
    snap = store.save(str(nested), period_index=1)

    # Delete the whole tree
    import shutil

    shutil.rmtree(tmp_path / "a")

    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.APPLIED
    assert nested.read_text() == "hello"


def test_restore_write_branch_refuses_directory(store: SnapshotStore, tmp_path: Path) -> None:
    target = tmp_path / "file_then_dir"
    target.write_text("original")
    snap = store.save(str(target), period_index=1)

    # Replace the file with a directory holding user data.
    target.unlink()
    target.mkdir()
    (target / "precious.txt").write_text("keep-me")

    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.FAILED
    assert "refusing to overwrite directory" in (result.reason or "")
    assert target.is_dir()
    assert (target / "precious.txt").read_text() == "keep-me"


def test_restore_write_branch_blob_missing(store: SnapshotStore, tmp_path: Path) -> None:
    """Snapshot points to a blob that's been deleted → FAILED."""
    target = tmp_path / "file.txt"
    target.write_text("original")
    snap = store.save(str(target), period_index=1)

    # Remove the blob file
    (store.mutations_dir / snap.content_hash).unlink()  # type: ignore[arg-type]

    # Also mutate the file so we actually try to write
    target.write_text("modified")

    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.FAILED
    assert "blob missing" in (result.reason or "")
    # Disk untouched — we didn't try to write empty content.
    assert target.read_text() == "modified"


def test_restore_write_branch_read_oserror_falls_through(
    store: SnapshotStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the hash-compare read raises OSError, we fall through to the write path."""
    target = tmp_path / "probe.txt"
    target.write_text("original")
    snap = store.save(str(target), period_index=1)
    target.write_text("modified")

    original_read_bytes = Path.read_bytes
    target_norm = Path(os.path.normpath(os.path.abspath(str(target))))

    # Only the HASH-CHECK read (on ``target``) should raise. The blob read must succeed.
    state = {"calls": 0}

    def _read(self: Path) -> bytes:
        if self == target_norm:
            state["calls"] += 1
            if state["calls"] == 1:
                raise OSError("transient read error")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _read)

    result = store.restore(snap)
    # Write path runs and succeeds
    assert result.outcome is RestoreOutcome.APPLIED
    assert target.read_text() == "original"


def test_restore_write_branch_write_oserror(
    store: SnapshotStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("original")
    snap = store.save(str(target), period_index=1)
    target.write_text("modified")

    def _boom(self: Path, data: bytes) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", _boom)

    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.FAILED
    assert "disk full" in (result.reason or "")


# Keep the real link-replacement assertion on every capable host, including
# Windows CI; only a proven missing capability may skip setup.
def test_restore_write_branch_replaces_symlink_with_real_file(store: SnapshotStore, tmp_path: Path) -> None:
    """If the path is a symlink at restore time, the link is removed and a real file is written."""
    real = tmp_path / "real.txt"
    real.write_text("original-real")
    snap = store.save(str(real), period_index=1)

    real.unlink()
    dest = tmp_path / "elsewhere.txt"
    dest.write_text("unrelated content")
    symlink_or_skip(real, dest)

    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.APPLIED
    assert real.is_file() and not real.is_symlink()
    assert real.read_text() == "original-real"
    # The unrelated symlink target must NOT have been overwritten
    assert dest.read_text() == "unrelated content"


# ── save_blob / save_data_as_blob / read_blob ─────────────────────────


def test_save_blob_missing_file_returns_none(store: SnapshotStore, tmp_path: Path) -> None:
    result = store.save_blob(str(tmp_path / "nope"))
    assert result.content_hash is None
    assert result.skip_reason is None


def test_save_blob_and_read_roundtrip(store: SnapshotStore, tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("payload")

    result = store.save_blob(str(f))
    assert result.content_hash is not None
    assert result.skip_reason is None
    assert store.read_blob(result.content_hash) == b"payload"


def test_save_blob_deduplicates(store: SnapshotStore, tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("same")
    f2.write_text("same")
    assert store.save_blob(str(f1)).content_hash == store.save_blob(str(f2)).content_hash


def test_save_blob_oserror_returns_none(store: SnapshotStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")

    original_read_bytes = Path.read_bytes
    f_norm = Path(os.path.normpath(os.path.abspath(str(f))))

    def _boom(self: Path) -> bytes:
        if self == f_norm:
            raise OSError("eio")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _boom)
    result = store.save_blob(str(f))
    assert result.content_hash is None
    assert result.skip_reason is SnapshotSkipReason.UNREADABLE


def test_save_data_as_blob_stores_and_dedupes(store: SnapshotStore) -> None:
    h1 = store.save_data_as_blob(b"payload").content_hash
    h2 = store.save_data_as_blob(b"payload").content_hash  # same data — same hash, single blob
    assert h1 is not None
    assert h1 == h2
    assert store.read_blob(h1) == b"payload"


def test_read_blob_missing_returns_none(store: SnapshotStore) -> None:
    assert store.read_blob("abc" * 20) is None


def test_read_content_returns_none_when_snapshot_says_absent(store: SnapshotStore) -> None:
    """``read_content`` short-circuits on ``existed=False`` without touching disk."""
    snap = FileSnapshot(path="/tmp/absent", period_index=1, existed=False)
    assert store.read_content(snap) is None


def test_read_content_returns_none_when_blob_missing(store: SnapshotStore, tmp_path: Path) -> None:
    """``existed=True`` but the blob file has been removed → None."""
    target = tmp_path / "f.txt"
    target.write_text("x")
    snap = store.save(str(target), period_index=1)
    (store.mutations_dir / snap.content_hash).unlink()  # type: ignore[arg-type]
    assert store.read_content(snap) is None


@pytest.mark.skipif(sys.platform == "win32", reason="symlink needs admin on Windows")
def test_restore_write_branch_symlink_unlink_oserror(
    store: SnapshotStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Symlink pre-unlink fails → FAILED with reason (write-branch)."""
    real = tmp_path / "real.txt"
    real.write_text("original")
    snap = store.save(str(real), period_index=1)

    # Replace the file with a symlink so the write-branch hits the
    # "unlink symlink before restore" path.
    real.unlink()
    dest = tmp_path / "dest.txt"
    dest.write_text("elsewhere")
    real.symlink_to(dest)

    # Patch unlink to raise OSError ONLY on symlinks (so fixture teardown
    # still works).
    original_unlink = Path.unlink

    def _selective_boom(self: Path, missing_ok: bool = False) -> None:
        if self.is_symlink():
            raise OSError("link locked")
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _selective_boom)

    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.FAILED
    assert "link locked" in (result.reason or "")


# ── remove_blobs / remove_all ──────────────────────────────────────────


def test_remove_blobs_skips_missing(store: SnapshotStore) -> None:
    # No exception raised for missing hashes
    store.remove_blobs({"not-a-real-hash", "another-fake"})


def test_remove_blobs_deletes_existing(store: SnapshotStore) -> None:
    h = store.save_data_as_blob(b"bye").content_hash
    assert h is not None
    assert store.read_blob(h) is not None
    store.remove_blobs({h})
    assert store.read_blob(h) is None


def test_remove_blobs_unlink_oserror_logged_not_raised(store: SnapshotStore, monkeypatch: pytest.MonkeyPatch) -> None:
    h = store.save_data_as_blob(b"data").content_hash
    assert h is not None

    def _boom(self: Path) -> None:
        raise OSError("locked")

    monkeypatch.setattr(Path, "unlink", _boom)
    # Must not raise
    store.remove_blobs({h})


def test_remove_all_removes_directory(store: SnapshotStore) -> None:
    store.save_data_as_blob(b"x")
    assert store.mutations_dir.is_dir()
    store.remove_all()
    assert not store.mutations_dir.is_dir()


def test_remove_all_when_absent_is_noop(store: SnapshotStore) -> None:
    # Never called save* → directory never created → remove_all is no-op
    assert not store.mutations_dir.exists()
    store.remove_all()
    assert not store.mutations_dir.exists()


# ════════════════════════════ GitignoreFilter ══════════════════════════


def test_gitignore_filter_basic(tmp_path: Path) -> None:
    f = GitignoreFilter(["*.log", "build/"])
    assert f.is_ignored("a.log")
    assert f.is_ignored("nested/b.log")
    assert f.is_ignored("build", is_dir=True)
    assert not f.is_ignored("build", is_dir=False)  # files named "build" not matched by "build/"
    assert not f.is_ignored("a.py")


def test_gitignore_filter_from_file_missing_returns_empty(tmp_path: Path) -> None:
    """Nonexistent .gitignore file → empty filter (no rules)."""
    f = GitignoreFilter.from_file(str(tmp_path / "nope.gitignore"))
    assert not f.is_ignored("anything")


def test_gitignore_filter_from_file(tmp_path: Path) -> None:
    gi = tmp_path / ".gitignore"
    gi.write_text("*.tmp\n")
    f = GitignoreFilter.from_file(str(gi))
    assert f.is_ignored("x.tmp")
    assert not f.is_ignored("x.py")


def test_gitignore_filter_from_text() -> None:
    f = GitignoreFilter.from_text("*.bak\n*.old\n")
    assert f.is_ignored("x.bak")
    assert f.is_ignored("x.old")


def test_gitignore_filter_normalises_os_sep_input() -> None:
    """``is_ignored`` accepts ``os.sep``-separated paths (Windows-friendly).

    Internally normalises via ``PurePath.as_posix()`` so callers can pass
    ``os.path``-derived relative paths directly.  On POSIX this is a no-op;
    on Windows it converts ``\\`` → ``/`` before pathspec matching.
    """
    f = GitignoreFilter(["build/*.log"])
    # POSIX-style input — works on all platforms.
    assert f.is_ignored("build/error.log")
    # ``os.sep``-separated input — converts on Windows, no-op on POSIX.
    assert f.is_ignored(os.path.join("build", "error.log"))


def test_gitignore_filter_handles_pure_windows_separator() -> None:
    """Explicit ``\\`` separator is normalised on Windows hosts.

    On POSIX hosts ``\\`` is a legal filename character, so a literal
    ``"build\\error.log"`` is treated as a single filename and would NOT
    match the ``build/*.log`` rule.  Test the platform-correct behaviour.
    """
    f = GitignoreFilter(["build/*.log"])
    if sys.platform == "win32":
        assert f.is_ignored("build\\error.log")
    else:
        # On POSIX: backslash is part of the filename, no normalisation.
        assert not f.is_ignored("build\\error.log")


def test_gitignore_filter_negation() -> None:
    gi = GitignoreFilter(["*.log", "!important.log"])
    assert gi.is_ignored("debug.log") is True
    assert gi.is_ignored("important.log") is False


def test_gitignore_filter_comments_and_blanks() -> None:
    gi = GitignoreFilter(["# comment", "", "*.tmp"])
    assert gi.is_ignored("foo.tmp") is True
    assert gi.is_ignored("foo.py") is False


def test_gitignore_filter_path_pattern() -> None:
    gi = GitignoreFilter(["docs/generated"])
    assert gi.is_ignored("docs/generated") is True
    assert gi.is_ignored("generated") is False  # no slash in rel path, pattern has slash


# ════════════════════════════ WorkspaceScanner ════════════════════════


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


def test_scan_finds_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / "b.py").write_text("b", encoding="utf-8")

    scanner = WorkspaceScanner(str(tmp_path))
    result = scanner.scan()
    assert len(result) == 2


def test_scan_excludes_default_dirs(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("code", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("lib", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "mod.pyc").write_bytes(b"\x00")

    scanner = WorkspaceScanner(str(tmp_path))
    result = scanner.scan()
    paths = list(result.keys())
    assert len(paths) == 1
    assert paths[0].endswith("main.py")


def test_scan_respects_max_depth(root: Path) -> None:
    """Files beyond max_depth are not discovered."""
    (root / "top.txt").write_text("x")
    deep = root / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "deep.txt").write_text("x")

    scanner = WorkspaceScanner(str(root), max_depth=1)
    result = scanner.scan()

    rel_paths = {os.path.relpath(p, str(root)) for p in result}
    assert "top.txt" in rel_paths
    assert not any("deep.txt" in p for p in rel_paths)


def test_scan_respects_gitignore(root: Path) -> None:
    (root / "keep.py").write_text("x")
    (root / "skip.log").write_text("x")

    scanner = WorkspaceScanner(str(root), gitignore=GitignoreFilter(["*.log"]))
    result = scanner.scan()
    rel = {os.path.relpath(p, str(root)) for p in result}
    assert "keep.py" in rel
    assert "skip.log" not in rel


def test_scan_skips_filename_excludes(root: Path) -> None:
    """Filenames in the exclude set (e.g. .DS_Store) are skipped."""
    (root / "real.txt").write_text("x")
    (root / ".DS_Store").write_text("x")
    scanner = WorkspaceScanner(str(root))
    rel = {os.path.relpath(p, str(root)) for p in scanner.scan()}
    assert "real.txt" in rel
    assert ".DS_Store" not in rel


@pytest.mark.skipif(sys.platform == "win32", reason="symlink needs admin on Windows")
def test_scan_skips_non_regular_files(root: Path) -> None:
    real = root / "real.txt"
    real.write_text("x")
    link = root / "link.txt"
    link.symlink_to(real)

    scanner = WorkspaceScanner(str(root))
    rel = {os.path.relpath(p, str(root)) for p in scanner.scan()}
    assert "real.txt" in rel
    # Symlinks are not regular files
    assert "link.txt" not in rel


def test_scan_since_ns_filters_old_files(root: Path) -> None:
    """``since_ns`` excludes files with older mtime_ns."""
    old = root / "old.txt"
    old.write_text("x")
    new = root / "new.txt"
    new.write_text("x")

    # Bump the new file's mtime far into the future
    future_ns = int(2e18)
    os.utime(new, ns=(future_ns, future_ns))

    scanner = WorkspaceScanner(str(root))
    result = scanner.scan(since_ns=future_ns - 1)
    rel = {os.path.relpath(p, str(root)) for p in result}
    assert "new.txt" in rel
    assert "old.txt" not in rel


def test_scan_lstat_oserror_invalidates_scan(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable entry must not become an apparently absent entry."""
    good = root / "good.txt"
    good.write_text("x")
    bad = root / "bad.txt"
    bad.write_text("x")

    real_lstat = os.lstat

    def _lstat(path: str):
        if path.endswith("bad.txt"):
            raise OSError("cannot stat")
        return real_lstat(path)

    monkeypatch.setattr(os, "lstat", _lstat)

    scanner = WorkspaceScanner(str(root))
    with pytest.raises(OSError, match="cannot stat"):
        scanner.scan()


def test_scan_max_files_cap_warns_and_returns_partial(root: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Hitting ``max_files`` stops the scan, logs a warning, returns partial results."""
    for i in range(10):
        (root / f"f{i}.txt").write_text("x")

    scanner = WorkspaceScanner(str(root), max_files=3)
    with caplog.at_level("WARNING", logger="chrys.service.mutations.scanner"):
        result = scanner.scan()
    # Stopped early — got exactly the cap
    assert len(result) == 3
    assert "max_files cap" in caplog.text


def test_scan_paths_excludes_filename_in_shallow_walk(root: Path) -> None:
    """Excluded filenames inside a directory scan are skipped."""
    d = root / "dir"
    d.mkdir()
    (d / "keep.py").write_text("x")
    (d / ".DS_Store").write_text("x")  # in default excludes

    scanner = WorkspaceScanner(str(root))
    result = scanner.scan_paths([str(d)], shallow_depth=1)
    names = {os.path.basename(p) for p in result}
    assert "keep.py" in names
    assert ".DS_Store" not in names


def test_is_excluded_dir_non_matching_returns_false(root: Path) -> None:
    """Directory not in excludes, not matching any wildcard, no gitignore → False."""
    scanner = WorkspaceScanner(str(root), excludes=frozenset({".git"}))
    assert scanner._is_excluded_dir("src", "src") is False


def test_is_excluded_dir_exact_name_match(root: Path) -> None:
    scanner = WorkspaceScanner(str(root), excludes=frozenset({".git"}))
    assert scanner._is_excluded_dir(".git", ".git") is True


def test_is_excluded_dir_via_gitignore(root: Path) -> None:
    scanner = WorkspaceScanner(
        str(root),
        excludes=frozenset(),
        gitignore=GitignoreFilter(["build/"]),
    )
    assert scanner._is_excluded_dir("build", "build") is True


def test_scan_excludes_wildcard_egg_info_pattern(root: Path) -> None:
    """Directories matching a ``*.egg-info`` pattern are excluded."""
    (root / "mypkg.egg-info").mkdir()
    (root / "mypkg.egg-info" / "PKG-INFO").write_text("x")
    (root / "src.py").write_text("x")

    scanner = WorkspaceScanner(str(root))
    rel = {os.path.relpath(p, str(root)) for p in scanner.scan()}
    assert "src.py" in rel
    assert not any("egg-info" in p for p in rel)


# ── scan_paths ─────────────────────────────────────────────────────────


def test_scan_paths_stats_single_file(root: Path) -> None:
    f = root / "a.txt"
    f.write_text("x")
    (root / "unrequested.txt").write_text("o")  # never asked for → must stay out of the result
    scanner = WorkspaceScanner(str(root))
    result = scanner.scan_paths([str(f)])
    assert len(result) == 1
    assert str(f) in result or os.path.normpath(str(f)) in result


def test_scan_paths_skips_missing(root: Path) -> None:
    scanner = WorkspaceScanner(str(root))
    result = scanner.scan_paths([str(root / "nope.txt")])
    assert result == {}


def test_scan_paths_walks_directory_shallow(root: Path) -> None:
    """Directory path → shallow walk at ``shallow_depth`` levels."""
    a = root / "dir"
    a.mkdir()
    (a / "top.txt").write_text("x")
    deep = a / "nested"
    deep.mkdir()
    (deep / "deep.txt").write_text("x")

    scanner = WorkspaceScanner(str(root))
    # shallow_depth=1 → only top.txt, not deep.txt
    result = scanner.scan_paths([str(a)], shallow_depth=1)
    names = {os.path.basename(p) for p in result}
    assert "top.txt" in names
    assert "deep.txt" not in names


def test_scan_paths_respects_excludes_in_dir_walk(root: Path) -> None:
    d = root / "pkg"
    d.mkdir()
    (d / "keep.py").write_text("x")
    pycache = d / "__pycache__"
    pycache.mkdir()
    (pycache / "junk.pyc").write_text("x")

    scanner = WorkspaceScanner(str(root))
    result = scanner.scan_paths([str(d)], shallow_depth=2)
    names = {os.path.basename(p) for p in result}
    assert "keep.py" in names
    assert "junk.pyc" not in names


def test_scan_paths_relative_paths_resolved_against_root(root: Path) -> None:
    (root / "x.txt").write_text("y")
    scanner = WorkspaceScanner(str(root))
    result = scanner.scan_paths(["x.txt"])
    assert any(p.endswith("x.txt") for p in result)


# ── _stat_file helper ─────────────────────────────────────────────────


def test_stat_file_helper_regular(root: Path) -> None:
    f = root / "x.txt"
    f.write_text("x")
    fs = WorkspaceScanner._stat_file(str(f))
    assert fs is not None
    assert fs.size == 1


@pytest.mark.skipif(sys.platform == "win32", reason="symlink needs admin on Windows")
def test_stat_file_helper_skips_symlink(root: Path) -> None:
    real = root / "real.txt"
    real.write_text("x")
    link = root / "link"
    link.symlink_to(real)
    assert WorkspaceScanner._stat_file(str(link)) is None


def test_stat_file_helper_oserror_returns_none() -> None:
    assert WorkspaceScanner._stat_file("/definitely/not/real/path") is None


# ── diff ──────────────────────────────────────────────────────────────


def test_diff_detects_all_three_ops() -> None:
    from chrys.service.mutations.types import MutationOp

    before = {"a": FileStat(100, 10), "b": FileStat(100, 10)}
    after = {"a": FileStat(100, 10), "b": FileStat(200, 20), "c": FileStat(100, 5)}
    changes = dict(WorkspaceScanner.diff(before, after))

    assert "a" not in changes  # unchanged
    assert changes["b"] is MutationOp.MODIFY
    assert changes["c"] is MutationOp.CREATE


def test_diff_detects_delete() -> None:
    from chrys.service.mutations.types import MutationOp

    before = {"gone.txt": FileStat(1, 1)}
    after: dict[str, FileStat] = {}
    changes = dict(WorkspaceScanner.diff(before, after))
    assert changes["gone.txt"] is MutationOp.DELETE


# ── module-level sanity ───────────────────────────────────────────────


def test_default_excludes_contains_common_entries() -> None:
    for name in {".git", "node_modules", "__pycache__", "target", ".venv"}:
        assert name in DEFAULT_EXCLUDES
