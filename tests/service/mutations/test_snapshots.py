# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for SnapshotStore save/restore semantics, including the symlink snapshot family."""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING

import pytest

from chrys.foundation.platform import get_platform
from chrys.service.mutations import store as mutation_store
from chrys.service.mutations.store import SnapshotStore
from chrys.service.mutations.types import FileSnapshot, RestoreOutcome

if TYPE_CHECKING:
    from pathlib import Path


class TestSnapshotStore:
    """SnapshotStore blob persistence and restore."""

    def test_save_and_read_content(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")

        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), period_index=1)
        assert snap.existed is True
        assert snap.content_hash is not None
        assert snap.size == len(b"hello world")

        content = store.read_content(snap)
        assert content == b"hello world"

    def test_save_nonexistent_file(self, tmp_path: Path) -> None:
        store = SnapshotStore(tmp_path)
        snap = store.save(str(tmp_path / "nope.txt"), period_index=1)
        assert snap.existed is False
        assert snap.content_hash is None
        assert store.read_content(snap) is None

    def test_restore_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "restore_me.txt"
        f.write_text("original", encoding="utf-8")

        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), period_index=1)

        f.write_text("overwritten", encoding="utf-8")
        assert store.restore(snap).ok
        assert f.read_text(encoding="utf-8") == "original"

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_symlink_snapshot_restores_link_not_file(self, tmp_path: Path) -> None:
        """A symlink is snapshotted as itself (target text, the Git blob
        representation) and restored as a link — never materialized as a
        regular file."""
        dest = tmp_path / "dest.txt"
        dest.write_text("dest content", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(dest)

        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), period_index=1)
        assert snap.existed is True
        assert snap.is_symlink is True
        assert snap.symlink_target_is_dir is False
        assert store.read_content(snap) == os.fsencode(str(dest))
        assert FileSnapshot.from_dict(snap.to_dict()).is_symlink is True

        other = tmp_path / "other.txt"
        other.write_text("other", encoding="utf-8")
        link.unlink()
        link.symlink_to(other)
        assert store.restore(snap).ok
        assert link.is_symlink()
        assert os.readlink(link) == str(dest)
        assert dest.read_text(encoding="utf-8") == "dest content"

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_symlink_snapshot_replaces_regular_file_and_noops_on_match(self, tmp_path: Path) -> None:
        dest = tmp_path / "dest.txt"
        dest.write_text("dest content", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(dest)
        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), period_index=1)

        link.unlink()
        link.write_text("plain file", encoding="utf-8")
        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.APPLIED
        assert link.is_symlink()
        assert os.readlink(link) == str(dest)

        assert store.restore(snap).outcome is RestoreOutcome.NOOP

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_dangling_symlink_snapshot_is_an_existing_entry(self, tmp_path: Path) -> None:
        link = tmp_path / "dangling"
        link.symlink_to(tmp_path / "missing.txt")
        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), period_index=1)
        assert snap.existed is True
        assert snap.is_symlink is True

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_save_blob_hashes_symlink_target_text(self, tmp_path: Path) -> None:
        """The entry's identity is the link: its blob is the target text,
        never the destination's content."""
        dest = tmp_path / "dest.txt"
        dest.write_text("dest content", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(dest)
        store = SnapshotStore(tmp_path / "session")
        result = store.save_blob(str(link))
        assert result.content_hash == SnapshotStore.content_hash(os.fsencode(str(dest)))

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_failed_symlink_restore_preserves_existing_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused link creation (Windows without privilege, filesystem
        without symlink support) reports FAILED with whatever sits at the
        path untouched — the replacement is built at a sibling temp path
        and swapped in only once it exists."""
        dest = tmp_path / "dest.txt"
        dest.write_text("dest content", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(dest)
        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), period_index=1)

        link.unlink()
        link.write_text("survivor", encoding="utf-8")

        def _refuse(*args: object, **kwargs: object) -> None:
            raise OSError("symlink creation refused")

        monkeypatch.setattr(os, "symlink", _refuse)
        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.FAILED
        assert not link.is_symlink()
        assert link.read_text(encoding="utf-8") == "survivor"
        assert not list(tmp_path.glob(".link.*"))

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_directory_symlink_snapshot_persists_target_type(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Windows needs ``target_is_directory`` to recreate a directory
        link; the snapshot persists the kind and restore forwards it."""
        folder = tmp_path / "folder"
        folder.mkdir()
        link = tmp_path / "dirlink"
        link.symlink_to(folder, target_is_directory=True)
        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), period_index=1)
        assert snap.is_symlink is True
        assert snap.symlink_target_is_dir is True
        assert FileSnapshot.from_dict(snap.to_dict()).symlink_target_is_dir is True

        link.unlink()
        real_symlink = os.symlink
        forwarded: dict[str, bool] = {}

        def _spy(src: str, dst: str, target_is_directory: bool = False) -> None:
            forwarded["target_is_directory"] = target_is_directory
            real_symlink(src, dst)

        monkeypatch.setattr(os, "symlink", _spy)
        assert store.restore(snap).ok
        assert forwarded["target_is_directory"] is True
        assert link.is_symlink()
        assert os.readlink(link) == str(folder)

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_symlink_restore_verifies_recorded_link_kind(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The kind participates in NOOP only where links are typed:
        POSIX links are untyped, so a matching text is NOOP regardless
        of the recorded kind — rebuilding could not change anything.
        Where links are typed (simulated), a mismatched kind rebuilds,
        a matching or unprovable kind NOOPs, and rebuilding with an
        unprovable kind infers it from what the target resolves to."""
        dest = tmp_path / "dest.txt"
        dest.write_text("dest content", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(dest)
        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), period_index=1)
        assert snap.symlink_target_is_dir is False

        mismatch = dataclasses.replace(snap, symlink_target_is_dir=True)
        assert store.restore(mismatch).outcome is RestoreOutcome.NOOP

        # Simulate a typed-link platform for the middle phase only — the
        # context manager takes the patch back out at the end of the block so
        # the last phase runs against the real (POSIX, untyped) probe again.
        with pytest.MonkeyPatch.context() as typed_links:
            typed_links.setattr(mutation_store, "on_disk_link_kind", lambda path: False)
            result = store.restore(mismatch)
            assert result.outcome is RestoreOutcome.APPLIED
            assert link.is_symlink()
            assert os.readlink(link) == str(dest)
            assert store.restore(snap).outcome is RestoreOutcome.NOOP
            unknown = dataclasses.replace(snap, symlink_target_is_dir=None)
            assert store.restore(unknown).outcome is RestoreOutcome.NOOP

        folder = tmp_path / "folder"
        folder.mkdir()
        dlink = tmp_path / "dlink"
        dlink.symlink_to(folder, target_is_directory=True)
        dsnap = dataclasses.replace(store.save(str(dlink), period_index=1), symlink_target_is_dir=None)
        dlink.unlink()
        real_symlink = os.symlink
        forwarded: dict[str, bool] = {}

        def _spy(src: str, dst: str, target_is_directory: bool = False) -> None:
            forwarded["target_is_directory"] = target_is_directory
            real_symlink(src, dst)

        monkeypatch.setattr(os, "symlink", _spy)
        assert store.restore(dsnap).ok
        assert forwarded["target_is_directory"] is True

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_symlink_restore_spares_unrelated_sibling_entries(self, tmp_path: Path) -> None:
        """The swap must never claim a name it did not create: a user
        entry that happens to sit at a would-be temporary name survives
        the restore untouched."""
        dest = tmp_path / "dest.txt"
        dest.write_text("dest content", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(dest)
        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), period_index=1)

        bystander = tmp_path / ".link.chrys-link-tmp"
        bystander.write_text("user data", encoding="utf-8")
        other = tmp_path / "other.txt"
        other.write_text("other", encoding="utf-8")
        link.unlink()
        link.symlink_to(other)

        assert store.restore(snap).outcome is RestoreOutcome.APPLIED
        assert os.readlink(link) == str(dest)
        assert bystander.read_text(encoding="utf-8") == "user data"
        assert not list(tmp_path.glob(".link.*.chrys-link-tmp"))

    def test_restore_deletes_created_file(self, tmp_path: Path) -> None:
        f = tmp_path / "created_later.txt"

        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), period_index=1)  # existed=False
        assert snap.existed is False

        f.write_text("I was created", encoding="utf-8")
        assert store.restore(snap).ok
        assert not f.exists()

    # -------- Revert edge cases --------
    #
    # Contract reminder: revert is unconditional — each selected file
    # is forced to the snapshot's pre-turn state.  The user has
    # already reviewed the target state in the rollback modal's diff
    # view and consented by leaving the path checked.  See
    # ``SnapshotStore.restore`` docstring for the full rationale.

    def test_restore_recreates_missing_file(self, tmp_path: Path) -> None:
        """Case 1: snapshot says restore-content but file is missing → APPLIED.

        User explicitly kept this path checked in the modal, so we
        recreate it.  If they wanted it gone, they'd have de-selected.
        """
        f = tmp_path / "was_here.txt"
        f.write_text("original", encoding="utf-8")
        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), period_index=1)

        f.unlink()
        assert not f.exists()

        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.APPLIED
        assert f.read_text(encoding="utf-8") == "original"

    def test_restore_overwrites_divergent_content(self, tmp_path: Path) -> None:
        """Case 2: file exists with unrelated content → APPLIED, overwritten.

        Matches the agreed "what the diff view shows IS what you get"
        contract — the user's checkbox IS the consent.
        """
        f = tmp_path / "conflict.txt"
        f.write_text("original", encoding="utf-8")
        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), period_index=1)

        f.write_text("manual edit", encoding="utf-8")
        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.APPLIED
        assert f.read_text(encoding="utf-8") == "original"
