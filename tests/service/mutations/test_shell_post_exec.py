# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for shell-detected mutations discovered after execution.

Covers the ``mv src dst/`` bug where :class:`WorkspaceScanner.scan_paths`
walks a newly-created dst directory and reports the moved file under its
new path.  That path was never pre-snapshotted, so reading it from disk
at :meth:`MutationTracker.record` time would capture the post-execution
state as the "turn-start" snapshot — producing a net-zero diff that drops
out of the rollback preview.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.app.tui.screens.diff.rollback_modal import _entries_for_target
from chrys.foundation.platform import get_platform
from chrys.service.mutations.detector import ShellMutationDetector
from chrys.service.mutations.pipeline import _record_shell_observations, _ShellObservation
from chrys.service.mutations.store import SnapshotStore
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.types import MutationLog, MutationOp, MutationSource

# ---------------------------------------------------------------------------
# Tracker-level regression
# ---------------------------------------------------------------------------


class TestShellCreateInNewDirectory:
    """``mv src dst/`` where ``dst`` was created by the same command.

    Reproduces the exact shape of the session-scoped bug:

    - Turn 1 creates ``README.md`` via ``write_file``.
    - Turn 2 runs ``mkdir -p docs && mv README.md docs/``.  The shell
      heuristic pre-snapshots ``README.md`` (DELETE) and ``docs`` (MOVE
      target; non-existent at turn start) — but NOT ``docs/README.md``.
    - Post-execution the workspace scanner walks ``docs/`` and emits a
      CREATE for ``docs/README.md``.  The tracker's ``record()`` runs
      for a path with no prior snapshot.

    The fix records a synthetic ``existed=False`` snapshot for the
    turn-start state so the CREATE mutation has a ``None`` before-hash
    and survives net-zero filtering in the rollback-diff builder.
    """

    def test_post_exec_create_has_null_before_hash(self, tmp_path: Path) -> None:
        """CREATE discovered post-execution records before_hash=None."""
        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        new_file = tmp_path / "docs" / "README.md"
        new_file.parent.mkdir()
        new_file.write_text("hello world", encoding="utf-8")

        mutation = tracker.record(str(new_file), MutationOp.CREATE, MutationSource.SHELL, "call-shell")

        assert mutation is not None
        assert mutation.before_hash is None
        assert mutation.after_hash is not None
        assert mutation.before_hash != mutation.after_hash

    def test_post_exec_create_snapshot_marked_non_existent(self, tmp_path: Path) -> None:
        """The turn-start snapshot for a post-exec CREATE records existed=False."""
        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        new_file = tmp_path / "out" / "generated.txt"
        new_file.parent.mkdir()
        new_file.write_text("generated", encoding="utf-8")

        tracker.record(str(new_file), MutationOp.CREATE, MutationSource.SHELL, "call-shell")

        snap = tracker.log.snapshots.get(MutationLog.snapshot_key(str(new_file), 1))
        assert snap is not None
        assert snap.existed is False
        assert snap.content_hash is None

    def test_presnapshotted_path_still_uses_real_snapshot(self, tmp_path: Path) -> None:
        """Pre-snapshotted paths keep reading their actual turn-start content.

        The guard must not kick in when ``pre_snapshot`` already saw the
        path — the normal ``rm`` / ``mv src`` shell pipeline must still
        capture pre-execution content for diff + rollback.
        """
        f = tmp_path / "existing.txt"
        f.write_text("before-exec", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        tracker.pre_snapshot([str(f)])
        f.unlink()

        mutation = tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "call-rm")
        assert mutation is not None
        assert mutation.before_hash is not None
        assert mutation.after_hash is None

    def test_file_tool_create_unchanged_by_fix(self, tmp_path: Path) -> None:
        """The guard is scoped to SHELL source only; ``write_file`` unaffected.

        ``write_file`` / ``edit_file`` call ``record()`` BEFORE executing
        the write, so ``_ensure_snapshot()`` already captures the correct
        ``existed=False`` state from disk.  Non-regression test for that.
        """
        new_file = tmp_path / "fresh.py"

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        mutation = tracker.record(str(new_file), MutationOp.CREATE, MutationSource.WRITE_FILE, "call-write")
        assert mutation is not None

        snap = tracker.log.snapshots.get(MutationLog.snapshot_key(str(new_file), 1))
        assert snap is not None
        assert snap.existed is False
        assert mutation.before_hash is None


# ---------------------------------------------------------------------------
# Rollback-diff aggregation regression
# ---------------------------------------------------------------------------


def _simulate_mv_into_new_dir(tmp_path: Path) -> MutationTracker:
    """Build a tracker mirroring the session from the bug report.

    Turn 1: ``write_file`` creates ``README.md``.
    Turn 2: shell ``mkdir -p docs && mv README.md docs/``.
    Turn 3: empty.
    """
    tracker = MutationTracker(SnapshotStore(tmp_path))

    tracker.start_turn(1)
    readme = tmp_path / "README.md"
    m1 = tracker.record(str(readme), MutationOp.CREATE, MutationSource.WRITE_FILE, "call-write")
    assert m1 is not None
    readme.write_text("readme content", encoding="utf-8")
    tracker.record_after(m1)

    tracker.start_turn(2)
    # Heuristic pre-snapshots src (mv-source) + dst token (``docs`` — a
    # non-existent dir at turn start, so its snapshot records existed=False).
    # It does NOT see ``docs/README.md``: that path only materialises when
    # the post-exec scanner walks the newly-created ``docs`` dir, which is
    # exactly the case the fix targets.
    tracker.pre_snapshot([str(readme), str(tmp_path / "docs")])

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    moved = docs_dir / "README.md"
    moved.write_text("readme content", encoding="utf-8")
    readme.unlink()

    tracker.record(str(moved), MutationOp.CREATE, MutationSource.SHELL, "call-shell")
    tracker.record(str(readme), MutationOp.DELETE, MutationSource.SHELL, "call-shell")

    tracker.start_turn(3)
    return tracker


class TestRollbackDiffAggregation:
    """``_entries_for_target`` must surface both files after ``mv A dir/``."""

    def test_keep_turn_1_shows_both_files(self, tmp_path: Path) -> None:
        """Rolling back to turn 1 previews both the CREATE-to-delete and
        the DELETE-to-restore entries.

        Before the fix ``docs/README.md`` was net-zero (before==after, both
        the post-exec hash) and was filtered out, leaving only the partial
        ``README.md`` restore in the preview.
        """
        tracker = _simulate_mv_into_new_dir(tmp_path)

        entries = _entries_for_target(tracker, target_turn=1, cwd=str(tmp_path))
        # Normalise to forward slashes — ``rel_path`` uses ``os.sep`` and the
        # test runs on Windows in CI.
        rels = {Path(e.rel_path).as_posix(): e for e in entries}

        assert "README.md" in rels
        assert "docs/README.md" in rels
        # Inverted for rollback preview: CREATE→DELETE, DELETE→CREATE.
        assert rels["README.md"].operation is MutationOp.CREATE
        assert rels["docs/README.md"].operation is MutationOp.DELETE

    def test_discard_all_shows_surviving_file(self, tmp_path: Path) -> None:
        """Rolling back to session start previews the file actually on disk.

        Before the fix, both entries came out net-zero and the preview was
        empty.  ``README.md`` is legitimately net-zero at session scope
        (created turn 1, deleted turn 2) and stays filtered.
        """
        tracker = _simulate_mv_into_new_dir(tmp_path)

        entries = _entries_for_target(tracker, target_turn=0, cwd=str(tmp_path))
        rels = {Path(e.rel_path).as_posix(): e for e in entries}

        assert "docs/README.md" in rels
        assert rels["docs/README.md"].operation is MutationOp.DELETE
        assert "README.md" not in rels


# ---------------------------------------------------------------------------
# Detector enhancement — ``mv A B/`` / ``cp A B/``
# ---------------------------------------------------------------------------


class TestDetectorDirectoryDestination:
    """``mv``/``cp`` into a directory destination synthesizes the inner target.

    The heuristic otherwise stops at the parsed dst token (e.g. ``B``),
    which is fine for a ``B`` that becomes a regular file but misses
    the synthesized ``B/basename(A)`` when ``B`` is a directory.  That
    miss forces the tracker's post-execution discovery path (the
    subject of fix #1); this enhancement closes the loop so the file's
    new path is pre-snapshotted too.
    """

    def test_mv_trailing_slash_emits_synthesized_target(self, tmp_path: Path) -> None:
        """``mv a.txt docs/`` → both ``docs`` AND ``docs/a.txt`` as targets."""
        targets = ShellMutationDetector.detect_targets("mv a.txt docs/", str(tmp_path))
        paths = {p for p, _ in targets}
        assert str(tmp_path / "docs") in paths
        assert str(tmp_path / "docs" / "a.txt") in paths
        assert str(tmp_path / "a.txt") in paths  # source DELETE still emitted

    def test_mv_existing_dir_destination_emits_synthesized_target(self, tmp_path: Path) -> None:
        """``mv a.txt docs`` where ``docs/`` already exists → synthesizes ``docs/a.txt``."""
        (tmp_path / "docs").mkdir()
        targets = ShellMutationDetector.detect_targets("mv a.txt docs", str(tmp_path))
        paths = {p for p, _ in targets}
        assert str(tmp_path / "docs" / "a.txt") in paths

    def test_mv_rename_to_file_unchanged(self, tmp_path: Path) -> None:
        """``mv old.txt new.txt`` (plain rename) does NOT synthesize extra targets."""
        targets = ShellMutationDetector.detect_targets("mv old.txt new.txt", str(tmp_path))
        paths = [p for p, _ in targets]
        # Expect exactly the two existing entries: dst MOVE + src DELETE.
        assert len(paths) == 2
        assert str(tmp_path / "new.txt") in paths
        assert str(tmp_path / "old.txt") in paths

    def test_cp_trailing_slash_emits_synthesized_target(self, tmp_path: Path) -> None:
        """``cp a.txt docs/`` → both ``docs`` AND ``docs/a.txt`` as CREATE targets."""
        targets = ShellMutationDetector.detect_targets("cp a.txt docs/", str(tmp_path))
        creates = {p for p, op in targets if op is MutationOp.CREATE}
        assert str(tmp_path / "docs") in creates
        assert str(tmp_path / "docs" / "a.txt") in creates

    def test_cp_rename_to_file_unchanged(self, tmp_path: Path) -> None:
        """``cp src.txt dst.txt`` (plain copy to new filename) unchanged."""
        targets = ShellMutationDetector.detect_targets("cp src.txt dst.txt", str(tmp_path))
        paths = [p for p, _ in targets]
        assert len(paths) == 1
        assert str(tmp_path / "dst.txt") in paths


class TestDetectorPowerShellContentCmdlets:
    """PowerShell content writers should feed shell mutation pre-snapshots."""

    def test_out_file_filepath_after_pipeline_detects_target(self, tmp_path: Path) -> None:
        """The user-reported ``$content | Out-File -FilePath ...`` shape is tracked."""
        command = "$content = @'\nhello\n'@\n$content | Out-File -FilePath 'out/SKILL.md' -Encoding UTF8 -Force"

        targets = ShellMutationDetector.detect_targets(command, str(tmp_path))
        paths = {p for p, _ in targets}

        assert str(tmp_path / "out" / "SKILL.md") in paths
        assert not any(path.endswith("UTF8") for path in paths)

    def test_set_content_path_detects_target(self, tmp_path: Path) -> None:
        """``Set-Content -Path`` uses the named path parameter, not ``-Value``."""
        targets = ShellMutationDetector.detect_targets("Set-Content -Path notes.txt -Value hello", str(tmp_path))
        paths = {p for p, _ in targets}

        assert str(tmp_path / "notes.txt") in paths
        assert not any(path.endswith("hello") for path in paths)

    def test_powershell_call_operator_before_content_cmdlet(self, tmp_path: Path) -> None:
        """A leading PowerShell call operator should not hide the cmdlet name."""
        plain_targets = ShellMutationDetector.detect_targets(
            "& Set-Content -Path notes.txt -Value hello", str(tmp_path)
        )
        quoted_targets = ShellMutationDetector.detect_targets(
            "& 'Set-Content' -Path quoted.txt -Value hello",
            str(tmp_path),
        )

        assert (str(tmp_path / "notes.txt"), MutationOp.MODIFY) in plain_targets
        assert (str(tmp_path / "quoted.txt"), MutationOp.MODIFY) in quoted_targets

    def test_set_content_positional_path_detects_target(self, tmp_path: Path) -> None:
        """``Set-Content "path" $content -NoNewline`` tracks only the path."""
        targets = ShellMutationDetector.detect_targets(
            'Set-Content "notes.txt" $content -NoNewline; Write-Host done',
            str(tmp_path),
        )
        paths = {p for p, _ in targets}

        assert str(tmp_path / "notes.txt") in paths
        assert not any(path.endswith("done") for path in paths)

    def test_content_common_switch_does_not_swallow_positional_path(self, tmp_path: Path) -> None:
        """PowerShell common switches such as ``-Verbose`` do not consume the path."""
        out_targets = ShellMutationDetector.detect_targets("Out-File -Verbose out.txt", str(tmp_path))
        set_targets = ShellMutationDetector.detect_targets("Set-Content -Debug notes.txt", str(tmp_path))

        assert str(tmp_path / "out.txt") in {p for p, _ in out_targets}
        assert str(tmp_path / "notes.txt") in {p for p, _ in set_targets}

    def test_add_content_literalpath_detects_target_case_insensitively(self, tmp_path: Path) -> None:
        """PowerShell cmdlets and parameter names are case-insensitive."""
        targets = ShellMutationDetector.detect_targets("add-content -LiteralPath notes.txt -Value hello", str(tmp_path))
        paths = {p for p, _ in targets}

        assert str(tmp_path / "notes.txt") in paths

    def test_powershell_unquoted_windows_path_keeps_backslashes(self, tmp_path: Path) -> None:
        """The PowerShell pass must not let POSIX shlex eat backslash separators."""
        targets = ShellMutationDetector.detect_targets(
            r"Set-Content -Path D:\program\Awesome\SKILL.md -Value hello",
            str(tmp_path),
        )
        paths = [p for p, _ in targets]

        assert len(paths) == 1
        assert r"D:\program\Awesome\SKILL.md" in paths[0]

    def test_powershell_legacy_cmdlet_case_and_backslashes(self, tmp_path: Path) -> None:
        """Legacy PowerShell cmdlets use non-POSIX parsing and case-insensitive names."""
        targets = ShellMutationDetector.detect_targets(r"remove-item C:\temp\a.txt", str(tmp_path))
        paths = {p for p, op in targets if op is MutationOp.DELETE}

        assert any(r"C:\temp\a.txt" in path for path in paths)
        assert not any("C:tempa.txt" in path for path in paths)

    def test_powershell_move_copy_new_item_keep_backslashes(self, tmp_path: Path) -> None:
        """Move-Item/Copy-Item/New-Item do not lose Windows path separators."""
        move_targets = ShellMutationDetector.detect_targets(r"Move-Item C:\temp\a.txt D:\out\a.txt", str(tmp_path))
        copy_targets = ShellMutationDetector.detect_targets(r"Copy-Item C:\temp\a.txt D:\out\a.txt", str(tmp_path))
        new_targets = ShellMutationDetector.detect_targets(r"New-Item C:\temp\a.txt -ItemType File", str(tmp_path))

        assert any(r"D:\out\a.txt" in path and op is MutationOp.MOVE for path, op in move_targets)
        assert any(r"C:\temp\a.txt" in path and op is MutationOp.DELETE for path, op in move_targets)
        assert any(r"D:\out\a.txt" in path and op is MutationOp.CREATE for path, op in copy_targets)
        assert any(r"C:\temp\a.txt" in path and op is MutationOp.CREATE for path, op in new_targets)

    def test_powershell_move_destination_before_source_keeps_source(self, tmp_path: Path) -> None:
        """``-Destination`` must not leak into the positional source slot."""
        targets = ShellMutationDetector.detect_targets("Move-Item -Destination b.txt a.txt", str(tmp_path))

        assert (str(tmp_path / "b.txt"), MutationOp.MOVE) in targets
        assert (str(tmp_path / "a.txt"), MutationOp.DELETE) in targets
        assert (str(tmp_path / "b.txt"), MutationOp.DELETE) not in targets

    def test_powershell_named_source_positional_destination(self, tmp_path: Path) -> None:
        """``-Path src dst`` uses the first positional as destination."""
        move_targets = ShellMutationDetector.detect_targets("Move-Item -Path a.txt b.txt", str(tmp_path))
        copy_targets = ShellMutationDetector.detect_targets("Copy-Item -LiteralPath a.txt b.txt", str(tmp_path))

        assert (str(tmp_path / "b.txt"), MutationOp.MOVE) in move_targets
        assert (str(tmp_path / "a.txt"), MutationOp.DELETE) in move_targets
        assert (str(tmp_path / "b.txt"), MutationOp.CREATE) in copy_targets

    def test_powershell_include_exclude_values_are_not_paths(self, tmp_path: Path) -> None:
        """``-Include``/``-Exclude`` values filter target selection; they are not targets."""
        copy_targets = ShellMutationDetector.detect_targets("Copy-Item -Include *.log src dst", str(tmp_path))
        remove_targets = ShellMutationDetector.detect_targets("Remove-Item -Exclude *.keep target.txt", str(tmp_path))

        assert (str(tmp_path / "dst"), MutationOp.CREATE) in copy_targets
        assert not any(path.endswith("src") and op is MutationOp.CREATE for path, op in copy_targets)
        assert (str(tmp_path / "target.txt"), MutationOp.DELETE) in remove_targets
        assert not any(path.endswith("*.keep") for path, _ in remove_targets)

    def test_powershell_new_item_name_without_path(self, tmp_path: Path) -> None:
        """``New-Item -Name`` creates the named item in the command cwd."""
        targets = ShellMutationDetector.detect_targets("New-Item -Name foo.txt -ItemType File", str(tmp_path))

        assert targets == [(str(tmp_path / "foo.txt"), MutationOp.CREATE)]

    def test_empty_powershell_path_is_ignored(self, tmp_path: Path) -> None:
        """An empty PowerShell path must not collapse to the command cwd."""
        targets = ShellMutationDetector.detect_targets("Set-Content '' -Value x", str(tmp_path))

        assert targets == []


# ---------------------------------------------------------------------------
# Detector — core command verbs
# ---------------------------------------------------------------------------


class TestDetectorCommandVerbs:
    """Heuristic shell command parser tests."""

    def test_rm_detects_delete(self, tmp_path: Path) -> None:
        targets = ShellMutationDetector.detect_targets("rm foo.txt bar.txt", str(tmp_path))
        assert len(targets) == 2
        assert all(op == MutationOp.DELETE for _, op in targets)

    def test_mv_detects_move(self, tmp_path: Path) -> None:
        targets = ShellMutationDetector.detect_targets("mv old.txt new.txt", str(tmp_path))
        ops = {op for _, op in targets}
        assert MutationOp.MOVE in ops
        assert MutationOp.DELETE in ops

    def test_cp_detects_create(self, tmp_path: Path) -> None:
        targets = ShellMutationDetector.detect_targets("cp src.txt dst.txt", str(tmp_path))
        assert len(targets) == 1
        assert targets[0][1] == MutationOp.CREATE

    def test_touch_detects_create(self, tmp_path: Path) -> None:
        targets = ShellMutationDetector.detect_targets("touch newfile.py", str(tmp_path))
        assert len(targets) == 1
        assert targets[0][1] == MutationOp.CREATE

    def test_redirect_detects_modify(self, tmp_path: Path) -> None:
        targets = ShellMutationDetector.detect_targets("echo hello > output.txt", str(tmp_path))
        assert any(t[0].endswith("output.txt") for t in targets)
        assert any(op == MutationOp.MODIFY for _, op in targets)

    def test_flags_skipped(self, tmp_path: Path) -> None:
        targets = ShellMutationDetector.detect_targets("rm -rf dir/", str(tmp_path))
        # -rf should be filtered as a flag, dir/ is the target
        paths = [p for p, _ in targets]
        assert not any(p.endswith("-rf") for p in paths)

    def test_empty_command(self) -> None:
        assert ShellMutationDetector.detect_targets("") == []

    def test_env_var_prefix_skipped(self, tmp_path: Path) -> None:
        targets = ShellMutationDetector.detect_targets("FOO=bar touch file.txt", str(tmp_path))
        assert len(targets) == 1
        assert targets[0][1] == MutationOp.CREATE


# ---------------------------------------------------------------------------
# Per-command observation merge
# ---------------------------------------------------------------------------


class TestShellObservationMergeKey:
    """One command's observations merge by route, never by resolved entry."""

    @pytest.mark.skipif(get_platform().is_windows, reason="POSIX symlinks")
    def test_symlink_and_destination_are_distinct_endpoints(self, tmp_path: Path) -> None:
        """A retargeted link and its destination must both survive the merge."""
        dest = tmp_path / "dest.txt"
        dest.write_text("payload", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(dest)
        tracker = MutationTracker(SnapshotStore(tmp_path / "session"))
        tracker.start_turn(1)

        mutations = _record_shell_observations(
            tracker,
            [
                _ShellObservation(str(link), MutationOp.MODIFY),
                _ShellObservation(str(dest), MutationOp.MODIFY),
            ],
            "call-shell",
        )

        assert sorted(mutation.path for mutation in mutations) == sorted([str(dest), str(link)])

    @pytest.mark.skipif(get_platform().is_windows, reason="POSIX symlinks")
    def test_aliased_directory_routes_to_one_file_still_merge(self, tmp_path: Path) -> None:
        """Observations reaching one file through a symlinked parent stay merged."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        target = real_dir / "f.txt"
        target.write_text("x", encoding="utf-8")
        link_dir = tmp_path / "linkdir"
        link_dir.symlink_to(real_dir)
        tracker = MutationTracker(SnapshotStore(tmp_path / "session"))
        tracker.start_turn(1)

        mutations = _record_shell_observations(
            tracker,
            [
                _ShellObservation(str(link_dir / "f.txt"), MutationOp.MODIFY),
                _ShellObservation(str(target), MutationOp.MODIFY),
            ],
            "call-shell",
        )

        assert len(mutations) == 1
