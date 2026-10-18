# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Shell content probes detect stat-invisible edits and report unreadable evidence."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from chrys.service.mutations.pipeline import finalize_mutation_tracking, prepare_mutation_tracking
from chrys.service.mutations.scanner import WorkspaceScanner
from chrys.service.mutations.store import FileProbe, SnapshotPolicy, SnapshotStore
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.types import MutationOp, SnapshotSkipReason
from tests.support.symlinks import symlink_or_skip

_FIXED_MTIME_NS = 1_700_000_000_000_000_000


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path.resolve() / "workspace"
    (root / "real").mkdir(parents=True)
    (root / "real" / "file.txt").write_bytes(b"original")
    return root


def _rewrite_at_mtime(path: Path, content: bytes, mtime_ns: int = _FIXED_MTIME_NS) -> None:
    path.write_bytes(content)
    os.utime(path, ns=(mtime_ns, mtime_ns))


@pytest.mark.parametrize("aliased", [False, True])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("changed", [False, True])
async def test_shell_content_probe_preserves_latest_content_and_earliest_backup(
    tmp_path: Path, workspace: Path, aliased: bool, resume: bool, changed: bool
) -> None:
    target = workspace / "real" / "file.txt"
    route = target
    if aliased:
        alias = workspace / "alias"
        await asyncio.to_thread(symlink_or_skip, alias, target.parent, target_is_directory=True)
        route = alias / target.name
    store = SnapshotStore(tmp_path / "session")
    tracker = MutationTracker(store)
    tracker.start_turn(1)
    context = await prepare_mutation_tracking(tracker, "edit_file", {"path": str(route)}, "edit", False, str(workspace))
    await asyncio.to_thread(_rewrite_at_mtime, route, b"edit 0")
    await finalize_mutation_tracking(tracker, context, "edit")
    if resume:
        tracker = MutationTracker.deserialize(tracker.serialize(), store)

    scanner = WorkspaceScanner(str(workspace))
    before_scan = await asyncio.to_thread(scanner.scan_paths, [str(target)])
    context = await prepare_mutation_tracking(
        tracker,
        "shell",
        {"command": f"touch {route.relative_to(workspace).as_posix()}"},
        "shell",
        True,
        str(workspace),
    )
    content = b"edit 1" if changed else b"edit 0"
    await asyncio.to_thread(_rewrite_at_mtime, route, content, before_scan[str(target)].mtime_ns)
    after_scan = await asyncio.to_thread(scanner.scan_paths, [str(target)])
    # The stat channel must have no evidence, regardless of filesystem precision.
    assert after_scan == before_scan
    result = await finalize_mutation_tracking(tracker, context, "shell")

    turn = tracker.get_turn_mutations(1)
    assert turn is not None and not turn.detection_truncated
    assert [row.tool_call_id for row in turn.mutations] == (["edit", "shell"] if changed else ["edit"])
    if changed:
        assert result.shell_snapshots is not None
        snapshot = result.shell_snapshots[str(target)]
        assert snapshot.before_text == "edit 0"
        assert snapshot.after_text == "edit 1"
        assert snapshot.operation == MutationOp.MODIFY.value
    else:
        assert result.shell_snapshots is None
    summary = tracker.get_turn_file_summary(1)
    assert set(summary) == {str(target)}
    assert summary[str(target)].before == store.content_hash(b"original")
    assert summary[str(target)].after == store.content_hash(content)
    assert len(tracker.log.snapshots) == 1
    restored = await asyncio.to_thread(tracker.rollback_turns, {1})
    assert restored and all(row.ok for row in restored)
    assert await asyncio.to_thread(target.read_bytes) == b"original"
    if aliased:
        assert route.parent.is_symlink()


@pytest.mark.parametrize("phase", ["before", "after"])
async def test_unreadable_shell_content_probe_marks_detection_incomplete(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    target = workspace / "real" / "file.txt"
    await asyncio.to_thread(_rewrite_at_mtime, target, b"edit 0")
    store = SnapshotStore(tmp_path / "session")
    tracker = MutationTracker(store)
    tracker.start_turn(1)
    # An existing healthy backup must not hide a later probe failure.
    await asyncio.to_thread(tracker.pre_snapshot, [str(target)])
    turn = tracker.get_turn_mutations(1)
    assert turn is not None and not turn.detection_truncated
    unreadable = create_autospec(store.probe, return_value=FileProbe(True, skip_reason=SnapshotSkipReason.UNREADABLE))
    scanner = WorkspaceScanner(str(workspace))
    before_scan = await asyncio.to_thread(scanner.scan_paths, [str(target)])
    with monkeypatch.context() as patch:
        if phase == "before":
            patch.setattr(store, "probe", unreadable)
        context = await prepare_mutation_tracking(
            tracker, "shell", {"command": "touch real/file.txt"}, "shell", True, str(workspace)
        )
    assert turn.detection_truncated is (phase == "before")
    await asyncio.to_thread(_rewrite_at_mtime, target, b"edit 1", before_scan[str(target)].mtime_ns)
    assert await asyncio.to_thread(scanner.scan_paths, [str(target)]) == before_scan
    with monkeypatch.context() as patch:
        if phase == "after":
            patch.setattr(store, "probe", unreadable)
        result = await finalize_mutation_tracking(tracker, context, "shell")

    unreadable.assert_called_once_with(str(target))
    assert turn.detection_truncated
    assert turn.mutations == []
    assert result.shell_snapshots is None
    assert tracker.get_turn_file_summary(1) == {}
    restored = MutationTracker.deserialize(tracker.serialize(), store).get_turn_mutations(1)
    assert restored is not None and restored.detection_truncated
    assert await asyncio.to_thread(target.read_bytes) == b"edit 1"


@pytest.mark.parametrize(
    ("before", "after", "policy"),
    [
        (b"edit 0", b"edit 1", SnapshotPolicy(max_file_bytes=4)),
        (b"\x89PNG\r\n\x1a\n0", b"\x89PNG\r\n\x1a\n1", SnapshotPolicy()),
        (b"text data", b"\x89PNG\r\n\x1a\n1", SnapshotPolicy()),
    ],
    ids=["oversized", "binary-before", "binary-after"],
)
async def test_policy_skipped_shell_content_is_not_an_incomplete_detection(
    tmp_path: Path, workspace: Path, before: bytes, after: bytes, policy: SnapshotPolicy
) -> None:
    target = workspace / "real" / "file.txt"
    await asyncio.to_thread(_rewrite_at_mtime, target, before)
    tracker = MutationTracker(SnapshotStore(tmp_path / "session", policy=policy))
    tracker.start_turn(1)
    scanner = WorkspaceScanner(str(workspace))
    before_scan = await asyncio.to_thread(scanner.scan_paths, [str(target)])
    context = await prepare_mutation_tracking(
        tracker, "shell", {"command": "touch real/file.txt"}, "shell", True, str(workspace)
    )
    await asyncio.to_thread(_rewrite_at_mtime, target, after, before_scan[str(target)].mtime_ns)
    assert await asyncio.to_thread(scanner.scan_paths, [str(target)]) == before_scan
    result = await finalize_mutation_tracking(tracker, context, "shell")

    turn = tracker.get_turn_mutations(1)
    assert turn is not None and not turn.detection_truncated
    assert turn.mutations == []
    assert result.shell_snapshots is None
    assert tracker.get_turn_file_summary(1) == {}
