# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Workspace baseline path containment, including persisted Windows device paths."""

from __future__ import annotations

import os
from ntpath import relpath as windows_relpath
from pathlib import Path, PureWindowsPath
from unittest.mock import create_autospec

import pytest

from chrys.foundation.models.workspace import Workspace
from chrys.service.mutations import workspace_changes
from chrys.service.mutations.git_state import GitHeadState, GitStatusResult, canonical_path
from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker


@pytest.mark.parametrize("scope", [(".",), ("child",)])
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (r"d:\repo\child\kept.txt", True),
        (r"d:\repo\child\deleted.txt", True),
        (r"\\.\nul", False),
        (r"\\.\CON", False),
        (r"c:\repo\child\other.txt", False),
        (r"\\server\share\child\other.txt", False),
        (r"d:\repo-other\child\other.txt", False),
        (r"d:\repo\..\outside.txt", False),
    ],
)
def test_windows_git_scope_rejects_paths_outside_repository(
    monkeypatch: pytest.MonkeyPatch, scope: tuple[str, ...], path: str, expected: bool
) -> None:
    # These are serialized Windows identities, not paths on the test host.
    # Use the real Windows lexical operations without changing the host OS.
    monkeypatch.setattr(os.path, "relpath", create_autospec(os.path.relpath, side_effect=windows_relpath))
    monkeypatch.setattr(workspace_changes, "PurePath", PureWindowsPath)

    assert workspace_changes._path_in_git_scope(r"d:\repo", path, scope) is expected


@pytest.mark.parametrize("scope", [".", "child"])
def test_restore_filters_foreign_mount_and_keeps_comparable_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    child = tmp_path / "child"
    child.mkdir()
    root = canonical_path(str(tmp_path))
    kept = os.path.join(root, "child", "kept.txt")
    outside = os.path.join(root, "..", "outside.txt")
    device = r"\\.\nul"
    state = {"exists": True, "mtime_ns": 1, "size": 2, "mode": None, "statuses": ["??"]}
    payload = {
        "version": 1,
        "turn_id": 7,
        "roots": {
            root: {
                "root": root,
                "mode": "git",
                "head": "saved-head",
                "branch": "saved-branch",
                "scope": ["."],
                "files": dict.fromkeys((kept, outside, device), state),
            }
        },
        "degraded": {},
        "pending_safety": [{"text": "Retained changes need review", "cwd": None}],
    }
    original_relpath = os.path.relpath

    def relpath(path: str, start: str = os.curdir) -> str:
        if path == device:
            # Reproduce ntpath's actual foreign-mount error on every host.
            return windows_relpath(path, r"d:\repo")
        return original_relpath(path, start)

    monkeypatch.setattr(os.path, "relpath", create_autospec(original_relpath, side_effect=relpath))
    monkeypatch.setattr(
        workspace_changes, "resolve_git_root", create_autospec(workspace_changes.resolve_git_root, return_value=root)
    )
    tracker = WorkspaceChangeTracker()
    tracker.restore(payload, Workspace.from_cwd(str(tmp_path if scope == "." else child)))

    baseline = tracker.baseline
    assert baseline is not None
    assert baseline.turn_id == 7
    assert baseline.degraded == {}
    assert baseline.roots[root].scope == (scope,)
    assert baseline.roots[root].head == "saved-head"
    assert set(baseline.roots[root].files) == {kept}
    serialized = tracker.serialize()
    assert serialized is not None
    assert set(serialized["roots"][root]["files"]) == {kept}
    assert tracker.take_pending_notice() == "Retained changes need review"


def test_git_capture_does_not_persist_paths_resolved_to_another_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = canonical_path(str(tmp_path))
    device = r"\\.\nul"
    original_lexical = workspace_changes.lexical_display_path
    original_relpath = os.path.relpath

    def lexical(path: str) -> str:
        # Model a Windows path normalizer resolving a reserved device name.
        return device if path == os.path.join(root, "nul") else original_lexical(path)

    def relpath(path: str, start: str = os.curdir) -> str:
        return windows_relpath(path, r"d:\repo") if path == device else original_relpath(path, start)

    monkeypatch.setattr(os.path, "relpath", create_autospec(original_relpath, side_effect=relpath))
    monkeypatch.setattr(
        workspace_changes, "lexical_display_path", create_autospec(original_lexical, side_effect=lexical)
    )
    monkeypatch.setattr(
        workspace_changes, "resolve_git_root", create_autospec(workspace_changes.resolve_git_root, return_value=root)
    )
    monkeypatch.setattr(
        workspace_changes,
        "read_git_head",
        create_autospec(workspace_changes.read_git_head, return_value=GitHeadState("head", "branch")),
    )
    monkeypatch.setattr(
        workspace_changes,
        "read_git_status",
        create_autospec(
            workspace_changes.read_git_status,
            return_value=GitStatusResult({"nul": ("??",), "deleted.txt": ("D ",)}),
        ),
    )
    tracker = WorkspaceChangeTracker()
    tracker.retarget_roots(Workspace.from_cwd(str(tmp_path)))

    baseline = tracker.capture_baseline(1)

    assert baseline.degraded == {}
    assert set(baseline.roots[root].files) == {os.path.join(root, "deleted.txt")}
    assert baseline.roots[root].files[os.path.join(root, "deleted.txt")].exists is False
