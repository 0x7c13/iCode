# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for BackendEventHandler live-diff accumulation from tool and shell mutations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chrys.app.tui.screens.main.diff_controller import LiveDiffTracker
from chrys.app.tui.screens.main.event_handlers import (
    BackendEventHandler,
)
from chrys.app.tui.screens.main.live_diff import LiveFileMutation, mutation_op_for_live_op
from chrys.service.mutations.types import FileMutationTextSnapshot, MutationOp, MutationSource
from tests.support.tui_helpers import (
    make_backend_handler,
    make_live_mutation,
)


def _make_handler() -> tuple[BackendEventHandler, dict[str, LiveFileMutation]]:
    """Create a BackendEventHandler with a minimal mock screen.

    Returns ``(handler, live_file_mutations_dict)`` — the dict is the
    mapping the screen's live-diff tracker accumulates into.
    """
    live: dict[str, LiveFileMutation] = {}
    screen = SimpleNamespace(_live_diff=LiveDiffTracker(file_mutations=live))
    handler = make_backend_handler(screen)
    return handler, live


def _shell_snapshot(
    before_text: str,
    after_text: str,
    operation: str,
    *,
    bytes_changed: bool | None = None,
    before_hash: str | None = None,
    after_hash: str | None = None,
    source: str = "",
) -> FileMutationTextSnapshot:
    return FileMutationTextSnapshot(
        before_text=before_text,
        after_text=after_text,
        operation=operation,
        bytes_changed=before_text != after_text if bytes_changed is None else bytes_changed,
        source=source,
        before_hash=before_hash,
        after_hash=after_hash,
    )


# ──────────── _accumulate_live_mutation ────────────────────────────────


def test_accumulate_new_file_create() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "", "content")
    assert live["/tmp/f.py"] == make_live_mutation("", "content", "create")


def test_accumulate_new_file_modify() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "old", "new")
    assert live["/tmp/f.py"] == make_live_mutation("old", "new", "modify")


def test_accumulate_preserves_first_before() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "original", "v1")
    handler._accumulate_live_mutation("/tmp/f.py", "v1", "v2")
    assert live["/tmp/f.py"] == make_live_mutation("original", "v2", "modify")


def test_accumulate_removes_live_mutation_that_reverts_to_original() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "original", "v1")
    handler._accumulate_live_mutation("/tmp/f.py", "v1", "original")
    assert "/tmp/f.py" not in live


def test_accumulate_removes_live_create_then_delete() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "", "generated", "create")
    handler._accumulate_live_mutation("/tmp/f.py", "generated", "", "delete")
    assert "/tmp/f.py" not in live


def test_accumulate_explicit_modify_empty_file_noop_is_ignored() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "", "", "modify")
    assert "/tmp/f.py" not in live


def test_accumulate_explicit_modify_empty_file_roundtrip_is_removed() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "", "content", "modify")
    handler._accumulate_live_mutation("/tmp/f.py", "content", "", "modify")
    assert "/tmp/f.py" not in live


def test_accumulate_with_explicit_op() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "old", "new", "delete")
    assert live["/tmp/f.py"] == make_live_mutation("old", "new", "delete")


def test_accumulate_explicit_op_ignored_on_update() -> None:
    """Once a path exists, op_str is always preserved from the first entry."""
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "", "v1", "create")
    handler._accumulate_live_mutation("/tmp/f.py", "v1", "v2", "modify")
    assert live["/tmp/f.py"] == make_live_mutation("", "v2", "create")


@pytest.mark.parametrize(
    ("path", "op", "bytes_changed", "before_hash", "after_hash"),
    [
        pytest.param("/tmp/f.py", "modify", True, "hash-with-bom", "hash-without-bom", id="metadata-only-modify"),
        pytest.param("/tmp/dst.py", "move", False, "same-hash", "same-hash", id="move-with-unchanged-bytes"),
    ],
)
def test_accumulate_keeps_live_mutation_whose_text_is_unchanged(
    path: str,
    op: str,
    bytes_changed: bool,
    before_hash: str,
    after_hash: str,
) -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation(
        path,
        "same text\n",
        "same text\n",
        op,
        bytes_changed=bytes_changed,
        before_hash=before_hash,
        after_hash=after_hash,
    )
    assert live[path] == make_live_mutation(
        "same text\n",
        "same text\n",
        op,
        bytes_changed=bytes_changed,
        before_hash=before_hash,
        after_hash=after_hash,
    )


def test_accumulate_hashes_remove_live_metadata_roundtrip() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation(
        "/tmp/f.py",
        "same text\n",
        "same text\n",
        "modify",
        bytes_changed=True,
        before_hash="h1",
        after_hash="h2",
    )
    handler._accumulate_live_mutation(
        "/tmp/f.py",
        "same text\n",
        "same text\n",
        "modify",
        bytes_changed=True,
        before_hash="h2",
        after_hash="h1",
    )
    assert "/tmp/f.py" not in live


# ──────────── _accumulate_shell_snapshots ─────────────────────────────


def test_accumulate_shell_snapshots_empty_metadata() -> None:
    handler, live = _make_handler()
    handler._accumulate_shell_snapshots({})
    assert live == {}


def test_accumulate_shell_snapshots_none_value() -> None:
    handler, live = _make_handler()
    handler._accumulate_shell_snapshots({"shell_file_snapshots": None})
    assert live == {}


def test_accumulate_shell_snapshots_basic() -> None:
    handler, live = _make_handler()
    handler._accumulate_shell_snapshots(
        {
            "shell_file_snapshots": {
                "/tmp/a.py": _shell_snapshot("old-a", "new-a", "modify"),
                "/tmp/b.py": _shell_snapshot("", "new-b", "create"),
            }
        }
    )
    assert live["/tmp/a.py"] == make_live_mutation("old-a", "new-a", "modify")
    assert live["/tmp/b.py"] == make_live_mutation("", "new-b", "create")


def test_accumulate_shell_snapshots_merges_with_existing() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/a.py", "original", "v1")
    handler._accumulate_shell_snapshots(
        {
            "shell_file_snapshots": {
                "/tmp/a.py": _shell_snapshot("v1", "v2", "modify"),
            }
        }
    )
    # Should keep "original" as before_text
    assert live["/tmp/a.py"] == make_live_mutation("original", "v2", "modify")


def test_accumulate_shell_snapshots_preserves_implicit_source() -> None:
    handler, live = _make_handler()
    handler._accumulate_shell_snapshots(
        {
            "shell_file_snapshots": {
                "/tmp/a.py": _shell_snapshot(
                    "original",
                    "v1",
                    "modify",
                    source=MutationSource.IMPLICIT.value,
                ),
            }
        }
    )
    handler._accumulate_live_mutation("/tmp/a.py", "v1", "v2")

    assert live["/tmp/a.py"] == make_live_mutation(
        "original",
        "v2",
        "modify",
        source=MutationSource.IMPLICIT.value,
    )


def test_accumulate_shell_snapshots_keeps_metadata_only_change() -> None:
    handler, live = _make_handler()
    handler._accumulate_shell_snapshots(
        {
            "shell_file_snapshots": {
                "/tmp/a.py": _shell_snapshot(
                    "same\n",
                    "same\n",
                    "modify",
                    bytes_changed=True,
                    before_hash="h1",
                    after_hash="h2",
                ),
            }
        }
    )
    assert live["/tmp/a.py"] == make_live_mutation(
        "same\n",
        "same\n",
        "modify",
        bytes_changed=True,
        before_hash="h1",
        after_hash="h2",
    )


# ──────────── mutation_op_for_live_op ─────────────────────────────────


def test_op_str_map_covers_all_ops() -> None:
    assert mutation_op_for_live_op("create") is MutationOp.CREATE
    assert mutation_op_for_live_op("modify") is MutationOp.MODIFY
    assert mutation_op_for_live_op("delete") is MutationOp.DELETE
    assert mutation_op_for_live_op("move") is MutationOp.MOVE
