# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SnapshotPolicy — per-file size cap and binary exclusion for mutation backups.

Covers the policy gate itself, its enforcement at every ``SnapshotStore``
blob-write entry point, the tracker-level consequences (skip metadata on
snapshots/mutations, summary annotations, rollback-plan exclusion), the
serialization round-trip, and the ``Settings`` env plumbing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.foundation.config.settings import Settings
from chrys.service.mutations.store import (
    DEFAULT_MAX_SNAPSHOT_FILE_BYTES,
    SnapshotPolicy,
    SnapshotStore,
)
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.types import (
    FileMutation,
    FileSnapshot,
    MutationOp,
    MutationSource,
    RestoreOutcome,
    SnapshotSkipReason,
    parse_skip_reason,
)

# PNG signature followed by NUL padding — trips both the magic-byte and
# NUL-byte binary heuristics regardless of sample size.
_BINARY_DATA = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

_CAP = 1024  # small size cap so tests don't need real 50MB files


@pytest.fixture
def store(tmp_path: Path) -> SnapshotStore:
    return SnapshotStore(tmp_path, policy=SnapshotPolicy(max_file_bytes=_CAP))


def _blob_count(store: SnapshotStore) -> int:
    if not store.mutations_dir.is_dir():
        return 0
    return sum(1 for p in store.mutations_dir.iterdir() if p.is_file())


# ════════════════════════════ SnapshotPolicy ════════════════════════════


def test_default_policy_values() -> None:
    policy = SnapshotPolicy()
    assert policy.max_file_bytes == DEFAULT_MAX_SNAPSHOT_FILE_BYTES == 50 * 1024 * 1024
    assert policy.skip_binary is True


def test_classify_size_boundary() -> None:
    policy = SnapshotPolicy(max_file_bytes=_CAP)
    assert policy.classify_size(_CAP) is None  # at the cap — allowed
    assert policy.classify_size(_CAP + 1) is SnapshotSkipReason.TOO_LARGE


def test_classify_size_cap_disabled() -> None:
    policy = SnapshotPolicy(max_file_bytes=0)
    assert policy.classify_size(10**12) is None


def test_classify_data_binary() -> None:
    policy = SnapshotPolicy(max_file_bytes=_CAP)
    assert policy.classify_data(_BINARY_DATA) is SnapshotSkipReason.BINARY
    assert policy.classify_data(b"plain text\n") is None
    assert policy.classify_data(b"") is None


def test_classify_data_binary_opt_out() -> None:
    policy = SnapshotPolicy(max_file_bytes=_CAP, skip_binary=False)
    assert policy.classify_data(_BINARY_DATA) is None


def test_classify_data_size_wins_over_binary() -> None:
    policy = SnapshotPolicy(max_file_bytes=_CAP)
    oversized_binary = _BINARY_DATA * (_CAP // len(_BINARY_DATA) + 1)
    assert policy.classify_data(oversized_binary) is SnapshotSkipReason.TOO_LARGE


def test_from_settings() -> None:
    settings = Settings.from_env(mutation_snapshot_max_file_mb=2, mutation_snapshot_skip_binary=False)
    policy = SnapshotPolicy.from_settings(settings)
    assert policy.max_file_bytes == 2 * 1024 * 1024
    assert policy.skip_binary is False


def test_settings_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_MUTATION_SNAPSHOT_MAX_FILE_MB", "7")
    monkeypatch.setenv("CHRYS_MUTATION_SNAPSHOT_SKIP_BINARY", "0")
    settings = Settings.from_env()
    assert settings.mutation_snapshot_max_file_mb == 7
    assert settings.mutation_snapshot_skip_binary is False


def test_settings_env_defaults_and_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHRYS_MUTATION_SNAPSHOT_MAX_FILE_MB", raising=False)
    monkeypatch.delenv("CHRYS_MUTATION_SNAPSHOT_SKIP_BINARY", raising=False)
    assert Settings.from_env().mutation_snapshot_max_file_mb == 50
    assert Settings.from_env().mutation_snapshot_skip_binary is True

    monkeypatch.setenv("CHRYS_MUTATION_SNAPSHOT_MAX_FILE_MB", "garbage")
    assert Settings.from_env().mutation_snapshot_max_file_mb == 50

    monkeypatch.setenv("CHRYS_MUTATION_SNAPSHOT_MAX_FILE_MB", "-3")
    assert Settings.from_env().mutation_snapshot_max_file_mb == 0  # no cap


# ════════════════════════════ SnapshotStore gating ════════════════════════════


def test_save_skips_oversized_file(store: SnapshotStore, tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (_CAP + 1))

    snap = store.save(str(f), period_index=1)
    assert snap.existed is True
    assert snap.content_hash is None
    assert snap.skip_reason is SnapshotSkipReason.TOO_LARGE
    assert snap.size == _CAP + 1
    assert snap.restorable is False
    assert _blob_count(store) == 0


def test_save_skips_binary_file(store: SnapshotStore, tmp_path: Path) -> None:
    f = tmp_path / "img.png"
    f.write_bytes(_BINARY_DATA)

    snap = store.save(str(f), period_index=1)
    assert snap.existed is True
    assert snap.content_hash is None
    assert snap.skip_reason is SnapshotSkipReason.BINARY
    assert snap.restorable is False
    assert _blob_count(store) == 0


def test_save_normal_file_unaffected(store: SnapshotStore, tmp_path: Path) -> None:
    f = tmp_path / "ok.txt"
    f.write_text("hello", encoding="utf-8")

    snap = store.save(str(f), period_index=1)
    assert snap.content_hash is not None
    assert snap.skip_reason is None
    assert snap.restorable is True
    assert _blob_count(store) == 1


def test_save_binary_allowed_when_opted_in(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path, policy=SnapshotPolicy(max_file_bytes=_CAP, skip_binary=False))
    f = tmp_path / "img.png"
    f.write_bytes(_BINARY_DATA)

    snap = store.save(str(f), period_index=1)
    assert snap.content_hash is not None
    assert snap.skip_reason is None


def test_save_blob_skips_oversized_and_binary(store: SnapshotStore, tmp_path: Path) -> None:
    big = tmp_path / "big.bin"
    big.write_bytes(b"y" * (_CAP + 1))
    png = tmp_path / "img.png"
    png.write_bytes(_BINARY_DATA)

    big_result = store.save_blob(str(big))
    assert big_result.content_hash is None
    assert big_result.skip_reason is SnapshotSkipReason.TOO_LARGE

    png_result = store.save_blob(str(png))
    assert png_result.content_hash is None
    assert png_result.skip_reason is SnapshotSkipReason.BINARY
    assert _blob_count(store) == 0


def test_save_data_as_blob_skips(store: SnapshotStore) -> None:
    big = store.save_data_as_blob(b"z" * (_CAP + 1))
    assert big.content_hash is None
    assert big.skip_reason is SnapshotSkipReason.TOO_LARGE

    binary = store.save_data_as_blob(_BINARY_DATA)
    assert binary.content_hash is None
    assert binary.skip_reason is SnapshotSkipReason.BINARY

    text = store.save_data_as_blob(b"fine")
    assert text.content_hash is not None
    assert text.skip_reason is None
    assert _blob_count(store) == 1


def test_restore_skipped_snapshot_fails_with_reason(store: SnapshotStore, tmp_path: Path) -> None:
    """Defense-in-depth: plans exclude skipped snapshots, but restore must not crash."""
    snap = FileSnapshot(
        path=str(tmp_path / "big.txt"),
        period_index=1,
        existed=True,
        skip_reason=SnapshotSkipReason.TOO_LARGE,
    )
    result = store.restore(snap)
    assert result.outcome is RestoreOutcome.FAILED
    assert "not stored" in result.reason
    assert "too_large" in result.reason


# ════════════════════════════ MutationTracker integration ════════════════════════════


def test_file_tool_mutation_on_binary_file(store: SnapshotStore, tmp_path: Path) -> None:
    """write_file/edit_file flow: record() before, record_after() after."""
    f = tmp_path / "img.png"
    f.write_bytes(_BINARY_DATA)

    tracker = MutationTracker(store)
    tracker.start_turn(1)
    m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call-1")
    assert m is not None
    assert m.before_hash is None
    assert m.before_skip is SnapshotSkipReason.BINARY

    f.write_bytes(_BINARY_DATA + b"\x00tail")
    tracker.record_after(m)
    assert m.after_hash is None
    assert m.after_skip is SnapshotSkipReason.BINARY
    assert _blob_count(store) == 0

    # Turn summary marks content unavailable; rollback plan excludes the file.
    summary = tracker.get_turn_file_summary(1)
    assert summary[m.path].content_unavailable is True
    assert tracker.get_rollback_plan(1).entries == []
    assert tracker.get_session_file_summary() == {}  # fully-skipped MODIFY nets out


def test_second_mutation_inherits_last_known_skip(store: SnapshotStore, tmp_path: Path) -> None:
    f = tmp_path / "img.png"
    f.write_bytes(_BINARY_DATA)

    tracker = MutationTracker(store)
    tracker.start_turn(1)
    m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call-1")
    assert m1 is not None
    tracker.record_after(m1)

    m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call-2")
    assert m2 is not None
    assert m2.before_hash is None
    assert m2.before_skip is SnapshotSkipReason.BINARY


def test_shell_created_oversized_file_still_rolls_back_via_delete(store: SnapshotStore, tmp_path: Path) -> None:
    """CREATE needs no before-content: rollback deletes the file."""
    f = tmp_path / "artifact.bin"
    f.write_bytes(b"b" * (_CAP + 1))

    tracker = MutationTracker(store)
    tracker.start_turn(1)
    m = tracker.record(str(f), MutationOp.CREATE, MutationSource.SHELL, "call-1")
    assert m is not None
    assert m.after_hash is None
    assert m.after_skip is SnapshotSkipReason.TOO_LARGE
    assert _blob_count(store) == 0

    plan = tracker.get_rollback_plan(1).entries
    assert len(plan) == 1
    assert plan[0][1].existed is False  # restore = delete

    results = tracker.rollback(1)
    assert all(r.ok for r in results)
    assert not f.exists()


def test_shell_modified_oversized_file_excluded_from_rollback(store: SnapshotStore, tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_bytes(b"a" * (_CAP + 1))

    tracker = MutationTracker(store)
    tracker.start_turn(1)
    tracker.pre_snapshot([str(f)])
    f.write_bytes(b"c" * (_CAP + 2))
    m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.SHELL, "call-1")
    assert m is not None
    assert m.before_skip is SnapshotSkipReason.TOO_LARGE
    assert m.after_skip is SnapshotSkipReason.TOO_LARGE

    assert tracker.get_rollback_plan(1).entries == []
    results = tracker.rollback(1)
    assert results == []
    assert f.read_bytes() == b"c" * (_CAP + 2)  # untouched


def test_record_calibrated_oversized_before_data(store: SnapshotStore, tmp_path: Path) -> None:
    f = tmp_path / "generated.txt"
    f.write_text("small now", encoding="utf-8")

    tracker = MutationTracker(store)
    tracker.start_turn(1)
    m = tracker.record_calibrated(str(f), MutationOp.MODIFY, before_data=b"h" * (_CAP + 1))
    assert m is not None
    assert m.before_hash is None
    assert m.before_skip is SnapshotSkipReason.TOO_LARGE
    assert m.after_hash is not None  # current content is small text — backed up
    assert m.after_skip is None

    snap = tracker.get_snapshot(str(f), 1)
    assert snap is not None
    assert snap.existed is True
    assert snap.skip_reason is SnapshotSkipReason.TOO_LARGE
    assert snap.restorable is False

    # Mixed before-skip/after-saved: summary flags content unavailable.
    summary = tracker.get_session_file_summary()
    assert summary[m.path].content_unavailable is True


# ═══════════════════ Move safety (rollback atomicity) ═══════════════════


def test_binary_move_with_old_path_excludes_destination(store: SnapshotStore, tmp_path: Path) -> None:
    """``mv`` of a binary file (explicit ``old_path`` pairing): the
    source's pre-move content was never backed up, so deleting the
    destination would destroy the only copy — the plan must leave BOTH
    paths untouched.
    """
    src = tmp_path / "src.bin"
    dst = tmp_path / "dst.bin"
    src.write_bytes(_BINARY_DATA)

    tracker = MutationTracker(store)
    tracker.start_turn(1)
    tracker.pre_snapshot([str(src), str(dst)])  # detector targets, pre-exec
    src.rename(dst)
    m = tracker.record(str(dst), MutationOp.MOVE, MutationSource.SHELL, "call-mv", old_path=str(src))
    assert m is not None

    assert tracker.get_rollback_plan(1).entries == []
    assert tracker.rollback(1) == []
    assert dst.read_bytes() == _BINARY_DATA  # moved content preserved
    assert not src.exists()


def test_scanner_decomposed_binary_move_keeps_destination(store: SnapshotStore, tmp_path: Path) -> None:
    """Production shell flow: ``mv`` is observed as independent
    CREATE(dst) + DELETE(src) with no pairing info.  The conservative
    escape rule (an unrecoverable-origin file was vacated in-window)
    must still keep the skip-marked destination out of the plan.
    """
    src = tmp_path / "src.bin"
    dst = tmp_path / "dst.bin"
    src.write_bytes(_BINARY_DATA)

    tracker = MutationTracker(store)
    tracker.start_turn(1)
    tracker.pre_snapshot([str(dst), str(src)])
    src.rename(dst)
    assert tracker.record(str(dst), MutationOp.CREATE, MutationSource.SHELL, "call-mv") is not None
    assert tracker.record(str(src), MutationOp.DELETE, MutationSource.SHELL, "call-mv") is not None

    assert tracker.get_rollback_plan(1).entries == []
    tracker.rollback(1)
    assert dst.read_bytes() == _BINARY_DATA


def test_chained_binary_move_poisons_transitively(store: SnapshotStore, tmp_path: Path) -> None:
    """``mv a b && mv b c``: the unrecoverable origin propagates through
    the chain — c holds the only copy and must survive rollback."""
    a, b, c = tmp_path / "a.bin", tmp_path / "b.bin", tmp_path / "c.bin"
    a.write_bytes(_BINARY_DATA)

    tracker = MutationTracker(store)
    tracker.start_turn(1)
    tracker.pre_snapshot([str(a), str(b)])
    a.rename(b)
    tracker.record(str(b), MutationOp.MOVE, MutationSource.SHELL, "call-1", old_path=str(a))
    tracker.pre_snapshot([str(c)])
    b.rename(c)
    tracker.record(str(c), MutationOp.MOVE, MutationSource.SHELL, "call-2", old_path=str(b))

    assert tracker.get_rollback_plan(1).entries == []
    tracker.rollback(1)
    assert c.read_bytes() == _BINARY_DATA


def test_text_move_still_rolls_back_both_sides(store: SnapshotStore, tmp_path: Path) -> None:
    """Restorable source: move rollback is unaffected by the safety
    filters — source restored from blob, destination deleted."""
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("hello", encoding="utf-8")

    tracker = MutationTracker(store)
    tracker.start_turn(1)
    tracker.pre_snapshot([str(src), str(dst)])
    src.rename(dst)
    m = tracker.record(str(dst), MutationOp.MOVE, MutationSource.SHELL, "call-mv", old_path=str(src))
    assert m is not None

    plan_paths = tracker.get_rollback_plan(1).paths
    assert plan_paths == {str(src), str(dst)}
    results = tracker.rollback(1)
    assert all(r.ok for r in results)
    assert src.read_text(encoding="utf-8") == "hello"
    assert not dst.exists()


def test_unrelated_skipped_create_survives_alongside_escape(store: SnapshotStore, tmp_path: Path) -> None:
    """When an unrecoverable origin escapes in-window, ALL skip-marked
    plan entries are conservatively preserved — even an unrelated
    oversized create (it might be the escaped content; the tracker
    cannot tell without content hashes)."""
    big = tmp_path / "big.bin"
    fresh = tmp_path / "fresh.bin"
    big.write_bytes(_BINARY_DATA)

    tracker = MutationTracker(store)
    tracker.start_turn(1)
    tracker.pre_snapshot([str(big)])
    big.unlink()
    assert tracker.record(str(big), MutationOp.DELETE, MutationSource.SHELL, "call-rm") is not None
    fresh.write_bytes(b"f" * (_CAP + 1))
    assert tracker.record(str(fresh), MutationOp.CREATE, MutationSource.SHELL, "call-gen") is not None

    assert tracker.get_rollback_plan(1).entries == []
    tracker.rollback(1)
    assert fresh.exists()  # conservatively left in place


# ═══════════════════ Diff surfaces (changed-detection) ═══════════════════


@pytest.mark.asyncio
async def test_pipeline_marks_skipped_edit_as_changed(store: SnapshotStore, tmp_path: Path) -> None:
    """An edit whose before AND after backups were both withheld must
    still report bytes-changed (``None == None`` hash equality proves
    nothing) and carry skip markers — otherwise live /diff treats a
    real oversized/binary edit as a no-op and drops it.
    """
    import chrys.service.agent_middleware  # noqa: F401  — break the pipeline<->middleware import cycle
    from chrys.service.mutations.pipeline import finalize_mutation_tracking, prepare_mutation_tracking

    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (_CAP + 1))
    tracker = MutationTracker(store)
    tracker.start_turn(1)

    ctx = await prepare_mutation_tracking(
        tracker, "write_file", {"path": str(f)}, "call-1", is_shell=False, workspace_cwd=str(tmp_path)
    )
    f.write_bytes(b"y" * (_CAP + 2))
    result = await finalize_mutation_tracking(tracker, ctx, "call-1")

    assert result.file_bytes_changed is True
    assert result.file_hashes is not None
    assert result.file_hashes.before_skip is SnapshotSkipReason.TOO_LARGE
    assert result.file_hashes.after_skip is SnapshotSkipReason.TOO_LARGE


def test_shell_snapshots_carry_skip_markers(store: SnapshotStore, tmp_path: Path) -> None:
    """Shell snapshot aggregation must not report a skipped mutation as
    unchanged via ``None == None`` hash comparison."""
    import chrys.service.agent_middleware  # noqa: F401  — break the pipeline<->middleware import cycle
    from chrys.service.mutations.pipeline import _read_shell_snapshots

    f = tmp_path / "big.bin"
    f.write_bytes(b"a" * (_CAP + 1))
    tracker = MutationTracker(store)
    tracker.start_turn(1)
    tracker.pre_snapshot([str(f)])
    f.write_bytes(b"b" * (_CAP + 2))
    m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.SHELL, "call-1")
    assert m is not None

    snap = _read_shell_snapshots(tracker, [m])[m.path]
    assert snap.bytes_changed is True
    assert snap.before_skip is SnapshotSkipReason.TOO_LARGE
    assert snap.after_skip is SnapshotSkipReason.TOO_LARGE


def test_turn_entries_show_content_omitted_row(store: SnapshotStore, tmp_path: Path) -> None:
    """The persisted per-turn /diff view keeps a row for a skip-marked
    mutation (marked ``content_omitted``) instead of silently dropping
    the event — matching what live /diff showed during the turn."""
    from chrys.app.tui.screens.diff.screen import _build_period_entries

    f = tmp_path / "img.png"
    f.write_bytes(_BINARY_DATA)
    tracker = MutationTracker(store)
    tracker.start_turn(1)
    m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call-1")
    assert m is not None
    f.write_bytes(_BINARY_DATA + b"\x01")
    tracker.record_after(m)

    turn = tracker.get_turn_mutations(1)
    assert turn is not None
    entries = _build_period_entries(tracker, turn, str(tmp_path))
    assert len(entries) == 1
    assert entries[0].content_omitted == "binary"
    assert entries[0].operation is MutationOp.MODIFY
    assert entries[0].bytes_changed is True


def test_session_summary_keeps_skipped_create_and_delete(store: SnapshotStore, tmp_path: Path) -> None:
    """Existence changes are provable net changes even with (None, None)
    hashes: a skipped create (absent → exists) and a delete of a file
    whose before-backup was withheld (exists → absent) must both stay in
    the session summary and the All view — only the fully-skipped
    MODIFY (exists → exists, nothing provable) is netted out.
    """
    from chrys.app.tui.screens.diff.screen import _build_session_entries

    created = tmp_path / "artifact.bin"
    deleted = tmp_path / "old.bin"
    deleted.write_bytes(b"a" * (_CAP + 1))

    tracker = MutationTracker(store)
    tracker.start_turn(1)
    m1 = tracker.record(str(created), MutationOp.CREATE, MutationSource.WRITE_FILE, "c1")
    assert m1 is not None
    created.write_bytes(b"x" * (_CAP + 1))
    tracker.record_after(m1)

    tracker.pre_snapshot([str(deleted)])
    deleted.unlink()
    assert tracker.record(str(deleted), MutationOp.DELETE, MutationSource.SHELL, "c2") is not None

    summary = tracker.get_session_file_summary()
    assert not summary[str(created)].is_net_zero
    assert not summary[str(deleted)].is_net_zero

    entries = {e.path: e for e in _build_session_entries(tracker, str(tmp_path))}
    assert entries[str(created)].operation is MutationOp.CREATE
    assert entries[str(created)].content_omitted == "too_large"
    assert entries[str(deleted)].operation is MutationOp.DELETE
    assert entries[str(deleted)].content_omitted == "too_large"


def test_session_entries_grown_file_is_modify_with_content_omitted(store: SnapshotStore, tmp_path: Path) -> None:
    """Session (All) view: a file that grew past the cap keeps a MODIFY
    row (skip-aware existence — after side exists despite a None hash)
    marked ``content_omitted``.  A fully-skipped file stays netted out
    by the tracker's session summary."""
    from chrys.app.tui.screens.diff.screen import _build_session_entries

    grown = tmp_path / "notes.txt"
    grown.write_text("small", encoding="utf-8")
    fully_skipped = tmp_path / "big.bin"
    fully_skipped.write_bytes(b"a" * (_CAP + 1))

    tracker = MutationTracker(store)
    tracker.start_turn(1)
    m1 = tracker.record(str(grown), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
    assert m1 is not None
    grown.write_bytes(b"y" * (_CAP + 1))
    tracker.record_after(m1)

    m2 = tracker.record(str(fully_skipped), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
    assert m2 is not None
    fully_skipped.write_bytes(b"b" * (_CAP + 2))
    tracker.record_after(m2)

    entries = _build_session_entries(tracker, str(tmp_path))
    assert [e.path for e in entries] == [str(grown)]
    entry = entries[0]
    assert entry.operation is MutationOp.MODIFY
    assert entry.content_omitted == "too_large"
    assert entry.before_text == "small"
    assert entry.after_text == ""


# ════════════════════════════ Serialization ════════════════════════════


def test_skip_metadata_roundtrip(store: SnapshotStore, tmp_path: Path) -> None:
    f = tmp_path / "img.png"
    f.write_bytes(_BINARY_DATA)

    tracker = MutationTracker(store)
    tracker.start_turn(1)
    m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call-1")
    assert m is not None
    tracker.record_after(m)

    restored = MutationTracker.deserialize(tracker.serialize(), store)
    rm = restored.get_all_turns()[0].mutations[0]
    assert rm.before_skip is SnapshotSkipReason.BINARY
    assert rm.after_skip is SnapshotSkipReason.BINARY
    rsnap = restored.get_snapshot(str(f), 1)
    assert rsnap is not None
    assert rsnap.skip_reason is SnapshotSkipReason.BINARY
    assert rsnap.restorable is False


def test_legacy_dicts_without_skip_fields_load(tmp_path: Path) -> None:
    snap = FileSnapshot.from_dict({"path": str(tmp_path / "a"), "turn_id": 1, "existed": True, "content_hash": "ab"})
    assert snap.skip_reason is None
    assert snap.restorable is True

    m = FileMutation.from_dict(
        {
            "path": str(tmp_path / "a"),
            "operation": "modify",
            "source": "edit_file",
            "tool_call_id": "c1",
            "timestamp": 1.0,
        }
    )
    assert m.before_skip is None
    assert m.after_skip is None


def test_parse_skip_reason_lenient() -> None:
    assert parse_skip_reason("too_large") is SnapshotSkipReason.TOO_LARGE
    assert parse_skip_reason("binary") is SnapshotSkipReason.BINARY
    assert parse_skip_reason(None) is None
    assert parse_skip_reason("") is None
    assert parse_skip_reason("from_the_future") is None
