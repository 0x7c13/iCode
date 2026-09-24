# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Mutation provenance (Layer 1) — schema, summaries, rollback plans.

Covers the ``MutationProvenance`` enum defaults and lenient
serialization, ``contested`` flag plumbing, foreign-aware summary
folding with ``contested``/``inferred`` badges, the calibration
already-tracked exclusion, ``RollbackPlan`` exclusions (existing and
provenance-based reasons), and the live-snapshot foreign filter.
Foreign/contested rows are constructed directly — nothing produces them
until the coordination registry lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.service.agent_middleware import _read_shell_snapshots
from chrys.service.mutations.store import SnapshotStore
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.types import (
    FileMutation,
    MutationOp,
    MutationProvenance,
    MutationSource,
    RollbackExclusionReason,
    RollbackPlan,
    default_provenance,
    parse_provenance,
)

# ════════════════════════════ schema ════════════════════════════


def test_default_provenance_by_source() -> None:
    assert default_provenance(MutationSource.WRITE_FILE) is MutationProvenance.PROVEN
    assert default_provenance(MutationSource.EDIT_FILE) is MutationProvenance.PROVEN
    assert default_provenance(MutationSource.SHELL) is MutationProvenance.ASSUMED
    assert default_provenance(MutationSource.IMPLICIT) is MutationProvenance.ASSUMED


def _mutation(source: MutationSource, **kwargs) -> FileMutation:
    return FileMutation(
        path="/tmp/x",
        operation=MutationOp.MODIFY,
        source=source,
        tool_call_id="c1",
        timestamp=1.0,
        **kwargs,
    )


def test_construction_derives_provenance_from_source() -> None:
    assert _mutation(MutationSource.WRITE_FILE).provenance is MutationProvenance.PROVEN
    assert _mutation(MutationSource.SHELL).provenance is MutationProvenance.ASSUMED
    # Explicit provenance wins over the source default.
    m = _mutation(MutationSource.SHELL, provenance=MutationProvenance.PROVEN)
    assert m.provenance is MutationProvenance.PROVEN


def test_to_dict_always_writes_provenance_and_contested_only_when_true() -> None:
    d = _mutation(MutationSource.EDIT_FILE).to_dict()
    assert d["provenance"] == "proven"
    assert "contested" not in d

    d = _mutation(MutationSource.SHELL, contested=True).to_dict()
    assert d["provenance"] == "assumed"
    assert d["contested"] is True


def test_from_dict_lenient_parsing() -> None:
    base = _mutation(MutationSource.SHELL).to_dict()

    # Missing key (legacy logs) → source-derived default.
    legacy = {k: v for k, v in base.items() if k != "provenance"}
    assert FileMutation.from_dict(legacy).provenance is MutationProvenance.ASSUMED
    legacy["source"] = "write_file"
    assert FileMutation.from_dict(legacy).provenance is MutationProvenance.PROVEN

    # Unknown future value → source-derived default, not an error.
    unknown = dict(base, provenance="quantum")
    assert FileMutation.from_dict(unknown).provenance is MutationProvenance.ASSUMED

    # Round-trip preserves explicit values.
    foreign = _mutation(MutationSource.SHELL, provenance=MutationProvenance.FOREIGN, contested=True)
    parsed = FileMutation.from_dict(foreign.to_dict())
    assert parsed.provenance is MutationProvenance.FOREIGN
    assert parsed.contested is True


def test_parse_provenance_helper() -> None:
    assert parse_provenance("foreign", MutationSource.SHELL) is MutationProvenance.FOREIGN
    assert parse_provenance(None, MutationSource.SHELL) is MutationProvenance.ASSUMED
    assert parse_provenance("", MutationSource.EDIT_FILE) is MutationProvenance.PROVEN
    assert parse_provenance("bogus", MutationSource.IMPLICIT) is MutationProvenance.ASSUMED


# ════════════════════════════ tracker recording ════════════════════════════


@pytest.fixture
def tracker(tmp_path: Path) -> MutationTracker:
    return MutationTracker(SnapshotStore(tmp_path))


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_record_defaults_and_explicit_provenance(tracker: MutationTracker, tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    _write(f, "one")
    tracker.start_turn(1)

    m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
    assert m1 is not None and m1.provenance is MutationProvenance.PROVEN

    _write(f, "two")
    m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.SHELL, "c2")
    assert m2 is not None and m2.provenance is MutationProvenance.ASSUMED

    _write(f, "three")
    m3 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.SHELL, "c3", provenance=MutationProvenance.PROVEN)
    assert m3 is not None and m3.provenance is MutationProvenance.PROVEN


def test_serialize_roundtrip_preserves_provenance(tracker: MutationTracker, tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    _write(f, "one")
    tracker.start_turn(1)
    _write(f, "two")
    m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.SHELL, "c1")
    assert m is not None
    m.provenance = MutationProvenance.FOREIGN

    restored = MutationTracker.deserialize(tracker.serialize(), tracker.store)
    rows = restored.get_all_turns()[0].mutations
    assert rows[0].provenance is MutationProvenance.FOREIGN


# ════════════════════════════ summaries ════════════════════════════


def _shell_modify(tracker: MutationTracker, path: Path, text: str, call_id: str) -> FileMutation:
    _write(path, text)
    m = tracker.record(str(path), MutationOp.MODIFY, MutationSource.SHELL, call_id)
    assert m is not None
    return m


def test_foreign_only_path_absent_from_summaries(tracker: MutationTracker, tmp_path: Path) -> None:
    f = tmp_path / "peer.txt"
    _write(f, "base")
    tracker.start_turn(1)
    tracker.pre_snapshot([str(f)])
    m = _shell_modify(tracker, f, "peer wrote this", "c1")
    m.provenance = MutationProvenance.FOREIGN

    assert tracker.get_turn_file_summary(1) == {}
    assert tracker.get_session_file_summary() == {}
    # Raw event log keeps the row for audit.
    assert len(tracker.get_all_turns()[0].mutations) == 1


def test_mixed_path_folds_only_our_endpoints_and_badges_contested(tracker: MutationTracker, tmp_path: Path) -> None:
    f = tmp_path / "shared.txt"
    _write(f, "base")
    tracker.start_turn(1)
    tracker.pre_snapshot([str(f)])
    ours = _shell_modify(tracker, f, "ours", "c1")
    peer = _shell_modify(tracker, f, "peer overwrote", "c2")
    peer.provenance = MutationProvenance.FOREIGN

    summary = tracker.get_turn_file_summary(1)
    diff = summary[str(f)]
    # After endpoint comes from OUR last row, not the peer's.
    assert diff.after == ours.after_hash
    assert diff.contested is True  # foreign rows mixed with ours
    assert diff.inferred is True  # shell rows are assumed

    session = tracker.get_session_file_summary()
    assert session[str(f)].after == ours.after_hash
    assert session[str(f)].contested is True


def test_contested_row_badges_without_demotion(tracker: MutationTracker, tmp_path: Path) -> None:
    f = tmp_path / "w.txt"
    _write(f, "base")
    tracker.start_turn(1)
    m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
    assert m is not None
    _write(f, "mine")
    tracker.record_after(m)
    m.contested = True

    diff = tracker.get_turn_file_summary(1)[str(f)]
    assert diff.contested is True
    assert diff.inferred is False  # proven write is not inference


def test_proven_only_path_has_no_badges(tracker: MutationTracker, tmp_path: Path) -> None:
    f = tmp_path / "clean.txt"
    _write(f, "base")
    tracker.start_turn(1)
    m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.WRITE_FILE, "c1")
    assert m is not None
    _write(f, "new")
    tracker.record_after(m)

    diff = tracker.get_turn_file_summary(1)[str(f)]
    assert diff.contested is False
    assert diff.inferred is False


def test_changed_files_set_excludes_foreign(tracker: MutationTracker, tmp_path: Path) -> None:
    ours = tmp_path / "ours.txt"
    theirs = tmp_path / "theirs.txt"
    _write(ours, "base")
    _write(theirs, "base")
    tracker.start_turn(1)
    tracker.pre_snapshot([str(ours), str(theirs)])
    _shell_modify(tracker, ours, "local", "c1")
    m = _shell_modify(tracker, theirs, "peer", "c2")
    m.provenance = MutationProvenance.FOREIGN

    # The demoted path stays eligible for fresh calibration detection.
    assert tracker.get_changed_files_set() == {str(ours)}


# ════════════════════════════ rollback plans ════════════════════════════


def test_plan_excludes_foreign_and_contested_with_reasons(tracker: MutationTracker, tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.txt"
    contested = tmp_path / "contested.txt"
    clean = tmp_path / "clean.txt"
    for p in (foreign, contested, clean):
        _write(p, "base")
    tracker.start_turn(1)
    tracker.pre_snapshot([str(foreign), str(contested), str(clean)])

    fm = _shell_modify(tracker, foreign, "peer", "c1")
    fm.provenance = MutationProvenance.FOREIGN
    cm = _shell_modify(tracker, contested, "both", "c2")
    cm.contested = True
    _shell_modify(tracker, clean, "ours", "c3")

    plan = tracker.get_rollback_plan(1)
    assert plan.paths == {str(clean)}
    assert dict(plan.exclusions) == {
        str(foreign): RollbackExclusionReason.FOREIGN,
        str(contested): RollbackExclusionReason.CONTESTED,
    }

    # rollback acts on entries only: peer/contested files stay on disk.
    results = tracker.rollback(1)
    assert {r.path for r in results} == {str(clean)}
    assert foreign.read_text(encoding="utf-8") == "peer"
    assert contested.read_text(encoding="utf-8") == "both"
    assert clean.read_text(encoding="utf-8") == "base"


def test_foreign_move_excludes_both_paths(tracker: MutationTracker, tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    _write(src, "content")
    tracker.start_turn(1)
    tracker.pre_snapshot([str(src)])
    src.rename(dst)
    m = tracker.record(str(dst), MutationOp.MOVE, MutationSource.SHELL, "c1", old_path=str(src))
    assert m is not None
    m.provenance = MutationProvenance.FOREIGN

    plan = tracker.get_rollback_plan(1)
    assert plan.entries == []
    reasons = dict(plan.exclusions)
    assert reasons[str(dst)] is RollbackExclusionReason.FOREIGN
    assert reasons[str(src)] is RollbackExclusionReason.FOREIGN


def test_unrestorable_exclusion_is_reported_not_silent(tmp_path: Path) -> None:
    from chrys.service.mutations.store import SnapshotPolicy

    cap = 64
    store = SnapshotStore(tmp_path, policy=SnapshotPolicy(max_file_bytes=cap))
    tracker = MutationTracker(store)
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (cap + 1))
    tracker.start_turn(1)
    m = tracker.record(str(big), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
    assert m is not None
    big.write_bytes(b"y" * (cap + 2))
    tracker.record_after(m)

    plan = tracker.get_rollback_plan(1)
    assert plan.entries == []
    assert plan.exclusions == [(str(big), RollbackExclusionReason.UNRESTORABLE)]


def test_rollback_turns_executes_from_prebuilt_plan(tracker: MutationTracker, tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    _write(f, "base")
    tracker.start_turn(1)
    m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
    assert m is not None
    _write(f, "changed")
    tracker.record_after(m)

    plan = tracker.get_rollback_plan_for_turns({1})
    assert plan.paths == {str(f)}
    results = tracker.rollback_turns({1}, plan=plan)
    assert [r.path for r in results] == [str(f)]
    assert f.read_text(encoding="utf-8") == "base"
    assert tracker.get_all_turns() == []


def test_empty_plan_shapes() -> None:
    plan = RollbackPlan(entries=[])
    assert plan.paths == set()
    assert plan.exclusions == []
    assert plan.warnings == []


# ════════════════════════════ live snapshots ════════════════════════════


def test_read_shell_snapshots_filters_foreign_and_threads_flags(tracker: MutationTracker, tmp_path: Path) -> None:
    ours = tmp_path / "ours.txt"
    theirs = tmp_path / "theirs.txt"
    _write(ours, "base")
    _write(theirs, "base")
    tracker.start_turn(1)
    tracker.pre_snapshot([str(ours), str(theirs)])
    local = _shell_modify(tracker, ours, "local", "c1")
    local.contested = True
    peer = _shell_modify(tracker, theirs, "peer", "c2")
    peer.provenance = MutationProvenance.FOREIGN

    snapshots = _read_shell_snapshots(tracker, [local, peer])
    assert set(snapshots) == {str(ours)}
    snap = snapshots[str(ours)]
    assert snap.provenance == "assumed"
    assert snap.contested is True


def test_read_shell_snapshots_marks_survivor_contested_on_mixed_path(tracker: MutationTracker, tmp_path: Path) -> None:
    f = tmp_path / "shared.txt"
    _write(f, "base")
    tracker.start_turn(1)
    tracker.pre_snapshot([str(f)])
    ours = _shell_modify(tracker, f, "ours", "c1")
    peer = _shell_modify(tracker, f, "peer", "c2")
    peer.provenance = MutationProvenance.FOREIGN

    # Foreign row dropped, but the surviving local row keeps the badge —
    # regardless of whether the foreign row comes before or after ours.
    for order in ([ours, peer], [peer, ours]):
        snapshots = _read_shell_snapshots(tracker, order)
        assert set(snapshots) == {str(f)}
        assert snapshots[str(f)].contested is True
        assert snapshots[str(f)].after_hash == ours.after_hash


# ════════════════════════════ diff screen entries ════════════════════════════


def test_turn_entries_ignore_foreign_metadata(tracker: MutationTracker, tmp_path: Path) -> None:
    """A peer's op/old_path/source must not leak onto our folded row."""
    from chrys.app.tui.screens.diff.screen import _build_period_entries

    f = tmp_path / "shared.txt"
    _write(f, "base")
    tracker.start_turn(1)
    m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
    assert m is not None
    _write(f, "ours")
    tracker.record_after(m)
    # Peer deletes the file afterwards (recorded as a foreign row).
    tracker.pre_snapshot([str(f)])
    f.unlink()
    peer = tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c2")
    assert peer is not None
    peer.provenance = MutationProvenance.FOREIGN

    turn = tracker.get_all_turns()[0]
    entries = _build_period_entries(tracker, turn, str(tmp_path))
    assert len(entries) == 1
    entry = entries[0]
    assert entry.operation is MutationOp.MODIFY  # ours, not the peer's DELETE
    assert entry.old_path is None
    assert entry.source == MutationSource.EDIT_FILE.value  # no shell folded in
    assert entry.contested is True


def test_session_entries_ignore_foreign_move_metadata(tracker: MutationTracker, tmp_path: Path) -> None:
    from chrys.app.tui.screens.diff.screen import _build_session_entries

    f = tmp_path / "shared.txt"
    other = tmp_path / "elsewhere.txt"
    _write(f, "base")
    tracker.start_turn(1)
    m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.WRITE_FILE, "c1")
    assert m is not None
    _write(f, "ours")
    tracker.record_after(m)
    # Peer "moves" another file onto this path (foreign MOVE row).
    _write(other, "peer content")
    tracker.pre_snapshot([str(f), str(other)])
    _write(f, "peer moved this here")
    peer = tracker.record(str(f), MutationOp.MOVE, MutationSource.SHELL, "c2", old_path=str(other))
    assert peer is not None
    peer.provenance = MutationProvenance.FOREIGN

    entries = _build_session_entries(tracker, str(tmp_path))
    by_path = {e.path: e for e in entries}
    entry = by_path[str(f)]
    assert entry.old_path is None  # peer's MOVE must not render as our rename
    assert entry.source == MutationSource.WRITE_FILE.value
    assert entry.contested is True
    assert entry.after_hash == m.after_hash  # our endpoint, not the peer's
