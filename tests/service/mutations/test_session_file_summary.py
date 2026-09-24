# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for MutationTracker net-diff aggregation: get_session_file_summary and get_turn_file_summary."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from chrys.service.mutations.store import SnapshotStore
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.types import MutationOp, MutationSource

if TYPE_CHECKING:
    from pathlib import Path


# ===========================================================================
# MutationTracker — session-wide file summary (ALL-tab aggregation)
# ===========================================================================


class TestSessionFileSummary:
    """get_session_file_summary: net before/after across all turns.

    before = earliest turn's snapshot (pre-session state)
    after = latest turn's last mutation's after_hash
    Net-zero churn (before == after, including both None) is filtered out.
    """

    def _mk(self, tmp_path: Path) -> MutationTracker:
        return MutationTracker(SnapshotStore(tmp_path))

    def _key(self, path: Path) -> str:
        return os.path.normpath(os.path.abspath(str(path)))

    def test_simple_modify_across_turns_keeps_pre_session_before(self, tmp_path: Path) -> None:
        """MODIFY v0→v1 (turn 1) then v1→v2 (turn 2): before=v0, after=v2."""
        f = tmp_path / "code.py"
        f.write_text("v0", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("v1", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("v2", encoding="utf-8")
        tracker.record_after(m2)

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before == m1.before_hash  # hash of "v0"
        assert diff.after == m2.after_hash  # hash of "v2"

    def test_create_then_delete_in_later_turn_is_net_zero(self, tmp_path: Path) -> None:
        """File didn't exist → CREATE (turn 1) → DELETE (turn 2): filtered."""
        f = tmp_path / "new.py"
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c1")
        f.write_text("hello", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c2")

        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary

    def test_create_delete_create_ends_as_create(self, tmp_path: Path) -> None:
        """File didn't exist → CREATE (t1) → DELETE (t2) → CREATE (t3): net CREATE."""
        f = tmp_path / "flaky.py"
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c1")
        f.write_text("first", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c2")

        tracker.start_turn(3)
        m3 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c3")
        f.write_text("second", encoding="utf-8")
        tracker.record_after(m3)

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before is None  # pre-session: file didn't exist
        assert diff.after == m3.after_hash  # final content = "second"

    def test_create_delete_create_edit_delete_is_net_zero(self, tmp_path: Path) -> None:
        """User-requested worst case: CREATE→DELETE→CREATE→EDIT→DELETE across turns.

        File never existed pre-session and doesn't exist post-session → net-zero → filtered.
        """
        f = tmp_path / "churn.py"
        tracker = self._mk(tmp_path)

        # Turn 1: CREATE
        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c1")
        f.write_text("a", encoding="utf-8")
        tracker.record_after(m1)

        # Turn 2: DELETE
        tracker.start_turn(2)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c2")

        # Turn 3: CREATE again
        tracker.start_turn(3)
        m3 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c3")
        f.write_text("b", encoding="utf-8")
        tracker.record_after(m3)

        # Turn 4: EDIT
        tracker.start_turn(4)
        m4 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c4")
        f.write_text("b-edited", encoding="utf-8")
        tracker.record_after(m4)

        # Turn 5: DELETE
        tracker.start_turn(5)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c5")

        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary  # net-zero filtered

    def test_existing_file_modified_then_deleted_then_recreated_same_content(self, tmp_path: Path) -> None:
        """Pre-existing X → MODIFY to Y → DELETE → CREATE with X: net-zero (back to X)."""
        f = tmp_path / "roundtrip.py"
        f.write_text("original", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("modified", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c2")

        tracker.start_turn(3)
        m3 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c3")
        f.write_text("original", encoding="utf-8")  # back to original
        tracker.record_after(m3)

        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary  # same content hash → filtered

    def test_existing_file_modified_then_deleted_then_recreated_different_content(self, tmp_path: Path) -> None:
        """Pre-existing X → MODIFY to Y → DELETE → CREATE with Z: before=X, after=Z."""
        f = tmp_path / "reshape.py"
        f.write_text("X", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("Y", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c2")

        tracker.start_turn(3)
        m3 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c3")
        f.write_text("Z", encoding="utf-8")
        tracker.record_after(m3)

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before == m1.before_hash  # hash("X")
        assert diff.after == m3.after_hash  # hash("Z")
        assert diff.before != diff.after

    def test_modify_bounce_back_same_content_is_net_zero(self, tmp_path: Path) -> None:
        """X → Y → X across turns: before == after → filtered."""
        f = tmp_path / "bounce.py"
        f.write_text("X", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("Y", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("X", encoding="utf-8")
        tracker.record_after(m2)

        assert m1.before_hash == m2.after_hash  # sanity: round-trip to same content
        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary

    def test_multiple_edits_in_same_turn_across_multiple_turns(self, tmp_path: Path) -> None:
        """Chained edits within each turn: earliest-before, latest-after."""
        f = tmp_path / "chained.py"
        f.write_text("v0", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m_a = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "a")
        f.write_text("v1", encoding="utf-8")
        tracker.record_after(m_a)
        m_b = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "b")
        f.write_text("v2", encoding="utf-8")
        tracker.record_after(m_b)

        tracker.start_turn(2)
        m_c = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c")
        f.write_text("v3", encoding="utf-8")
        tracker.record_after(m_c)
        m_d = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "d")
        f.write_text("v4", encoding="utf-8")
        tracker.record_after(m_d)

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before == m_a.before_hash  # v0
        assert diff.after == m_d.after_hash  # v4

    def test_multiple_independent_files_tracked_separately(self, tmp_path: Path) -> None:
        """Several files mutated across turns — each gets its own aggregation."""
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        c = tmp_path / "c.py"
        a.write_text("a0", encoding="utf-8")
        b.write_text("b0", encoding="utf-8")
        # c doesn't exist initially
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        # a: MODIFY
        ma = tracker.record(str(a), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        a.write_text("a1", encoding="utf-8")
        tracker.record_after(ma)
        # c: CREATE
        mc = tracker.record(str(c), MutationOp.CREATE, MutationSource.WRITE_FILE, "c1")
        c.write_text("c_content", encoding="utf-8")
        tracker.record_after(mc)

        tracker.start_turn(2)
        # b: DELETE
        tracker.pre_snapshot([str(b)])
        b.unlink()
        tracker.record(str(b), MutationOp.DELETE, MutationSource.SHELL, "c2")
        # a: MODIFY back to a0 (net-zero)
        ma2 = tracker.record(str(a), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        a.write_text("a0", encoding="utf-8")
        tracker.record_after(ma2)

        summary = tracker.get_session_file_summary()

        # a: bounced back → net-zero → not in summary
        assert self._key(a) not in summary
        # b: was X, now deleted → before=hash_X, after=None
        b_diff = summary[self._key(b)]
        assert b_diff.before is not None
        assert b_diff.after is None
        # c: didn't exist, now exists → before=None, after=hash
        c_diff = summary[self._key(c)]
        assert c_diff.before is None
        assert c_diff.after == mc.after_hash

    def test_delete_then_create_same_name_same_content_is_net_zero(self, tmp_path: Path) -> None:
        """Pre-existing X → DELETE (t1) → CREATE with same content X (t2): filtered."""
        f = tmp_path / "recreated.py"
        f.write_text("hello", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c1")

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c2")
        f.write_text("hello", encoding="utf-8")  # same content as original
        tracker.record_after(m2)

        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary  # hash(hello) == hash(hello) → filtered

    def test_delete_then_create_same_name_different_content_is_modify(self, tmp_path: Path) -> None:
        """Pre-existing X → DELETE (t1) → CREATE with Y (t2): MODIFY, before=X, after=Y."""
        f = tmp_path / "replaced.py"
        f.write_text("old", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        m1 = tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c1")

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c2")
        f.write_text("new", encoding="utf-8")
        tracker.record_after(m2)

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before == m1.before_hash  # hash("old") — captured by turn-1 snapshot
        assert diff.after == m2.after_hash  # hash("new")
        assert diff.before != diff.after

    def test_two_edits_same_turn_netting_to_original_is_filtered(self, tmp_path: Path) -> None:
        """Single turn: A → B → A. Two edits, no net change → filtered."""
        from chrys.app.tui.screens.diff.screen import _build_period_entries

        f = tmp_path / "reverted.py"
        f.write_text("A", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("B", encoding="utf-8")
        tracker.record_after(m1)

        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("A", encoding="utf-8")  # reverted
        tracker.record_after(m2)

        assert m1.before_hash == m2.after_hash  # sanity: back to original hash
        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary
        turn = tracker.get_turn_mutations(1)
        assert turn is not None
        assert _build_period_entries(tracker, turn, str(tmp_path)) == []

    def test_eol_only_modify_is_kept_as_zero_line_diff_entry(self, tmp_path: Path) -> None:
        """Line-ending-only changes have no line-count delta, but still changed bytes."""
        from chrys.app.tui.screens.diff.screen import _build_period_entries, _build_session_entries

        f = tmp_path / "Power.yaml"
        f.write_bytes(b"name: Power\n")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_bytes(b"name: Power\r\n")
        tracker.record_after(m)

        turn = tracker.get_turn_mutations(1)
        assert turn is not None
        assert m.before_hash != m.after_hash
        turn_entries = _build_period_entries(tracker, turn, str(tmp_path))
        session_entries = _build_session_entries(tracker, str(tmp_path))
        assert len(turn_entries) == 1
        assert len(session_entries) == 1
        assert turn_entries[0].before_text == "name: Power\n"
        assert turn_entries[0].after_text == "name: Power\r\n"

    def test_bom_only_modify_is_kept_as_metadata_only_diff_entry(self, tmp_path: Path) -> None:
        """Byte-level encoding changes can decode to identical text and still be real changes."""
        from chrys.app.tui.screens.diff.screen import _build_period_entries, _build_session_entries

        f = tmp_path / "Power.yaml"
        f.write_bytes(b"\xef\xbb\xbfname: Power\n")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_bytes(b"name: Power\n")
        tracker.record_after(m)

        turn = tracker.get_turn_mutations(1)
        assert turn is not None
        assert m.before_hash != m.after_hash
        turn_entries = _build_period_entries(tracker, turn, str(tmp_path))
        session_entries = _build_session_entries(tracker, str(tmp_path))
        assert len(turn_entries) == 1
        assert len(session_entries) == 1
        assert turn_entries[0].before_text == "name: Power\n"
        assert turn_entries[0].after_text == "name: Power\n"
        assert turn_entries[0].bytes_changed is True

    def test_two_edits_same_turn_with_net_change_is_modify(self, tmp_path: Path) -> None:
        """Single turn: A → B → C. Two edits, net change → MODIFY (before=A, after=C)."""
        f = tmp_path / "stepwise.py"
        f.write_text("A", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("B", encoding="utf-8")
        tracker.record_after(m1)

        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("C", encoding="utf-8")
        tracker.record_after(m2)

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before == m1.before_hash  # hash("A")
        assert diff.after == m2.after_hash  # hash("C")

    def test_two_failed_edits_same_content_after_both_is_filtered(self, tmp_path: Path) -> None:
        """Two edits both leaving file untouched (failed tool calls) → filtered."""
        f = tmp_path / "unchanged.py"
        f.write_text("stable", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        # Tool ran but made no change (e.g. edit_file with replacement == original)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        tracker.record_after(m1)  # file bytes unchanged on disk
        assert m1.before_hash == m1.after_hash

        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        tracker.record_after(m2)
        assert m2.before_hash == m2.after_hash

        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary

    def test_move_mutation_tracks_both_paths(self, tmp_path: Path) -> None:
        """MOVE with old_path: old_path shows as implicit DELETE in summary."""
        src = tmp_path / "src.py"
        dst = tmp_path / "dst.py"
        src.write_text("moved content", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        # Snapshot BOTH paths before the shell command runs (realistic flow
        # from WorkspaceScanner) — otherwise dst's post-move snapshot would
        # equal its after_hash and collapse to net-zero.
        tracker.pre_snapshot([str(src), str(dst)])
        src.rename(dst)
        # Synthesize a MOVE record (old_path is the only code path that
        # produces FileMutation.old_path).
        m = tracker.record(
            str(dst),
            MutationOp.MOVE,
            MutationSource.SHELL,
            "c1",
            old_path=str(src),
        )
        assert m is not None

        summary = tracker.get_session_file_summary()
        # Destination: didn't exist → now exists with moved content.
        dst_diff = summary[self._key(dst)]
        assert dst_diff.before is None
        assert dst_diff.after is not None
        # Source (old_path): implicit DELETE → before=moved content, after=None.
        src_diff = summary[self._key(src)]
        assert src_diff.before is not None  # captured by pre_snapshot
        assert src_diff.after is None  # implicit delete via MOVE

    def test_turn_diff_keeps_move_when_destination_bytes_are_unchanged(self, tmp_path: Path) -> None:
        """A MOVE remains visible even when the destination hash itself is net-zero."""
        from chrys.app.tui.screens.diff.screen import _build_period_entries

        src = tmp_path / "src.py"
        dst = tmp_path / "dst.py"
        src.write_text("same content", encoding="utf-8")
        dst.write_text("same content", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        tracker.pre_snapshot([str(src), str(dst)])
        src.replace(dst)
        mutation = tracker.record(str(dst), MutationOp.MOVE, MutationSource.SHELL, "c1", old_path=str(src))
        assert mutation is not None

        turn = tracker.get_turn_mutations(1)
        assert turn is not None
        summary = tracker.get_turn_file_summary(1)
        assert summary[self._key(dst)].is_net_zero

        entries = _build_period_entries(tracker, turn, str(tmp_path))
        assert len(entries) == 1
        assert entries[0].operation is MutationOp.MOVE
        assert entries[0].old_path == self._key(src)
        assert entries[0].bytes_changed is False

    def test_net_modify_when_existing_file_deleted_then_recreated_different(self, tmp_path: Path) -> None:
        """Regression: existing X → DELETE → CREATE(Y) must surface as MODIFY, not CREATE.

        The last recorded mutation op is CREATE, but the *net* effect on
        a pre-existing file is MODIFY (file existed, still exists with
        different content).  This test guards against the
        last-op-fallback pitfall in _build_session_entries.
        """
        from chrys.app.tui.screens.diff.screen import _build_session_entries

        f = tmp_path / "replaced.py"
        f.write_text("X", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c1")

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c2")
        f.write_text("Y", encoding="utf-8")
        tracker.record_after(m2)

        entries = _build_session_entries(tracker, str(tmp_path))
        assert len(entries) == 1
        assert entries[0].operation == MutationOp.MODIFY  # NOT CREATE
        assert entries[0].before_text == "X"
        assert entries[0].after_text == "Y"

    def test_empty_tracker_returns_empty(self, tmp_path: Path) -> None:
        """No turns recorded → empty summary."""
        tracker = self._mk(tmp_path)
        assert tracker.get_session_file_summary() == {}

    def test_empty_turns_return_empty(self, tmp_path: Path) -> None:
        """Turns started but no mutations → empty summary."""
        tracker = self._mk(tmp_path)
        tracker.start_turn(1)
        tracker.start_turn(2)
        assert tracker.get_session_file_summary() == {}

    def test_non_monotonic_turn_order_uses_turn_id_not_insertion(self, tmp_path: Path) -> None:
        """Session restore/retry can leave ``_log.turns`` with non-monotonic
        ``turn_id`` values — the summary must aggregate chronologically by
        ``turn_id``, not by list insertion order.

        Regression: without sorting, the reversed list would take turn 3's
        ``before`` as the session start and turn 1's ``after`` as the final
        state, collapsing to net-zero (hash(v1) == hash(v1)) and dropping
        the entry entirely.
        """
        f = tmp_path / "x.py"
        f.write_text("v0", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("v1", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(3)
        m3 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c3")
        f.write_text("v3", encoding="utf-8")
        tracker.record_after(m3)

        # Simulate restore path where the saved turns deserialize in a
        # different order than their turn_ids (the private reverse() is
        # the simplest faithful reproduction of that state).
        tracker._log.turns.reverse()

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before == m1.before_hash  # pre-session = v0
        assert diff.after == m3.after_hash  # final = v3

    def test_survives_serialize_roundtrip(self, tmp_path: Path) -> None:
        """Summary is identical before and after tracker serialize/deserialize."""
        f = tmp_path / "persist.py"
        f.write_text("v0", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("v1", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("v2", encoding="utf-8")
        tracker.record_after(m2)

        before_summary = tracker.get_session_file_summary()
        data = tracker.serialize()
        restored = MutationTracker.deserialize(data, SnapshotStore(tmp_path))
        after_summary = restored.get_session_file_summary()

        assert before_summary == after_summary
        diff = after_summary[self._key(f)]
        assert diff.before == m1.before_hash
        assert diff.after == m2.after_hash
