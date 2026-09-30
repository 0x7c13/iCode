# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for the session todo list in system reminders.

Two distinct surfaces, with different freshness contracts:

* Tier-1 per-turn reminder — snapshotted by ``prepare_turn`` via the
  ``todo_state_provider`` and frozen for the turn (byte-identical on a
  preserving retry, even when the tracker changed mid-attempt).
* LAST_WORDS dynamic section — CAPTURED once at ``set_last_words`` /
  ``restore_last_words`` and rendered INSIDE the ``[LAST_WORDS]``-prefixed
  string; never read live (the builder runs on every LLM call, so a live
  read would break the KV-cache prefix after each post-compaction
  ``todo_write``).
"""

from __future__ import annotations

from functools import partial

import pytest

from chrys.foundation.models.todos import TodoItem
from chrys.orchestration.engine.build.builder import _render_todo_reminder
from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
from chrys.service.todos.reminder import format_todo_reminder
from chrys.service.todos.tracker import TodoTracker
from tests.support.reminder_stack import reminder_pair

# ---------------------------------------------------------------------------
# format_todo_reminder
# ---------------------------------------------------------------------------


class TestFormatTodoReminder:
    def test_empty_returns_none(self) -> None:
        assert format_todo_reminder(()) is None

    def test_renders_header_and_status_marks(self) -> None:
        items = (
            TodoItem(content="done thing", status="completed"),
            TodoItem(content="active thing", status="in_progress", active_form="Doing active thing"),
            TodoItem(content="future thing", status="pending"),
        )
        text = format_todo_reminder(items)
        assert text is not None
        lines = text.splitlines()
        assert lines[0] == "Current todo list (todo_write to update):"
        assert lines[1] == "- [x] done thing"
        assert lines[2] == "- [>] active thing"
        assert lines[3] == "- [ ] future thing"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tracker_with(*contents: str) -> TodoTracker:
    tracker = TodoTracker()
    # Writers are async; seed via the sync internals the same way restore
    # would land them (tests never race the lock here).
    tracker._items = tuple(TodoItem(content=c) for c in contents)
    return tracker


def _todo_reminders(mw: SystemReminderMiddleware) -> list[str]:
    return [r for r in mw._build_reminders() if r.startswith("Current todo list")]


# ---------------------------------------------------------------------------
# Tier-1 per-turn reminder
# ---------------------------------------------------------------------------


class TestTierOneTodoReminder:
    def test_present_when_provider_wired_and_tracker_non_empty(self) -> None:
        tracker = _tracker_with("write tests")
        mw = SystemReminderMiddleware(todo_state_provider=partial(_render_todo_reminder, tracker))
        mw.prepare_turn(usage={})
        todo_reminders = _todo_reminders(mw)
        assert len(todo_reminders) == 1
        assert "- [ ] write tests" in todo_reminders[0]

    def test_absent_when_provider_is_none(self) -> None:
        mw = SystemReminderMiddleware(todo_state_provider=None)
        mw.prepare_turn(usage={})
        assert _todo_reminders(mw) == []

    def test_absent_when_tracker_empty(self) -> None:
        tracker = TodoTracker()
        mw = SystemReminderMiddleware(todo_state_provider=partial(_render_todo_reminder, tracker))
        mw.prepare_turn(usage={})
        assert _todo_reminders(mw) == []

    def test_absent_when_provider_raises(self) -> None:
        def _boom() -> str | None:
            raise RuntimeError("provider failure")

        mw = SystemReminderMiddleware(todo_state_provider=_boom)
        mw.prepare_turn(usage={})
        assert _todo_reminders(mw) == []

    def test_snapshot_frozen_for_the_turn(self) -> None:
        """Mid-turn tracker changes do not alter the current turn's reminder."""
        tracker = _tracker_with("original item")
        mw = SystemReminderMiddleware(todo_state_provider=partial(_render_todo_reminder, tracker))
        mw.prepare_turn(usage={})
        first = _todo_reminders(mw)
        tracker._items = (TodoItem(content="changed mid-turn"),)
        second = _todo_reminders(mw)
        assert first == second
        assert "original item" in first[0]

    def test_preserving_retry_keeps_the_turns_todo_snapshot(self) -> None:
        """A retry (preserve_turn_reminders) keeps the turn's todo snapshot
        instead of re-reading the tracker; what the retried request sends is
        the opener's reminder record (``test_system_reminder_record.py``)."""
        tracker = _tracker_with("step one")
        mw = SystemReminderMiddleware(todo_state_provider=partial(_render_todo_reminder, tracker))
        mw.prepare_turn(usage={})
        first = mw._build_reminders()
        tracker._items = (TodoItem(content="step one", status="completed"), TodoItem(content="step two"))
        mw.prepare_turn(usage={}, preserve_turn_reminders=True)
        retry = mw._build_reminders()
        assert retry == first

    def test_fresh_turn_resnapshots_tracker(self) -> None:
        tracker = _tracker_with("step one")
        mw = SystemReminderMiddleware(todo_state_provider=partial(_render_todo_reminder, tracker))
        mw.prepare_turn(usage={})
        tracker._items = (TodoItem(content="step two"),)
        mw.prepare_turn(usage={})
        todo_reminders = _todo_reminders(mw)
        assert len(todo_reminders) == 1
        assert "step two" in todo_reminders[0]
        assert "step one" not in todo_reminders[0]


# ---------------------------------------------------------------------------
# LAST_WORDS dynamic todo section (captured, never live)
# ---------------------------------------------------------------------------


class TestLastWordsReminderWording:
    """§5.4: task-anchored wording — the reminder rides the LAST user message,
    which on an injected turn is an injection or a synthetic ``continue``, so
    "the above request" would anchor the agent on the wrong message."""

    def test_task_anchored_wording_and_prefix_stability(self) -> None:
        mw, lw = reminder_pair()
        mw.prepare_turn(usage={})
        lw.set_last_words("[progress note]")
        reminders = lw.render()
        assert len(reminders) == 1
        text = reminders[0]
        # The prefix is load-bearing: refresh strips reminder contents by
        # byte-prefix match (_is_last_words_reminder_content).
        assert text.startswith("[LAST_WORDS] ")
        assert "progress on the current task" in text
        assert "follow-up messages" in text
        assert "the above request" not in text
        # Anti-restart guard: a note that (wrongly) claims completion must
        # steer the agent toward delivering the reply, not re-researching.
        assert "compose and deliver that reply now" in text


class TestLastWordsTodoSection:
    def test_todo_section_rides_inside_last_words_block(self) -> None:
        tracker = _tracker_with("current task")
        mw, lw = reminder_pair(todo_state_provider=partial(_render_todo_reminder, tracker))
        mw.prepare_turn(usage={})
        lw.set_last_words("[progress note]")
        reminders = lw.render()
        assert len(reminders) == 1
        # One string: the todo section must live INSIDE the [LAST_WORDS]-
        # prefixed text (refresh strips only that prefix; a separate block
        # would accumulate stale copies across refreshes).
        assert reminders[0].startswith("[LAST_WORDS] ")
        assert "[progress note]" in reminders[0]
        assert "- [ ] current task" in reminders[0]

    def test_no_todo_section_without_note(self) -> None:
        tracker = _tracker_with("current task")
        mw, lw = reminder_pair(todo_state_provider=partial(_render_todo_reminder, tracker))
        mw.prepare_turn(usage={})
        assert lw.render() == []

    def test_captured_at_set_not_read_live(self) -> None:
        """A todo_write AFTER set_last_words must not change the rendered
        block until the next refresh (KV-cache: the builder runs per call)."""
        tracker = _tracker_with("captured state")
        mw, lw = reminder_pair(todo_state_provider=partial(_render_todo_reminder, tracker))
        mw.prepare_turn(usage={})
        lw.set_last_words("[note]")
        before = lw.render()
        tracker._items = (TodoItem(content="post-capture change"),)
        after = lw.render()
        assert after == before
        assert "captured state" in after[0]
        assert "post-capture change" not in after[0]

    def test_recaptured_when_note_refreshed(self) -> None:
        tracker = _tracker_with("first state")
        mw, lw = reminder_pair(todo_state_provider=partial(_render_todo_reminder, tracker))
        mw.prepare_turn(usage={})
        lw.set_last_words("[note v1]")
        tracker._items = (TodoItem(content="second state"),)
        lw.set_last_words("[note v2]")
        reminders = lw.render()
        assert "second state" in reminders[0]
        assert "first state" not in reminders[0]

    def test_clearing_note_drops_captured_todo(self) -> None:
        tracker = _tracker_with("task")
        mw, lw = reminder_pair(todo_state_provider=partial(_render_todo_reminder, tracker))
        mw.prepare_turn(usage={})
        lw.set_last_words("[note]")
        lw.set_last_words(None)
        assert lw.render() == []
        state = mw._turns.current()
        assert state is not None
        assert state.last_words is not None
        assert state.last_words.todo is None

    def test_captured_todo_survives_preserving_retry(self) -> None:
        tracker = _tracker_with("captured state")
        mw, lw = reminder_pair(todo_state_provider=partial(_render_todo_reminder, tracker))
        mw.prepare_turn(usage={})
        lw.set_last_words("[note]")
        tracker._items = (TodoItem(content="mid-attempt change"),)
        mw.prepare_turn(usage={}, preserve_last_words=True)
        reminders = lw.render()
        assert len(reminders) == 1
        assert "captured state" in reminders[0]
        assert "mid-attempt change" not in reminders[0]

    def test_restore_captures_from_already_hydrated_tracker(self) -> None:
        """Session restore hydrates ``chrys_todos`` BEFORE restore_last_words;
        the restored note re-captures its todo section from that tracker."""
        tracker = _tracker_with("restored task")
        mw, lw = reminder_pair(todo_state_provider=partial(_render_todo_reminder, tracker))
        lw.restore_last_words("[persisted note]")
        mw.prepare_turn(usage={}, preserve_last_words=True)
        reminders = lw.render()
        assert len(reminders) == 1
        assert "[persisted note]" in reminders[0]
        assert "- [ ] restored task" in reminders[0]

    def test_restored_capture_visible_before_any_turn(self) -> None:
        """Restore → save-without-resume still renders the captured section
        (mirrors get_last_words falling back to the restored note)."""
        tracker = _tracker_with("restored task")
        _, lw = reminder_pair(todo_state_provider=partial(_render_todo_reminder, tracker))
        lw.restore_last_words("[persisted note]")
        reminders = lw.render()
        assert len(reminders) == 1
        assert "- [ ] restored task" in reminders[0]

    def test_fresh_turn_discards_restored_capture(self) -> None:
        tracker = _tracker_with("restored task")
        mw, lw = reminder_pair(todo_state_provider=partial(_render_todo_reminder, tracker))
        lw.restore_last_words("[persisted note]")
        mw.prepare_turn(usage={})
        assert lw.render() == []
        assert lw._restored_last_words_todo is None


# ---------------------------------------------------------------------------
# Builder provider glue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tracker", [None, TodoTracker()])
def test_render_todo_reminder_handles_absent_or_empty_tracker(tracker: TodoTracker | None) -> None:
    assert _render_todo_reminder(tracker) is None


def test_render_todo_reminder_renders_snapshot() -> None:
    tracker = _tracker_with("wire the builder")
    text = _render_todo_reminder(tracker)
    assert text is not None
    assert "- [ ] wire the builder" in text
