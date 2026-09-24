# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for history-marker visibility (interrupted, turn, continuation, injected) in state-store message counting."""

from __future__ import annotations

from pathlib import Path

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.kernel import Message
from chrys.service.state.store import JsonFileStateStore, _is_visible_message


def _make_interrupted_marker() -> Message:
    m = Message("assistant", ["Execution interrupted"])
    m.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.INTERRUPTED
    m.additional_properties["_interrupted_by"] = "user"
    return m


def _make_turn_marker() -> Message:
    m = Message("assistant", [""])
    m.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    return m


def test_interrupted_marker_not_visible():
    """Interrupted markers should not be counted as visible messages."""
    m = _make_interrupted_marker()
    assert not _is_visible_message(m)


def test_turn_marker_not_visible():
    """Turn markers should not be counted as visible messages."""
    m = _make_turn_marker()
    assert not _is_visible_message(m)


def test_user_message_visible():
    """User messages should be counted as visible."""
    m = Message("user", ["hello"])
    assert _is_visible_message(m)


def test_assistant_with_text_visible():
    """Assistant messages with non-empty text should be counted as visible."""
    m = Message("assistant", ["I can help with that."])
    assert _is_visible_message(m)


def test_assistant_empty_text_not_visible():
    """Assistant messages with empty text should not be counted as visible."""
    m = Message("assistant", [""])
    assert not _is_visible_message(m)


def test_is_visible_message_excludes_continuation_counts_injected() -> None:
    """A synthetic ``continue`` nudge is a orchestration placeholder, not user
    content; an injected mid-turn message IS user input."""
    nudge = Message("user", ["continue"])
    nudge.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
    injected = Message("user", ["mid-turn guidance"])
    injected.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True

    assert not _is_visible_message(nudge)
    assert _is_visible_message(injected)
    assert _is_visible_message(Message("user", ["real question"]))


async def test_message_count_stable_across_continuation_nudge(tmp_path: Path) -> None:
    """A crash-leftover flagged nudge must not inflate message_count — and,
    through it, must not advance updated_at on the next save."""
    store = JsonFileStateStore(tmp_path)
    user = Message("user", ["do the thing"])
    answer = Message("assistant", ["done"])
    await store.save_session("s1", {"messages": [user, answer], "compressed_msgs": []})
    first = (await store.list_sessions())[0]
    assert first.message_count == 2

    nudge = Message("user", ["continue"])
    nudge.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
    await store.save_session("s1", {"messages": [user, answer, nudge], "compressed_msgs": []})
    second = (await store.list_sessions())[0]
    assert second.message_count == 2
    assert second.updated_at == first.updated_at

    injected = Message("user", ["also check the docs"])
    injected.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    await store.save_session("s1", {"messages": [user, answer, nudge, injected], "compressed_msgs": []})
    third = (await store.list_sessions())[0]
    assert third.message_count == 3
