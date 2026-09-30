# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for crash-recovery checkpoint shaping."""

from __future__ import annotations

from typing import Any

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.trajectory.metadata import ensure_analytics_item_id, read_analytics_item_id
from chrys.kernel import Content, LoopRecorder, Message
from chrys.service.agent_middleware.injection import ConsumedInjection, InjectionAnchor
from chrys.service.agent_middleware.reminders.archive_pointer import CATALOG_POINTER_RECORD_COUNT_STATE_KEY
from chrys.service.session.checkpoint import build_recovery_state
from chrys.service.session.history import SessionHistoryManager
from chrys.service.session.message_metadata import MESSAGE_CREATED_AT_KEY
from chrys.service.session.runtime_metadata import SessionRuntimeMetadata
from chrys.service.trajectory.preparation import PreparationScope, PreparationTrace
from tests.service.trajectory._fakes import make_context


class _MutationTracker:
    def serialize(self) -> dict[str, object]:
        return {"turns": [{"turn_id": 7}]}


def _marker_kind(message: Message) -> object:
    return message.additional_properties.get(HistoryMarkerKind.KEY)


def test_checkpoint_adds_missing_user_prompt_without_mutating_live_state() -> None:
    live_state = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    capture = LoopRecorder()
    capture._initial_count = 1
    capture._captured = [Message("user", ["hello"])]

    recovered = build_recovery_state(
        live_state,
        capture,
        mutation_tracker=_MutationTracker(),
        runtime_meta=SessionRuntimeMetadata(total_session_tokens=42),
        user_text="hello",
        user_contents=None,
        user_created_at="2026-01-01T00:00:00+00:00",
    )

    assert live_state["messages"] == []
    assert recovered is not None
    messages = recovered["messages"]
    assert [message.role for message in messages] == ["user", "assistant", "assistant"]
    assert messages[0].text == "hello"
    assert _marker_kind(messages[1]) == HistoryMarkerKind.INTERRUPTED
    assert messages[1].additional_properties["_interrupted_by"] == ""
    assert (
        messages[1].additional_properties[HistoryMarkerKind.STATUS_CODE_KEY] == HistoryMarkerKind.STATUS_SESSION_CLOSED
    )
    assert messages[1].text.encode("utf-8") == b"Session closed before the turn finished"
    assert _marker_kind(messages[2]) == HistoryMarkerKind.TURN
    assert recovered["chrys_mutations"] == {"turns": [{"turn_id": 7}]}
    assert recovered["total_session_tokens"] == 42


def test_checkpoint_preserves_first_prompt_from_empty_new_session_state() -> None:
    live_state: dict[str, object] = {}
    capture = LoopRecorder()
    capture._initial_count = 1
    capture._captured = [Message("user", ["first prompt"])]

    recovered = build_recovery_state(
        live_state,
        capture,
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="first prompt",
        user_contents=None,
        user_created_at="2026-01-01T00:00:00+00:00",
    )

    assert live_state == {}
    assert recovered is not None
    messages = recovered["messages"]
    assert [message.role for message in messages] == ["user", "assistant", "assistant"]
    assert messages[0].text == "first prompt"
    assert _marker_kind(messages[1]) == HistoryMarkerKind.INTERRUPTED
    assert _marker_kind(messages[2]) == HistoryMarkerKind.TURN


def test_checkpoint_preserves_completed_tool_batch() -> None:
    user = Message("user", ["do work"])
    assistant = Message("assistant", [Content.from_function_call("call-1", "read_file", arguments={})])
    tool = Message("tool", [Content.from_function_result("call-1", result="done")])
    live_state = {"messages": [user], "compressed_msgs": [], "turn_counter": 0}
    capture = LoopRecorder()
    capture._initial_count = 1
    capture._captured = [user, assistant, tool]

    recovered = build_recovery_state(
        live_state,
        capture,
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text=None,
        user_contents=None,
        user_created_at=None,
    )

    assert recovered is not None
    messages = recovered["messages"]
    assert [message.role for message in messages] == ["user", "assistant", "tool", "assistant", "assistant"]
    assert messages[1].contents[0].type == "function_call"
    assert messages[2].contents[0].type == "function_result"
    assert _marker_kind(messages[-2]) == HistoryMarkerKind.INTERRUPTED
    assert _marker_kind(messages[-1]) == HistoryMarkerKind.TURN


def test_checkpoint_trim_does_not_mutate_captured_loop_messages() -> None:
    user = Message("user", ["do work"])
    assistant = Message(
        "assistant",
        [
            Content.from_function_call("call-1", "read_file", arguments={}),
            Content.from_function_call("call-2", "write_file", arguments={}),
        ],
    )
    tool = Message("tool", [Content.from_function_result("call-1", result="done")])
    live_state = {"messages": [user], "compressed_msgs": [], "turn_counter": 0}
    capture = LoopRecorder()
    capture._initial_count = 1
    capture._captured = [user, assistant, tool]

    recovered = build_recovery_state(
        live_state,
        capture,
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text=None,
        user_contents=None,
        user_created_at=None,
    )

    original_call_ids = [content.call_id for content in assistant.contents if content.type == "function_call"]
    assert original_call_ids == ["call-1", "call-2"]
    assert recovered is not None
    recovered_call_ids = [
        content.call_id for content in recovered["messages"][1].contents if content.type == "function_call"
    ]
    assert recovered_call_ids == ["call-1"]


def test_bare_resume_checkpoint_does_not_synthesize_empty_user_message() -> None:
    original = Message("user", ["original prompt"])
    live_state = {"messages": [original], "compressed_msgs": [], "turn_counter": 0}
    capture = LoopRecorder()
    capture._initial_count = 1
    capture._captured = [original]

    recovered = build_recovery_state(
        live_state,
        capture,
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="",
        user_contents=None,
        user_created_at=None,
    )

    assert recovered is not None
    user_messages = [message for message in recovered["messages"] if message.role == "user"]
    assert len(user_messages) == 1
    assert user_messages[0].text == "original prompt"


def test_empty_resume_checkpoint_keeps_new_exchange_after_retained_pass_work() -> None:
    user = Message("user", ["original prompt"])
    first_call = Message("assistant", [Content.from_function_call("call-1", "read_file", arguments={})])
    first_result = Message("tool", [Content.from_function_result("call-1", result="first")])
    second_call = Message("assistant", [Content.from_function_call("call-2", "read_file", arguments={})])
    second_result = Message("tool", [Content.from_function_result("call-2", result="second")])
    live_state = {
        "messages": [user, first_call, first_result],
        "compressed_msgs": [],
        "turn_counter": 0,
    }
    capture = LoopRecorder()
    capture._initial_count = 0
    capture._captured = [second_call, second_result]

    recovered = build_recovery_state(
        live_state,
        capture,
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="",
        user_contents=None,
        user_created_at=None,
        insert_index=3,
    )

    assert recovered is not None
    exchanges = [
        content.call_id
        for message in recovered["messages"]
        for content in message.contents
        if content.type == "function_call"
    ]
    assert exchanges == ["call-1", "call-2"]


def test_checkpoint_embeds_last_words() -> None:
    """The Phase 4 note travels with the sidecar — crash recovery of a
    compacted turn must not resume from an amputated transcript."""
    live_state = {"messages": [Message("user", ["hello"])], "compressed_msgs": [], "turn_counter": 0}
    capture = LoopRecorder()
    capture._initial_count = 1
    capture._captured = [Message("user", ["hello"])]

    recovered = build_recovery_state(
        live_state,
        capture,
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="hello",
        user_contents=None,
        user_created_at=None,
        last_words="[LAST_WORDS] phase 4 progress note",
    )

    assert recovered is not None
    assert recovered["last_words"] == "[LAST_WORDS] phase 4 progress note"
    assert "last_words" not in live_state


def test_checkpoint_embeds_manifest_and_breaker_without_mutating_live_state() -> None:
    live_state = {"messages": [Message("user", ["hello"])], "compressed_msgs": [], "turn_counter": 0}
    capture = LoopRecorder()
    capture._initial_count = 1
    capture._captured = [Message("user", ["hello"])]
    manifest = [{"record_id": "r1", "relative_path": "compactions/dropped/turn001/a.md"}]
    breaker = {
        "version": 1,
        "attempts": 2,
        "consecutive_no_progress": 1,
        "tail_override": True,
        "disabled": False,
        "side_call_tokens": 456,
    }

    recovered = build_recovery_state(
        live_state,
        capture,
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="hello",
        user_contents=None,
        user_created_at=None,
        last_words_manifest=manifest,
        last_words_breaker=breaker,
        catalog_pointer_record_count=0,
    )

    assert recovered is not None
    assert recovered["last_words_manifest"] == manifest
    assert recovered["last_words_breaker"] == breaker
    assert recovered[CATALOG_POINTER_RECORD_COUNT_STATE_KEY] == 0
    assert "last_words_manifest" not in live_state
    assert "last_words_breaker" not in live_state
    assert CATALOG_POINTER_RECORD_COUNT_STATE_KEY not in live_state
    assert recovered["last_words_manifest"] is not manifest
    assert recovered["last_words_breaker"] is not breaker


def test_checkpoint_embeds_todos() -> None:
    """The session todo list travels with the sidecar so crash recovery
    restores the tracker alongside the transcript."""
    live_state = {"messages": [Message("user", ["hello"])], "compressed_msgs": [], "turn_counter": 0}
    capture = LoopRecorder()
    capture._initial_count = 1
    capture._captured = [Message("user", ["hello"])]

    todos = [{"content": "recover me", "status": "in_progress", "active_form": "Recovering"}]
    recovered = build_recovery_state(
        live_state,
        capture,
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="hello",
        user_contents=None,
        user_created_at=None,
        todos=todos,
    )

    assert recovered is not None
    assert recovered["chrys_todos"] == todos
    assert "chrys_todos" not in live_state


def test_checkpoint_drops_stale_todos_when_tracker_empty() -> None:
    """A stale ``chrys_todos`` left in live state by an earlier save is not
    carried into a sidecar for a session whose tracker is now empty."""
    live_state = {
        "messages": [Message("user", ["hello"])],
        "compressed_msgs": [],
        "turn_counter": 0,
        "chrys_todos": [{"content": "stale", "status": "pending", "active_form": ""}],
    }
    capture = LoopRecorder()
    capture._initial_count = 1
    capture._captured = [Message("user", ["hello"])]

    recovered = build_recovery_state(
        live_state,
        capture,
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="hello",
        user_contents=None,
        user_created_at=None,
        todos=[],
    )

    assert recovered is not None
    assert "chrys_todos" not in recovered
    assert live_state["chrys_todos"] == [{"content": "stale", "status": "pending", "active_form": ""}]


def test_checkpoint_drops_stale_last_words_when_note_cleared() -> None:
    """A stale note left in live state by an earlier save is not carried
    into a sidecar for a turn that has no note."""
    live_state = {
        "messages": [Message("user", ["hello"])],
        "compressed_msgs": [],
        "turn_counter": 0,
        "last_words": "[stale note from a previous turn]",
    }
    capture = LoopRecorder()
    capture._initial_count = 1
    capture._captured = [Message("user", ["hello"])]

    recovered = build_recovery_state(
        live_state,
        capture,
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="hello",
        user_contents=None,
        user_created_at=None,
        last_words=None,
    )

    assert recovered is not None
    assert "last_words" not in recovered
    assert live_state["last_words"] == "[stale note from a previous turn]"


# ---------------------------------------------------------------------------
# Kind threading — recovery re-creates the input with the right flags (§2.5)
# ---------------------------------------------------------------------------


def _injected_flag(message: Message) -> object:
    return message.additional_properties.get(HistoryMarkerKind.INJECTED_KEY)


def _recorder(*captured: Message) -> LoopRecorder:
    capture = LoopRecorder()
    capture._initial_count = 1
    capture._captured = list(captured)
    return capture


def test_checkpoint_recorder_dedup_preserves_consumed_injection_timestamp() -> None:
    opener = Message("user", ["task"])
    recorded = Message("user", ["note"])
    recorded.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    recorded.additional_properties[HistoryMarkerKind.INJECTION_ID_KEY] = "inj-1"
    live_state = {"messages": [opener], "compressed_msgs": [], "turn_counter": 0}

    recovered = build_recovery_state(
        live_state,
        _recorder(opener, recorded),
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text=None,
        user_contents=None,
        user_created_at=None,
        consumed_injections=[
            ConsumedInjection(
                text="note",
                anchor=InjectionAnchor.from_message(opener),
                created_at="2026-07-14T13:46:00+00:00",
                consumption_id="inj-1",
                wire_properties={HistoryMarkerKind.SYSTEM_REMINDERS_KEY: [{"kind": "event", "text": "hook"}]},
            )
        ],
    )

    assert recovered is not None
    notes = [message for message in recovered["messages"] if message.text == "note"]
    assert len(notes) == 1
    assert notes[0].additional_properties[MESSAGE_CREATED_AT_KEY] == "2026-07-14T13:46:00.000000+00:00"
    assert notes[0].additional_properties[HistoryMarkerKind.SYSTEM_REMINDERS_KEY] == [{"kind": "event", "text": "hook"}]


def test_checkpoint_replays_injection_that_carries_its_live_preparation_trace() -> None:
    """Replay reads a consumed injection and never copies it: the preparation
    trace it carries holds live trajectory state (a lock, settlement tasks)
    that cannot be deep-copied, so copying it would fail every checkpoint."""
    opener = Message("user", ["task"])
    live_state = {"messages": [opener], "compressed_msgs": [], "turn_counter": 0}
    trace = PreparationTrace.open(scope=PreparationScope.PRE_TURN, phase="input_admission", context=make_context())
    assert trace is not None
    injection = ConsumedInjection(
        text="note",
        anchor=InjectionAnchor.from_message(opener),
        consumption_id="inj-1",
        preparation=trace,
        wire_properties={HistoryMarkerKind.SYSTEM_REMINDERS_KEY: [{"kind": "event", "text": "hook"}]},
    )

    recovered = build_recovery_state(
        live_state,
        _recorder(opener),
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text=None,
        user_contents=None,
        user_created_at=None,
        consumed_injections=[injection],
    )

    assert recovered is not None
    notes = [message for message in recovered["messages"] if message.text == "note"]
    assert len(notes) == 1
    assert notes[0].additional_properties[HistoryMarkerKind.INJECTION_ID_KEY] == "inj-1"
    assert notes[0].additional_properties[HistoryMarkerKind.SYSTEM_REMINDERS_KEY] == [{"kind": "event", "text": "hook"}]
    assert injection.preparation is trace


def test_checkpoint_guided_retry_recovers_guidance_flagged_injected() -> None:
    """A crash during a guided retry restores the guidance as mid-turn input."""
    opener = Message("user", ["original task"])
    live_state = {"messages": [opener], "compressed_msgs": [], "turn_counter": 0}

    recovered = build_recovery_state(
        live_state,
        _recorder(opener),
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="also check env vars",
        user_contents=None,
        user_created_at=None,
        user_kind="injected",
    )

    assert recovered is not None
    guidance = [m for m in recovered["messages"] if m.role == "user" and m.text == "also check env vars"]
    assert len(guidance) == 1
    assert _injected_flag(guidance[0]) is True
    # The opener stays unflagged.
    opener_copies = [m for m in recovered["messages"] if m.role == "user" and m.text == "original task"]
    assert all(not _injected_flag(m) for m in opener_copies)


def test_checkpoint_fresh_run_opener_stays_unflagged() -> None:
    live_state = {"messages": [], "compressed_msgs": [], "turn_counter": 0}

    recovered = build_recovery_state(
        live_state,
        _recorder(Message("user", ["hello"])),
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="hello",
        user_contents=None,
        user_created_at=None,
    )

    assert recovered is not None
    user_messages = [m for m in recovered["messages"] if m.role == "user"]
    assert len(user_messages) == 1
    assert not _injected_flag(user_messages[0])


def test_checkpoint_recovered_opener_keeps_the_reminders_it_was_sent_with() -> None:
    """A crash before the opener was stored recovers it with the record the
    request wrote, so the resumed turn re-renders the same reminders, and
    with the item id the trajectory already announced for it."""
    live_state = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    sent: dict[str, Any] = {
        HistoryMarkerKind.SYSTEM_REMINDERS_KEY: [
            {"kind": "turn", "text": "runtime"},
            {"kind": "catalog", "text": "skills"},
        ]
    }
    live_item_id = ensure_analytics_item_id(sent)

    recovered = build_recovery_state(
        live_state,
        _recorder(Message("user", ["hello"])),
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="hello",
        user_contents=None,
        user_created_at=None,
        user_reminder_source=sent,
    )

    assert recovered is not None
    (opener,) = [m for m in recovered["messages"] if m.role == "user"]
    assert read_analytics_item_id(opener.additional_properties) == live_item_id
    record = opener.additional_properties[HistoryMarkerKind.SYSTEM_REMINDERS_KEY]
    assert record == sent[HistoryMarkerKind.SYSTEM_REMINDERS_KEY]
    # The writer thread gets its own copy of the live record.
    assert record is not sent[HistoryMarkerKind.SYSTEM_REMINDERS_KEY]
    assert all(
        entry is not source for entry, source in zip(record, sent[HistoryMarkerKind.SYSTEM_REMINDERS_KEY], strict=True)
    )


def test_checkpoint_guidance_worded_like_opener_appends_second_flagged_message() -> None:
    """Kind-aware dedup: same-text guidance never dedups against the opener."""
    opener = Message("user", ["do the thing"])
    live_state = {"messages": [opener], "compressed_msgs": [], "turn_counter": 0}

    recovered = build_recovery_state(
        live_state,
        _recorder(opener),
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="do the thing",
        user_contents=None,
        user_created_at=None,
        user_kind="injected",
    )

    assert recovered is not None
    same_text = [m for m in recovered["messages"] if m.role == "user" and m.text == "do the thing"]
    assert len(same_text) == 2
    assert [bool(_injected_flag(m)) for m in same_text] == [False, True]


def test_checkpoint_already_persisted_flagged_guidance_dedups() -> None:
    opener = Message("user", ["task"])
    guidance = Message("user", ["my note"])
    guidance.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    live_state = {"messages": [opener, guidance], "compressed_msgs": [], "turn_counter": 0}

    recovered = build_recovery_state(
        live_state,
        _recorder(opener),
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="my note",
        user_contents=None,
        user_created_at=None,
        user_kind="injected",
    )

    assert recovered is not None
    notes = [m for m in recovered["messages"] if m.role == "user" and m.text == "my note"]
    assert len(notes) == 1


# ---------------------------------------------------------------------------
# Consumed-injection mirror replay (§2.5 crash durability)
# ---------------------------------------------------------------------------


def _consumed(text: str, anchor_msg: Message | None = None, consumption_id: str | None = None) -> ConsumedInjection:
    anchor = InjectionAnchor.from_message(anchor_msg) if anchor_msg is not None else InjectionAnchor()
    return ConsumedInjection(text=text, anchor=anchor, consumption_id=consumption_id)


def test_checkpoint_replays_crash_dropped_consumed_injection_flagged() -> None:
    """An injection consumed mid-run but not yet finalized survives in the sidecar."""
    opener = Message("user", ["task"])
    live_state = {"messages": [opener], "compressed_msgs": [], "turn_counter": 0}

    recovered = build_recovery_state(
        live_state,
        _recorder(opener),
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="task",
        user_contents=None,
        user_created_at=None,
        consumed_injections=[_consumed("mid-run note")],
    )

    assert recovered is not None
    notes = [m for m in recovered["messages"] if m.role == "user" and m.text == "mid-run note"]
    assert len(notes) == 1
    assert _injected_flag(notes[0]) is True
    assert live_state["messages"] == [opener]


def test_checkpoint_consumed_injection_already_persisted_no_duplicate() -> None:
    opener = Message("user", ["task"])
    persisted = Message("user", ["mid-run note"])
    persisted.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    live_state = {"messages": [opener, persisted], "compressed_msgs": [], "turn_counter": 0}

    recovered = build_recovery_state(
        live_state,
        _recorder(opener),
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="task",
        user_contents=None,
        user_created_at=None,
        consumed_injections=[_consumed("mid-run note")],
    )

    assert recovered is not None
    notes = [m for m in recovered["messages"] if m.role == "user" and m.text == "mid-run note"]
    assert len(notes) == 1


def test_checkpoint_new_same_text_injection_not_suppressed_by_earlier_one() -> None:
    """Crash-recovery must not drop a new injection that repeats an old one's text.

    Restored resumed-turn shape: the prior run already persisted an injected
    "same note" (own ``_injection_id``); the user injects the same text again
    and Chrys crashes before finalization.  The checkpoint replay dedups by
    identity, so the second note lands instead of being text-matched away.
    """
    opener = Message("user", ["task"])
    earlier = Message("user", ["same note"])
    earlier.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    earlier.additional_properties[HistoryMarkerKind.INJECTION_ID_KEY] = "id-old"
    done = Message("assistant", ["done"])
    live_state = {"messages": [opener, earlier, done], "compressed_msgs": [], "turn_counter": 0}

    recovered = build_recovery_state(
        live_state,
        _recorder(opener),
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="task",
        user_contents=None,
        user_created_at=None,
        consumed_injections=[_consumed("same note", done, consumption_id="id-new")],
    )

    assert recovered is not None
    notes = [m for m in recovered["messages"] if m.role == "user" and m.text == "same note"]
    assert len(notes) == 2
    assert [m.additional_properties.get(HistoryMarkerKind.INJECTION_ID_KEY) for m in notes] == [
        "id-old",
        "id-new",
    ]
    assert all(_injected_flag(m) for m in notes)


def test_checkpoint_multiple_consumed_injections_restore_in_consumption_order() -> None:
    opener = Message("user", ["task"])
    live_state = {"messages": [opener], "compressed_msgs": [], "turn_counter": 0}

    recovered = build_recovery_state(
        live_state,
        _recorder(opener),
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="task",
        user_contents=None,
        user_created_at=None,
        consumed_injections=[_consumed("first", opener), _consumed("second", opener)],
    )

    assert recovered is not None
    user_texts = [m.text for m in recovered["messages"] if m.role == "user"]
    assert user_texts == ["task", "first", "second"]


def test_checkpoint_anchor_miss_injection_lands_before_terminal_markers() -> None:
    """Anchor-miss append still precedes INTERRUPTED+TURN (pre-shaping window).

    A trailing user message AFTER the markers would hide the status marker
    from ``trailing_status_marker`` and leak into the next turn region.
    """
    opener = Message("user", ["task"])
    live_state = {"messages": [opener], "compressed_msgs": [], "turn_counter": 0}

    recovered = build_recovery_state(
        live_state,
        _recorder(opener),
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text="task",
        user_contents=None,
        user_created_at=None,
        consumed_injections=[_consumed("orphaned note")],
    )

    assert recovered is not None
    messages = recovered["messages"]
    tail_kinds = [_marker_kind(m) for m in messages[-2:]]
    assert tail_kinds == [HistoryMarkerKind.INTERRUPTED, HistoryMarkerKind.TURN]
    assert messages[-3].role == "user"
    assert messages[-3].text == "orphaned note"

    manager = SessionHistoryManager()
    manager.bind(recovered)
    marker = manager.trailing_status_marker()
    assert marker is not None
    assert marker[0] == HistoryMarkerKind.INTERRUPTED


def test_checkpoint_stamps_legacy_items_and_preserves_existing_occurrence_ids() -> None:
    from chrys.foundation.trajectory.metadata import read_analytics_item_id
    from chrys.service.session.history import stamp_history_item_ids

    retained = Message("user", ["retained"], message_id="msg_1")
    legacy = Message("user", ["legacy"], message_id="msg_1")
    stamp_history_item_ids({"messages": [retained]})
    retained_id = read_analytics_item_id(retained.additional_properties)
    live = {"messages": [retained, legacy], "compressed_msgs": [], "turn_counter": 0}
    recovered = build_recovery_state(
        live,
        LoopRecorder(),
        mutation_tracker=None,
        runtime_meta=SessionRuntimeMetadata(),
        user_text=None,
        user_contents=None,
        user_created_at=None,
    )
    assert recovered is not None
    messages = recovered["messages"]
    assert read_analytics_item_id(messages[0].additional_properties) == retained_id
    legacy_id = read_analytics_item_id(messages[1].additional_properties)
    assert legacy_id
    assert legacy_id != retained_id
    assert read_analytics_item_id(legacy.additional_properties) is None
    assert [message.message_id for message in messages[:2]] == ["msg_1", "msg_1"]
