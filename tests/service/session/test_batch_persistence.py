# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Keyed batch/text persistence invariants (turn-boundary plan §2.1.1).

``persist_batch_ids`` matches drained ``ToolBatchRecord``s to un-batched
assistant function-call messages by ``(provider_call_id, tool_name)`` —
never by position, never by ``tool_order``, never by chrys/UI id.
``persist_intermediate_texts`` assigns captured texts purely by
``batch_id`` lookup through the anchors ``persist_batch_ids`` returns.
For both, loss beats misattribution.
"""

from __future__ import annotations

import logging

import pytest

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.kernel import Content, Message
from chrys.service.agent_middleware.events.tool_events import ToolBatchRecord
from chrys.service.context.providers.history import PRE_OUTPUT_HISTORY_LEN_STATE_KEY
from chrys.service.session.history import SessionHistoryManager

_HISTORY_LOGGER = "chrys.service.session.history"


def _call_msg(*calls: tuple[str, str]) -> Message:
    """Assistant message with ``(call_id, tool_name)`` function calls."""
    return Message(
        "assistant",
        [Content.from_function_call(call_id=call_id, name=name, arguments={}) for call_id, name in calls],
    )


def _turn_marker() -> Message:
    marker = Message("assistant", [Content.from_text("")])
    marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    return marker


def _bind(messages: list[Message], **extra_state: object) -> SessionHistoryManager:
    history = SessionHistoryManager()
    history.bind({"messages": messages, **extra_state})
    return history


def _record(provider_call_id: str, tool_name: str, tool_order: int, batch_id: int) -> ToolBatchRecord:
    return ToolBatchRecord(
        provider_call_id=provider_call_id,
        tool_name=tool_name,
        tool_order=tool_order,
        batch_id=batch_id,
    )


def _batch_id(msg: Message) -> object:
    return msg.additional_properties.get("_batch_id")


class TestPersistBatchIds:
    def test_assigns_by_provider_call_id_and_tool_name(self) -> None:
        msg_a = _call_msg(("call_0", "echo"))
        msg_b = _call_msg(("call_1", "read_file"))
        history = _bind([Message("user", ["go"]), msg_a, msg_b])

        anchors = history.persist_batch_ids([_record("call_0", "echo", 0, 1), _record("call_1", "read_file", 1, 2)])

        assert _batch_id(msg_a) == 1
        assert _batch_id(msg_b) == 2
        assert anchors == {1: msg_a, 2: msg_b}

    def test_record_without_provider_id_is_dropped_never_order_matched(self, caplog: pytest.LogCaptureFixture) -> None:
        """A record with no provider id assigns nothing, even with a slot at its tool_order."""
        msg = _call_msg(("call_0", "echo"))
        history = _bind([Message("user", ["go"]), msg])
        caplog.set_level(logging.DEBUG, logger=_HISTORY_LOGGER)

        anchors = history.persist_batch_ids([_record("", "echo", 0, 1)])

        assert "_batch_id" not in msg.additional_properties
        assert anchors == {}
        assert "dropping record without provider_call_id" in caplog.text

    def test_chrys_uuid_id_never_matches(self) -> None:
        """The chrys/UI id is never a match key: a uuid record assigns nothing."""
        msg = _call_msg(("call_0", "echo"))
        history = _bind([Message("user", ["go"]), msg])

        anchors = history.persist_batch_ids([_record("a1b2c3d4e5f6", "echo", 0, 1)])

        assert "_batch_id" not in msg.additional_properties
        assert anchors == {}

    def test_stale_straggler_never_consumes_a_current_run_record(self) -> None:
        """A pre-run un-batched slot under the same key loses to the newest slot."""
        stale = _call_msg(("call_0", "echo"))
        current = _call_msg(("call_0", "echo"))
        history = _bind([Message("user", ["go"]), stale, Message("tool", ["ok"]), current])

        anchors = history.persist_batch_ids([_record("call_0", "echo", 0, 5)])

        assert "_batch_id" not in stale.additional_properties
        assert _batch_id(current) == 5
        assert anchors == {5: current}

    def test_tool_order_zero_record_does_not_hit_stale_slot(self) -> None:
        """Per-run tool_order=0 + a stale differently-keyed slot: no order fallback fires."""
        stale = _call_msg(("call_9", "echo"))
        current = _call_msg(("call_0", "echo"))
        history = _bind([Message("user", ["go"]), stale, current])

        anchors = history.persist_batch_ids([_record("call_0", "echo", 0, 3)])

        assert "_batch_id" not in stale.additional_properties
        assert _batch_id(current) == 3
        assert anchors == {3: current}

    def test_duplicate_keys_assign_chronologically_never_reversed(self) -> None:
        """Repeated (provider id, tool) keys bucket to the NEWEST N slots in order."""
        stale = _call_msg(("call_0", "echo"))
        first = _call_msg(("call_0", "echo"))
        second = _call_msg(("call_0", "echo"))
        history = _bind([Message("user", ["go"]), stale, first, second])

        anchors = history.persist_batch_ids([_record("call_0", "echo", 0, 3), _record("call_0", "echo", 1, 7)])

        assert "_batch_id" not in stale.additional_properties
        assert _batch_id(first) == 3
        assert _batch_id(second) == 7
        assert anchors == {3: first, 7: second}

    def test_multi_call_message_two_tools_single_batch_id(self) -> None:
        msg = _call_msg(("call_0", "echo"), ("call_1", "read_file"))
        history = _bind([Message("user", ["go"]), msg])

        anchors = history.persist_batch_ids([_record("call_0", "echo", 0, 4), _record("call_1", "read_file", 1, 4)])

        assert _batch_id(msg) == 4
        assert anchors == {4: msg}

    def test_multi_call_message_duplicate_same_tool_calls(self) -> None:
        """Identical (provider id, tool) twice in one message: both records resolve there."""
        msg = _call_msg(("call_0", "echo"), ("call_0", "echo"))
        history = _bind([Message("user", ["go"]), msg])

        anchors = history.persist_batch_ids([_record("call_0", "echo", 0, 4), _record("call_0", "echo", 1, 4)])

        assert _batch_id(msg) == 4
        assert anchors == {4: msg}

    def test_later_same_message_record_with_mismatched_id_never_overwrites(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        msg = _call_msg(("call_0", "echo"), ("call_1", "read_file"))
        history = _bind([Message("user", ["go"]), msg])
        caplog.set_level(logging.DEBUG, logger=_HISTORY_LOGGER)

        anchors = history.persist_batch_ids([_record("call_0", "echo", 0, 4), _record("call_1", "read_file", 1, 9)])

        assert _batch_id(msg) == 4
        assert anchors == {4: msg}
        assert "batch id mismatch" in caplog.text

    def test_out_of_region_slots_are_ignored(self) -> None:
        """No global scan: an un-batched slot before the last turn marker never matches."""
        old = _call_msg(("call_0", "echo"))
        history = _bind([Message("user", ["old"]), old, _turn_marker(), Message("user", ["new"])])

        anchors = history.persist_batch_ids([_record("call_0", "echo", 0, 2)])

        assert "_batch_id" not in old.additional_properties
        assert anchors == {}

    def test_interrupted_pass_records_still_assign(self) -> None:
        """Records are pre-execution: a call with no tool result still gets its batch id."""
        msg = _call_msg(("call_0", "zsh"))
        # No tool-result message follows — the pass was interrupted mid-call.
        history = _bind([Message("user", ["go"]), msg])

        anchors = history.persist_batch_ids([_record("call_0", "zsh", 0, 1)])

        assert _batch_id(msg) == 1
        assert anchors == {1: msg}


class TestPersistIntermediateTexts:
    def test_same_pass_pre_injection_batches_still_receive_texts(self) -> None:
        msg_a = _call_msg(("call_0", "echo"))
        injected = Message("user", ["mid-turn note"])
        injected.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
        msg_b = _call_msg(("call_1", "read_file"))
        history = _bind([Message("user", ["go"]), msg_a, injected, msg_b])

        anchors = history.persist_batch_ids([_record("call_0", "echo", 0, 1), _record("call_1", "read_file", 1, 2)])
        history.persist_intermediate_texts({1: "before injection", 2: "after injection"}, anchors)

        assert msg_a.additional_properties["_intermediate_text"] == "before injection"
        assert msg_b.additional_properties["_intermediate_text"] == "after injection"

    def test_pre_mid_pass_compression_batches_still_receive_texts(self) -> None:
        """No strict numeric floor: a batch predating the PRE_OUTPUT floor still matches."""
        msg_a = _call_msg(("call_0", "echo"))
        msg_b = _call_msg(("call_1", "read_file"))
        messages = [Message("user", ["go"]), msg_a, msg_b]
        # Mid-pass compression left the floor AFTER msg_a — identity matching
        # must not care.
        history = _bind(messages, **{PRE_OUTPUT_HISTORY_LEN_STATE_KEY: 2})

        anchors = history.persist_batch_ids([_record("call_0", "echo", 0, 1), _record("call_1", "read_file", 1, 2)])
        history.persist_intermediate_texts({1: "pre-compression", 2: "post-compression"}, anchors)

        assert msg_a.additional_properties["_intermediate_text"] == "pre-compression"
        assert msg_b.additional_properties["_intermediate_text"] == "post-compression"

    def test_pre_resume_batches_never_receive_fresh_texts(self) -> None:
        """A restored session's stale ``_batch_id`` colliding with a fresh id gets no text."""
        stale = _call_msg(("call_0", "echo"))
        stale.additional_properties["_batch_id"] = 1  # earlier process, counter restarted
        fresh = _call_msg(("call_0", "echo"))
        history = _bind([Message("user", ["go"]), stale, Message("tool", ["ok"]), fresh])

        anchors = history.persist_batch_ids([_record("call_0", "echo", 0, 1)])
        history.persist_intermediate_texts({1: "fresh text"}, anchors)

        assert "_intermediate_text" not in stale.additional_properties
        assert fresh.additional_properties["_intermediate_text"] == "fresh text"
        assert _batch_id(fresh) == 1

    def test_validation_failed_message_stays_unbatched_without_shifting(self, caplog: pytest.LogCaptureFixture) -> None:
        """All-failed message: un-batched, no positional shift, orphan text dropped."""
        failed = _call_msg(("call_0", "echo"))  # every call failed validation → no record
        dispatched = _call_msg(("call_1", "read_file"))
        history = _bind([Message("user", ["go"]), failed, dispatched])
        caplog.set_level(logging.DEBUG, logger=_HISTORY_LOGGER)

        anchors = history.persist_batch_ids([_record("call_1", "read_file", 0, 3)])
        history.persist_intermediate_texts({2: "orphan text", 3: "dispatched text"}, anchors)

        assert "_batch_id" not in failed.additional_properties
        assert "_intermediate_text" not in failed.additional_properties
        assert _batch_id(dispatched) == 3
        assert dispatched.additional_properties["_intermediate_text"] == "dispatched text"
        assert "dropping text for unmatched batch 2" in caplog.text

    def test_mixed_message_is_batched_by_its_dispatched_call(self) -> None:
        """One failed + one dispatched call in a message: the dispatched record batches it."""
        mixed = _call_msg(("call_0", "echo"), ("call_1", "read_file"))
        history = _bind([Message("user", ["go"]), mixed])

        # Only call_1 entered the middleware pipeline.
        anchors = history.persist_batch_ids([_record("call_1", "read_file", 0, 2)])

        assert _batch_id(mixed) == 2
        assert anchors == {2: mixed}

    def test_empty_text_writes_nothing(self) -> None:
        msg = _call_msg(("call_0", "echo"))
        history = _bind([Message("user", ["go"]), msg])

        anchors = history.persist_batch_ids([_record("call_0", "echo", 0, 1)])
        history.persist_intermediate_texts({1: ""}, anchors)

        assert "_intermediate_text" not in msg.additional_properties

    def test_text_already_in_message_contents_is_not_duplicated(self) -> None:
        msg = Message(
            "assistant",
            [
                Content.from_text("Thinking..."),
                Content.from_function_call(call_id="call_0", name="echo", arguments={}),
            ],
        )
        history = _bind([Message("user", ["go"]), msg])

        anchors = history.persist_batch_ids([_record("call_0", "echo", 0, 1)])
        history.persist_intermediate_texts({1: "Thinking..."}, anchors)

        assert "_intermediate_text" not in msg.additional_properties
