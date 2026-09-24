# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Unit tests for _trim_to_last_complete_tool_results logic.

Tests the engine's ability to trim incomplete tool call exchanges
from the conversation history on interrupt.
"""

from __future__ import annotations

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.foundation.tool_result_metadata import TOOL_ERROR_KIND_METADATA_KEY
from chrys.kernel import ChatResponse, Content, Message
from chrys.service.session.history import SessionHistoryManager
from tests.support.transcript_invariants import assert_transcript_invariants


def _user(text: str = "go") -> Message:
    return Message("user", [text])


def _call(call_id: object, name: str = "echo", *, informational: bool = False) -> Content:
    content = Content.from_function_call(call_id, name, arguments={})
    if informational:
        content.informational_only = True
    return content


def _result(call_id: object, *, error_kind: str | None = None) -> Content:
    content = Content.from_function_result(call_id, result=f"result_{call_id!r}")
    if error_kind is not None:
        content.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] = error_kind
    return content


def _assistant(*contents: Content | str) -> Message:
    return Message("assistant", list(contents))


def _tool(*contents: Content) -> Message:
    return Message("tool", list(contents))


def _marker(*contents: Content | str, kind: str = HistoryMarkerKind.TURN) -> Message:
    message = Message("assistant", list(contents) or ["marker"])
    message.additional_properties[HistoryMarkerKind.KEY] = kind
    return message


def _trimmed(messages: list[Message]) -> list[Message]:
    history = SessionHistoryManager()
    history.bind({"messages": messages})
    history.trim_to_last_complete_tool_results()
    return history.messages


def _content_ids(message: Message) -> list[tuple[str, object]]:
    return [(c.type, c.call_id) for c in message.contents if getattr(c, "type", None) is not None]


class TestTrimToLastCompleteToolResults:
    """Test the trim logic in isolation by directly manipulating messages."""

    def _make_user_msg(self, text: str) -> Message:
        return Message("user", [text])

    def _make_assistant_text(self, text: str) -> Message:
        return Message("assistant", [text])

    def _make_assistant_with_calls(self, call_ids: list[str]) -> Message:
        contents = [Content.from_function_call(cid, f"tool_{cid}", arguments={}) for cid in call_ids]
        return Message("assistant", contents)

    def _make_tool_result(self, call_id: str) -> Message:
        return Message("tool", [Content.from_function_result(call_id, result=f"result_{call_id}")])

    def _immediate_tool_result_call_ids(self, messages: list[Message], assistant_idx: int) -> set[str]:
        result_call_ids: set[str] = set()
        for msg in messages[assistant_idx + 1 :]:
            if msg.role != "tool":
                break
            for c in msg.contents:
                if getattr(c, "type", "") == "function_result":
                    cid = getattr(c, "call_id", None)
                    if cid:
                        result_call_ids.add(cid)
        return result_call_ids

    def _trim(self, messages: list[Message]) -> list[Message]:
        """Simulate _trim_to_last_complete_tool_results on a message list."""
        # Walk backwards
        i = len(messages) - 1
        while i >= 0:
            msg = messages[i]
            if msg.role != "assistant":
                i -= 1
                continue

            call_ids_in_msg = []
            for c in msg.contents:
                if getattr(c, "type", "") == "function_call":
                    cid = getattr(c, "call_id", None)
                    if cid:
                        call_ids_in_msg.append(cid)

            if not call_ids_in_msg:
                break

            result_call_ids = self._immediate_tool_result_call_ids(messages, i)
            matched = [cid for cid in call_ids_in_msg if cid in result_call_ids]
            unmatched = [cid for cid in call_ids_in_msg if cid not in result_call_ids]

            if not unmatched:
                break

            if matched:
                unmatched_set = set(unmatched)
                msg.contents = [
                    c
                    for c in msg.contents
                    if getattr(c, "type", "") != "function_call" or getattr(c, "call_id", None) not in unmatched_set
                ]
                break
            messages.pop(i)
            i -= 1

        return messages

    def test_all_complete_no_change(self):
        """If all tool calls have results, nothing is trimmed."""
        msgs = [
            self._make_user_msg("do X"),
            self._make_assistant_with_calls(["c1", "c2"]),
            self._make_tool_result("c1"),
            self._make_tool_result("c2"),
        ]
        result = self._trim(msgs)
        assert len(result) == 4

    def test_one_unexecuted_removed(self):
        """If 2 calls requested but only 1 has result, unexecuted call is removed."""
        msgs = [
            self._make_user_msg("do X"),
            self._make_assistant_with_calls(["c1", "c2"]),
            self._make_tool_result("c1"),
            # c2 never executed
        ]
        result = self._trim(msgs)
        assert len(result) == 3  # user + assistant(c1 only) + tool_result(c1)
        # Assistant should only have c1
        assistant = result[1]
        call_ids = [
            getattr(c, "call_id", None) for c in assistant.contents if getattr(c, "type", "") == "function_call"
        ]
        assert call_ids == ["c1"]

    def test_no_results_removes_assistant(self):
        """If no tool calls have results, the entire assistant message is removed."""
        msgs = [
            self._make_user_msg("do X"),
            self._make_assistant_with_calls(["c1", "c2"]),
            # neither executed
        ]
        result = self._trim(msgs)
        assert len(result) == 1  # only user msg remains

    def test_text_only_assistant_preserved(self):
        """Assistant messages with only text (no function_calls) are preserved."""
        msgs = [
            self._make_user_msg("do X"),
            self._make_assistant_text("thinking..."),
        ]
        result = self._trim(msgs)
        assert len(result) == 2

    def test_multiple_exchanges(self):
        """Multiple complete exchanges + one incomplete at end."""
        msgs = [
            self._make_user_msg("do X"),
            self._make_assistant_with_calls(["c1"]),
            self._make_tool_result("c1"),
            self._make_assistant_with_calls(["c2", "c3"]),
            self._make_tool_result("c2"),
            # c3 not executed
        ]
        result = self._trim(msgs)
        # c3 should be removed from the last assistant msg
        assert len(result) == 5  # user + assistant(c1) + result(c1) + assistant(c2) + result(c2)
        last_assistant = result[3]
        call_ids = [
            getattr(c, "call_id", None) for c in last_assistant.contents if getattr(c, "type", "") == "function_call"
        ]
        assert call_ids == ["c2"]

    def test_previous_duplicate_call_id_does_not_complete_current_call(self):
        """An older tool result with a reused call_id must not keep a dangling current call."""
        msgs = [
            self._make_user_msg("old"),
            self._make_assistant_with_calls(["shared"]),
            self._make_tool_result("shared"),
            self._make_user_msg("current"),
            self._make_assistant_with_calls(["shared"]),
            # Current call never produced a result.
        ]
        result = self._trim(msgs)
        assert result == msgs[:4]

    def test_manager_previous_duplicate_call_id_does_not_complete_current_call(self):
        """SessionHistoryManager scopes result matching to the current exchange."""
        msgs = [
            self._make_user_msg("old"),
            self._make_assistant_with_calls(["shared"]),
            self._make_tool_result("shared"),
            self._make_user_msg("current"),
            self._make_assistant_with_calls(["shared"]),
        ]
        expected = msgs[:4]
        history = SessionHistoryManager()
        history.bind({"messages": msgs})

        history.trim_to_last_complete_tool_results()

        assert history.messages == expected

    def test_awaiting_marker_shields_dangling_sub_agent_call(self):
        """Checkpoint trim must leave a paused sub-agent call for reload repair."""
        user_message = self._make_user_msg("delegate")
        call_message = Message(
            "assistant",
            [Content.from_function_call("sub-1", "Explore", arguments={})],
        )
        awaiting_marker = Message("assistant", ["Awaiting 1 sub-agent"])
        awaiting_marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.AWAITING_SUB_AGENTS
        history = SessionHistoryManager()
        history.bind({"messages": [user_message, call_message, awaiting_marker]})

        history.trim_to_last_complete_tool_results()

        assert len(history.messages) == 3
        assert history.messages[0] is user_message
        assert history.messages[1] is call_message
        assert history.messages[2] is awaiting_marker

    def test_result_only_assistant_answer_survives_trim(self):
        """A result-only assistant starts the output block for the preceding call."""
        call = Content.from_function_call("c1", "echo", arguments={})
        call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
        result = Content.from_function_result("c1", result="done")
        user_message = self._make_user_msg("run")
        call_message = Message("assistant", [call])
        result_message = Message("assistant", [result])
        messages = [user_message, call_message, result_message]
        history = SessionHistoryManager()
        history.bind({"messages": messages})
        assert_transcript_invariants(ChatResponse(messages=list(messages)))

        history.trim_to_last_complete_tool_results()

        assert history.messages == [user_message, call_message, result_message]
        assert_transcript_invariants(ChatResponse(messages=list(history.messages)))

    def test_empty_messages(self):
        """Empty message list doesn't crash."""
        result = self._trim([])
        assert result == []

    def test_only_user_message(self):
        """User message alone is preserved."""
        msgs = [self._make_user_msg("hi")]
        result = self._trim(msgs)
        assert len(result) == 1


class TestTrimExchangePreservation:
    """Preservation pins: behaviors trim must keep verbatim through the exchange migration."""

    def test_marker_message_carrying_results_shields_dangling_call(self):
        """Any trailing history marker halts trim, even one carrying tool results."""
        messages = [_user(), _assistant(_call("c1")), _marker(_result("c1"))]
        expected = list(messages)
        assert _trimmed(messages) == expected

    def test_marker_message_before_tool_block_shields_everything(self):
        """A marker between the call and a real tool block still halts trim."""
        messages = [
            _user(),
            _assistant(_call("c1")),
            _marker(_result("c1")),
            _tool(_result("c2")),
        ]
        expected = list(messages)
        assert _trimmed(messages) == expected

    def test_informational_only_exchange_halts_trim(self):
        """An exchange whose only call is informational stops the backward walk."""
        earlier_dangling = _assistant(_call("c1"))
        messages = [_user(), earlier_dangling, _user("again"), _assistant(_call("h1", informational=True))]
        expected = list(messages)
        assert _trimmed(messages) == expected

    def test_text_only_run_head_halts_trim_after_sibling_pop(self):
        """A text-only run head stops the walk once its dangling sibling is popped."""
        earlier_dangling = _assistant(_call("c1"))
        text_head = _assistant("thinking aloud")
        messages = [_user(), earlier_dangling, _user("again"), text_head, _assistant(_call("c2"))]
        trimmed = _trimmed(messages)
        assert trimmed == [messages[0], earlier_dangling, messages[2], text_head]

    def test_none_id_call_answered_by_idless_prevalidation_failure_kept(self):
        """An id-less prevalidation failure in the block answers an id-less call."""
        messages = [
            _user(),
            _assistant(_call(None)),
            _tool(_result(None, error_kind="argument_parsing")),
        ]
        expected = list(messages)
        assert _trimmed(messages) == expected

    def test_empty_id_call_answered_by_empty_id_result_kept(self):
        """An empty-id call paired positionally with an empty-id result stays complete."""
        messages = [_user(), _assistant(_call("")), _tool(_result(""))]
        expected = list(messages)
        assert _trimmed(messages) == expected

    def test_falsy_numeric_call_with_falsy_result_kept(self):
        """A falsy non-string call id answered by a falsy non-string result stays put."""
        messages = [_user(), _assistant(_call(0)), _tool(_result(False))]
        expected = list(messages)
        assert _trimmed(messages) == expected

    def test_falsy_unhashable_call_with_falsy_unhashable_result_kept(self):
        """A falsy unhashable call id answered by a falsy unhashable result stays put."""
        messages = [_user(), _assistant(_call([])), _tool(_result({}))]
        expected = list(messages)
        assert _trimmed(messages) == expected


class TestTrimExchangeCompleteness:
    """Exchange-level completeness: every response sibling's calls are checked, embedded
    results and the id-less/malformed-id policies participate, and per-sibling deletion
    follows the same branches trim applies to a single message today."""

    def test_user_role_result_only_message_does_not_resolve_call(self):
        """A user-role message carrying only results is not part of the output block."""
        result_message = Message("user", [_result("c1")])
        messages = [_user(), _assistant(_call("c1")), result_message]
        trimmed = _trimmed(messages)
        assert trimmed == [messages[0], result_message]

    def test_dangling_call_in_earlier_sibling_is_trimmed(self):
        """Completeness is per sibling: a later sibling's answered call keeps only itself."""
        keep_assistant = _assistant(_call("c2"))
        keep_tool = _tool(_result("c2"))
        messages = [_user(), _assistant(_call("c1")), keep_assistant, keep_tool]
        trimmed = _trimmed(messages)
        assert trimmed == [messages[0], keep_assistant, keep_tool]

    def test_sibling_with_text_and_dangling_call_pops_whole(self):
        """A sibling whose calls are all unanswered pops entirely, text included."""
        keep_assistant = _assistant(_call("c2"))
        keep_tool = _tool(_result("c2"))
        messages = [_user(), _assistant("thinking", _call("c1")), keep_assistant, keep_tool]
        trimmed = _trimmed(messages)
        assert trimmed == [messages[0], keep_assistant, keep_tool]

    def test_all_unanswered_narrated_exchange_drops_the_atomic_assistant_set(self):
        user = _user()
        narrated = _assistant("I will inspect both files.", _call("c1"), _call("c2"))

        trimmed = _trimmed([user, narrated])

        assert trimmed == [user]

    def test_sibling_with_informational_call_trims_surgically(self):
        """A sibling keeping a hosted informational call loses only its unanswered call."""
        mixed = _assistant(_call("h1", informational=True), _call("c1"))
        keep_assistant = _assistant(_call("c2"))
        keep_tool = _tool(_result("c2"))
        messages = [_user(), mixed, keep_assistant, keep_tool]
        trimmed = _trimmed(messages)
        assert trimmed == [messages[0], mixed, keep_assistant, keep_tool]
        assert _content_ids(mixed) == [("function_call", "h1")]

    def test_partially_answered_sibling_trims_surgically(self):
        """A sibling with some answered calls loses only the unanswered ones."""
        mixed = _assistant(_call("c1"), _call("c1b"))
        keep_assistant = _assistant(_call("c2"))
        keep_tool = _tool(_result("c1"), _result("c2"))
        messages = [_user(), mixed, keep_assistant, keep_tool]
        trimmed = _trimmed(messages)
        assert trimmed == [messages[0], mixed, keep_assistant, keep_tool]
        assert _content_ids(mixed) == [("function_call", "c1")]

    def test_duplicate_id_sibling_calls_share_one_result(self):
        """Duplicate truthy ids are one invocation: a single result answers every sibling occurrence."""
        first = _assistant(_call("x"))
        second = _assistant(_call("x"))
        messages = [_user(), first, second, _tool(_result("x"))]
        expected = list(messages)
        assert _trimmed(messages) == expected

    def test_duplicate_id_calls_in_one_message_share_one_result(self):
        """Duplicate truthy ids within one message all count answered by the one result."""
        doubled = _assistant(_call("x"), _call("x"))
        messages = [_user(), doubled, _tool(_result("x"))]
        expected = list(messages)
        assert _trimmed(messages) == expected
        assert _content_ids(doubled) == [("function_call", "x"), ("function_call", "x")]

    def test_duplicate_id_surgical_trim_keeps_both_answered_occurrences(self):
        """Surgical trim drops only the unanswered id, never a duplicate of an answered one."""
        mixed = _assistant(_call("x"), _call("x"), _call("y"))
        messages = [_user(), mixed, _tool(_result("x"))]
        trimmed = _trimmed(messages)
        assert trimmed == [messages[0], mixed, messages[2]]
        assert _content_ids(mixed) == [("function_call", "x"), ("function_call", "x")]

    def test_none_id_call_without_idless_failure_is_deleted(self):
        """An id-less call with no id-less failure result anywhere fails closed."""
        messages = [_user(), _assistant(_call(None))]
        assert _trimmed(messages) == [messages[0]]

    def test_empty_id_call_without_empty_id_result_is_deleted(self):
        """An empty-id call with no positional empty-id result fails closed."""
        messages = [_user(), _assistant(_call(""))]
        assert _trimmed(messages) == [messages[0]]

    def test_none_id_sibling_deleted_alongside_complete_sibling(self):
        """An unanswered id-less call is trimmed even when its sibling is complete."""
        keep_assistant = _assistant(_call("c1"))
        keep_tool = _tool(_result("c1"))
        messages = [_user(), keep_assistant, _assistant(_call(None)), keep_tool]
        trimmed = _trimmed(messages)
        assert trimmed == [messages[0], keep_assistant, keep_tool]

    def test_embedded_result_completes_exchange(self):
        """A result embedded in the call's own message answers that call."""
        messages = [_user(), _assistant(_call("c1"), _result("c1"))]
        expected = list(messages)
        assert _trimmed(messages) == expected

    def test_embedded_result_keeps_message_while_dangling_call_removed(self):
        """Embedded answers keep the message alive; only the unanswered call goes."""
        mixed = _assistant(_call("c1"), _call("c2"), _result("c1"))
        messages = [_user(), mixed]
        trimmed = _trimmed(messages)
        assert trimmed == [messages[0], mixed]
        assert _content_ids(mixed) == [("function_call", "c1"), ("function_result", "c1")]

    def test_numeric_call_id_pairs_with_equal_digit_string_result(self):
        """Malformed ids pair by their string forms: int 7 matches result id "7"."""
        messages = [_user(), _assistant(_call(7)), _tool(_result("7"))]
        expected = list(messages)
        assert _trimmed(messages) == expected

    def test_separately_constructed_nan_ids_pair(self):
        """Two distinct NaN objects pair through their identical string forms."""
        messages = [_user(), _assistant(_call(float("nan"))), _tool(_result(float("nan")))]
        expected = list(messages)
        assert _trimmed(messages) == expected

    def test_falsy_numeric_call_unanswered_is_deleted(self):
        """A falsy non-string call id joins the empty-id stream and fails closed."""
        messages = [_user(), _assistant(_call(0))]
        assert _trimmed(messages) == [messages[0]]

    def test_int_call_id_does_not_pair_with_bool_result_id(self):
        """Python-equal ids of different types split by string form: 1 is not True."""
        stray_tool = _tool(_result(True))
        messages = [_user(), _assistant(_call(1)), stray_tool]
        assert _trimmed(messages) == [messages[0], stray_tool]

    def test_truthy_unhashable_call_id_processed_without_crash(self):
        """An unhashable call id is stringified and processed, never a TypeError."""
        messages = [_user(), _assistant(_call(["x"]))]
        assert _trimmed(messages) == [messages[0]]

    def test_truthy_unhashable_call_id_with_unrelated_result_processed(self):
        """An unhashable call id does not crash pairing against real result ids."""
        stray_tool = _tool(_result("z"))
        messages = [_user(), _assistant(_call(["x"])), stray_tool]
        assert _trimmed(messages) == [messages[0], stray_tool]

    def test_falsy_unhashable_call_unanswered_is_deleted(self):
        """A falsy unhashable call id joins the empty-id stream and fails closed."""
        messages = [_user(), _assistant(_call([]))]
        assert _trimmed(messages) == [messages[0]]
