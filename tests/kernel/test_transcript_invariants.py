# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Failure-path pins for the transcript-invariant oracle itself.

The oracle (``tests.support.transcript_invariants``) guards the whole loop
suite silently: every loop test passes through it, so nothing would notice if
a sub-check were deleted, weakened, or skipped by a refactor — the suite
would simply stay green while the guard rotted into a no-op. These tests hold
each invariant to red cases (it must fire on the defect shape it exists for)
and green cases (it must stay quiet on the legitimate shapes it was
deliberately relaxed to admit), so any change to the oracle's strictness is a
visible test change, not a silent one.
"""

from __future__ import annotations

import pytest

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.kernel.types import ChatResponse, Content, Message
from tests.support.transcript_invariants import assert_transcript_invariants


def _call(call_id: str, *, order: int | None = None, informational: bool = False) -> Content:
    content = Content.from_function_call(
        call_id=call_id, name="echo", arguments={"text": call_id}, informational_only=informational
    )
    if order is not None:
        content.additional_properties[TOOL_INVOCATION_ORDER_KEY] = order
    return content


def _result(call_id: str) -> Content:
    return Content.from_function_result(call_id, result="ok")


def _mcp_call(call_id: str) -> Content:
    # Hosted MCP calls are informational_only by construction — never
    # dispatched, never stamped — but they still open pairing scope for
    # their assistant-role results.
    return Content.from_mcp_server_tool_call(call_id, "search")


def _mcp_result(call_id: str) -> Content:
    return Content.from_mcp_server_tool_result(call_id, output="ok")


def _image_call(image_id: str) -> Content:
    # Image generation pairs on image_id; its call_id is always None.
    return Content.from_image_generation_tool_call(image_id=image_id)


def _image_result(image_id: str) -> Content:
    return Content.from_image_generation_tool_result(image_id=image_id)


def _response(*messages: Message) -> ChatResponse:
    return ChatResponse(messages=list(messages))


class TestOrdinalIntegrity:
    """I1: stamped invocation orders form one consecutive increasing run."""

    def test_fires_on_unstamped_actionable_call(self) -> None:
        with pytest.raises(AssertionError, match=r"I1.*no invocation-order stamp"):
            assert_transcript_invariants(_response(Message("assistant", [_call("c1")])))

    def test_fires_on_ordinal_gap(self) -> None:
        transcript = _response(
            Message("assistant", [_call("c1", order=0)]),
            Message("tool", [_result("c1")]),
            Message("assistant", [_call("c2", order=2)]),
            Message("tool", [_result("c2")]),
        )
        with pytest.raises(AssertionError, match=r"I1.*consecutive increasing run"):
            assert_transcript_invariants(transcript)

    def test_fires_on_duplicate_ordinal(self) -> None:
        # Two calls sharing a stamp is the defect that renumbered echoes (or a
        # stale foreign stamp) produce; distinct objects keep I2 quiet so the
        # failure is attributable to I1 alone.
        transcript = _response(
            Message("assistant", [_call("c1", order=0), _call("c2", order=0)]),
            Message("tool", [_result("c1"), _result("c2")]),
        )
        with pytest.raises(AssertionError, match=r"I1.*consecutive increasing run"):
            assert_transcript_invariants(transcript)

    def test_accepts_run_starting_above_zero(self) -> None:
        # The middleware-termination exit returns the current response without
        # the accumulated prepend, so a final transcript may be a trailing
        # window of the run's numbering — deliberately admitted.
        transcript = _response(
            Message("assistant", [_call("c3", order=3), _call("c4", order=4)]),
            Message("tool", [_result("c3"), _result("c4")]),
        )
        assert_transcript_invariants(transcript)

    def test_ignores_unstamped_informational_calls(self) -> None:
        # Hosted (informational) calls are never dispatched and never stamped;
        # the oracle must not demand stamps from them.
        transcript = _response(
            Message("assistant", [_call("hosted", informational=True), _call("c1", order=0)]),
            Message("tool", [_result("c1")]),
        )
        assert_transcript_invariants(transcript)


class TestObjectIdentityUniqueness:
    """I2: no non-usage Content or Message object appears twice."""

    def test_fires_on_repeated_content_object(self) -> None:
        # Text content, so the duplicate trips I2 alone (a duplicated stamped
        # call would trip I1's duplicate-ordinal check first).
        text = Content.from_text("hello")
        transcript = _response(Message("assistant", [text]), Message("assistant", [text]))
        with pytest.raises(AssertionError, match=r"I2.*Content object"):
            assert_transcript_invariants(transcript)

    def test_fires_on_repeated_message_object(self) -> None:
        message = Message("assistant", [Content.from_text("hello")])
        with pytest.raises(AssertionError, match=r"I2.*Message object"):
            assert_transcript_invariants(_response(message, message))

    def test_accepts_repeated_usage_object(self) -> None:
        usage = Content.from_usage(
            usage_details={"input_token_count": 3, "output_token_count": 1, "total_token_count": 4}
        )
        transcript = _response(Message("assistant", [usage]), Message("assistant", [usage]))

        assert_transcript_invariants(transcript)


class TestExchangeScopedResultPairing:
    """I3: each wire-id result answers its own exchange's call, exactly once."""

    def test_fires_on_orphaned_result(self) -> None:
        transcript = _response(
            Message("assistant", [_call("c1", order=0)]),
            Message("tool", [_result("other")]),
        )
        with pytest.raises(AssertionError, match=r"I3.*answers no call"):
            assert_transcript_invariants(transcript)

    def test_fires_on_twice_answered_call(self) -> None:
        # Two DISTINCT result objects with one call id — the echoed-result
        # shape after the echo was copied rather than removed. Distinct
        # objects keep I2 quiet so the failure is attributable to I3 alone.
        transcript = _response(
            Message("assistant", [_call("c1", order=0)]),
            Message("tool", [_result("c1"), _result("c1")]),
        )
        with pytest.raises(AssertionError, match=r"I3.*answered twice"):
            assert_transcript_invariants(transcript)

    def test_fires_on_duplicate_answer_across_assistant_role_results(self) -> None:
        # Hosted MCP results ride result-only assistant messages; those
        # messages continue the exchange's output run, so a second same-id
        # answer there must fire exactly like one in a tool-role message.
        transcript = _response(
            Message("assistant", [_call("c1", order=0)]),
            Message("assistant", [_result("c1")]),
            Message("assistant", [_result("c1")]),
        )
        with pytest.raises(AssertionError, match=r"I3.*answered twice"):
            assert_transcript_invariants(transcript)

    def test_fires_on_duplicate_answer_split_between_tool_and_assistant_roles(self) -> None:
        transcript = _response(
            Message("assistant", [_call("c1", order=0)]),
            Message("tool", [_result("c1")]),
            Message("assistant", [_result("c1")]),
        )
        with pytest.raises(AssertionError, match=r"I3.*answered twice"):
            assert_transcript_invariants(transcript)

    def test_accepts_result_only_assistant_answer(self) -> None:
        # The legitimate hosted-MCP shape: the answer arrives assistant-role.
        transcript = _response(
            Message("assistant", [_call("c1", order=0)]),
            Message("assistant", [_result("c1")]),
        )
        assert_transcript_invariants(transcript)

    def test_accepts_call_id_reuse_across_exchanges(self) -> None:
        # Providers mint per-response counter ids, so call_00-style reuse
        # across iterations is legitimate — pairing is exchange-scoped, and a
        # result-only assistant answer must not tear the exchange boundary.
        transcript = _response(
            Message("assistant", [_call("c1", order=0)]),
            Message("tool", [_result("c1")]),
            Message("assistant", [_call("c1", order=1)]),
            Message("assistant", [_result("c1")]),
        )
        assert_transcript_invariants(transcript)

    def test_accepts_results_without_call_id(self) -> None:
        # An id-less result is unpairable by construction (the kernel copies
        # the call's empty id onto its result); I2 covers echoes of it.
        transcript = _response(
            Message("assistant", [_call("c1", order=0)]),
            Message("tool", [_result("c1"), _result("")]),
        )
        assert_transcript_invariants(transcript)

    def test_fires_on_duplicate_hosted_mcp_answer(self) -> None:
        # Hosted MCP pairing uses the shared call/result TYPE SETS, not just
        # function_call/function_result — two distinct same-id
        # mcp_server_tool_result objects are the echoed-result shape and must
        # fire exactly like their function_result counterpart.
        transcript = _response(
            Message("assistant", [_mcp_call("m1")]),
            Message("assistant", [_mcp_result("m1")]),
            Message("assistant", [_mcp_result("m1")]),
        )
        with pytest.raises(AssertionError, match=r"I3.*answered twice"):
            assert_transcript_invariants(transcript)

    def test_accepts_hosted_mcp_exchange(self) -> None:
        transcript = _response(
            Message("assistant", [_mcp_call("m1")]),
            Message("assistant", [_mcp_result("m1")]),
        )
        assert_transcript_invariants(transcript)

    def test_fires_on_duplicate_image_generation_answer(self) -> None:
        # Image generation pairs on image_id, not call_id (which is always
        # None for it) — a call_id-only check would skip these entirely and
        # accept the echoed-result shape.
        transcript = _response(
            Message("assistant", [_image_call("img1")]),
            Message("assistant", [_image_result("img1")]),
            Message("assistant", [_image_result("img1")]),
        )
        with pytest.raises(AssertionError, match=r"I3.*answered twice"):
            assert_transcript_invariants(transcript)

    def test_accepts_image_generation_exchange(self) -> None:
        transcript = _response(
            Message("assistant", [_image_call("img1")]),
            Message("assistant", [_image_result("img1")]),
        )
        assert_transcript_invariants(transcript)

    def test_fires_on_image_result_matching_only_a_call_id(self) -> None:
        # The pairing key is namespaced by field: an image result whose
        # image_id happens to equal a function call's call_id answers nothing.
        transcript = _response(
            Message("assistant", [_call("c1", order=0)]),
            Message("tool", [_result("c1")]),
            Message("assistant", [_image_result("c1")]),
        )
        with pytest.raises(AssertionError, match=r"I3.*answers no call"):
            assert_transcript_invariants(transcript)

    def test_accepts_hosted_same_message_call_result_pair(self) -> None:
        # A hosted same-message pair (call and its result in ONE assistant
        # message): the message's calls register before its embedded results
        # are checked, so the self-paired result passes — there is no blanket
        # exemption (see the embedded-orphan red case below).
        transcript = _response(
            Message("assistant", [_call("hosted", informational=True), _result("hosted"), _call("c1", order=0)]),
            Message("tool", [_result("c1")]),
        )
        assert_transcript_invariants(transcript)

    def test_fires_on_embedded_result_orphaned_by_exchange_boundary(self) -> None:
        # A call-carrying assistant message starts a NEW exchange; an embedded
        # result whose id belongs to the PREVIOUS exchange (or to no call at
        # all) is an orphan and must fire — riding next to fresh calls does
        # not exempt it from pairing.
        transcript = _response(
            Message("assistant", [_call("c1", order=0)]),
            Message("tool", [_result("c1")]),
            Message("assistant", [_call("c2", order=1), _result("c1")]),
            Message("tool", [_result("c2")]),
        )
        with pytest.raises(AssertionError, match=r"I3.*answers no call"):
            assert_transcript_invariants(transcript)


def _marker(*contents: Content) -> Message:
    message = Message("assistant", list(contents))
    message.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    return message


class TestMarkerExchangeBoundary:
    """History markers are exchange-neutral chrome injected between turns: a
    marker-stamped message never carries an exchange's output run, and a
    sibling call run answered by one shared result block is one exchange."""

    def test_fires_on_result_carried_by_marker_message(self) -> None:
        # The marker closes the exchange, so the result riding it can no
        # longer answer the call — the transcript is torn and must fire.
        transcript = _response(
            Message("user", ["go"]),
            Message("assistant", [_call("c1", order=0)]),
            _marker(_result("c1")),
        )
        with pytest.raises(AssertionError, match=r"I3.*answers no call"):
            assert_transcript_invariants(transcript)

    def test_fires_on_result_crossing_a_falsy_valued_marker(self) -> None:
        # KEY presence marks the boundary even when the kind value is
        # falsy; the result behind it answers no call and must fire.
        stamped = Message("assistant", ["pause"])
        stamped.additional_properties[HistoryMarkerKind.KEY] = ""
        transcript = _response(
            Message("user", ["go"]),
            Message("assistant", [_call("c1", order=0)]),
            stamped,
            Message("tool", [_result("c1")]),
        )
        with pytest.raises(AssertionError, match=r"I3.*answers no call"):
            assert_transcript_invariants(transcript)

    def test_accepts_shared_result_block_for_sibling_call_run(self) -> None:
        # Consecutive call-carrying assistant siblings answered by one tool
        # message are a single exchange; pairing must not tear them apart.
        transcript = _response(
            Message("user", ["go"]),
            Message("assistant", [_call("ca", order=0)]),
            Message("assistant", [_call("cb", order=1)]),
            Message("tool", [_result("ca"), _result("cb")]),
        )
        assert_transcript_invariants(transcript)
