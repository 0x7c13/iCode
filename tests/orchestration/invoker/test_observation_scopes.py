# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Existing observation scopes; no future evidence DTO or merge engine lives here."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.kernel import Content, LoopRecorder, Message
from chrys.orchestration.invoker.acp_protocol import AcpUpdateTranslator
from chrys.service.acp_client import AcpPromptUsage
from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware
from tests.orchestration.sub_agents._acp_fakes import make_controller
from tests.service.agent_middleware._response_validation_fakes import (
    _assistant,
    _FakeCallNext,
    _final_response,
    _make_context,
)


def test_same_slot_duplicate_commit_is_ignored_and_restore_preserves_answered_count() -> None:
    recorder = LoopRecorder()
    before = recorder.snapshot()
    call = Content.from_function_call("reused", "write_file", arguments={})
    call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    (commit,) = recorder.stage_exchange([Message("assistant", [call])], [call], result_carrier_item_id="a" * 32)
    result = Content.from_function_result("reused", result="written")
    commit.commit_final(result)
    commit.commit_final(Content.from_function_result("reused", result="duplicate"))
    assert recorder.committed_count == 1
    recorder.restore(before)
    assert recorder.committed_count == 1
    messages = recorder.loop_messages
    assert messages is not None
    assert messages[-1].contents[0] is result
    recorder.reset()
    assert recorder.committed_count == 0
    assert recorder.loop_messages is None
    # Reset is the present pass boundary; it does not retain invocation totals.
    commit.commit_final(result)
    assert recorder.committed_count == 0


def test_cancel_placeholder_does_not_erase_commits_or_count_as_an_answer() -> None:
    recorder = LoopRecorder()
    before = recorder.snapshot()
    calls = [Content.from_function_call("same", "write_file", arguments={}) for _ in range(3)]
    for index, call in enumerate(calls):
        call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = index
    commits = recorder.stage_exchange([Message("assistant", calls)], calls, result_carrier_item_id="a" * 32)
    commits[0].commit_final(Content.from_function_result("same", result="one"))
    commits[1].commit_interrupted(calls[1], None)
    recorder.restore(before)
    assert recorder.committed_count == 1
    commits[2].commit_final(Content.from_function_result("same", result="two"))
    assert recorder.committed_count == 2
    assert recorder.loop_messages is not None
    assert [content.result for content in recorder.loop_messages[-1].contents] == [
        "one",
        "Error: Tool execution was interrupted. The operation may have completed; inspect current state before retrying.",
        "two",
    ]


@pytest.mark.parametrize("stream", [False, True])
async def test_one_hosted_shell_has_two_labels_but_only_one_observed_lower_bound(stream: bool) -> None:
    response = _assistant(
        [
            Content.from_shell_tool_call(call_id="sh1", commands=["touch output"]),
            Content.from_shell_tool_result(call_id="sh1", outputs=[]),
            Content.from_text("done"),
        ]
    )
    context = _make_context(stream=stream)
    fake = _FakeCallNext([response, _assistant([Content.from_text("next wire")])], stream=stream)
    fake.bind(context)
    middleware = ResponseValidationMiddleware(backoff_schedule=[0.0])
    await middleware.process(context, fake)
    await _final_response(context, stream=stream)
    labels = middleware.hosted_commits_observed()
    assert labels == ("shell", "shell_tool_result")
    assert middleware.hosted_commits_in_flight() == labels
    # F-25's reporting projection, not a new production Count implementation.
    # The future Count projection uses int(bool(labels)); it is not a current production API.
    await middleware.process(context, fake)
    await _final_response(context, stream=stream)
    assert middleware.hosted_commits_in_flight() == ()
    assert middleware.hosted_commits_observed() == labels
    assert fake.call_count == 2


@pytest.mark.parametrize("second_reported", [False, True])
async def test_acp_usage_keeps_known_spend_and_marks_unreported_attempts(tmp_path: Path, second_reported: bool) -> None:
    bus = EventBus()
    controller = make_controller(bus, tmp_path)
    first = AcpUpdateTranslator(
        event_bus=bus,
        session_id="parent",
        agent_name="External",
        invocation_id="inv",
        attempt=1,
        origin=InvocationOrigin("sub_agent", "parent", "inv", None),
    )
    second = AcpUpdateTranslator(
        event_bus=bus,
        session_id="parent",
        agent_name="External",
        invocation_id="inv",
        attempt=2,
        origin=InvocationOrigin("sub_agent", "parent", "inv", None),
    )
    known = AcpPromptUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    await controller.policy.backend._account_usage(1, known, first)
    await controller.policy.backend._account_usage(1, known, first)
    assert controller.policy.backend.total_usage_tokens == 2
    await controller.policy.backend._account_usage(2, known if second_reported else None, second)
    assert controller.policy.backend.total_usage_tokens == (4 if second_reported else 2)
    assert controller.policy.backend.usage_unreported_attempts == (0 if second_reported else 1)
    await controller.policy.backend._account_usage(1, known, first)
    assert controller.policy.backend.total_usage_tokens == (4 if second_reported else 2)
