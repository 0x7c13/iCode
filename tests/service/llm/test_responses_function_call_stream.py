# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Streamed Responses function calls: each is sent once, whole, in output order."""

from __future__ import annotations

from typing import Any

import pytest

from chrys.foundation.hosted_tools import HeldHostedEvidence
from chrys.kernel import ChatResponse, ChatResponseUpdate, Content, Message
from chrys.service.llm.openai_responses.client import OPENAI_RESPONSES
from chrys.service.llm.openai_responses.request import build_request
from tests.service.llm._responses_wire import Script, call_item, mcp_item, respond, tool_runs
from tests.support.wire_cases._kit import WeatherReport, resp_message

_PARIS = '{"city": "Paris"}'
_ROME = '{"city": "Rome"}'


def _calls(response: ChatResponse) -> list[tuple[str | None, str | None, Any]]:
    return [
        (content.call_id, content.name, content.arguments)
        for message in response.messages
        for content in message.contents
        if content.type == "function_call"
    ]


def _sent_calls(updates: list[ChatResponseUpdate]) -> list[tuple[int, str | None]]:
    """Which update sent which call, by update position."""
    return [
        (position, content.call_id)
        for position, update in enumerate(updates)
        for content in update.contents
        if content.type == "function_call"
    ]


async def test_interleaved_calls_keep_their_own_arguments() -> None:
    script = (
        Script()
        .started()
        .call_added(0, "fc_1", "call_1")
        .call_added(1, "fc_2", "call_2")
        .call_deltas(1, "fc_2", _ROME)
        .call_deltas(0, "fc_1", _PARIS)
        .call_done(0, "fc_1", "call_1", arguments=_PARIS)
        .call_done(1, "fc_2", "call_2", arguments=_ROME)
        .finished(call_item("fc_1", "call_1", _PARIS), call_item("fc_2", "call_2", _ROME))
    )

    response, updates = await respond(script.reply(), stream=True)

    assert _calls(response) == [("call_1", "lookup", _PARIS), ("call_2", "lookup", _ROME)]
    assert [call_id for _, call_id in _sent_calls(updates)] == ["call_1", "call_2"]


async def test_a_call_sent_only_as_a_done_item_is_kept() -> None:
    script = Script().started().call_done(0, "fc_1", "call_1").finished(call_item("fc_1", "call_1"))

    response, _ = await respond(script.reply(), stream=True)

    assert _calls(response) == [("call_1", "lookup", _PARIS)]


async def test_the_whole_arguments_of_a_done_event_win_over_the_deltas() -> None:
    script = (
        Script()
        .started()
        .call_added(0, "fc_1", "call_1")
        .call_deltas(0, "fc_1", '{"city": "Par')
        .call_arguments_done(0, "fc_1", _PARIS)
        .call_done(0, "fc_1", "call_1", arguments="")
        .finished(call_item("fc_1", "call_1"))
    )

    response, _ = await respond(script.reply(), stream=True)

    assert _calls(response) == [("call_1", "lookup", _PARIS)]


async def test_deltas_and_a_done_item_repeating_them_are_not_doubled() -> None:
    script = Script().started().call(0, "fc_1", "call_1").finished(call_item("fc_1", "call_1"))

    response, updates = await respond(script.reply(), stream=True)

    assert _calls(response) == [("call_1", "lookup", _PARIS)]
    assert len(_sent_calls(updates)) == 1


async def test_a_call_done_first_waits_for_the_calls_before_it() -> None:
    script = (
        Script()
        .started()
        .call_added(0, "fc_1", "call_1")
        .call_added(1, "fc_2", "call_2")
        .call_deltas(0, "fc_1", _PARIS)
        .call_deltas(1, "fc_2", _ROME)
        .call_done(1, "fc_2", "call_2", arguments=_ROME)
    )
    held_until = len(script.events)
    script.call_done(0, "fc_1", "call_1", arguments=_PARIS)
    script.finished(call_item("fc_1", "call_1", _PARIS), call_item("fc_2", "call_2", _ROME))

    response, updates = await respond(script.reply(), stream=True)

    assert _calls(response) == [("call_1", "lookup", _PARIS), ("call_2", "lookup", _ROME)]
    # Both leave with the earlier call's done event, the later one second.
    assert _sent_calls(updates) == [(held_until, "call_1"), (held_until, "call_2")]


async def test_calls_between_other_items_keep_their_output_order() -> None:
    script = (
        Script()
        .started()
        .text(0, "msg_1", "Checking.")
        .call(1, "fc_1", "call_1", _PARIS)
        .text(2, "msg_2", "And Rome.")
        .call(3, "fc_2", "call_2", _ROME)
        .finished(
            resp_message("msg_1", "Checking."),
            call_item("fc_1", "call_1", _PARIS),
            resp_message("msg_2", "And Rome."),
            call_item("fc_2", "call_2", _ROME),
        )
    )

    response, _ = await respond(script.reply(), stream=True)

    assert [(content.type, content.text or content.call_id) for content in response.messages[0].contents][:4] == [
        ("text", "Checking."),
        ("function_call", "call_1"),
        ("text", "And Rome."),
        ("function_call", "call_2"),
    ]


def _order(response: ChatResponse) -> list[tuple[str, Any]]:
    return [
        (content.type, content.text or content.call_id or content.arguments)
        for message in response.messages
        for content in message.contents
        if content.type != "usage"
    ]


async def test_text_after_a_call_not_yet_done_waits_for_it() -> None:
    script = Script().started().call_added(0, "fc_1", "call_1").call_deltas(0, "fc_1", _PARIS)
    script.text(1, "msg_1", "After the call.")
    held_until = len(script.events)
    script.call_done(0, "fc_1", "call_1", arguments=_PARIS)
    script.finished(call_item("fc_1", "call_1", _PARIS), resp_message("msg_1", "After the call."))

    response, updates = await respond(script.reply(), stream=True)

    assert _order(response) == [("function_call", "call_1"), ("text", "After the call.")]
    sent_text = [position for position, update in enumerate(updates) for content in update.contents if content.text]
    assert sent_text == [held_until]


async def test_text_between_two_calls_keeps_its_place_when_the_later_call_is_done_first() -> None:
    script = Script().started().call_added(0, "fc_1", "call_1").call_deltas(0, "fc_1", _PARIS)
    script.text(1, "msg_1", "Between calls.").call(2, "fc_2", "call_2", _ROME)
    script.call_done(0, "fc_1", "call_1", arguments=_PARIS)
    script.finished(
        call_item("fc_1", "call_1", _PARIS),
        resp_message("msg_1", "Between calls."),
        call_item("fc_2", "call_2", _ROME),
    )

    response, _ = await respond(script.reply(), stream=True)

    assert _order(response) == [
        ("function_call", "call_1"),
        ("text", "Between calls."),
        ("function_call", "call_2"),
    ]


async def test_calls_sharing_a_call_id_with_text_between_stay_two_calls() -> None:
    script = Script().started().call_added(0, "fc_1", "same").call_deltas(0, "fc_1", _PARIS)
    script.text(1, "msg_1", "And Rome.").call(2, "fc_2", "same", _ROME)
    script.call_done(0, "fc_1", "same", arguments=_PARIS)
    script.finished(
        call_item("fc_1", "same", _PARIS), resp_message("msg_1", "And Rome."), call_item("fc_2", "same", _ROME)
    )

    response, _ = await respond(script.reply(), stream=True)

    assert _calls(response) == [("same", "lookup", _PARIS), ("same", "lookup", _ROME)]
    assert _order(response)[1] == ("text", "And Rome.")


async def test_adjacent_calls_sharing_a_call_id_stay_two_calls() -> None:
    script = Script().started().call(0, "fc_1", "same", _PARIS).call(1, "fc_2", "same", _ROME)
    script.finished(call_item("fc_1", "same", _PARIS), call_item("fc_2", "same", _ROME))

    response, _ = await respond(script.reply(), stream=True)

    assert _calls(response) == [("same", "lookup", _PARIS), ("same", "lookup", _ROME)]


async def test_hosted_work_behind_a_call_is_reported_at_once_and_sent_after_the_call() -> None:
    script = Script().started().call_added(0, "fc_1", "call_1").call_deltas(0, "fc_1", _PARIS)
    item = mcp_item("mcp_1")
    script.emit("response.output_item.added", output_index=1, item={**item, "status": "in_progress"})
    script.emit("response.mcp_call.in_progress", output_index=1, item_id="mcp_1")
    script.emit("response.output_item.done", output_index=1, item=item)
    held_until = len(script.events)
    script.call_done(0, "fc_1", "call_1", arguments=_PARIS)
    script.finished(call_item("fc_1", "call_1", _PARIS), item)

    response, updates = await respond(script.reply(), stream=True)

    evidence = [
        (position, update.raw_representation.contents)
        for position, update in enumerate(updates)
        if isinstance(update.raw_representation, HeldHostedEvidence)
    ]
    # The added item's call, then the done item's result, each reported ahead
    # of the update of the event that brought it; the status and done events
    # refresh the call in place, so it is not reported again.
    assert [[content.type for content in contents] for _, contents in evidence] == [
        ["mcp_server_tool_call"],
        ["mcp_server_tool_result"],
    ]
    assert all(not updates[position].contents for position, _ in evidence)
    hosted_sent = [
        (position, content)
        for position, update in enumerate(updates)
        for content in update.contents
        if content.provider_hosted
    ]
    # Evidence updates shift positions: the done event's update sits after both.
    assert [position for position, _ in hosted_sent] == [held_until + len(evidence)] * 2
    # What was reported is what goes out, once each.
    reported = [content for _, contents in evidence for content in contents]
    assert all(sent is held for (_, sent), held in zip(hosted_sent, reported, strict=True))
    assert [content.type for content in response.messages[0].contents if content.type != "usage"] == [
        "function_call",
        "mcp_server_tool_call",
        "mcp_server_tool_result",
    ]


async def test_hosted_work_after_a_finished_call_is_sent_at_once() -> None:
    script = Script().started().call(0, "fc_1", "call_1", _PARIS)
    sent_from = len(script.events)
    script.hosted(1, mcp_item("mcp_1")).finished(call_item("fc_1", "call_1", _PARIS), mcp_item("mcp_1"))

    _, updates = await respond(script.reply(), stream=True)

    assert not any(isinstance(update.raw_representation, HeldHostedEvidence) for update in updates)
    hosted_sent = [
        position for position, update in enumerate(updates) for content in update.contents if content.provider_hosted
    ]
    assert hosted_sent == [sent_from, sent_from + 1, sent_from + 1]


@pytest.mark.parametrize("options", [{}, {"response_format": WeatherReport}], ids=["create", "parsed"])
async def test_an_event_the_client_does_not_know_between_deltas_changes_nothing(options: dict[str, Any]) -> None:
    script = Script().started().call_added(0, "fc_1", "call_1").call_deltas(0, "fc_1", '{"city": ')
    script.emit("response.heartbeat")
    script.call_deltas(0, "fc_1", '"Paris"}').call_done(0, "fc_1", "call_1", arguments=_PARIS)
    script.finished(call_item("fc_1", "call_1", _PARIS))

    response, updates = await respond(script.reply(), stream=True, options=options)

    assert _calls(response) == [("call_1", "lookup", _PARIS)]
    assert len(_sent_calls(updates)) == 1


async def test_calls_at_scattered_indexes_after_a_text_in_two_parts_keep_their_order() -> None:
    where = {"item_id": "msg_1", "output_index": 0}
    message = {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []}
    script = Script().started().emit("response.output_item.added", output_index=0, item=message)
    for part_index, text in enumerate(["Checking ", "both."]):
        part = {"type": "output_text", "text": text, "annotations": []}
        script.emit("response.content_part.added", **where, content_index=part_index, part={**part, "text": ""})
        script.emit("response.output_text.delta", **where, content_index=part_index, delta=text, logprobs=[])
        script.emit("response.content_part.done", **where, content_index=part_index, part=part)
    script.call_added(5, "fc_2", "call_2").call_added(2, "fc_1", "call_1")
    script.call_deltas(5, "fc_2", _ROME).call_deltas(2, "fc_1", _PARIS)
    script.call_done(5, "fc_2", "call_2", arguments=_ROME)
    held_until = len(script.events)
    script.call_done(2, "fc_1", "call_1", arguments=_PARIS)
    script.finished(
        resp_message("msg_1", "Checking both."),
        call_item("fc_1", "call_1", _PARIS),
        call_item("fc_2", "call_2", _ROME),
    )

    response, updates = await respond(script.reply(), stream=True)

    assert response.text == "Checking both."
    assert _calls(response) == [("call_1", "lookup", _PARIS), ("call_2", "lookup", _ROME)]
    assert _sent_calls(updates) == [(held_until, "call_1"), (held_until, "call_2")]


async def test_a_call_still_held_at_the_terminal_event_is_sent_with_it() -> None:
    script = (
        Script()
        .started()
        .call_added(0, "fc_1", "call_1")
        .call_deltas(0, "fc_1", _PARIS)
        .finished(call_item("fc_1", "call_1"))
    )

    response, updates = await respond(script.reply(), stream=True)

    assert _calls(response) == [("call_1", "lookup", _PARIS)]
    assert _sent_calls(updates) == [(len(updates) - 1, "call_1")]


async def test_a_call_without_a_name_is_dropped() -> None:
    nameless = {**call_item("fc_1", "call_1"), "name": ""}
    script = Script().started().emit("response.output_item.done", output_index=0, item=nameless).finished(nameless)

    response, _ = await respond(script.reply(), stream=True)

    assert _calls(response) == []


@pytest.mark.parametrize(
    ("ending", "code"),
    [("failed", "server_error"), ("error", "server_error"), ("cut", "stream_truncated")],
    ids=["failed", "error_event", "cut_off"],
)
async def test_a_response_that_does_not_finish_runs_none_of_the_calls_it_streamed(ending: str, code: str) -> None:
    script = Script().started().call(0, "fc_1", "call_1")
    if ending == "failed":
        script.failed()
    elif ending == "error":
        script.error("server_error")

    result = await tool_runs(script.reply())

    assert result.runs == []
    assert len(result.requests) == 1
    assert result.error is not None
    assert result.error.code == code


async def test_a_streamed_call_replays_as_the_item_it_came_from() -> None:
    script = Script().started().call(0, "fc_live", "call_1").finished(call_item("fc_live", "call_1"))
    response, _ = await respond(script.reply(), stream=True)
    [call] = [content for content in response.messages[0].contents if content.type == "function_call"]

    assert call.additional_properties == {"output_index": 0, "fc_id": "fc_live", "status": "completed"}

    history = [
        Message("user", ["What is the weather in Paris?"]),
        Message("assistant", [call]),
        Message("tool", [Content.from_function_result(call_id="call_1", result="Sunny.")]),
    ]
    request = build_request(history, {}, model="gpt-test", variant=OPENAI_RESPONSES)
    [item] = [item for item in request["input"] if item.get("type") == "function_call"]
    assert item == {
        "call_id": "call_1",
        "id": "fc_live",
        "type": "function_call",
        "name": "lookup",
        "arguments": _PARIS,
        "status": "completed",
    }
