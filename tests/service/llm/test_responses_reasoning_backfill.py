# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A reasoning payload only the terminal response carries still reaches local history."""

from __future__ import annotations

import json
from typing import Any

import pytest

from chrys.kernel import ChatResponse, Content
from chrys.service.llm.openai_responses import DeepSeekResponsesApiClient, ResponsesApiClient
from tests.service.llm._responses_wire import Script, call_item, respond, tool_runs
from tests.support.wire_cases._kit import resp_message, resp_reasoning

_LOCAL = {"store": False}


def _reasoning(*, done: str | None = None, terminal: str | None = "enc-final", incomplete: str | None = None) -> Script:
    """A reasoning item streamed without a payload, its done item carrying *done*, the terminal one *terminal*."""
    where = {"item_id": "rs_1", "output_index": 0, "summary_index": 0}
    script = (
        Script()
        .started()
        .emit("response.output_item.added", output_index=0, item=resp_reasoning("rs_1"))
        .emit("response.reasoning_summary_part.added", **where, part={"type": "summary_text", "text": ""})
        .emit("response.reasoning_summary_text.delta", **where, delta="Weather")
        .emit("response.reasoning_summary_text.delta", **where, delta=" first.")
        .emit("response.reasoning_summary_text.done", **where, text="Weather first.")
        .emit(
            "response.output_item.done", output_index=0, item=resp_reasoning("rs_1", "Weather first.", encrypted=done)
        )
        .text(1, "msg_1", "Sunny.")
    )
    final = resp_reasoning("rs_1", "Weather first.", encrypted=terminal)
    return script.finished(final, resp_message("msg_1", "Sunny."), incomplete=incomplete)


def _reasoning_contents(response: ChatResponse) -> list[Content]:
    return [
        content for message in response.messages for content in message.contents if content.type == "text_reasoning"
    ]


@pytest.mark.parametrize("incomplete", [None, "max_output_tokens"], ids=["completed", "incomplete"])
async def test_a_payload_only_the_terminal_response_carries_lands_on_the_streamed_reasoning(
    incomplete: str | None,
) -> None:
    response, _ = await respond(_reasoning(incomplete=incomplete).reply(), stream=True, options=_LOCAL)

    [reasoning] = _reasoning_contents(response)
    assert (reasoning.id, reasoning.text, reasoning.protected_data) == ("rs_1", "Weather first.", "enc-final")


@pytest.mark.parametrize(
    ("done", "terminal"), [(None, "enc-final"), ("enc-final", None)], ids=["terminal_payload", "done_payload"]
)
async def test_the_payload_goes_onto_the_last_content_already_sent(done: str | None, terminal: str | None) -> None:
    _, updates = await respond(_reasoning(done=done, terminal=terminal).reply(), stream=True, options=_LOCAL)

    sent = [content for update in updates for content in update.contents if content.type == "text_reasoning"]
    assert [(content.text, content.protected_data) for content in sent] == [
        ("", None),
        ("Weather", None),
        (" first.", "enc-final"),
    ]


async def test_a_reasoning_item_done_after_a_later_item_stays_one_occurrence_and_replays_once() -> None:
    where = {"item_id": "rs_1", "output_index": 0, "summary_index": 0}
    first = (
        Script()
        .started()
        .emit("response.output_item.added", output_index=0, item=resp_reasoning("rs_1"))
        .emit("response.reasoning_summary_text.delta", **where, delta="Weather first.")
        .call(1, "fc_1", "call_1")
        .emit(
            "response.output_item.done",
            output_index=0,
            item=resp_reasoning("rs_1", "Weather first.", encrypted="enc-done"),
        )
        .finished(resp_reasoning("rs_1", "Weather first.", encrypted="enc-done"), call_item("fc_1", "call_1"))
    )
    answer = Script().started().text(0, "msg_2", "Sunny.").finished(resp_message("msg_2", "Sunny."))

    result = await tool_runs(first.reply(), answer.reply(), options=_LOCAL)

    assert result.error is None
    replayed: list[dict[str, Any]] = json.loads(result.requests[1].content)["input"]
    assert [(item.get("type"), item.get("id")) for item in replayed if item.get("type") != "message"][:2] == [
        ("reasoning", "rs_1"),
        ("function_call", "fc_1"),
    ]
    [item] = [item for item in replayed if item.get("type") == "reasoning"]
    assert (item["encrypted_content"], item["summary"]) == (
        "enc-done",
        [{"type": "summary_text", "text": "Weather first."}],
    )


async def test_the_backfilled_payload_is_replayed_in_the_next_request() -> None:
    where = {"item_id": "rs_1", "output_index": 0, "summary_index": 0}
    first = (
        Script()
        .started()
        .emit("response.output_item.added", output_index=0, item=resp_reasoning("rs_1"))
        .emit("response.reasoning_summary_text.delta", **where, delta="Weather first.")
        .emit("response.output_item.done", output_index=0, item=resp_reasoning("rs_1", "Weather first."))
        .call(1, "fc_1", "call_1")
        .finished(resp_reasoning("rs_1", "Weather first.", encrypted="enc-final"), call_item("fc_1", "call_1"))
    )
    answer = Script().started().text(0, "msg_2", "Sunny.").finished(resp_message("msg_2", "Sunny."))

    result = await tool_runs(first.reply(), answer.reply(), options=_LOCAL)

    assert result.error is None
    replayed: list[dict[str, Any]] = json.loads(result.requests[1].content)["input"]
    [item] = [item for item in replayed if item.get("type") == "reasoning"]
    assert (item["id"], item["encrypted_content"]) == ("rs_1", "enc-final")
    assert item["summary"] == [{"type": "summary_text", "text": "Weather first."}]


async def test_a_payload_the_done_item_carried_is_kept() -> None:
    response, _ = await respond(_reasoning(done="enc-done").reply(), stream=True, options=_LOCAL)

    [reasoning] = _reasoning_contents(response)
    assert reasoning.protected_data == "enc-done"


async def test_a_payload_the_done_item_carries_replaces_the_one_the_item_started_with() -> None:
    script = (
        Script()
        .started()
        .emit("response.output_item.added", output_index=0, item=resp_reasoning("rs_1", encrypted="enc-added"))
        .emit("response.output_item.done", output_index=0, item=resp_reasoning("rs_1", encrypted="enc-done"))
        .text(1, "msg_1", "Sunny.")
        .finished(resp_reasoning("rs_1", encrypted="enc-done"), resp_message("msg_1", "Sunny."))
    )

    response, _ = await respond(script.reply(), stream=True, options=_LOCAL)

    assert [content.protected_data for content in _reasoning_contents(response)] == ["enc-done"]


@pytest.mark.parametrize(
    ("client_type", "options"),
    [
        pytest.param(ResponsesApiClient, {}, id="service_side_storage"),
        pytest.param(DeepSeekResponsesApiClient, _LOCAL, id="no_encrypted_reasoning"),
    ],
)
async def test_nothing_is_backfilled_where_local_history_does_not_replay_payloads(
    client_type: type[ResponsesApiClient], options: dict[str, Any]
) -> None:
    response, _ = await respond(_reasoning().reply(), stream=True, options=options, client_type=client_type)

    assert [content.protected_data for content in _reasoning_contents(response)] == [None]


async def test_a_reasoning_item_the_stream_never_sent_is_not_added() -> None:
    script = (
        Script()
        .started()
        .text(0, "msg_1", "Sunny.")
        .finished(resp_message("msg_1", "Sunny."), resp_reasoning("rs_unsent", "Hidden.", encrypted="enc-unsent"))
    )

    response, _ = await respond(script.reply(), stream=True, options=_LOCAL)

    assert _reasoning_contents(response) == []
