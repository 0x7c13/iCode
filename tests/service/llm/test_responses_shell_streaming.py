# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Regression tests for OpenAI Responses streaming shell items."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chrys.kernel import Content, Message
from chrys.service.llm.openai_responses import RawOpenAIChatClient


class _FakeAsyncOpenAI:
    base_url = "https://api.test"


def _client() -> RawOpenAIChatClient:
    return RawOpenAIChatClient(model="gpt-test", async_client=_FakeAsyncOpenAI())


def _event(event_type: str, item: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, item=item, output_index=0)


def _local_shell_tool_and_result(client: RawOpenAIChatClient) -> tuple[object, Content]:
    async def run_shell(command: str) -> str:
        return f"ran {command}"

    tool = RawOpenAIChatClient.get_shell_tool(func=run_shell, name="bash")
    item = SimpleNamespace(
        type="local_shell_call",
        id="lsc_1",
        call_id="call_1",
        action=SimpleNamespace(command=["echo", "hi"], timeout_ms=1000),
        status="completed",
    )

    update = client._parse_chunk_from_openai(_event("response.output_item.done", item), {"tools": [tool]}, {})
    call = update.contents[0]
    result = Content.from_function_result(
        call_id=call.call_id,
        result="ok",
        additional_properties=call.additional_properties,
    )
    return tool, result


@pytest.mark.parametrize(
    "item",
    [
        SimpleNamespace(type="shell_call", call_id="call_1", action=SimpleNamespace(commands=[])),
        SimpleNamespace(type="local_shell_call", call_id="call_1", action=SimpleNamespace(command=[])),
    ],
)
def test_streaming_hosted_shell_added_emits_one_start_carrier(item: SimpleNamespace) -> None:
    update = _client()._parse_chunk_from_openai(_event("response.output_item.added", item), {}, {})

    assert len(update.contents) == 1
    assert update.contents[0].type == "shell_tool_call"
    assert update.contents[0].provider_phase == "start"


def test_streaming_shell_output_added_defers_until_done() -> None:
    item = SimpleNamespace(type="shell_call_output", call_id="call_1", output=[])

    update = _client()._parse_chunk_from_openai(_event("response.output_item.added", item), {}, {})

    assert update.contents == []


def test_streaming_shell_done_parses_shell_call() -> None:
    item = SimpleNamespace(
        type="shell_call",
        call_id="call_1",
        action=SimpleNamespace(commands=["echo hi"], timeout_ms=1000, max_output_length=2000),
        status="completed",
    )

    update = _client()._parse_chunk_from_openai(_event("response.output_item.done", item), {}, {})

    content = update.contents[0]
    assert content.type == "shell_tool_call"
    assert content.call_id == "call_1"
    assert content.commands == ["echo hi"]
    assert content.timeout_ms == 1000
    assert content.max_output_length == 2000
    assert content.status == "completed"


def test_streaming_shell_done_parses_local_shell_call() -> None:
    item = SimpleNamespace(
        type="local_shell_call",
        call_id="call_1",
        action=SimpleNamespace(command=["echo", "hi"], timeout_ms=1000),
        status="completed",
    )

    update = _client()._parse_chunk_from_openai(_event("response.output_item.done", item), {}, {})

    content = update.contents[0]
    assert content.type == "shell_tool_call"
    assert content.call_id == "call_1"
    assert content.commands == ["echo hi"]
    assert content.timeout_ms == 1000
    assert content.status == "completed"


@pytest.mark.parametrize("request_uses_service_side_storage", [False, True])
def test_streaming_local_shell_result_serializes_as_local_shell_output(
    request_uses_service_side_storage: bool,
) -> None:
    client = _client()
    _tool, result = _local_shell_tool_and_result(client)

    prepared = client._prepare_messages_for_openai(
        [Message(role="tool", contents=[result])],
        request_uses_service_side_storage=request_uses_service_side_storage,
    )

    assert prepared[0]["type"] == "local_shell_call_output"
    assert prepared[0]["id"] == "call_1"
    assert prepared[0]["output"] == '{"stdout": "ok", "exit_code": 0}'


async def test_stored_local_shell_result_keeps_request_input() -> None:
    client = _client()
    tool, result = _local_shell_tool_and_result(client)

    prepared = await client._prepare_options(
        [Message(role="tool", contents=[result])],
        {"conversation_id": "resp_123", "tools": [tool]},
    )

    assert prepared["input"][0]["type"] == "local_shell_call_output"
    assert prepared["input"][0]["id"] == "call_1"


def test_streaming_shell_done_parses_shell_call_output() -> None:
    item = SimpleNamespace(
        type="shell_call_output",
        call_id="call_1",
        output=[
            SimpleNamespace(
                stdout="ok",
                stderr="",
                outcome=SimpleNamespace(type="exit", exit_code=0),
            )
        ],
        max_output_length=2000,
    )

    update = _client()._parse_chunk_from_openai(_event("response.output_item.done", item), {}, {})

    content = update.contents[0]
    assert content.type == "shell_tool_result"
    assert content.call_id == "call_1"
    assert content.max_output_length == 2000
    assert content.outputs[0].type == "shell_command_output"
    assert content.outputs[0].stdout == "ok"
    assert content.outputs[0].exit_code == 0
    assert content.outputs[0].timed_out is False
