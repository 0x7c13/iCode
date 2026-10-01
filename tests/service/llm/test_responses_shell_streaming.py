# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""OpenAI Responses shell items: streamed hosted calls and the stored local-shell results replayed from history."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from chrys.kernel import Content, Message
from chrys.service.llm.openai_responses.client import OPENAI_RESPONSES
from chrys.service.llm.openai_responses.history import (
    OPENAI_SHELL_OUTPUT_TYPE_KEY,
    OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL,
    OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL,
)
from chrys.service.llm.openai_responses.replay import encode_input
from chrys.service.llm.openai_responses.request import build_request
from chrys.service.llm.openai_responses.stream import StreamState


def _stream() -> StreamState:
    return StreamState({}, model="gpt-test", variant=OPENAI_RESPONSES)


def _event(event_type: str, item: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(type=event_type, item=item, output_index=0)


def _stored_local_shell_result() -> Content:
    """A local shell result as stored history carries it: a function result marked with its output item type."""
    return Content.from_function_result(
        call_id="call_1",
        result="ok",
        additional_properties={OPENAI_SHELL_OUTPUT_TYPE_KEY: OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL},
    )


@pytest.mark.parametrize(
    "item",
    [
        SimpleNamespace(type="shell_call", call_id="call_1", action=SimpleNamespace(commands=[])),
        SimpleNamespace(type="local_shell_call", call_id="call_1", action=SimpleNamespace(command=[])),
    ],
)
def test_streaming_hosted_shell_added_emits_one_start_carrier(item: SimpleNamespace) -> None:
    update = _stream().update_for(_event("response.output_item.added", item))

    assert len(update.contents) == 1
    assert update.contents[0].type == "shell_tool_call"
    assert update.contents[0].provider_phase == "start"


def test_streaming_shell_output_added_defers_until_done() -> None:
    item = SimpleNamespace(type="shell_call_output", call_id="call_1", output=[])

    update = _stream().update_for(_event("response.output_item.added", item))

    assert update.contents == []


def test_streaming_shell_done_parses_shell_call() -> None:
    item = SimpleNamespace(
        type="shell_call",
        call_id="call_1",
        action=SimpleNamespace(commands=["echo hi"], timeout_ms=1000, max_output_length=2000),
        status="completed",
    )

    update = _stream().update_for(_event("response.output_item.done", item))

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

    update = _stream().update_for(_event("response.output_item.done", item))

    content = update.contents[0]
    assert content.type == "shell_tool_call"
    assert content.call_id == "call_1"
    assert content.commands == ["echo hi"]
    assert content.timeout_ms == 1000
    assert content.status == "completed"


@pytest.mark.parametrize("request_uses_service_side_storage", [False, True])
def test_stored_local_shell_result_serializes_as_local_shell_output(
    request_uses_service_side_storage: bool,
) -> None:
    prepared = encode_input(
        [Message(role="tool", contents=[_stored_local_shell_result()])],
        service_side=request_uses_service_side_storage,
        variant=OPENAI_RESPONSES,
    )

    assert prepared[0]["type"] == "local_shell_call_output"
    assert prepared[0]["id"] == "call_1"
    assert prepared[0]["output"] == '{"stdout": "ok", "exit_code": 0}'


@pytest.mark.parametrize(
    ("output_type", "output"),
    [
        (OPENAI_SHELL_OUTPUT_TYPE_LOCAL_SHELL_CALL, '{"stdout": "Error: Function failed.", "exit_code": 1}'),
        (
            OPENAI_SHELL_OUTPUT_TYPE_SHELL_CALL,
            [{"stdout": "Error: Function failed.", "stderr": "", "outcome": {"type": "exit", "exit_code": 1}}],
        ),
    ],
    ids=["local_shell_call", "shell_call"],
)
def test_failed_local_shell_result_keeps_its_exception_record_off_the_wire(output_type: str, output: object) -> None:
    result = Content.from_function_result(
        call_id="call_1",
        result="Error: Function failed.",
        exception="ToolExecutionException: Failed. (caused by OSError: /home/me/.secret)",
        additional_properties={OPENAI_SHELL_OUTPUT_TYPE_KEY: output_type},
    )

    prepared = encode_input([Message(role="tool", contents=[result])], service_side=False, variant=OPENAI_RESPONSES)

    assert prepared[0]["output"] == output
    assert "/home/me/.secret" not in json.dumps(prepared)


def test_stored_local_shell_result_keeps_request_input() -> None:
    prepared = build_request(
        [Message(role="tool", contents=[_stored_local_shell_result()])],
        {"conversation_id": "resp_123"},
        model="gpt-test",
        variant=OPENAI_RESPONSES,
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

    update = _stream().update_for(_event("response.output_item.done", item))

    content = update.contents[0]
    assert content.type == "shell_tool_result"
    assert content.call_id == "call_1"
    assert content.max_output_length == 2000
    assert content.outputs[0].type == "shell_command_output"
    assert content.outputs[0].stdout == "ok"
    assert content.outputs[0].exit_code == 0
    assert content.outputs[0].timed_out is False
