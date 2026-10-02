# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Responses that fail or break off, through the main turn's whole-run lane under service-side storage.

The engine, its retry owner and the OpenAI SDK are real; only the HTTP
answers are scripted. The parsed stream (structured output) is not reachable
from a chat turn: its ending and failures are covered at the adapter
(``test_responses_terminal_events.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from chrys.foundation.events.types import Error, InvocationMessage
from chrys.service.profiles.models.schema import API_STYLE_RESPONSES
from tests.service.llm._responses_wire import RESPONSE_ID, Script, blocking, call_item, mcp_item, paths
from tests.support.mock_provider_turns import ProviderTurn, mock_provider_profile, run_mock_provider_turn
from tests.support.wire_cases import Reply
from tests.support.wire_cases._kit import resp_message

_CREATE = "POST /v1/responses"
_POLL = f"GET /v1/responses/{RESPONSE_ID}"
_FAILURE = {"code": "server_error", "message": "The model failed."}
_SHELL = {
    "type": "shell_call",
    "id": "sh_1",
    "call_id": "call_sh",
    "action": {"commands": ["rm -rf build"]},
    "status": "completed",
}


def _answer(*, stream: bool) -> Reply:
    if stream:
        return Script().started().text(0, "msg_1", "Sunny.").finished(resp_message("msg_1", "Sunny.")).reply()
    return blocking(resp_message("msg_1", "Sunny."))


def _answers(*replies: Reply) -> Callable[[httpx.Request], httpx.Response]:
    queue = list(replies)

    def respond(request: httpx.Request) -> httpx.Response:
        if not queue:
            raise AssertionError(f"unscripted request: {request.method} {request.url}")
        reply = queue.pop(0)
        return httpx.Response(reply.status, headers=list(reply.headers), content=reply.body, request=request)

    return respond


async def _turn(agent_engine: Any, monkeypatch: pytest.MonkeyPatch, *replies: Reply, stream: bool) -> ProviderTurn:
    profile = mock_provider_profile(
        "openai", stream=stream, api_style=API_STYLE_RESPONSES, chat_options='{"store": true}'
    )
    return await run_mock_provider_turn(agent_engine, monkeypatch, profile, _answers(*replies))


def _answered(turn: ProviderTurn) -> str | None:
    return turn.terminal.text if isinstance(turn.terminal, InvocationMessage) else None


async def test_a_stream_cut_after_its_response_started_is_resumed_by_polling_it(
    agent_engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The call the cut stream finished never runs: the response it belongs to is polled again.
    cut = Script().started().call(0, "fc_1", "call_1").reply()

    turn = await _turn(agent_engine, monkeypatch, cut, _answer(stream=True), stream=True)

    assert paths(turn.requests) == [_CREATE, _POLL]
    assert [retry.scope for retry in turn.retries] == ["run"]
    assert _answered(turn) == "Sunny."


@pytest.mark.parametrize(
    ("ending", "stream"),
    [("failed", True), ("error_event", True), ("failed", False)],
    ids=["streaming", "error_event", "blocking"],
)
async def test_a_failed_response_without_hosted_work_is_created_again(
    ending: str, stream: bool, agent_engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not stream:
        failed = blocking(call_item("fc_1", "call_1"), status="failed", error=_FAILURE)
    elif ending == "error_event":
        failed = Script().started().call(0, "fc_1", "call_1").error("server_error").reply()
    else:
        failed = Script().started().call(0, "fc_1", "call_1").failed(call_item("fc_1", "call_1")).reply()

    turn = await _turn(agent_engine, monkeypatch, failed, _answer(stream=stream), stream=stream)

    # The failed response is over: the retry creates a new one instead of polling it.
    assert paths(turn.requests) == [_CREATE, _CREATE]
    assert [retry.scope for retry in turn.retries] == ["run"]
    assert _answered(turn) == "Sunny."


@pytest.mark.parametrize(
    ("replies", "stream", "sent", "retried"),
    [
        pytest.param(
            [Script().started().hosted(0, mcp_item("mcp_1")).failed(mcp_item("mcp_1")).reply()],
            True,
            [_CREATE],
            0,
            id="streamed_mcp",
        ),
        pytest.param(
            [Script().started().failed(mcp_item("mcp_1")).reply()], True, [_CREATE], 0, id="terminal_only_mcp"
        ),
        pytest.param([Script().started().failed(_SHELL).reply()], True, [_CREATE], 0, id="terminal_only_shell"),
        pytest.param(
            [Script().started().call_added(0, "fc_1", "call_1").hosted(1, mcp_item("mcp_1")).failed().reply()],
            True,
            [_CREATE],
            0,
            id="mcp_held_behind_a_call",
        ),
        pytest.param(
            [blocking(mcp_item("mcp_1"), status="failed", error=_FAILURE)], False, [_CREATE], 0, id="blocking_mcp"
        ),
        pytest.param(
            [Script().started().reply(), Script().hosted(0, mcp_item("mcp_1")).failed(mcp_item("mcp_1")).reply()],
            True,
            [_CREATE, _POLL],
            1,
            id="polled_mcp",
        ),
    ],
)
async def test_a_failed_response_that_ran_hosted_work_is_not_sent_again(
    replies: list[Reply],
    stream: bool,
    sent: list[str],
    retried: int,
    agent_engine: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turn = await _turn(agent_engine, monkeypatch, *replies, _answer(stream=stream), stream=stream)

    assert paths(turn.requests) == sent
    assert [retry.scope for retry in turn.retries] == ["run"] * retried
    assert isinstance(turn.terminal, Error)
