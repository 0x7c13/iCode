# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""OpenAI Responses continuation-resume and request-error regressions."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import BadRequestError

from chrys.kernel import Content, Message
from chrys.kernel.exceptions import ChatClientException
from chrys.service.llm.openai_exceptions import OpenAIContentFilterException
from chrys.service.llm.openai_responses import ResponsesApiClient


def _response(*, response_id: str, status: str, output: list[object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=response_id,
        created_at=0,
        model="gpt-test",
        metadata={},
        output=output or [],
        usage=None,
        conversation=None,
        status=status,
        incomplete_details=None,
    )


def _raw(response: object) -> SimpleNamespace:
    return SimpleNamespace(parse=lambda: response, headers={})


def _client(*, retrieve: object, create: object | None = None) -> tuple[ResponsesApiClient, AsyncMock, AsyncMock]:
    retrieve_mock = AsyncMock(side_effect=retrieve if isinstance(retrieve, BaseException) else None)
    if not isinstance(retrieve, BaseException):
        retrieve_mock.return_value = retrieve
    create_mock = AsyncMock()
    if create is not None:
        create_mock.return_value = create
    with_raw_response = SimpleNamespace(retrieve=retrieve_mock, create=create_mock)
    async_client = SimpleNamespace(
        base_url="https://api.test", responses=SimpleNamespace(with_raw_response=with_raw_response)
    )
    return ResponsesApiClient(model="gpt-test", sdk_client=async_client), retrieve_mock, create_mock


@pytest.mark.asyncio
async def test_background_continuation_retrieves_and_preserves_pending_token() -> None:
    client, retrieve, _create = _client(retrieve=_raw(_response(response_id="resp-pending", status="in_progress")))
    options = {
        "background": True,
        "continuation_token": {"response_id": "resp-pending"},
    }

    response = await client._inner_get_response(messages=[Message("user", ["continue"])], options=options)

    retrieve.assert_awaited_once_with("resp-pending")
    assert response.continuation_token == {"response_id": "resp-pending"}
    assert options["continuation_token"] == {"response_id": "resp-pending"}


@pytest.mark.asyncio
async def test_completed_continuation_posts_tool_result_on_next_iteration() -> None:
    tool_call = SimpleNamespace(
        type="function_call",
        call_id="call-1",
        name="lookup",
        arguments='{"query":"chrys"}',
        id="fc-1",
        status="completed",
    )
    client, retrieve, create = _client(
        retrieve=_raw(_response(response_id="resp-done", status="completed", output=[tool_call])),
        create=_raw(_response(response_id="resp-next", status="completed")),
    )
    options = {
        "background": True,
        "continuation_token": {"response_id": "resp-done"},
    }

    first = await client._inner_get_response(messages=[Message("user", ["look it up"])], options=options)
    assert "continuation_token" not in options
    assert options["background"] is True

    messages = [
        Message("user", ["look it up"]),
        *first.messages,
        Message("tool", [Content.from_function_result("call-1", result="found")]),
    ]
    await client._inner_get_response(messages=messages, options=options)

    retrieve.assert_awaited_once_with("resp-done")
    create.assert_awaited_once()
    request_input = create.await_args.kwargs["input"]
    assert any(item.get("type") == "function_call_output" and item.get("call_id") == "call-1" for item in request_input)


@pytest.mark.asyncio
async def test_content_filter_bad_request_maps_at_retrieve_boundary() -> None:
    error = BadRequestError(
        "filtered",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.test/v1/responses")),
        body={"code": "content_filter"},
    )
    client, _retrieve, _create = _client(retrieve=error)

    with pytest.raises(OpenAIContentFilterException) as exc_info:
        await client._inner_get_response(
            messages=[Message("user", ["continue"])],
            options={"continuation_token": {"response_id": "resp-filtered"}},
        )

    assert exc_info.value.args[1] is error


@pytest.mark.asyncio
async def test_generic_retrieve_failure_maps_to_chat_client_exception() -> None:
    error = RuntimeError("transport failed")
    client, _retrieve, _create = _client(retrieve=error)

    with pytest.raises(ChatClientException) as exc_info:
        await client._inner_get_response(
            messages=[Message("user", ["continue"])],
            options={"continuation_token": {"response_id": "resp-failed"}},
        )

    assert exc_info.value.args[1] is error
