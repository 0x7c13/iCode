# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real provider-SDK exceptions for error-handling tests.

Status and in-band stream errors come from the real OpenAI and Anthropic SDKs
answering through ``httpx.MockTransport``; network failures come from real SDK
requests whose DNS or connect step fails through
:mod:`tests.support.network_faults`.  No request leaves the process.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, NoReturn

import anthropic
import httpcore
import httpx
import openai

from tests.support.network_faults import NetworkFaults, network_faults

API_HOST = "api.example.test"
_CHAT = [{"role": "user", "content": "hi"}]


def raised_from(wrapper: BaseException, cause: BaseException | None) -> BaseException:
    """Return *wrapper* after ``raise wrapper from cause``, with a real traceback."""
    try:
        raise wrapper from cause
    except BaseException as exc:
        return exc


def raised_while_handling(stale: BaseException, raise_fresh: Callable[[], NoReturn]) -> BaseException:
    """Return the exception *raise_fresh* raises while *stale* is being handled.

    The fresh chain inherits *stale* only as implicit ``__context__``, the way a
    retry that fails inside an ``except`` block does.
    """
    try:
        try:
            raise stale
        except BaseException:
            raise_fresh()
    except BaseException as exc:
        return exc
    raise AssertionError("raise_fresh must raise")


def api_request() -> httpx.Request:
    """Return a request to the fake provider host."""
    return httpx.Request("POST", f"https://{API_HOST}/v1/chat/completions")


async def openai_status(status: int, body: Any) -> BaseException:
    """Return the real OpenAI SDK error for one HTTP error response."""

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as http:
        client = openai.AsyncOpenAI(
            api_key="sk-test", base_url=f"https://{API_HOST}/v1", max_retries=0, http_client=http
        )
        try:
            await client.chat.completions.create(model="m", messages=_CHAT)
        except openai.APIError as exc:
            return exc
    raise AssertionError(f"HTTP {status} did not raise")


OPENAI_CONTEXT_OVERFLOW_BODY = {
    "error": {
        "type": "invalid_request_error",
        "code": "context_length_exceeded",
        "message": "This model's maximum context length is 131072 tokens. "
        "However, your messages resulted in 140000 tokens.",
    }
}


async def openai_context_overflow() -> BaseException:
    """Return the 400 an OpenAI-compatible service answers when the input exceeds the model's window."""
    return await openai_status(400, OPENAI_CONTEXT_OVERFLOW_BODY)


async def anthropic_status(status: int, body: Any) -> BaseException:
    """Return the real Anthropic SDK error for one HTTP error response."""

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as http:
        client = anthropic.AsyncAnthropic(
            api_key="sk-test", base_url=f"https://{API_HOST}", max_retries=0, http_client=http
        )
        try:
            await client.messages.create(model="m", max_tokens=8, messages=_CHAT)
        except anthropic.APIError as exc:
            return exc
    raise AssertionError(f"HTTP {status} did not raise")


ANTHROPIC_THINKING_BINDING_MESSAGE = (
    "messages.1.content.2: Invalid `signature` in `thinking` block. The block is bound to a different "
    "conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to `drop_block`."
)
"""What Anthropic says when replayed thinking no longer matches the conversation before it."""


def anthropic_thinking_binding_body(*, names_the_window: bool = False) -> dict[str, Any]:
    """The 400 body refusing replayed thinking; *names_the_window* adds text an overflow matches too."""
    message = ANTHROPIC_THINKING_BINDING_MESSAGE
    if names_the_window:
        message += " The context window the block was signed in has changed."
    return {"type": "error", "error": {"type": "invalid_request_error", "message": message}}


async def anthropic_thinking_binding_rejection(*, names_the_window: bool = False) -> BaseException:
    """Return the real Anthropic SDK error refusing replayed thinking as bound to a different conversation."""
    return await anthropic_status(400, anthropic_thinking_binding_body(names_the_window=names_the_window))


async def openai_stream_error(error: dict[str, Any]) -> BaseException:
    """Return the bare ``APIError`` the OpenAI SDK raises for an in-band stream error."""

    def answer(request: httpx.Request) -> httpx.Response:
        payload = f"data: {json.dumps({'error': error})}\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=payload, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as http:
        client = openai.AsyncOpenAI(
            api_key="sk-test", base_url=f"https://{API_HOST}/v1", max_retries=0, http_client=http
        )
        try:
            stream = await client.chat.completions.create(model="m", messages=_CHAT, stream=True)
            async for _chunk in stream:
                pass
        except openai.APIError as exc:
            return exc
    raise AssertionError("the stream did not fail")


async def anthropic_stream_error(data: str) -> BaseException:
    """Return the error the Anthropic SDK raises for an ``error`` event after its 200."""

    def answer(request: httpx.Request) -> httpx.Response:
        payload = f"event: error\ndata: {data}\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=payload, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as http:
        client = anthropic.AsyncAnthropic(
            api_key="sk-test", base_url=f"https://{API_HOST}", max_retries=0, http_client=http
        )
        try:
            stream = await client.messages.create(model="m", max_tokens=8, messages=_CHAT, stream=True)
            async for _event in stream:
                pass
        except anthropic.APIError as exc:
            return exc
    raise AssertionError("the stream did not fail")


def anthropic_error_event(error_type: str, message: str, *, as_json_text: bool = False) -> str:
    """Return the ``data:`` of an Anthropic error event; *as_json_text* JSON-encodes it once more."""
    data = json.dumps({"type": "error", "error": {"type": error_type, "message": message}})
    return json.dumps(data) if as_json_text else data


async def openai_transport_error(build: Callable[[httpx.Request], BaseException]) -> BaseException:
    """Return the OpenAI SDK error for a transport failure *build* returns."""

    def answer(request: httpx.Request) -> httpx.Response:
        raise build(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as http:
        client = openai.AsyncOpenAI(
            api_key="sk-test", base_url=f"https://{API_HOST}/v1", max_retries=0, http_client=http
        )
        try:
            await client.chat.completions.create(model="m", messages=_CHAT)
        except openai.APIError as exc:
            return exc
    raise AssertionError("the request did not fail")


async def openai_network(
    arrange: Callable[[NetworkFaults], None], *, connect_timeout: float = 5.0, host: str = API_HOST
) -> BaseException:
    """Return the real OpenAI SDK error for a request whose DNS or connect step fails."""
    with network_faults() as faults:
        arrange(faults)
        timeout = httpx.Timeout(5.0, connect=connect_timeout)
        async with httpx.AsyncClient(trust_env=False, timeout=timeout) as http:
            client = openai.AsyncOpenAI(
                api_key="sk-test", base_url=f"http://{host}/v1", max_retries=0, http_client=http
            )
            try:
                await client.chat.completions.create(model="m", messages=_CHAT)
            except openai.APIError as exc:
                return exc
    raise AssertionError("the request did not fail")


async def anthropic_network(arrange: Callable[[NetworkFaults], None]) -> BaseException:
    with network_faults() as faults:
        arrange(faults)
        async with httpx.AsyncClient(trust_env=False) as http:
            client = anthropic.AsyncAnthropic(
                api_key="sk-test", base_url=f"http://{API_HOST}", max_retries=0, http_client=http
            )
            try:
                await client.messages.create(model="m", max_tokens=8, messages=_CHAT)
            except anthropic.APIError as exc:
                return exc
    raise AssertionError("the request did not fail")


def httpcore_connect_failure(leaf: BaseException) -> BaseException:
    """Return the SDK chain for a connect failure whose root cause httpcore keeps in ``args[0]``."""
    core = httpcore.ConnectError(leaf)
    transport = raised_from(httpx.ConnectError(str(leaf), request=api_request()), core)
    return raised_from(openai.APIConnectionError(request=api_request()), transport)
