# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A real Responses client over scripted HTTP, and the event streams it reads.

:class:`Script` writes a stream event by event, numbering them in order, so a
test can leave out, repeat or reorder what a well-behaved service sends.
Items are the wire dictionaries of ``tests/support/wire_cases``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Self

import httpx
from openai import AsyncOpenAI

from chrys.foundation.errors import ProviderResponseError
from chrys.kernel import ChatResponse, ChatResponseUpdate, Message, ResponseStream, tool
from chrys.kernel.middleware import ChatMiddlewareLayer
from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware
from chrys.service.llm.openai_responses import ResponsesApiClient
from tests.support.scripted_wire import ScriptedWire
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer
from tests.support.wire_cases import Reply
from tests.support.wire_cases._kit import LOOKUP_ARGUMENTS, json_reply, resp_response, sse_reply

RESPONSE_ID = "resp_1"
_BASE_URL = "https://responses.test/v1"
# Arguments arrive in deltas of this many characters.
_SPLIT = 6


def snapshot(
    *output: Mapping[str, Any],
    status: str = "completed",
    error: Mapping[str, Any] | None = None,
    incomplete: str | None = None,
) -> dict[str, Any]:
    """The response object with *output*, as a terminal event or a blocking reply carries it."""
    response = resp_response(response_id=RESPONSE_ID, output=output)
    response["status"] = status
    response["error"] = dict(error) if error is not None else None
    response["incomplete_details"] = {"reason": incomplete} if incomplete is not None else None
    return response


def refusal_item(item_id: str, text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "refusal", "refusal": text}],
    }


def call_item(item_id: str, call_id: str, arguments: str = LOOKUP_ARGUMENTS) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": "lookup",
        "arguments": arguments,
        "status": "completed",
    }


def mcp_item(item_id: str) -> dict[str, Any]:
    """A hosted MCP call that ran, with side effects the client cannot undo."""
    return {
        "type": "mcp_call",
        "id": item_id,
        "server_label": "github",
        "name": "create_issue",
        "arguments": "{}",
        "output": "created #42",
        "status": "completed",
    }


@dataclass(slots=True)
class Script:
    """The events of one stream, written in the order the service sends them."""

    events: list[tuple[str | None, Any]] = field(default_factory=list)
    # The connection is lost after the events.
    broken: bool = False

    def emit(self, event_type: str, **fields: Any) -> Self:
        self.events.append((event_type, {"type": event_type, "sequence_number": len(self.events), **fields}))
        return self

    def started(self, *, status: str = "in_progress") -> Self:
        running = {**snapshot(status=status), "usage": None}
        return self.emit("response.created", response=running).emit("response.in_progress", response=running)

    def text(self, index: int, item_id: str, text: str) -> Self:
        where = {"item_id": item_id, "output_index": index, "content_index": 0}
        part = {"type": "output_text", "text": text, "annotations": []}
        added = {"type": "message", "id": item_id, "role": "assistant", "status": "in_progress", "content": []}
        self.emit("response.output_item.added", output_index=index, item=added)
        self.emit("response.content_part.added", **where, part={**part, "text": ""})
        self.emit("response.output_text.delta", **where, delta=text, logprobs=[])
        self.emit("response.output_text.done", **where, text=text, logprobs=[])
        self.emit("response.content_part.done", **where, part=part)
        done = {**added, "status": "completed", "content": [part]}
        return self.emit("response.output_item.done", output_index=index, item=done)

    def refusal(self, index: int, item_id: str, text: str) -> Self:
        where = {"item_id": item_id, "output_index": index, "content_index": 0}
        added = {"type": "message", "id": item_id, "role": "assistant", "status": "in_progress", "content": []}
        self.emit("response.output_item.added", output_index=index, item=added)
        self.emit("response.content_part.added", **where, part={"type": "refusal", "refusal": ""})
        self.emit("response.refusal.delta", **where, delta=text)
        self.emit("response.refusal.done", **where, refusal=text)
        self.emit("response.content_part.done", **where, part={"type": "refusal", "refusal": text})
        return self.emit("response.output_item.done", output_index=index, item=refusal_item(item_id, text))

    def call_added(self, index: int, item_id: str, call_id: str, *, arguments: str = "") -> Self:
        added = {**call_item(item_id, call_id, arguments), "status": "in_progress"}
        return self.emit("response.output_item.added", output_index=index, item=added)

    def call_deltas(self, index: int, item_id: str, arguments: str = LOOKUP_ARGUMENTS) -> Self:
        for start in range(0, len(arguments), _SPLIT):
            self.emit(
                "response.function_call_arguments.delta",
                item_id=item_id,
                output_index=index,
                delta=arguments[start : start + _SPLIT],
            )
        return self

    def call_arguments_done(self, index: int, item_id: str, arguments: str = LOOKUP_ARGUMENTS) -> Self:
        return self.emit(
            "response.function_call_arguments.done", item_id=item_id, output_index=index, arguments=arguments
        )

    def call_done(self, index: int, item_id: str, call_id: str, *, arguments: str = LOOKUP_ARGUMENTS) -> Self:
        return self.emit("response.output_item.done", output_index=index, item=call_item(item_id, call_id, arguments))

    def call(self, index: int, item_id: str, call_id: str, arguments: str = LOOKUP_ARGUMENTS) -> Self:
        """A call as a well-behaved service streams it: added, deltas, arguments done, item done."""
        self.call_added(index, item_id, call_id).call_deltas(index, item_id, arguments)
        return self.call_arguments_done(index, item_id, arguments).call_done(
            index, item_id, call_id, arguments=arguments
        )

    def hosted(self, index: int, item: Mapping[str, Any]) -> Self:
        self.emit("response.output_item.added", output_index=index, item={**item, "status": "in_progress"})
        return self.emit("response.output_item.done", output_index=index, item=dict(item))

    def finished(self, *output: Mapping[str, Any], incomplete: str | None = None) -> Self:
        if incomplete is None:
            return self.emit("response.completed", response=snapshot(*output))
        response = snapshot(*output, status="incomplete", incomplete=incomplete)
        return self.emit("response.incomplete", response=response)

    def failed(
        self, *output: Mapping[str, Any], code: str = "server_error", message: str = "The model failed."
    ) -> Self:
        response = snapshot(*output, status="failed", error={"code": code, "message": message})
        return self.emit("response.failed", response=response)

    def error(self, code: str | None, message: str = "Something went wrong.") -> Self:
        return self.emit("error", code=code, message=message, param=None)

    def breaks_off(self) -> Self:
        """Lose the connection after the events so far: reading on fails as a reset connection."""
        self.broken = True
        return self

    def reply(self) -> Reply:
        return replace(sse_reply(self.events), breaks_off=self.broken)


def blocking(*output: Mapping[str, Any], **fields: Any) -> Reply:
    """A blocking reply: the whole response at once."""
    return json_reply(snapshot(*output, **fields))


@asynccontextmanager
async def responses_client(
    *replies: Reply, client_type: type[ResponsesApiClient] = ResponsesApiClient
) -> AsyncIterator[tuple[ResponsesApiClient, ScriptedWire]]:
    """A production Responses client whose requests are answered from *replies*."""
    wire = ScriptedWire(replies)
    # No proxy from the environment may route around the scripted transport.
    http_client = httpx.AsyncClient(transport=wire.transport, trust_env=False)
    sdk_client = AsyncOpenAI(api_key="sk-test", base_url=_BASE_URL, max_retries=0, http_client=http_client)
    client = client_type(model="gpt-test", sdk_client=sdk_client)
    try:
        yield client, wire
    finally:
        await client.aclose()


async def respond(
    reply: Reply,
    *,
    stream: bool,
    options: Mapping[str, Any] | None = None,
    client_type: type[ResponsesApiClient] = ResponsesApiClient,
) -> tuple[ChatResponse, list[ChatResponseUpdate]]:
    """The response one scripted reply decodes to, and the updates a stream sent on the way."""
    async with responses_client(reply, client_type=client_type) as (client, _):
        result = client._inner_get_response(
            messages=[Message("user", ["What is the weather in Paris?"])], options=dict(options or {}), stream=stream
        )
        if isinstance(result, ResponseStream):
            updates = [update async for update in result]
            return await result.get_final_response(), updates
        return await result, []


@dataclass(frozen=True, slots=True)
class ToolRuns:
    """What a tool loop over scripted replies did."""

    runs: list[str]
    requests: list[httpx.Request]
    error: ProviderResponseError | None


async def tool_runs(*replies: Reply, stream: bool = True, options: Mapping[str, Any] | None = None) -> ToolRuns:
    """Drive *replies* through the tool loop with a ``lookup`` tool that records each city it is run for."""
    runs: list[str] = []

    @tool(name="lookup", description="Look up the weather in a city.")
    def lookup(city: str) -> str:
        runs.append(city)
        return f"Sunny in {city}."

    async with responses_client(*replies) as (client, wire):
        layer = InvariantCheckedToolLoopLayer(
            ChatMiddlewareLayer(client, middleware=[ResponseValidationMiddleware(backoff_schedule=(0,))])
        )
        result = layer.get_response(
            [Message("user", ["What is the weather in Paris?"])],
            stream=stream,
            options={**(options or {}), "tools": [lookup]},
        )
        try:
            if isinstance(result, ResponseStream):
                await result.get_final_response()
            else:
                await result
        except ProviderResponseError as error:
            return ToolRuns(runs, wire.requests, error)
        return ToolRuns(runs, wire.requests, None)


def paths(requests: Sequence[httpx.Request]) -> list[str]:
    return [f"{request.method} {request.url.path}" for request in requests]
