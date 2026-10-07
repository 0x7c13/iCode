# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A real ``AsyncOpenAI`` client whose Chat Completions requests are answered from a script.

Codec tests drive the production request path (the SDK's raw-response
wrapper, SSE decoding and lenient model construction) instead of a
hand-written ``chat.completions`` double. A scripted stream is a sequence of
chunks sent as ``text/event-stream`` events and closed with ``[DONE]``; a
scripted ``ChatCompletion`` is sent as one JSON body. Models built with
``model_construct`` serialize only the fields they set, so absent and
literal-null wire fields reach the client as written. No request leaves the
process; ``done=False`` ends the streams at EOF instead, ``breaks_off=True``
loses the connection after the last chunk and ``held_open=True`` keeps it
open, sending nothing more. A string in a stream is sent as an event's data
verbatim, for data a chunk model cannot express (no chunk at all, or a chunk
with fields left out or null); a lone surrogate escape in it (``"\\udcff"``)
sends its byte, for data that is not UTF-8. Parser tests that skip HTTP use
``parse_stream_chunks``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from chrys.kernel import ChatResponseUpdate
from chrys.service.llm.chat_completions import ChatCompletionsClient
from chrys.service.llm.chat_completions.stream import StreamState

type ChatReply = Sequence[ChatCompletionChunk | str] | ChatCompletion
"""A streamed reply (its chunks, or raw event data) or a blocking one (its completion)."""


def wire_payload(model: ChatCompletionChunk | ChatCompletion) -> dict[str, Any]:
    """Return the JSON object *model* is on the wire: its explicitly set fields only."""
    return model.model_dump(mode="json", exclude_unset=True)


class _EventStream(httpx.AsyncByteStream):
    """One SSE event per read, each sent after awaiting *pace* when one is given."""

    def __init__(
        self,
        chunks: Sequence[ChatCompletionChunk | str],
        pace: Callable[[], Awaitable[object]] | None,
        *,
        done: bool,
        breaks_off: bool = False,
        held_open: bool = False,
    ) -> None:
        self._events = [
            f"data: {chunk if isinstance(chunk, str) else json.dumps(wire_payload(chunk))}\n\n".encode(
                "utf-8", "surrogateescape"
            )
            for chunk in chunks
        ]
        if done and not (breaks_off or held_open):
            self._events.append(b"data: [DONE]\n\n")
        self._pace = pace
        self._breaks_off = breaks_off
        self._held_open = held_open
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self._events:
            if self._pace is not None:
                await self._pace()
            yield event
        if self._breaks_off:
            raise httpx.ReadError("Connection reset by peer")
        if self._held_open:
            # Until the reader gives up: it is cancelled.
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class ScriptedOpenAI:
    """The scripted client, the JSON bodies it sent and the stream bodies it was answered with."""

    client: AsyncOpenAI
    requests: list[dict[str, Any]]
    streams: list[_EventStream]


@asynccontextmanager
async def scripted_openai(
    replies: Sequence[ChatReply] | Callable[[dict[str, Any]], ChatReply],
    *,
    base_url: str = "https://api.test/v1",
    pace: Callable[[], Awaitable[object]] | None = None,
    done: bool = True,
    breaks_off: bool = False,
    held_open: bool = False,
) -> AsyncIterator[ScriptedOpenAI]:
    """Yield a real ``AsyncOpenAI`` answering request *n* with ``replies[n]``.

    *replies* may instead be a callable that picks the reply from the decoded
    request body. A request past the end of the script fails the test when the
    block exits, whether the connection error the SDK raised for it was caught
    inside the block or escapes it. Every stream awaits *pace* before each event, so
    ``asyncio.Barrier(n).wait`` steps *n* concurrent streams in lockstep.
    """
    script = [] if callable(replies) else list(replies)
    requests: list[dict[str, Any]] = []
    streams: list[_EventStream] = []
    overruns: list[int] = []

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if callable(replies):
            reply = replies(body)
        elif script:
            reply = script.pop(0)
        else:
            overruns.append(len(requests))
            raise AssertionError(f"unscripted request #{len(requests)}")
        if isinstance(reply, ChatCompletion):
            return httpx.Response(200, json=wire_payload(reply), request=request)
        stream = _EventStream(reply, pace, done=done, breaks_off=breaks_off, held_open=held_open)
        streams.append(stream)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream, request=request)

    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as http_client:
            client = AsyncOpenAI(api_key="sk-test", base_url=base_url, max_retries=0, http_client=http_client)
            yield ScriptedOpenAI(client, requests, streams)
    finally:
        # Also when the error the SDK raised for the overrun leaves the block:
        # an outer ``pytest.raises`` must not accept it.
        if overruns:
            raise AssertionError(f"unscripted request(s) {overruns} ran past the script")


def parse_stream_chunks(client: ChatCompletionsClient, *chunks: ChatCompletionChunk) -> list[ChatResponseUpdate]:
    """Parse *chunks* as the events of one stream, which share its assembly state.

    Streamed tool calls are only accumulated here: the stream loop emits them
    at a finish reason or end of stream, so assert on them through
    ``scripted_openai`` instead.
    """
    state = StreamState(client.VARIANT)
    return [state.update_for(chunk) for chunk in chunks]
