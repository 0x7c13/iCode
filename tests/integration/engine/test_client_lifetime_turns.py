# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Turns keep working across a rebuild, and a rebuild leaves only the current build's clients open.

Real provider SDK clients talk HTTP to a loopback stub, so a stack closed while
still in use would surface as a failed turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import pytest

from chrys.foundation.events.types import Error, InvocationMessage, UserMessage
from tests.support.event_capture import capture_events
from tests.support.llm_client_engines import (
    ALT_MODEL,
    MAIN_MODEL,
    SUB_AGENT,
    SUB_MODEL,
    ClientEngine,
    loopback_model,
    start_client_engine,
    switch_model,
)
from tests.support.waiting import ENGINE_TURN_TIMEOUT, await_run_task_chain, wait_for

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support.engines import AgentEngineFactory
    from tests.support.llm_http_clients import HttpClientLedger

# Real SDK clients must reach the loopback stub, not a proxy.
pytestmark = pytest.mark.usefixtures("direct_route")


def _completion(model: str, message: dict[str, Any], finish_reason: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _text(text: str) -> tuple[dict[str, Any], str]:
    return {"role": "assistant", "content": text}, "stop"


def _delegate(prompt: str) -> tuple[dict[str, Any], str]:
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": SUB_AGENT, "arguments": json.dumps({"prompt": prompt})},
    }
    return {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"


class _StubProvider:
    """A keep-alive HTTP/1.1 server answering Chat Completions from per-model scripts."""

    def __init__(self, scripts: dict[str, list[tuple[dict[str, Any], str]]]) -> None:
        self._scripts = scripts
        self.models: list[str] = []

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while await reader.readline():
                headers: dict[str, str] = {}
                while (line := await reader.readline()) not in (b"\r\n", b"\n", b""):
                    name, _, value = line.decode("latin-1").partition(":")
                    headers[name.strip().lower()] = value.strip()
                request = json.loads(await reader.readexactly(int(headers.get("content-length", "0"))))
                model = request["model"]
                self.models.append(model)
                message, finish_reason = self._scripts[model].pop(0)
                body = json.dumps(_completion(model, message, finish_reason)).encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
        except asyncio.IncompleteReadError, ConnectionError:
            pass
        finally:
            writer.close()


@contextlib.asynccontextmanager
async def _serve(provider: _StubProvider) -> AsyncIterator[str]:
    server = await asyncio.start_server(provider.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.close()
        server.close_clients()
        await server.wait_closed()


def _turn_answers(messages: list[InvocationMessage]) -> list[str]:
    return [message.text for message in messages if message.is_final and message.origin.kind == "turn"]


async def _turn(started: ClientEngine, messages: list[InvocationMessage], text: str) -> str:
    answered = len(_turn_answers(messages))
    await started.bus.publish(UserMessage(text=text))
    await wait_for(
        lambda: len(_turn_answers(messages)) > answered,
        timeout=ENGINE_TURN_TIMEOUT,
        description=f"final answer to {text!r}",
    )
    await await_run_task_chain(started.engine, turn_state=started.engine.turns.turn_state)
    [answer] = _turn_answers(messages)[answered:]
    return answer


async def test_turns_across_a_rebuild_use_live_clients_and_close_the_old_ones(
    agent_engine: AgentEngineFactory, tmp_path: Path, http_client_ledger: HttpClientLedger
) -> None:
    main_id, alt_id, sub_id = (loopback_model(name).model_id for name in (MAIN_MODEL, ALT_MODEL, SUB_MODEL))
    provider = _StubProvider(
        {
            main_id: [_delegate("look around"), _text("first answer")],
            sub_id: [_text("explored")],
            alt_id: [_text("second answer")],
        }
    )
    async with _serve(provider) as base_url:
        started = await start_client_engine(agent_engine, tmp_path, base_url=base_url)
        errors = await capture_events(started.bus, Error)
        messages = await capture_events(started.bus, InvocationMessage)

        assert await _turn(started, messages, "first") == "first answer"
        await switch_model(started, ALT_MODEL)
        assert await _turn(started, messages, "second") == "second answer"

        assert errors == []
        assert provider.models == [main_id, sub_id, main_id, alt_id]
        [old_main] = http_client_ledger.for_profile(MAIN_MODEL)
        [new_main] = http_client_ledger.for_profile(ALT_MODEL)
        old_sub, new_sub = http_client_ledger.for_profile(SUB_MODEL)
        assert old_main.is_closed and old_sub.is_closed
        assert http_client_ledger.open() == [new_main, new_sub]

        await started.engine.shutdown()
        assert http_client_ledger.open() == []
