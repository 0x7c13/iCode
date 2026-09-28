# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A resolver that fails after the provider has answered is retried; a malformed lookup is not.

"Name not found" for a host this process has never reached is a typo and
fails at once. Once a response has come back through the same first hop, the
same answer means the resolver or network is flapping, so the turn's wire
lane retries it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import pytest

from chrys.foundation.errors import Origin
from chrys.foundation.events.types import Error, InvocationMessage, InvocationRetryAttempt, UserMessage
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.service.llm.route_facts import has_reached
from tests.support.event_capture import capture_events
from tests.support.llm_client_engines import ClientEngine, start_client_engine
from tests.support.network_faults import NetworkFaults, gaierror, network_faults
from tests.support.provider_errors import API_HOST
from tests.support.waiting import ENGINE_TURN_TIMEOUT, await_run_task_chain, wait_for

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support.engines import AgentEngineFactory


def _completion(text: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 0,
            "model": "stub",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


async def _answer_and_close(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Answer one request, then close: every request opens a connection, so every one resolves."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
        length = next(
            (
                int(line.split(b":", 1)[1])
                for line in head.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            ),
            0,
        )
        await reader.readexactly(length)
        body = _completion("answer")
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        await writer.drain()
    except asyncio.IncompleteReadError, ConnectionError:
        pass
    finally:
        writer.close()


@contextlib.asynccontextmanager
async def _provider() -> AsyncIterator[int]:
    server = await asyncio.start_server(_answer_and_close, "127.0.0.1", 0)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        server.close()
        server.close_clients()
        await server.wait_closed()


class _Resolver:
    """``API_HOST`` resolves to loopback, except for failures queued ahead of it."""

    def __init__(self, faults: NetworkFaults) -> None:
        self.failures: list[int] = []
        self.always_fail: int | None = None
        faults.resolve_to(API_HOST, "127.0.0.1")
        self._loopback = faults.resolve_rules[API_HOST]
        faults.resolve_rules[API_HOST] = self._answer

    def _answer(self) -> list[tuple[Any, ...]] | BaseException:
        if self.always_fail is not None:
            return gaierror(self.always_fail)
        if self.failures:
            return gaierror(self.failures.pop(0))
        return self._loopback()


def _final_answers(messages: list[InvocationMessage]) -> int:
    return sum(1 for message in messages if message.is_final and message.origin.kind == "turn")


async def _turn(started: ClientEngine, messages: list[InvocationMessage], errors: list[Error], text: str) -> None:
    """Run one turn to its end, whether that is an answer or an error."""
    ended = _final_answers(messages) + len(errors)
    await started.bus.publish(UserMessage(text=text))
    await wait_for(
        lambda: _final_answers(messages) + len(errors) > ended,
        timeout=ENGINE_TURN_TIMEOUT,
        description=f"turn {text!r} to end",
    )
    await await_run_task_chain(started.engine, turn_state=started.engine.turns.turn_state)


@pytest.mark.usefixtures("direct_route")
async def test_noname_after_a_reached_first_hop_is_retried_and_badflags_is_not(
    agent_engine: AgentEngineFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(TurnBindings, "_BACKOFF_SCHEDULE", (0,))
    async with _provider() as port:
        with network_faults() as faults:
            resolver = _Resolver(faults)
            started = await start_client_engine(
                agent_engine, tmp_path, sub_agent=False, base_url=f"http://{API_HOST}:{port}/v1"
            )
            messages = await capture_events(started.bus, InvocationMessage)
            retries = await capture_events(started.bus, InvocationRetryAttempt)
            errors = await capture_events(started.bus, Error)

            await _turn(started, messages, errors, "first")
            assert (_final_answers(messages), errors) == (1, [])
            assert has_reached(Origin("http", API_HOST, port))
            assert (len(faults.resolve_calls), retries) == (1, [])

            resolver.failures.append(socket.EAI_NONAME)
            await _turn(started, messages, errors, "second")
            assert (errors, _final_answers(messages)) == ([], 2)
            assert len(faults.resolve_calls) == 3
            [retry] = retries
            assert (retry.origin.kind, retry.scope) == ("turn", "wire")

            resolver.always_fail = socket.EAI_BADFLAGS
            await _turn(started, messages, errors, "third")
            # A malformed lookup can never succeed: one attempt, no retry.
            assert len(errors) == 1
            assert len(faults.resolve_calls) == 4
            assert len(retries) == 1
            assert _final_answers(messages) == 2
