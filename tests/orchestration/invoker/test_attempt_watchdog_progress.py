# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The service-side stream watchdog times idle gaps, not whole pulls.

Service-side storage has no kernel wire policy, so this watchdog alone times
the pull whose first byte waits on compaction and its LAST_WORDS side call.
That side call reports progress chunk by chunk; the watchdog must accept it,
still stall once it stops, and ignore reports that outlive its timer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.retry import StreamStall
from chrys.kernel import Agent, ChatResponse, Message, ResponseStream, report_wire_progress
from chrys.service.llm.mock import MockChatClient
from chrys.service.llm.one_shot import get_final_response
from tests.kernel.test_wire_retry import _text_update
from tests.orchestration.sub_agents._controller_fixtures import _make_controller
from tests.support.waiting import wait_for

_TIMEOUT = 1.0


def _side_call_client(monkeypatch: pytest.MonkeyPatch, *, chunks: int, then_hang: bool = False) -> MockChatClient:
    """A LAST_WORDS-shaped side call that streams one chunk per tenth of the timeout."""
    client = MockChatClient()

    def wire(*, messages, stream, options, **kwargs):
        async def updates():
            for index in range(chunks):
                await asyncio.sleep(_TIMEOUT / 10)
                yield _text_update(f"note {index} ")
            if then_hang:
                await asyncio.Event().wait()

        return ResponseStream(updates(), finalizer=ChatResponse.from_updates)

    monkeypatch.setattr(client, "_inner_get_response", create_autospec(client._inner_get_response, side_effect=wire))
    return client


def _service_side_attempt(
    monkeypatch: pytest.MonkeyPatch, before_first_byte: Callable[[], Awaitable[None]]
) -> asyncio.Task:
    client = MockChatClient()

    def wire(*, messages, stream, options, **kwargs):
        async def updates():
            await before_first_byte()
            yield _text_update("answer")

        return ResponseStream(updates(), finalizer=ChatResponse.from_updates)

    monkeypatch.setattr(client, "_inner_get_response", create_autospec(client._inner_get_response, side_effect=wire))
    controller = _make_controller(
        Agent(client=client), EventBus(), stream=True, run_kwargs={"options": {"store": True}}
    )
    controller.policy._stream_attempt_timeout = _TIMEOUT
    return asyncio.create_task(
        controller.policy._attempts._stream_single_attempt(
            controller.policy._active_run_input, controller.policy._run_kwargs
        )
    )


async def test_side_call_progress_keeps_the_pull_alive_past_the_stall_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    side_client = _side_call_client(monkeypatch, chunks=12)
    loop = asyncio.get_running_loop()

    async def last_words_side_call() -> None:
        await get_final_response(side_client, [Message("user", ["summarize"])], stream=True)

    started = loop.time()
    task = _service_side_attempt(monkeypatch, last_words_side_call)
    try:
        await wait_for(task.done, description="attempt past a long side call finished")
        response = await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert response.text == "answer"
    # 1.2 timeouts of side call in total; each gap leaves 0.9 of slack.
    assert loop.time() - started > _TIMEOUT


async def test_a_pull_still_stalls_once_its_side_call_stops_reporting(monkeypatch: pytest.MonkeyPatch) -> None:
    side_client = _side_call_client(monkeypatch, chunks=3, then_hang=True)
    loop = asyncio.get_running_loop()

    async def hung_side_call() -> None:
        await get_final_response(side_client, [Message("user", ["summarize"])], stream=True)

    started = loop.time()
    task = _service_side_attempt(monkeypatch, hung_side_call)
    try:
        await wait_for(task.done, description="attempt with a hung side call stalled")
        with pytest.raises(StreamStall):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert side_client._inner_get_response.call_count == 1
    # The last report landed 0.3 timeouts in; the stall came a timeout later,
    # past where a timer started with the pull would have fired.
    assert loop.time() - started >= 1.1 * _TIMEOUT


async def test_a_report_that_outlives_its_pull_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    late = asyncio.Event()
    strays: list[asyncio.Task[None]] = []

    async def late_report() -> None:
        await late.wait()
        report_wire_progress()

    async def spawn_stray_reporter() -> None:
        # Copies the pull's context, callback included, and outlives it.
        strays.append(asyncio.create_task(late_report()))

    task = _service_side_attempt(monkeypatch, spawn_stray_reporter)
    try:
        await wait_for(task.done, description="attempt finished before the stray report")
        response = await task
        late.set()
        await wait_for(strays[0].done, description="stray report delivered")
        await strays[0]
    finally:
        late.set()
        task.cancel()
        await asyncio.gather(task, *strays, return_exceptions=True)

    assert response.text == "answer"


async def test_a_report_made_while_the_stall_cancels_the_pull_keeps_it_a_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def report_while_unwinding() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            report_wire_progress()

    task = _service_side_attempt(monkeypatch, report_while_unwinding)
    try:
        await wait_for(task.done, description="attempt stalled")
        with pytest.raises(StreamStall):
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
