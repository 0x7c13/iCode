# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for EventBus — pub/sub and streaming."""

from __future__ import annotations

import asyncio
import logging

import pytest

from chrys.foundation.events import bus as bus_module
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationMessage, UserMessage
from chrys.foundation.models.invocations import InvocationOrigin


@pytest.mark.asyncio
async def test_publish_subscribe() -> None:
    """Subscriber receives published events."""
    bus = EventBus()
    received: list[UserMessage] = []

    async def handler(event: UserMessage) -> None:
        received.append(event)

    await bus.subscribe(UserMessage, handler)
    await bus.publish(UserMessage(text="hello"))

    assert len(received) == 1
    assert received[0].text == "hello"


@pytest.mark.asyncio
async def test_subscribe_only_matching_type() -> None:
    """Subscriber does not receive events of different types."""
    bus = EventBus()
    received: list[UserMessage] = []

    async def handler(event: UserMessage) -> None:
        received.append(event)

    await bus.subscribe(UserMessage, handler)
    await bus.publish(InvocationMessage(text="agent says hi", origin=InvocationOrigin("turn", "", "turn-test", None)))

    assert len(received) == 0


@pytest.mark.asyncio
async def test_unsubscribe() -> None:
    """Unsubscribed handler stops receiving events."""
    bus = EventBus()
    received: list[UserMessage] = []

    async def handler(event: UserMessage) -> None:
        received.append(event)

    await bus.subscribe(UserMessage, handler)
    await bus.publish(UserMessage(text="first"))
    await bus.unsubscribe(UserMessage, handler)
    await bus.publish(UserMessage(text="second"))

    assert len(received) == 1
    assert received[0].text == "first"


@pytest.mark.asyncio
async def test_stream() -> None:
    """Stream yields events of specified types."""
    bus = EventBus()
    collected: list[str] = []

    async def producer() -> None:
        await asyncio.sleep(0.01)
        await bus.publish(UserMessage(text="msg1"))
        await bus.publish(InvocationMessage(text="agent1", origin=InvocationOrigin("turn", "", "turn-test", None)))
        await bus.publish(UserMessage(text="msg2"))

    stream = bus.stream(UserMessage)

    async with stream:
        task = asyncio.create_task(producer())

        # Collect 2 UserMessage events
        for _ in range(2):
            event = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
            collected.append(event.text)

        await task

    assert collected == ["msg1", "msg2"]


@pytest.mark.asyncio
async def test_handler_exception_does_not_block_other_subscribers() -> None:
    """One failing callback must not prevent later callbacks or streams."""
    bus = EventBus()
    received: list[str] = []

    async def bad_handler(_event: UserMessage) -> None:
        raise RuntimeError("boom")

    async def good_handler(event: UserMessage) -> None:
        received.append(event.text)

    await bus.subscribe(UserMessage, bad_handler)
    await bus.subscribe(UserMessage, good_handler)

    stream = bus.stream(UserMessage)
    async with stream:
        await bus.publish(UserMessage(text="hello"))
        streamed = await asyncio.wait_for(stream.__anext__(), timeout=1.0)

    assert received == ["hello"]
    assert streamed.text == "hello"


@pytest.mark.asyncio
async def test_raise_handler_errors_fans_out_before_reraising() -> None:
    """Command publishers can fail fast without starving later subscribers."""
    bus = EventBus()
    received: list[str] = []

    async def bad_handler(_event: UserMessage) -> None:
        raise RuntimeError("boom")

    async def good_handler(event: UserMessage) -> None:
        received.append(event.text)

    await bus.subscribe(UserMessage, bad_handler)
    await bus.subscribe(UserMessage, good_handler)

    stream = bus.stream(UserMessage)
    async with stream:
        with pytest.raises(RuntimeError, match="boom"):
            await bus.publish(UserMessage(text="hello"), raise_handler_errors=True)
        streamed = await asyncio.wait_for(stream.__anext__(), timeout=1.0)

    assert received == ["hello"]
    assert streamed.text == "hello"


@pytest.mark.asyncio
async def test_stream_queue_is_unbounded_and_lossless() -> None:
    """A slow stream consumer retains every event in order — never drops.

    Dropping here would silently truncate streamed LLM output on the
    headless/ACP path, so the queue is unbounded by design: a lagging consumer
    accumulates events rather than losing them, and ``publish`` never aborts.
    """
    bus = EventBus()

    async with bus.stream(UserMessage) as stream:
        # Publish far more than any bounded queue would hold, without consuming.
        for i in range(1000):
            await bus.publish(UserMessage(text=f"msg{i}"))

        # Nothing was dropped: every event drains, in publish order.
        collected = [(await asyncio.wait_for(stream.__anext__(), timeout=1.0)).text for _ in range(1000)]

    assert collected == [f"msg{i}" for i in range(1000)]


@pytest.mark.asyncio
async def test_stream_backlog_emits_one_shot_high_water_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A badly-lagging consumer is observable via a single warning — still lossless."""
    monkeypatch.setattr(bus_module, "_STREAM_QUEUE_HIGH_WATER", 3)
    bus = EventBus()

    async with bus.stream(UserMessage) as stream:
        with caplog.at_level(logging.WARNING, logger="chrys.foundation.events.bus"):
            for i in range(10):
                await bus.publish(UserMessage(text=f"m{i}"))

        warnings = [r for r in caplog.records if "backlog exceeded" in r.getMessage()]
        assert len(warnings) == 1  # one-shot, not once-per-event

        # Warning is diagnostic only — all 10 events are still retained.
        collected = [(await asyncio.wait_for(stream.__anext__(), timeout=1.0)).text for _ in range(10)]

    assert collected == [f"m{i}" for i in range(10)]
