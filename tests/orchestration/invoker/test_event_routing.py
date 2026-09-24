# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Headless boundary, immutable origins, and explicitly captured callbacks."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationMessage
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.orchestration.invoker.origin import (
    BoundEmitter,
    InvocationPublishers,
    current_invocation_origin,
    invocation_routing_key,
)
from chrys.orchestration.session_host import allows_headless_event
from tests.support.invocation_events import ORIGIN_IDS, ORIGINS, PHASES, PROSE_PHASES, has_chat_card, projection_event


@pytest.mark.parametrize("origin", ORIGINS, ids=ORIGIN_IDS)
@pytest.mark.parametrize("phase", PHASES)
def test_headless_projection_matrix(origin: InvocationOrigin, phase: str) -> None:
    expected = origin.kind == "turn" or (has_chat_card(origin) and phase not in PROSE_PHASES)
    assert allows_headless_event(projection_event(origin, phase)) is expected


def test_live_fact_requires_origin_and_cannot_be_retargeted() -> None:
    with pytest.raises(TypeError, match="origin"):
        InvocationMessage()  # type: ignore[missing-argument]
    with pytest.raises(ValueError, match="origin"):
        InvocationMessage(origin=None)  # type: ignore[invalid-argument-type]
    event = InvocationMessage(origin=ORIGINS[0])
    with pytest.raises(FrozenInstanceError):
        event.origin = ORIGINS[1]
    with pytest.raises(FrozenInstanceError):
        event.session_id = "other"
    with pytest.raises(FrozenInstanceError):
        del event.origin


async def test_concurrent_delayed_callbacks_keep_bound_ancestry_after_registry_release() -> None:
    bus = EventBus()
    registry = InvocationPublishers(bus)
    parent = ORIGINS[0]
    origins = [InvocationOrigin("sub_agent", "s1", str(index), parent) for index in range(2)]
    captured = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()
    events = []

    async def record(event: InvocationMessage) -> None:
        events.append(event)

    async def callback(index: int) -> None:
        token = invocation_routing_key.set(origins[index].invocation_id)
        try:
            emitter = registry.capture()
            captured[index].set()
            await release.wait()
            await emitter.publish(InvocationMessage(origin=emitter.origin, text=str(index)))
        finally:
            invocation_routing_key.reset(token)

    await bus.subscribe(InvocationMessage, record)
    for origin in origins:
        registry.bind(origin)
    tasks = [asyncio.create_task(callback(index)) for index in range(2)]
    try:
        await asyncio.gather(*(ready.wait() for ready in captured))
        for origin in origins:
            registry.unbind(origin)
        token = current_invocation_origin.set(InvocationOrigin("turn", "s1", "later", None))
        try:
            release.set()
            await asyncio.gather(*tasks)
        finally:
            current_invocation_origin.reset(token)
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert sorted((event.text, event.origin) for event in events) == list(zip(("0", "1"), origins, strict=True))
    assert all(event.origin.parent is parent for event in events)
    with pytest.raises(ValueError, match="unbound"):
        registry.capture()
    with pytest.raises(ValueError, match="origin"):
        await BoundEmitter(bus, origins[0]).publish(InvocationMessage(origin=origins[1]))


async def test_finished_turn_rejects_publication_with_stale_origin(executor) -> None:
    with pytest.raises(ValueError, match="active turn"):
        _ = executor._emitter
    await executor.backend.run(executor.inputs.fresh_request(["work"]))
    with pytest.raises(ValueError, match="active turn"):
        _ = executor._emitter
    with pytest.raises(ValueError, match="unbound"):
        executor._invocation_publishers.capture()
