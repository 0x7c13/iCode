# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A real child control publisher rejects invalid bound origins before the bus."""

from __future__ import annotations

from dataclasses import replace

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationResumed
from chrys.kernel import Agent
from chrys.service.llm.mock import MockChatClient
from tests.orchestration.sub_agents._controller_fixtures import _make_controller


async def test_child_control_publisher_rejects_missing_and_foreign_origin():
    bus = EventBus()
    controller = _make_controller(Agent(client=MockChatClient()), bus)
    events = []

    async def capture(event):
        events.append(event)

    await bus.subscribe(InvocationResumed, capture)
    try:
        await controller.publish_resumed()
        assert len(events) == 1
        original = controller._emitter
        with pytest.raises(ValueError, match="origin"):
            controller._emitter = replace(original, origin=None)
        assert controller._emitter is original
        controller._emitter = replace(original, origin=replace(controller.origin, session_id="foreign"))
        with pytest.raises(ValueError, match="session"):
            await controller.publish_resumed()
        assert len(events) == 1
        assert events[0].origin.invocation_id == controller.origin.invocation_id
    finally:
        await bus.unsubscribe(InvocationResumed, capture)
        await controller.policy.backend.owner.aclose()
