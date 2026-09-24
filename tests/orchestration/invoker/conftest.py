# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real main pass resources for contract and attempt tests."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.kernel import Agent, AgentSession, LoopRecorder
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.orchestration.invoker.resources import Conversation
from chrys.service.agent_middleware import ApprovalMiddleware, AskUserMiddleware
from chrys.service.agent_middleware.injection import InjectionMiddleware
from chrys.service.approval.policy import ApprovalPolicy
from chrys.service.llm.mock import MockChatClient
from chrys.service.profiles.agents.schema import ApprovalConfig


@pytest.fixture
async def executor() -> AsyncIterator[TurnBindings]:
    bus = EventBus()
    instance = TurnBindings(
        conversation=Conversation(),
        agent=Agent(client=MockChatClient()),
        session=AgentSession(),
        event_bus=bus,
        approval_middleware=ApprovalMiddleware(ApprovalPolicy(ApprovalConfig()), bus),
        ask_user_middleware=AskUserMiddleware(bus),
        injection_middleware=InjectionMiddleware(),
        loop_recorder=LoopRecorder(),
    )
    instance.resource_scope.own(instance.approval.close)
    try:
        yield instance
    finally:
        await instance.resource_scope.aclose()
