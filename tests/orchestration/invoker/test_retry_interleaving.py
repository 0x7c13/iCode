# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Validation exemptions and transient budgets remain independent in both shells."""

from __future__ import annotations

from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationRetryAttempt
from chrys.kernel import Agent, AgentResponse, AgentSession, Message
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.orchestration.invoker.resources import Conversation
from chrys.service.agent_middleware.control.approval import ApprovalMiddleware
from chrys.service.agent_middleware.control.ask_user import AskUserMiddleware
from chrys.service.agent_middleware.injection import InjectionMiddleware
from chrys.service.agent_middleware.response_validation import (
    RetryableResponseValidationError,
    ValidationRetryExemption,
)
from chrys.service.approval.policy import ApprovalPolicy
from chrys.service.llm.mock import MockChatClient
from chrys.service.profiles.agents.schema import ApprovalConfig
from tests.orchestration.invoker._main_pass import continuation_pass
from tests.orchestration.sub_agents._controller_fixtures import _make_controller
from tests.support.event_capture import capture_event_sequence


@pytest.mark.parametrize("shell", ["turn", "child"])
async def test_validation_transient_interleaving_preserves_budgets_and_pass_hooks(
    monkeypatch: pytest.MonkeyPatch, shell: str
) -> None:
    bus = EventBus()
    hooks: list[str] = []
    outcomes = [
        RetryableResponseValidationError(
            "empty", exemption=ValidationRetryExemption(attempt=1, max_attempts=3, delay_seconds=0)
        ),
        ConnectionError("transport one"),
        RetryableResponseValidationError(
            "whitespace", exemption=ValidationRetryExemption(attempt=2, max_attempts=3, delay_seconds=0)
        ),
        ConnectionError("transport two"),
        AgentResponse(messages=[Message("assistant", ["done"])]),
    ]
    agent = create_autospec(Agent, instance=True)
    agent.client = MockChatClient()

    async def next_result() -> AgentResponse:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    # The autospecced boundary validates Agent.run's full signature. Each call
    # returns a fresh coroutine, like the real non-streaming Agent.run.
    agent.run.side_effect = lambda *args, **kwargs: next_result()
    async with capture_event_sequence(bus, InvocationRetryAttempt, InvocationRetryAttempt) as events:
        if shell == "turn":
            executor = TurnBindings(
                conversation=Conversation(),
                agent=agent,
                session=AgentSession(),
                event_bus=bus,
                approval_middleware=ApprovalMiddleware(ApprovalPolicy(ApprovalConfig()), bus),
                ask_user_middleware=AskUserMiddleware(bus),
                injection_middleware=InjectionMiddleware(),
                run_cycle_start_hooks=(lambda: hooks.append("pass"),),
            )
            executor.resource_scope.own(executor.approval.close)
            executor._chat_options = {"store": True}
            executor._max_retries_override = 2
            monkeypatch.setattr(executor, "_BACKOFF_SCHEDULE", (0,))
            try:
                await continuation_pass(executor, [Message("user", ["work"])])
                assert not executor.state.run_failed
            finally:
                await executor.resource_scope.aclose()
        else:
            controller = _make_controller(
                agent,
                bus,
                max_retries=2,
                run_kwargs={"options": {"store": True}},
                pass_start_hooks=(lambda: hooks.append("pass"),),
            )
            assert await controller.run() == "done"
    assert hooks == ["pass"]
    assert agent.run.call_count == 5
    assert outcomes == []
    assert [(event.attempt, event.max_attempts) for event in events] == [(1, 3), (1, 2), (2, 3), (2, 2)]
    inputs = [[message.text for message in call.args[0]] for call in agent.run.call_args_list]
    assert inputs == [["work" if shell == "turn" else "do the thing"]] * 5
