# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real main/child shells install the local wire policy in both run modes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import create_autospec

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import InvocationPaused, InvocationRetryAttempt
from chrys.foundation.models.session_env import SessionEnvironment
from chrys.foundation.platform import get_platform
from chrys.kernel import Agent, AgentSession, ChatResponse, ResponseStream
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.orchestration.invoker.resources import Conversation
from chrys.orchestration.sub_agents import tools as tools_module
from chrys.orchestration.sub_agents.kernel_policy import KernelSubAgentPolicy
from chrys.orchestration.sub_agents.tools import SubAgentTools
from chrys.service.agent_middleware.control.approval import ApprovalMiddleware
from chrys.service.agent_middleware.control.ask_user import AskUserMiddleware
from chrys.service.agent_middleware.injection import InjectionMiddleware
from chrys.service.approval.policy import ApprovalPolicy
from chrys.service.llm.mock import MockChatClient
from chrys.service.profiles.agents.schema import (
    AgentProfile,
    ApprovalConfig,
    CompactionConfig,
    SubAgentRef,
    ToolsConfig,
)
from chrys.service.profiles.models.schema import ModelProfile
from tests.kernel.test_wire_retry import _ScriptedWire, _text_response, _text_update
from tests.orchestration.invoker._main_pass import fresh_pass
from tests.support.event_capture import capture_event_sequence
from tests.support.waiting import wait_for


@pytest.mark.parametrize("shell", ["turn", "child"])
@pytest.mark.parametrize("mode", ["blocking", "stream", "stall"])
async def test_local_shell_policy_records_wire_inputs_and_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shell: str, mode: str
) -> None:
    stream = mode != "blocking"
    stream_timeout = 0.02 if mode == "stall" else 5.0

    def stalled() -> ResponseStream:
        async def updates():
            yield _text_update("discarded")
            await asyncio.Event().wait()

        return ResponseStream(updates(), finalizer=ChatResponse.from_updates)

    if mode == "stall":
        outcomes = [ConnectionError("transient"), stalled, stalled]
        if shell == "turn":
            outcomes += [ConnectionError("fallback transient"), _text_response("done")]
    else:
        outcomes = [ConnectionError("transient"), [_text_update("done")] if stream else _text_response("done")]
    wire = _ScriptedWire(outcomes)
    client = MockChatClient()
    monkeypatch.setattr(
        client, "_inner_get_response", create_autospec(client._inner_get_response, side_effect=wire.get_response)
    )
    bus = EventBus()
    tools = None
    executor = None
    task = None
    original_init = KernelSubAgentPolicy.__init__

    def controller_init(*args, **kwargs):
        kwargs["backoff_schedule"] = (0,)
        original_init(*args, **kwargs)

    monkeypatch.setattr(KernelSubAgentPolicy, "__init__", create_autospec(original_init, side_effect=controller_init))
    try:
        async with capture_event_sequence(
            bus, InvocationRetryAttempt, InvocationRetryAttempt, InvocationPaused
        ) as events:
            if shell == "turn":
                agent = Agent(client=client, instructions="fixed instruction")
                executor = TurnBindings(
                    conversation=Conversation(),
                    agent=agent,
                    session=AgentSession(),
                    event_bus=bus,
                    approval_middleware=ApprovalMiddleware(ApprovalPolicy(ApprovalConfig()), bus),
                    ask_user_middleware=AskUserMiddleware(bus),
                    injection_middleware=InjectionMiddleware(),
                )
                executor.resource_scope.own(executor.approval.close)
                executor._max_retries_override = 1
                executor._stream_attempt_timeout = stream_timeout
                executor._stream = stream
                executor._chat_options = {"store": False}
                monkeypatch.setattr(executor, "_BACKOFF_SCHEDULE", (0,))
                await fresh_pass(executor, ["fixed input"])
            else:
                monkeypatch.setattr(
                    tools_module, "create_client", create_autospec(tools_module.create_client, return_value=client)
                )
                tools = SubAgentTools(event_bus=bus, session_id="parent", session_dir=tmp_path, max_transient_retries=1)
                await tools.register(
                    SubAgentRef(profile="Explore", tool_name="Explore"),
                    AgentProfile(
                        name="Explore",
                        instructions="fixed instruction",
                        tools=ToolsConfig(builtins=[]),
                        compaction=CompactionConfig(enabled=False),
                    ),
                    SessionEnvironment(cwd=str(tmp_path), platform=get_platform()),
                    settings=Settings(),
                    fallback_profile=ModelProfile(
                        id="mock",
                        name="mock",
                        provider="mock",
                        model_id="mock",
                        stream=stream,
                        http_read_timeout=stream_timeout,
                    ),
                )
                task = asyncio.create_task(tools.get_tools()[0].func(prompt="fixed input"))
                if mode == "stall":
                    await wait_for(
                        lambda: any((isinstance(e, InvocationPaused) and e.origin.kind == "sub_agent") for e in events),
                        description="real local streaming stall pauses child",
                    )
                    controller = next(iter(tools._controllers.values()))
                    assert controller.policy._stream is True
                    controller.request_abort()
                    assert (await task).startswith("Error:")
                else:
                    await wait_for(
                        lambda: (
                            task.done()
                            or any(
                                (isinstance(event, InvocationPaused) and event.origin.kind == "sub_agent")
                                for event in events
                            )
                        ),
                        description="child local retry finishes without a manual pause",
                    )
                    assert task.done(), "local wire retry unexpectedly paused"
                    assert await task == "done"
        expected_flags = ([True] * 3 + ([False] * 2 if shell == "turn" else [])) if mode == "stall" else [stream] * 2
        assert [call["stream"] for call in wire.calls] == expected_flags
        assert wire.outcomes == []
        # Compare all submitted contents, including shell-provided context, for every retry.
        first = [(m.role, m.text) for m in wire.calls[0]["messages"]]
        assert [(role, text.split(" <system-reminder>", 1)[0]) for role, text in first] == [("user", "fixed input")]
        assert all([(m.role, m.text) for m in call["messages"]] == first for call in wire.calls)
        retry_events = [event for event in events if isinstance(event, InvocationRetryAttempt | InvocationRetryAttempt)]
        expected = [(1, 1)] if mode != "stall" else [(1, 1), (1, 1)] + ([(2, 2), (1, 1)] if shell == "turn" else [])
        assert [(event.attempt, event.max_attempts) for event in retry_events] == expected
    finally:
        if task is not None:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if tools is not None:
            await tools.cleanup()
        if executor is not None:
            await executor.resource_scope.aclose()
