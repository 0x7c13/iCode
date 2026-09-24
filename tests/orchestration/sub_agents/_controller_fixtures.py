# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Controller construction and scripted outcomes shared by sub-agent tests."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from chrys.foundation.events.bus import EventBus
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.kernel import AgentSession, LoopRecorder, Message
from chrys.kernel.client import BaseChatClient
from chrys.orchestration.invoker.resources import Conversation
from chrys.orchestration.sub_agents.kernel_policy import KernelSubAgentPolicy
from chrys.orchestration.sub_agents.shell import SubAgentToolShell
from chrys.service.agent_middleware.events.sub_agent_events import SubAgentEventMiddleware
from chrys.service.llm.mock import MockChatClient


@dataclass
class _StubResponse:
    text: str = ""
    messages: list[Message] = field(default_factory=list)


@dataclass
class _ScriptedAgent:
    """Tiny ``agent.run()`` stand-in driven by a scripted outcome list.

    Each ``await run()`` call pops the next outcome: ``"ok"`` returns a
    successful response, any Exception instance is raised.  Empty-list
    behaviour raises ``AssertionError`` so test authors can't accidentally
    under-script.
    """

    outcomes: list[Any] = field(default_factory=list)
    calls: list[list[Any]] = field(default_factory=list)
    client: BaseChatClient = field(default_factory=MockChatClient)

    async def run(self, prompt, **kwargs):
        self.calls.append(list(prompt))
        assert self.outcomes, "ScriptedAgent: ran out of scripted outcomes"
        outcome = self.outcomes.pop(0)
        if callable(outcome):
            outcome = outcome(prompt, kwargs)
            if inspect.isawaitable(outcome):
                outcome = await outcome
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, _StubResponse):
            return outcome
        return _StubResponse(text=outcome)


async def _collect(bus: EventBus, event_type: type):
    """Subscribe + return a list that grows as events fire."""
    captured: list = []

    async def _on(ev):
        captured.append(ev)

    await bus.subscribe(event_type, _on)
    return captured


def _make_controller(
    agent,
    bus,
    *,
    backoff=(0, 0, 0, 0, 0),
    max_retries=2,
    sleep_middleware=None,
    session: AgentSession | None = None,
    loop_recorder: LoopRecorder | None = None,
    prompt: str = "do the thing",
    tool_event_middleware: SubAgentEventMiddleware | None = None,
    run_kwargs: dict | None = None,
    parent_interrupted_result_commit=None,
    pass_start_hooks=(),
    hosted_commits_probe=None,
    stream: bool = False,
    human_failure_decisions: bool = True,
):
    return kernel_shell(
        conversation=Conversation(),
        human_failure_decisions=human_failure_decisions,
        parent_origin=None,
        invocation_id="inv-1",
        tool_name="Explore",
        agent_name="Explore",
        agent=agent,
        **_controller_runtime(session=session, loop_recorder=loop_recorder),
        prompt=prompt,
        run_kwargs=run_kwargs if run_kwargs is not None else {},
        event_bus=bus,
        session_id="s-1",
        max_retries=max_retries,
        backoff_schedule=backoff,
        sleep_middleware=sleep_middleware,
        tool_event_middleware=tool_event_middleware,
        parent_interrupted_result_commit=parent_interrupted_result_commit,
        pass_start_hooks=pass_start_hooks,
        hosted_commits_probe=hosted_commits_probe,
        stream=stream,
    )


def _controller_runtime(
    *,
    session: AgentSession | None = None,
    loop_recorder: LoopRecorder | None = None,
) -> dict[str, Any]:
    return {"session": session or AgentSession(), "loop_recorder": loop_recorder or LoopRecorder()}


def kernel_shell(
    *,
    invocation_id,
    conversation,
    parent_origin,
    tool_name,
    agent_name,
    event_bus,
    session_id=None,
    parent_interrupted_result_commit=None,
    human_failure_decisions=True,
    **policy_options,
):
    """Compose the real shared shell and kernel policy with the test's exact recipe."""
    origin = InvocationOrigin("sub_agent", session_id or "", invocation_id, parent_origin)
    shell = SubAgentToolShell(
        origin=origin,
        tool_name=tool_name,
        agent_name=agent_name,
        event_bus=event_bus,
        human_failure_decisions=human_failure_decisions,
    )
    if parent_interrupted_result_commit is not None:
        shell.bind_parent_interrupt_commit(parent_interrupted_result_commit)
    shell.attach_policy(KernelSubAgentPolicy(shell=shell, conversation=conversation, **policy_options))
    return shell
