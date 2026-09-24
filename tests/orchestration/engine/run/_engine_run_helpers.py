# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Profile, registry, event-filter, and run-drain helpers shared by the engine run tests."""

from __future__ import annotations

from chrys.foundation.events.types import InvocationMessage
from chrys.orchestration.engine.engine import AgentEngine
from chrys.service.llm.mock import MockChatClient
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import AgentProfile, ApprovalConfig, CompactionConfig, ToolsConfig
from tests.support.waiting import ENGINE_TURN_TIMEOUT, await_run_task_chain, wait_for

_PROFILE = AgentProfile(
    name="Code",
    display_name="Code Agent",
    instructions="You are a coding assistant.",
    tools=ToolsConfig(builtins=[]),
    approval=ApprovalConfig(default="auto"),
    compaction=CompactionConfig(enabled=False),
)


def _make_registry() -> AgentProfileRegistry:
    registry = AgentProfileRegistry()
    registry.register(_PROFILE)
    return registry


def _filter(events: list, cls: type) -> list:
    return [e for e in events if isinstance(e, cls)]


def _history_messages(engine: AgentEngine) -> list:
    """Extract messages from the engine's current session history."""
    assert engine.current.loaded is not None
    state = engine.current.loaded.bindings.backend.history_state
    return state.get("messages", [])


async def _final_agent_messages_after_run(
    engine: AgentEngine,
    events: list,
    timeout: float = ENGINE_TURN_TIMEOUT,
) -> list:
    """Return final messages after a non-strict boundary drain.

    An internally cancelled run is reported as a drain outcome rather than
    raised; these resume tests diagnose it through their event assertions.
    """
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state, timeout=timeout)
    return [e for e in events if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final]


async def _final_agent_messages_after_strict_run(
    engine: AgentEngine,
    events: list,
    timeout: float = ENGINE_TURN_TIMEOUT,
) -> list:
    """Return final messages after a strict run-lifecycle drain.

    Unlike :func:`_final_agent_messages_after_run`, this one preserves an
    internal ``CancelledError`` because cancellation is material to
    interrupt tests.
    """
    await await_run_task_chain(
        engine,
        turn_state=engine.turns.turn_state,
        timeout=timeout,
        propagate_inner_cancel=True,
        expect_installed=True,
    )
    return [e for e in events if (isinstance(e, InvocationMessage) and e.origin.kind == "turn") and e.is_final]


async def _wait_for_call_count(
    client: MockChatClient,
    count: int,
    timeout: float = ENGINE_TURN_TIMEOUT,
) -> None:
    """Poll until the mock client has begun serving its ``count``-th LLM call.

    ``call_count`` bumps synchronously when a call reaches the raw client, so
    this marks "the run is mid-stream".  A fixed sleep instead flakes on slow
    runners (Windows CI): the interrupt can land before the run reaches the
    LLM, become a no-op, and the run completes with no interrupted marker.
    """
    await wait_for(
        lambda: client.call_count >= count,
        timeout=timeout,
        interval=0.01,
        description=f"mock client call count >= {count}",
    )
