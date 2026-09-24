# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Engine-level tests for sub-agent pause / resume / abort handlers.

``AgentEngine`` glues three moving parts together when a sub-agent pauses:

1. The ``SUB_AGENT_PAUSED`` / ``SUB_AGENT_RESOLVED`` FSM transitions
2. The ``awaiting_sub_agents`` history marker (upsert on pause, remove
   on last resolution)
3. Refusal of ``UserRetry`` while ``is_awaiting_sub_agents()``

FSM transitions and marker helpers each have their own unit tests; this
file exercises the *glue* — edge-triggering on 0↔1 paused controllers
and the refusal-with-Warning behaviour.

The tests construct a minimal :class:`AgentEngine` without ``start()`` —
we drive the FSM into ``RUNNING`` manually and bind a minimal history
dict. The handlers only touch ``self._fsm``, ``self._history``,
``self._turn_state.paused_sub_agents`` and ``self._bus``, so the heavy-weight
agent/executor setup is unnecessary for these unit tests.
"""

from __future__ import annotations

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    InvocationAborted,
    InvocationCascadeAborted,
    InvocationPaused,
    InvocationResumed,
    UserRetry,
    Warning,
)
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.invocations import InvocationOrigin
from chrys.kernel import Message
from chrys.orchestration.engine.assembly import assemble_agent_engine
from chrys.orchestration.engine.engine import AgentEngine
from chrys.orchestration.engine.state.machine import EngineState, Trigger
from tests.support.loaded_agents import install_loaded_agent


def _make_engine(*, engine_services) -> AgentEngine:
    """Build a minimally-initialized engine whose FSM is RUNNING.

    Skips ``start()``: no agent, no executor, no subscriptions. History
    is bound to a fresh empty state dict so marker helpers can mutate
    it. The executor is left as ``None`` — the only handler that
    dereferences it (``_on_user_retry``) is tested for the refuse path,
    which short-circuits before the executor is touched.
    """
    engine = assemble_agent_engine(EventBus(), settings=Settings())
    engine_services(engine).fsm.try_transition(Trigger.START)
    engine_services(engine).fsm.try_transition(Trigger.USER_MESSAGE)
    assert engine.state is EngineState.RUNNING
    engine_services(engine).history.bind({"messages": [Message("user", ["run the child agent"])]})
    # ``_on_user_retry`` short-circuits on ``_turn_bindings is None``; the
    # refusal-while-awaiting branch comes AFTER that check, so the test
    # needs a truthy placeholder to reach it. We never invoke the
    # executor itself — the awaiting branch returns early.
    install_loaded_agent(engine, bindings=object())  # type: ignore[assignment]
    return engine


def _paused_event(invocation_id: str, *, last_error: str = "boom") -> InvocationPaused:
    return InvocationPaused(
        agent_name="Explore",
        tool_name="Explore",
        reason="framework_exc",
        last_error=last_error,
        retry_attempts=0,
        origin=InvocationOrigin("sub_agent", "", invocation_id or "event-test", None),
    )


def _marker_ids(engine: AgentEngine, *, engine_services) -> list[str] | None:
    for m in engine_services(engine).history.messages:
        if m.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.AWAITING_SUB_AGENTS:
            return list(m.additional_properties.get("_invocation_ids", []))
    return None


# --- FSM + marker glue ---------------------------------------------------


@pytest.mark.asyncio
async def test_first_pause_transitions_fsm_and_inserts_marker(*, engine_services) -> None:
    engine = _make_engine(engine_services=engine_services)
    await engine._on_sub_agent_paused(_paused_event("inv-1"))

    assert engine.state is EngineState.AWAITING_SUB_AGENTS
    assert engine_services(engine).fsm.is_awaiting_sub_agents() is True
    assert engine.turns.turn_state.paused_sub_agents == {"inv-1"}
    assert _marker_ids(engine, engine_services=engine_services) == ["inv-1"]


@pytest.mark.asyncio
async def test_second_pause_updates_marker_only_not_fsm(*, engine_services) -> None:
    """Edge-triggered: FSM transition only fires on 0→1, second pause
    merely augments the set and refreshes the marker."""
    engine = _make_engine(engine_services=engine_services)
    await engine._on_sub_agent_paused(_paused_event("inv-1"))
    # FSM already in AWAITING_SUB_AGENTS — no extra transition.
    prev_state = engine.state
    await engine._on_sub_agent_paused(_paused_event("inv-2"))

    assert engine.state is prev_state
    assert engine.turns.turn_state.paused_sub_agents == {"inv-1", "inv-2"}
    assert _marker_ids(engine, engine_services=engine_services) == ["inv-1", "inv-2"]


@pytest.mark.asyncio
async def test_resume_intermediate_keeps_fsm_in_awaiting_and_updates_marker(*, engine_services) -> None:
    engine = _make_engine(engine_services=engine_services)
    await engine._on_sub_agent_paused(_paused_event("a"))
    await engine._on_sub_agent_paused(_paused_event("b"))

    await engine._on_sub_agent_unpaused(
        InvocationResumed(agent_name="Explore", origin=InvocationOrigin("sub_agent", "", "a", None))
    )

    assert engine.state is EngineState.AWAITING_SUB_AGENTS
    assert engine.turns.turn_state.paused_sub_agents == {"b"}
    assert _marker_ids(engine, engine_services=engine_services) == ["b"]


@pytest.mark.asyncio
async def test_last_resolution_drops_marker_and_returns_to_running(*, engine_services) -> None:
    engine = _make_engine(engine_services=engine_services)
    await engine._on_sub_agent_paused(_paused_event("inv-1"))
    await engine._on_sub_agent_unpaused(
        InvocationAborted(
            agent_name="Explore",
            last_error="x",
            origin=InvocationOrigin("sub_agent", "", "inv-1", None),
        )
    )

    assert engine.state is EngineState.RUNNING
    assert engine.turns.turn_state.paused_sub_agents == set()
    assert _marker_ids(engine, engine_services=engine_services) is None


@pytest.mark.asyncio
async def test_a_workflow_node_child_pause_never_touches_the_chat_turn(*, engine_services) -> None:
    """A workflow run's sub-agents pause under their node's shell, not the chat FSM: the engine's
    handlers ignore them even while a chat turn happens to be running."""
    engine = _make_engine(engine_services=engine_services)
    node = InvocationOrigin("workflow_node", "", "node-run", None)
    await engine._on_sub_agent_paused(
        InvocationPaused(
            agent_name="Child",
            tool_name="Child",
            reason="framework_exc",
            last_error="boom",
            retry_attempts=0,
            origin=InvocationOrigin("sub_agent", "", "inv-1", node),
        )
    )

    assert engine.state is EngineState.RUNNING
    assert engine.turns.turn_state.paused_sub_agents == set()
    assert _marker_ids(engine, engine_services=engine_services) is None

    # Nor does a workflow child's resume resolve a chat child that shares its invocation id.
    await engine._on_sub_agent_paused(_paused_event("inv-1"))
    await engine._on_sub_agent_unpaused(
        InvocationResumed(agent_name="Child", origin=InvocationOrigin("sub_agent", "", "inv-1", node))
    )

    assert engine.state is EngineState.AWAITING_SUB_AGENTS
    assert engine.turns.turn_state.paused_sub_agents == {"inv-1"}


@pytest.mark.asyncio
async def test_unpause_with_unknown_invocation_is_noop(*, engine_services) -> None:
    """A stale/late event for an id we never tracked shouldn't alter state."""
    engine = _make_engine(engine_services=engine_services)
    await engine._on_sub_agent_paused(_paused_event("inv-1"))

    await engine._on_sub_agent_unpaused(
        InvocationCascadeAborted(agent_name="Explore", origin=InvocationOrigin("sub_agent", "", "ghost", None))
    )

    assert engine.state is EngineState.AWAITING_SUB_AGENTS
    assert engine.turns.turn_state.paused_sub_agents == {"inv-1"}
    assert _marker_ids(engine, engine_services=engine_services) == ["inv-1"]


@pytest.mark.asyncio
async def test_cascade_abort_event_unpauses_same_as_abort(*, engine_services) -> None:
    """Cascade uses the shared ``_on_sub_agent_unpaused`` handler."""
    engine = _make_engine(engine_services=engine_services)
    await engine._on_sub_agent_paused(_paused_event("inv-1"))
    await engine._on_sub_agent_unpaused(
        InvocationCascadeAborted(agent_name="Explore", origin=InvocationOrigin("sub_agent", "", "inv-1", None))
    )

    assert engine.state is EngineState.RUNNING
    assert engine.turns.turn_state.paused_sub_agents == set()


# --- UserRetry refusal ---------------------------------------------------


@pytest.mark.asyncio
async def test_late_pause_after_run_terminated_is_ignored(*, engine_services) -> None:
    """A stale ``InvocationPaused`` event whose dispatch lands after the
    parent run has already transitioned to a terminal state (IDLE /
    INTERRUPTED / FAILED) must not re-insert a marker on top of the
    terminal one, nor repopulate ``turn_state.paused_sub_agents``.

    The controller-side invariant "parent run awaits every sub-agent"
    should keep this unreachable in practice, but the engine guards
    defensively so history stays well-formed if that invariant ever
    breaks (e.g. a bus-dispatch reordering bug).
    """
    engine = _make_engine(engine_services=engine_services)
    # Simulate the parent run having ended cleanly.
    engine_services(engine).fsm.try_transition(Trigger.RUN_COMPLETED)
    assert engine.state is EngineState.IDLE

    await engine._on_sub_agent_paused(_paused_event("late"))

    assert engine.state is EngineState.IDLE
    assert engine.turns.turn_state.paused_sub_agents == set()
    assert _marker_ids(engine, engine_services=engine_services) is None


@pytest.mark.asyncio
async def test_late_pause_after_interrupt_is_ignored(*, engine_services) -> None:
    """Same invariant — but for the interrupted terminal state."""
    engine = _make_engine(engine_services=engine_services)
    engine_services(engine).fsm.try_transition(Trigger.RUN_INTERRUPTED)
    assert engine.state is EngineState.INTERRUPTED

    await engine._on_sub_agent_paused(_paused_event("late-after-interrupt"))

    assert engine.state is EngineState.INTERRUPTED
    assert engine.turns.turn_state.paused_sub_agents == set()
    assert _marker_ids(engine, engine_services=engine_services) is None


@pytest.mark.asyncio
async def test_user_retry_refused_while_awaiting_sub_agents(*, engine_services) -> None:
    """While a sub-agent is paused the parent tool call is still in flight.
    A main-agent retry in this state would hang forever; the handler must
    refuse and publish a Warning instead."""
    engine = _make_engine(engine_services=engine_services)
    warnings: list[Warning] = []

    async def _on(ev: Warning) -> None:
        warnings.append(ev)

    await engine.event_bus.subscribe(Warning, _on)
    await engine._on_sub_agent_paused(_paused_event("inv-1"))

    await engine._on_user_retry(UserRetry())

    # The handler must have short-circuited before trying to resume —
    # nothing changed on the FSM or paused set.
    assert engine.state is EngineState.AWAITING_SUB_AGENTS
    assert engine.turns.turn_state.paused_sub_agents == {"inv-1"}
    # Allow the bus to dispatch the Warning.
    import asyncio as _asyncio

    await _asyncio.sleep(0)
    assert len(warnings) == 1
    warning = warnings[0]
    assert (warning.code, warning.message, warning.session_id) == (
        "sub_agent_paused",
        "Resolve paused sub-agent(s) first (Retry/Abort on the card) before retrying the main run.",
        None,
    )
    assert warning.display_message is not None
    assert warning.display_message.definition.key == "retry.sub_agent_paused"
    assert dict(warning.display_message.args) == {}
