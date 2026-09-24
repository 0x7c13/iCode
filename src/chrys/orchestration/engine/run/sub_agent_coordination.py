# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Engine-internal sub-agent pause coordination."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from chrys.orchestration.engine.state.machine import Trigger

if TYPE_CHECKING:
    from chrys.foundation.events.types import (
        InvocationAborted,
        InvocationAbortRequested,
        InvocationCascadeAborted,
        InvocationPaused,
        InvocationResumed,
        InvocationRetryRequested,
    )
    from chrys.orchestration.engine.run.turn_state import TurnRuntimeState
    from chrys.orchestration.engine.state.current_agent import CurrentAgent
    from chrys.orchestration.engine.state.machine import EngineStateMachine
    from chrys.orchestration.engine.trajectory import TrajectoryRecorder
    from chrys.service.session.history import SessionHistoryManager


logger = logging.getLogger(__name__)


async def on_sub_agent_retry(current: CurrentAgent, event: InvocationRetryRequested) -> None:
    """Route a user's per-card Retry click to the owning controller."""
    if current.loaded is None or current.loaded.sub_agent_tools is None:
        return
    current.loaded.sub_agent_tools.request_retry(event.invocation_id)


async def on_sub_agent_abort(current: CurrentAgent, event: InvocationAbortRequested) -> None:
    """Route a user's per-card Abort click to the owning controller."""
    if current.loaded is None or current.loaded.sub_agent_tools is None:
        return
    current.loaded.sub_agent_tools.request_abort(event.invocation_id)


async def on_sub_agent_paused(
    turn_state: TurnRuntimeState,
    history: SessionHistoryManager,
    fsm: EngineStateMachine,
    trajectory_recorder: TrajectoryRecorder,
    event: InvocationPaused,
) -> None:
    """Track a newly paused sub-agent and drive FSM / marker.

    Idempotent on the paused set — if the same id arrives twice
    (controller re-publishes after retry exhaustion, for example)
    the FSM transition only fires on the 0→1 edge.

    Defensive FSM guard: if the parent run has already terminated
    (e.g. the bus is dispatching a stale pause event that was queued
    before the parent's task was cancelled and ``_post_run`` ran),
    drop the event silently rather than re-inserting a marker on top
    of an already-terminal ``interrupted``/``error`` marker.  The
    parent invariant ("parent run ends only after all sub-agents
    resolve") should make this unreachable, but defense in depth
    keeps history well-formed if that invariant ever breaks.
    """
    if event.origin.kind != "sub_agent" or event.origin.root.kind == "workflow_node":
        return
    if not fsm.is_running():
        logger.debug(
            "Ignoring InvocationPaused for %s: FSM=%s (parent run already terminated)",
            event.origin.invocation_id,
            fsm.state.name,
        )
        return
    first = not turn_state.paused_sub_agents
    turn_state.paused_sub_agents.add(event.origin.invocation_id)
    if first:
        fsm.try_transition(Trigger.SUB_AGENT_PAUSED)
        await trajectory_recorder.turn_suspended()
    if history.is_bound:
        history.upsert_awaiting_sub_agents_marker(sorted(turn_state.paused_sub_agents))


async def on_sub_agent_unpaused(
    turn_state: TurnRuntimeState,
    history: SessionHistoryManager,
    fsm: EngineStateMachine,
    trajectory_recorder: TrajectoryRecorder,
    event: InvocationResumed | InvocationAborted | InvocationCascadeAborted,
) -> None:
    """Common handler — retry/abort/cascade all remove the invocation from the paused set.

    FSM transitions only on the N→0 edge (last paused sub-agent
    resolved).  Marker is updated or stripped accordingly.
    """
    if event.origin.kind != "sub_agent" or event.origin.root.kind == "workflow_node":
        return
    if event.origin.invocation_id not in turn_state.paused_sub_agents:
        return
    turn_state.paused_sub_agents.discard(event.origin.invocation_id)
    if not turn_state.paused_sub_agents:
        fsm.try_transition(Trigger.SUB_AGENT_RESOLVED)
        await trajectory_recorder.turn_resumed()
        if history.is_bound:
            history.remove_awaiting_sub_agents_marker()
    elif history.is_bound:
        history.upsert_awaiting_sub_agents_marker(sorted(turn_state.paused_sub_agents))
    # Defensive invariant: FSM ``AWAITING_SUB_AGENTS`` must mirror a
    # non-empty paused set.  A desync here would mean either a stale
    # pause event leaked past the ``_fsm.is_running()`` guard in
    # ``_on_sub_agent_paused`` or a ``try_transition`` silently no-op'd.
    # Log instead of raising so a surprise in production doesn't crash
    # the engine mid-run — the log points straight at the bug.
    if bool(turn_state.paused_sub_agents) != fsm.is_awaiting_sub_agents():
        logger.error(
            "FSM/paused-set desync: paused=%s FSM=%s",
            sorted(turn_state.paused_sub_agents),
            fsm.state.name,
        )


def clear_parent_paused_state(turn_state: TurnRuntimeState, history: SessionHistoryManager) -> None:
    """Clear engine-side paused sub-agent state after a parent run ends."""
    turn_state.paused_sub_agents.clear()
    history.remove_awaiting_sub_agents_marker()
