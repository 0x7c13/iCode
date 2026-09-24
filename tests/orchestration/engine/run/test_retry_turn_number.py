# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests that retry/resume does not increment the turn number.

When a user interrupts an agent run and then resumes (or retries after an
error), the resumed work belongs to the same logical user turn.  The mutation
tracker should record mutations under the same turn_id, not a new one.

Regression: prior to the fix, pre-run setup always incremented
``_turn_number`` and called ``start_turn()``, so an interrupt+resume
sequence created an extra empty turn and the real mutations landed under
a wrong (incremented) turn_id.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import create_autospec

import pytest

import chrys.orchestration.engine.build.builder as builder_module
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    Error,
    InvocationMessage,
    SessionReady,
    SessionSaved,
    UsageUpdate,
    UserInterrupt,
    UserMessage,
    UserRetry,
)
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.workspace import Workspace
from chrys.kernel import ChatResponseUpdate, FinishReason, FinishReasonLiteral
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.state.store import JsonFileStateStore
from tests.orchestration.engine.run._engine_run_helpers import _PROFILE, _make_registry
from tests.support.event_capture import collect_events
from tests.support.pipeline_helpers import make_mock_settings_and_registry
from tests.support.waiting import (
    ENGINE_TEST_WAIT_TIMEOUT,
    ENGINE_TURN_TIMEOUT,
    await_run_task_chain,
    wait_for,
    with_wait_deadline,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterable
    from pathlib import Path


class _InterruptibleClient(MockChatClient):
    """Hold the first stream until the engine cancels it; shutdown owns teardown."""

    def __init__(self, responses: list[MockResponse]) -> None:
        super().__init__(responses=responses)
        self.stream_entered = asyncio.Event()
        self.stream_cancelled = asyncio.Event()
        self.release_stream = asyncio.Event()

    async def _stream_updates(
        self,
        resp: MockResponse,
        model_id: str,
        finish_reason: FinishReasonLiteral | FinishReason,
    ) -> AsyncIterable[ChatResponseUpdate]:
        hold_first_chunk = self.call_count == 1
        async for update in super()._stream_updates(resp, model_id, finish_reason):
            yield update
            if hold_first_chunk:
                hold_first_chunk = False
                self.stream_entered.set()
                try:
                    await self.release_stream.wait()
                except asyncio.CancelledError:
                    self.stream_cancelled.set()
                    raise


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_retry_does_not_increment_turn_number(
    tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch, *, engine_services
):
    """Interrupt + resume should keep the same turn number.

    Flow:
    1. User sends message → turn 1 starts
    2. Agent streams slowly, user interrupts
    3. User retries → should still be turn 1 (not turn 2)
    4. Agent completes
    5. Verify: turn_number == 1, mutation tracker has no extra turns
    """
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, InvocationMessage, Error, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: collect_events(_events, e))

    settings, model_registry = make_mock_settings_and_registry(stream=True)
    # An interrupt still finalizes and saves. Advisory workspace capture has
    # its own 15s deadline, so scanning the CI checkout can consume the entire
    # former run-wait budget without testing anything about turn numbering.
    settings = replace(settings, workspace_change_notice=False)
    state_store = JsonFileStateStore(tmp_path)
    registry = _make_registry()
    client = _InterruptibleClient([MockResponse(text="interrupted", chunk_delay=0), MockResponse(text="Done.")])
    monkeypatch.setattr(
        builder_module, "create_client", create_autospec(builder_module.create_client, return_value=client)
    )
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=registry,
        model_registry=model_registry,
        state_store=state_store,
        initial_workspace=Workspace.from_cwd(str(tmp_path)),
    )
    await engine.start(_PROFILE)

    # Wait for an actual stream chunk, then keep that stream in flight until
    # interruption. A call count alone only proves the request started.
    await bus.publish(UserMessage(text="Hello"))
    await wait_for(client.stream_entered.is_set, timeout=ENGINE_TURN_TIMEOUT, description="first stream entered")
    assert engine.session.turn_number == 1

    # Interrupt — wait for the interrupted run task to finish rolling back
    # rather than sleeping a fixed duration (flaky on slow Windows CI).
    await bus.publish(UserInterrupt())
    await wait_for(client.stream_cancelled.is_set, timeout=ENGINE_TURN_TIMEOUT, description="first stream cancelled")
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state, timeout=ENGINE_TEST_WAIT_TIMEOUT)

    # Turn number should still be 1 after interrupt
    assert engine.session.turn_number == 1
    picker_revision = engine.conversation_revision
    assert picker_revision == 1

    # Resume/retry → should NOT increment turn number
    await bus.publish(UserRetry())
    await wait_for(
        lambda: engine.conversation_revision == picker_revision + 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="retry conversation revision",
    )
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state, timeout=ENGINE_TEST_WAIT_TIMEOUT)

    assert engine.session.turn_number == 1, (
        f"Expected turn_number to stay at 1 after retry, got {engine.session.turn_number}"
    )
    assert engine.conversation_revision == picker_revision + 1

    # If mutation tracker exists, verify no extra turns
    if engine.session.mutation_tracker is not None:
        all_turns = engine.session.mutation_tracker.get_all_turns()
        turn_ids = [t.turn_id for t in all_turns]
        assert 1 in turn_ids, f"Turn 1 should exist, got {turn_ids}"
        assert 2 not in turn_ids, f"Turn 2 should NOT exist after retry, got {turn_ids}"

    # Persisted-state invariant: ``turn_counter`` must equal the count
    # of turn markers in history, and marker ``_turn`` values must be
    # sequential starting at 1.  Regression for the bug where
    # ``remove_trailing_markers`` popped a turn marker during retry but
    # left ``turn_counter`` advanced, so the next ``insert_turn_marker``
    # skipped slot 1 and produced ``_turn=2`` / ``turn_counter=2`` for
    # what was actually the first (retried) turn — which then restored
    # as ``Turn 2 (Current)`` in the rollback picker.
    state = engine_services(engine).history.state
    turn_markers = [
        m for m in state["messages"] if m.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.TURN
    ]
    marker_indices = [m.additional_properties.get("_turn") for m in turn_markers]
    assert state["turn_counter"] == 1, (
        f"turn_counter must be 1 after retry of turn 1, got {state['turn_counter']} (markers: {marker_indices})"
    )
    assert marker_indices == [1], f"Expected exactly one turn marker with _turn=1, got {marker_indices}"
    assert state["turn_counter"] == len(turn_markers), (
        f"turn_counter must equal len(turn_markers): counter={state['turn_counter']} markers={marker_indices}"
    )


@pytest.mark.asyncio
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_new_message_after_retry_increments_turn(
    tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch, *, engine_services
):
    """After a retry completes, a new user message should start a new turn.

    Flow:
    1. User sends message → turn 1
    2. Interrupt + retry → still turn 1
    3. New message → turn 2
    """
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, InvocationMessage, Error, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: collect_events(_events, e))

    settings, model_registry = make_mock_settings_and_registry(stream=True)
    settings = replace(settings, workspace_change_notice=False)
    state_store = JsonFileStateStore(tmp_path)
    registry = _make_registry()

    client = _InterruptibleClient(
        [
            MockResponse(text="interrupted", chunk_delay=0),
            MockResponse(text="Done with turn 1."),
            MockResponse(text="Done with turn 2."),
        ]
    )
    monkeypatch.setattr(
        builder_module, "create_client", create_autospec(builder_module.create_client, return_value=client)
    )
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=registry,
        model_registry=model_registry,
        state_store=state_store,
        initial_workspace=Workspace.from_cwd(str(tmp_path)),
    )
    await engine.start(_PROFILE)

    # Turn 1: send + interrupt + retry. Poll for the turn-start increment
    # and the interrupted run's completion rather than fixed settles
    # (flaky on slow Windows CI).
    await bus.publish(UserMessage(text="First"))
    await wait_for(client.stream_entered.is_set, timeout=ENGINE_TURN_TIMEOUT, description="first stream entered")
    await bus.publish(UserInterrupt())
    await wait_for(client.stream_cancelled.is_set, timeout=ENGINE_TURN_TIMEOUT, description="first stream cancelled")
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state, timeout=ENGINE_TEST_WAIT_TIMEOUT)
    assert engine.session.turn_number == 1

    await bus.publish(UserRetry())
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state, timeout=ENGINE_TEST_WAIT_TIMEOUT)
    assert engine.session.turn_number == 1, "Retry should not increment turn"

    # Turn 2: new message
    await bus.publish(UserMessage(text="Second"))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state, timeout=ENGINE_TEST_WAIT_TIMEOUT)
    assert engine.session.turn_number == 2, (
        f"New message after retry should be turn 2, got {engine.session.turn_number}"
    )

    # Persisted-state invariant: two turns completed → two turn markers
    # with sequential ``_turn`` indices ``[1, 2]`` and ``turn_counter=2``.
    # This is the exact scenario that reproduced the user-visible
    # "Turn 3 (Current) for a 2-turn session" bug: before the fix the
    # retry path advanced ``turn_counter`` without re-inserting the
    # matching marker, so a fresh turn 2 ended up labelled ``_turn=3``
    # on disk, which restored as ``Turn 3`` in the rollback picker.
    state = engine_services(engine).history.state
    turn_markers = [
        m for m in state["messages"] if m.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.TURN
    ]
    marker_indices = [m.additional_properties.get("_turn") for m in turn_markers]
    assert marker_indices == [1, 2], f"Expected sequential markers [1, 2], got {marker_indices}"
    assert state["turn_counter"] == 2, f"turn_counter must be 2, got {state['turn_counter']}"
    assert state["turn_counter"] == len(turn_markers)
