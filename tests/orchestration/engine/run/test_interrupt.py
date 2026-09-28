# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Integration tests — interrupt preserves completed work in session history.

Verifies:
1. Interrupting a running agent preserves completed work (user message + any
   completed tool calls) and appends a interrupted marker
2. Saved session file contains the preserved state
3. No spurious Error events are published for user-requested interrupts
4. Agent can continue chatting normally after an interrupt
5. Interrupt while idle has no effect
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import create_autospec

import pytest

if TYPE_CHECKING:
    from pathlib import Path

import chrys.orchestration.engine.build.builder as builder_module
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    Error,
    InvocationMessage,
    SessionReady,
    SessionSaved,
    UsageUpdate,
    UserInterrupt,
    UserMessage,
)
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.state.store import JsonFileStateStore
from tests.orchestration.engine.run._engine_run_helpers import (
    _PROFILE,
    _filter,
    _final_agent_messages_after_strict_run,
    _history_messages,
    _make_registry,
    _wait_for_call_count,
)
from tests.support.event_capture import collect_events
from tests.support.pipeline_helpers import make_mock_settings_and_registry
from tests.support.waiting import ENGINE_TURN_TIMEOUT, await_run_task_chain, wait_for

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_for_session_ready(events: list, timeout: float = ENGINE_TURN_TIMEOUT) -> None:
    """Poll until ``engine.start()`` has published its ``SessionReady`` event.

    ``start()`` awaits this publish inline, so it is already collected by the
    time ``start()`` returns; polling it (instead of a fixed settle sleep) just
    asserts the engine is fully built and idle before the test drives it.
    """
    await wait_for(
        lambda: any(isinstance(e, SessionReady) for e in events),
        timeout=timeout,
        interval=0.01,
        description="engine session ready event",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_preserves_user_message(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Interrupt mid-stream → user message preserved, interrupted marker appended.

    Flow:
    1. Normal chat: user → agent response (establishes baseline history)
    2. User sends another message, agent streams slowly
    3. User interrupts mid-stream
    4. History should contain Phase 1 + Phase 2 user message + interrupted marker
    5. Saved session should contain the preserved state
    """
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, InvocationMessage, Error, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: collect_events(_events, e))

    settings, model_registry = make_mock_settings_and_registry(stream=True)
    # This test owns turn finalization and persistence, not workspace-change
    # detection. Keep unrelated Git/filesystem probes out of its timing.
    settings = replace(settings, workspace_change_notice=False)
    state_store = JsonFileStateStore(tmp_path)
    registry = _make_registry()

    mock_clients: list[MockChatClient] = []

    async def _mock_create_client(s=None, **kw):
        client = MockChatClient(
            responses=[
                # Phase 1: fast response
                MockResponse(text="Hello! How can I help?"),
                # Phase 2: slow streaming — gives time to interrupt
                MockResponse(text="A" * 20, chunk_size=2, chunk_delay=0.05),
                # Phase 3: post-interrupt response (if user chats again)
                MockResponse(text="Back to normal."),
            ]
        )
        mock_clients.append(client)
        return client

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=registry,
        model_registry=model_registry,
        state_store=state_store,
    )
    await engine.start(_PROFILE)
    await _wait_for_session_ready(events)

    # ---- Phase 1: normal conversation ----
    await bus.publish(UserMessage(text="Hello"))

    finals = await _final_agent_messages_after_strict_run(engine, events)
    assert len(finals) == 1, f"Expected 1 final message, got {len(finals)}"

    msgs_after_phase1 = _history_messages(engine)
    baseline_len = len(msgs_after_phase1)
    assert baseline_len >= 2  # at least user + assistant

    # ---- Phase 2: send message, then interrupt mid-stream ----
    await bus.publish(UserMessage(text="Tell me something long"))
    await _wait_for_call_count(mock_clients[0], 2)

    await bus.publish(UserInterrupt())
    await await_run_task_chain(
        engine, turn_state=engine.turns.turn_state, propagate_inner_cancel=True, expect_installed=True
    )

    # ---- Verify: history contains Phase 1 + Phase 2 user message ----
    msgs_after_interrupt = _history_messages(engine)
    # Should be more than baseline (Phase 2 user msg preserved)
    assert len(msgs_after_interrupt) > baseline_len, (
        f"History should preserve interrupted run, got {len(msgs_after_interrupt)} (baseline={baseline_len})"
    )

    # User message from Phase 2 should be in history
    user_msgs = [m for m in msgs_after_interrupt if getattr(m, "role", "") == "user"]
    assert len(user_msgs) == 2, f"Expected 2 user messages, got {len(user_msgs)}"

    # Cancelled marker should be present
    cancelled = [
        m
        for m in msgs_after_interrupt
        if getattr(m, "additional_properties", {}).get(HistoryMarkerKind.KEY) == HistoryMarkerKind.INTERRUPTED
    ]
    assert len(cancelled) == 1, "Expected 1 interrupted marker"

    # ---- Verify: no Error events for the interrupt ----
    errors = _filter(events, Error)
    assert not errors, f"Unexpected errors: {[e.message for e in errors]}"

    # ---- Verify: saved session file has preserved state ----
    session_id = engine.session.session_id
    saved_msgs = await state_store.load_session_raw(session_id)
    assert saved_msgs is not None
    user_msgs_in_save = [m for m in saved_msgs if m.get("role") == "user"]
    assert len(user_msgs_in_save) == 2, f"Saved session should have 2 user messages, got {len(user_msgs_in_save)}"


@pytest.mark.asyncio
async def test_interrupt_during_rollback_snapshot_never_starts_executor(
    tmp_path: Path,
    agent_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop during worker snapshot I/O must terminate before any model/tool execution."""
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, InvocationMessage, Error, UsageUpdate]:
        await bus.subscribe(cls, lambda event, _events=events: collect_events(_events, event))

    mock_client = MockChatClient(responses=[MockResponse(text="must not run")])
    monkeypatch.setattr(
        builder_module, "create_client", create_autospec(builder_module.create_client, return_value=mock_client)
    )
    engine = agent_engine(
        bus,
        settings=Settings(workspace_change_notice=False),
        agent_registry=_make_registry(),
        state_store=JsonFileStateStore(tmp_path),
    )
    await engine.start(_PROFILE)
    await _wait_for_session_ready(events)
    assert engine.current.loaded is not None
    engine.current.loaded.bindings.inputs.pending_continuation_token = "stale-response"

    snapshot_started = threading.Event()
    release_snapshot = threading.Event()

    def capture_snapshot_writer(session, settings_handle) -> Callable[[], None]:
        assert session is engine.session
        assert settings_handle is engine.settings_handle

        def write_snapshot() -> None:
            snapshot_started.set()
            release_snapshot.wait(timeout=ENGINE_TURN_TIMEOUT)

        return write_snapshot

    monkeypatch.setattr("chrys.orchestration.engine.run.runner.capture_snapshot_writer", capture_snapshot_writer)

    await bus.publish(UserMessage(text="stop before model"))
    await wait_for(
        snapshot_started.is_set,
        timeout=ENGINE_TURN_TIMEOUT,
        description="rollback snapshot writer start",
    )
    await bus.publish(UserInterrupt())
    release_snapshot.set()
    await await_run_task_chain(
        engine, turn_state=engine.turns.turn_state, propagate_inner_cancel=True, expect_installed=True
    )

    assert mock_client.call_count == 0
    assert engine.current.loaded is not None
    assert engine.current.loaded.bindings.state.was_interrupted is True
    assert engine.current.loaded.bindings.inputs.pending_continuation_token is None
    messages = _history_messages(engine)
    assert any(message.role == "user" and message.text == "stop before model" for message in messages)
    assert any(
        message.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.INTERRUPTED
        for message in messages
    )


@pytest.mark.asyncio
async def test_interrupt_during_post_run_save_does_not_cancel_next_turn(
    tmp_path: Path,
    agent_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A late Stop belongs to the finalizing task, never the following fresh task."""
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, InvocationMessage, Error, UsageUpdate]:
        await bus.subscribe(cls, lambda event, _events=events: collect_events(_events, event))

    mock_client = MockChatClient(responses=[MockResponse(text="first complete"), MockResponse(text="second must run")])
    monkeypatch.setattr(
        builder_module, "create_client", create_autospec(builder_module.create_client, return_value=mock_client)
    )
    engine = agent_engine(
        bus,
        settings=Settings(workspace_change_notice=False),
        agent_registry=_make_registry(),
        state_store=JsonFileStateStore(tmp_path),
    )
    await engine.start(_PROFILE)
    await _wait_for_session_ready(events)

    save_started = asyncio.Event()
    release_save = asyncio.Event()
    block_next_save = True
    save_current_session = engine.writer.save_current_session

    async def blocked_save_current_session() -> bool:
        nonlocal block_next_save
        if block_next_save:
            block_next_save = False
            save_started.set()
            await release_save.wait()
        return await save_current_session()

    monkeypatch.setattr(engine.writer, "save_current_session", blocked_save_current_session)

    await bus.publish(UserMessage(text="first"))
    # Generous ceiling like every other wait here: the budget covers a whole
    # model turn reaching its post-run save on a loaded CI runner, and the
    # test is about ordering, not latency.
    await wait_for(
        save_started.is_set,
        timeout=ENGINE_TURN_TIMEOUT,
        description="post-run save start",
    )
    assert engine.current.loaded is not None
    assert engine.current.loaded.bindings.state.running is False

    await bus.publish(UserInterrupt())
    release_save.set()
    await await_run_task_chain(
        engine, turn_state=engine.turns.turn_state, propagate_inner_cancel=True, expect_installed=True
    )
    assert engine.current.loaded.bindings.state.was_interrupted is False

    events.clear()
    await bus.publish(UserMessage(text="second"))
    finals = await _final_agent_messages_after_strict_run(engine, events)

    assert [event.text for event in finals] == ["second must run"]
    assert mock_client.call_count == 2


@pytest.mark.asyncio
async def test_chat_works_after_interrupt(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """After an interrupt, the user can send a new message and get a response."""
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, InvocationMessage, Error, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: collect_events(_events, e))

    settings, model_registry = make_mock_settings_and_registry(stream=True)
    settings = replace(settings, workspace_change_notice=False)
    state_store = JsonFileStateStore(tmp_path)
    registry = _make_registry()

    mock_clients: list[MockChatClient] = []

    async def _mock_create_client(s=None, **kw):
        client = MockChatClient(
            responses=[
                # Message 1: will be interrupted
                MockResponse(text="B" * 20, chunk_size=2, chunk_delay=0.02),
                # Message 2: after interrupt, normal response
                MockResponse(text="All good now."),
            ]
        )
        mock_clients.append(client)
        return client

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=registry,
        model_registry=model_registry,
        state_store=state_store,
    )
    await engine.start(_PROFILE)
    await _wait_for_session_ready(events)

    # ---- Send message and interrupt ----
    await bus.publish(UserMessage(text="Start something"))
    # Let the run reach the LLM (mid-stream) before interrupting, so the
    # interrupt lands on an in-flight run rather than racing ahead of it.
    await _wait_for_call_count(mock_clients[0], 1)
    await bus.publish(UserInterrupt())
    # Wait for the interrupted run to actually finish unwinding rather
    # than relying on a fixed sleep — otherwise on slow runners the
    # next UserMessage can race the still-running run task.
    await await_run_task_chain(
        engine, turn_state=engine.turns.turn_state, propagate_inner_cancel=True, expect_installed=True
    )

    # History should have user message + interrupted marker (not empty)
    msgs = _history_messages(engine)
    user_msgs = [m for m in msgs if getattr(m, "role", "") == "user"]
    assert len(user_msgs) >= 1, "Interrupted user message should be preserved"

    # ---- Send another message — should work normally ----
    events.clear()
    await bus.publish(UserMessage(text="Try again"))
    finals = await _final_agent_messages_after_strict_run(engine, events)
    assert len(finals) == 1, f"Expected 1 final message after retry, got {len(finals)}"

    # History should now have the new conversation too.
    # The interrupted run's leaked text response is dropped by
    # _remove_trailing_agent_text, so we expect:
    #   user("Start something") + user("Try again") + assistant("All good now.")
    # plus any turn/status markers.
    msgs = _history_messages(engine)
    assert len(msgs) >= 3  # interrupted user + new user + new assistant

    # No errors
    errors = _filter(events, Error)
    assert not errors, f"Unexpected errors: {[e.message for e in errors]}"


@pytest.mark.asyncio
async def test_interrupt_no_effect_when_idle(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Publishing UserInterrupt when agent is idle should be a no-op."""
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, InvocationMessage, Error, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: collect_events(_events, e))

    settings = Settings(workspace_change_notice=False)
    state_store = JsonFileStateStore(tmp_path)
    registry = _make_registry()

    async def _mock_create_client(s=None, **kw):
        return MockChatClient(
            responses=[MockResponse(text="Hello!"), MockResponse(text="Still responsive after idle Stop.")]
        )

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=registry,
        state_store=state_store,
    )
    await engine.start(_PROFILE)
    await _wait_for_session_ready(events)

    # Normal chat
    await bus.publish(UserMessage(text="Hi"))
    finals = await _final_agent_messages_after_strict_run(engine, events)
    assert len(finals) == 1, f"Expected 1 final message, got {len(finals)}"

    baseline_len = len(_history_messages(engine))
    assert baseline_len >= 2

    # Interrupt while idle — should be harmless. ``bus.publish`` awaits the
    # handler inline and the idle interrupt spawns no run task, so the
    # no-op is fully processed by the time publish returns; a fixed settle
    # sleep here would only add dead time.
    await bus.publish(UserInterrupt())

    # History should be untouched
    assert len(_history_messages(engine)) == baseline_len

    # No errors
    errors = _filter(events, Error)
    assert not errors

    # The idle Stop must not leave an interrupt flag that cancels the next turn.
    events.clear()
    await bus.publish(UserMessage(text="Still there?"))
    finals = await _final_agent_messages_after_strict_run(engine, events)
    assert [event.text for event in finals] == ["Still responsive after idle Stop."]
