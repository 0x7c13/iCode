# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for backend session fork orchestration."""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from unittest.mock import create_autospec

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Error, SessionFork, SessionForked
from chrys.foundation.i18n import DisplayBlock
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.kernel import Message
from chrys.orchestration.engine.assembly import assemble_agent_engine
from chrys.orchestration.engine.engine import AgentEngine
from chrys.service.context.providers.history import CompressedBlock
from chrys.service.state.store import JsonFileStateStore
from tests.support.loaded_agents import install_loaded_agent


async def _collect(out: list[object], event: object) -> None:
    out.append(event)


def _executor_with_state(
    messages: list[Message],
    compressed_msgs: list[CompressedBlock] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        backend=SimpleNamespace(history_state={"messages": messages, "compressed_msgs": compressed_msgs or []})
    )


def _executor_with_messages(messages: list[Message]) -> SimpleNamespace:
    return _executor_with_state(messages)


def _assert_display_message(event: Error, key: str, args: dict[str, object]) -> None:
    reference = event.display_message
    assert reference is not None
    assert reference.definition.key == key
    assert dict(reference.args) == args


async def _raise_timeout(_self: AgentEngine, **_kwargs: object) -> None:
    raise TimeoutError("write lock busy")


async def _raise_runtime_error(_self: AgentEngine, **_kwargs: object) -> None:
    raise RuntimeError("disk full")


@pytest.mark.asyncio
async def test_session_fork_saves_then_publishes_forked_without_owning_fork(tmp_path) -> None:
    store = JsonFileStateStore(tmp_path)
    bus = EventBus()
    events: list[object] = []
    await bus.subscribe(SessionForked, lambda event: _collect(events, event))
    engine = assemble_agent_engine(bus, settings=Settings(), state_store=store)
    parent_id = "parent-session"
    engine.session.session_id = parent_id
    install_loaded_agent(engine, bindings=_executor_with_messages([Message("user", ["hello"])]))
    assert engine.session.guard.ensure(parent_id)
    try:
        await engine._on_session_fork(SessionFork(session_id=parent_id))
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, SessionForked)
        assert event.session_id == parent_id
        assert event.parent_session_id == parent_id
        assert event.new_session_id
        assert engine.session.guard.owns(parent_id)
        assert not engine.session.guard.owns(event.new_session_id)
    finally:
        engine.session.guard.release()

    fork_meta = await store.load_session_meta(event.new_session_id)
    assert fork_meta is not None
    assert fork_meta.parent_session_id == parent_id
    loaded = await store.load_session(event.new_session_id)
    assert loaded is not None
    assert loaded["messages"][0].text == "hello"


@pytest.mark.asyncio
async def test_session_fork_rejects_empty_session(tmp_path) -> None:
    store = JsonFileStateStore(tmp_path)
    bus = EventBus()
    errors: list[object] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = assemble_agent_engine(bus, settings=Settings(), state_store=store)
    parent_id = "parent-session"
    engine.session.session_id = parent_id
    install_loaded_agent(engine, bindings=_executor_with_messages([]))
    assert engine.session.guard.ensure(parent_id)
    try:
        await engine._on_session_fork(SessionFork(session_id=parent_id))
    finally:
        engine.session.guard.release()

    assert len(errors) == 1
    error = errors[0]
    assert isinstance(error, Error)
    assert error.code == "session_fork_empty"
    assert error.message == "Cannot fork an empty session."
    _assert_display_message(error, "engine.fork_empty_session", {})
    assert not store.session_dir(parent_id).exists()


@pytest.mark.asyncio
async def test_session_fork_allows_fully_compacted_session(tmp_path) -> None:
    store = JsonFileStateStore(tmp_path)
    bus = EventBus()
    events: list[object] = []
    await bus.subscribe(SessionForked, lambda event: _collect(events, event))
    engine = assemble_agent_engine(bus, settings=Settings(), state_store=store)
    parent_id = "parent-session"
    summary = Message("assistant", ["[Compressed context: ctx_1]\nSummary: old turn"])
    summary.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.SUMMARY
    compressed = CompressedBlock(
        compressed_context_id="ctx_1",
        messages=[Message("user", ["old real turn"])],
        summary_text="old turn",
        marker_id="turn_1",
        turn_range=(1, 1),
        created_at="2026-03-17T00:00:00+00:00",
    )
    engine.session.session_id = parent_id
    install_loaded_agent(engine, bindings=_executor_with_state([summary], [compressed]))
    assert engine.session.guard.ensure(parent_id)
    try:
        await engine._on_session_fork(SessionFork(session_id=parent_id))
    finally:
        engine.session.guard.release()

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SessionForked)
    loaded = await store.load_session(event.new_session_id)
    assert loaded is not None
    assert loaded["compressed_msgs"][0].messages[0].text == "old real turn"


@pytest.mark.asyncio
async def test_session_fork_without_ready_session_publishes_not_ready(tmp_path) -> None:
    store = JsonFileStateStore(tmp_path)
    bus = EventBus()
    errors: list[object] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = assemble_agent_engine(bus, settings=Settings(), state_store=store)

    await engine._on_session_fork(SessionFork(session_id="requested-session"))

    assert len(errors) == 1
    error = errors[0]
    assert isinstance(error, Error)
    assert error.code == "session_fork_not_ready"
    assert error.session_id == "requested-session"
    assert error.message == "Cannot fork before a session is ready."
    _assert_display_message(error, "engine.fork_not_ready", {})


@pytest.mark.asyncio
async def test_session_fork_stale_error_targets_requested_session(tmp_path) -> None:
    store = JsonFileStateStore(tmp_path)
    bus = EventBus()
    errors: list[object] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = assemble_agent_engine(bus, settings=Settings(), state_store=store)
    engine.session.session_id = "active-session"
    install_loaded_agent(engine, bindings=_executor_with_messages([Message("user", ["hello"])]))

    await engine._on_session_fork(SessionFork(session_id="requested-session"))

    assert len(errors) == 1
    error = errors[0]
    assert isinstance(error, Error)
    assert error.code == "session_fork_stale"
    assert error.session_id == "requested-session"
    assert error.message == "Cannot fork because the active session changed."
    _assert_display_message(error, "engine.fork_session_changed", {})


@pytest.mark.asyncio
async def test_session_fork_save_timeout_publishes_busy_error(tmp_path) -> None:
    store = JsonFileStateStore(tmp_path)
    bus = EventBus()
    errors: list[object] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = assemble_agent_engine(bus, settings=Settings(), state_store=store)
    parent_id = "parent-session"
    engine.session.session_id = parent_id
    install_loaded_agent(engine, bindings=_executor_with_messages([Message("user", ["hello"])]))
    engine.writer.save_current_session = MethodType(_raise_timeout, engine)
    assert engine.session.guard.ensure(parent_id)
    try:
        await engine._on_session_fork(SessionFork(session_id=parent_id))
    finally:
        engine.session.guard.release()

    assert len(errors) == 1
    error = errors[0]
    assert isinstance(error, Error)
    assert error.code == "session_fork_busy"
    assert error.message == "Timed out preparing session for fork: write lock busy"
    _assert_display_message(error, "engine.fork_prepare_timeout", {"detail": DisplayBlock("write lock busy")})


@pytest.mark.asyncio
async def test_session_fork_save_failure_publishes_failed_error(tmp_path) -> None:
    store = JsonFileStateStore(tmp_path)
    bus = EventBus()
    errors: list[object] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = assemble_agent_engine(bus, settings=Settings(), state_store=store)
    parent_id = "parent-session"
    engine.session.session_id = parent_id
    install_loaded_agent(engine, bindings=_executor_with_messages([Message("user", ["hello"])]))
    engine.writer.save_current_session = MethodType(_raise_runtime_error, engine)
    assert engine.session.guard.ensure(parent_id)
    try:
        await engine._on_session_fork(SessionFork(session_id=parent_id))
    finally:
        engine.session.guard.release()

    assert len(errors) == 1
    error = errors[0]
    assert isinstance(error, Error)
    assert error.code == "session_fork_failed"
    assert error.message == "Failed to prepare session for fork: disk full"
    _assert_display_message(error, "engine.fork_prepare_failed", {"detail": DisplayBlock("disk full")})


@pytest.mark.asyncio
async def test_session_fork_store_save_failure_does_not_copy_stale_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("parent-session", {"messages": [Message("user", ["old"])], "compressed_msgs": []})
    bus = EventBus()
    errors: list[object] = []
    forked: list[object] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    await bus.subscribe(SessionForked, lambda event: _collect(forked, event))

    async def fail_save_session(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "save_session", fail_save_session)
    engine = assemble_agent_engine(bus, settings=Settings(), state_store=store)
    parent_id = "parent-session"
    engine.session.session_id = parent_id
    install_loaded_agent(engine, bindings=_executor_with_messages([Message("user", ["new"])]))
    assert engine.session.guard.ensure(parent_id)
    try:
        await engine._on_session_fork(SessionFork(session_id=parent_id))
    finally:
        engine.session.guard.release()

    assert forked == []
    assert len(errors) == 1
    error = errors[0]
    assert isinstance(error, Error)
    assert error.code == "session_fork_failed"
    sessions = await store.list_sessions()
    assert [session.session_id for session in sessions] == [parent_id]


async def test_fork_prelude_keeps_identity_captured_before_reload(tmp_path, monkeypatch, agent_engine) -> None:
    import asyncio

    from chrys.foundation.trajectory.event_types import EventType
    from chrys.foundation.trajectory.reader import read_trajectory
    from chrys.service.llm.clients import create_client
    from chrys.service.llm.mock import MockChatClient
    from chrys.service.profiles.agents.schema import AgentProfile, CompactionConfig, ToolsConfig
    from chrys.service.trajectory import fork as fork_module
    from chrys.service.trajectory.session import trajectory_events_path

    monkeypatch.setattr(
        "chrys.orchestration.engine.build.builder.create_client",
        create_autospec(create_client, side_effect=lambda *args, **kwargs: MockChatClient()),
    )
    bus = EventBus()
    store = JsonFileStateStore(tmp_path)
    engine = agent_engine(bus, settings=Settings(), state_store=store)
    events: list[SessionForked] = []

    async def record_forked(event: SessionForked) -> None:
        events.append(event)

    await bus.subscribe(SessionForked, record_forked)
    profile = AgentProfile(
        name="First",
        instructions="First agent",
        tools=ToolsConfig(builtins=[]),
        compaction=CompactionConfig(enabled=False),
    )
    await engine.start(profile)
    engine.current.loaded.bindings.backend.history_state["messages"] = [Message("user", ["fork this history"])]
    captured = engine.loader.trajectory_session_start_info()
    assert captured.agent_profile_fingerprint
    assert captured.model_profile_fingerprint
    entered = asyncio.Event()
    activate = asyncio.Event()
    write_prelude = fork_module._write_prelude

    async def held_prelude(trajectory, origin_session_id, forked_at_sequence):
        assert not trajectory.is_active
        entered.set()
        await activate.wait()
        return await write_prelude(trajectory, origin_session_id, forked_at_sequence)

    monkeypatch.setattr(fork_module, "_write_prelude", held_prelude)
    fork_task = asyncio.create_task(engine.lifecycle.on_session_fork(SessionFork(session_id=engine.session_id)))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        replacement = AgentProfile(
            name="Second",
            instructions="Second agent",
            tools=ToolsConfig(builtins=[]),
            compaction=CompactionConfig(enabled=False),
        )
        await engine.lifecycle.reload(replacement)
        assert engine.current.manifest.agent_profile_fingerprint != captured.agent_profile_fingerprint
    finally:
        activate.set()
        await asyncio.wait_for(fork_task, timeout=5)

    assert len(events) == 1
    log = read_trajectory(trajectory_events_path(store.session_dir(events[0].new_session_id)))
    assert log.corrupt_lines == []
    started = [event for event in log.events if event.event_type == EventType.SESSION_STARTED]
    assert len(started) == 1
    assert started[0].payload["agent_profile_fingerprint"] == captured.agent_profile_fingerprint
    assert started[0].payload["model_profile_fingerprint"] == captured.model_profile_fingerprint
