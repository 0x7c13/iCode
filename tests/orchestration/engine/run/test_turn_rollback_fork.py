# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A rollback inside a fork retires the turns it inherited by identity.

A fork keeps the conversation but clears the registry's log sequences: the
fork's log is a new one, numbered from one. Retrying an inherited turn fills its
sequence again, so the range a later rollback supersedes opens at that retry —
and the turn the snapshot names as its own is the one the fork re-ran.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any
from unittest.mock import create_autospec

import pytest

import chrys.orchestration.engine.build.builder as builder_module
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    RollbackResult,
    SessionFork,
    SessionForked,
    SessionRestore,
    SessionRestored,
    TurnStarted,
    UserMessage,
    UserRetry,
    UserRollback,
)
from chrys.foundation.models.workspace import Workspace
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.reader import read_trajectory
from chrys.kernel import ChatResponseUpdate
from chrys.orchestration.engine.state.machine import EngineState
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.mutations.snapshot_files import rollback_snapshot_paths, snapshot_target_turn
from chrys.service.state.store import JsonFileStateStore
from chrys.service.trajectory.session import trajectory_events_path
from chrys.service.trajectory.state import turn_records
from tests.orchestration.engine.run._engine_run_helpers import _PROFILE, _make_registry
from tests.support.event_capture import collect_events
from tests.support.pipeline_helpers import make_mock_settings_and_registry
from tests.support.waiting import ENGINE_TURN_TIMEOUT, await_run_task_chain, wait_for

if TYPE_CHECKING:
    from collections.abc import AsyncIterable
    from pathlib import Path

_SOURCE_ID = "f0a1b2c3-d4e5-4f60-8a91-b2c3d4e5f602"


class ProviderRefusal(RuntimeError):
    """A provider failure the classifier cannot call transient, so no retry runs."""


class _FailSecondClient(MockChatClient):
    """Answer the first wire call, refuse the second, then answer every later one.

    Refusing before the scripted response is consumed leaves the answers for the
    passes that are expected to succeed, including the retry after the fork.
    """

    def __init__(self, responses: list[MockResponse], *, fail_at: int = 2) -> None:
        super().__init__(responses=responses)
        self.calls = 0
        self.fail_at = fail_at

    async def _stream_updates(
        self,
        resp: MockResponse,
        model_id: str,
        finish_reason: str,
    ) -> AsyncIterable[ChatResponseUpdate]:
        self.calls += 1
        if self.calls == self.fail_at:
            raise ProviderRefusal("provider refused the call")
        async for update in super()._stream_updates(resp, model_id, finish_reason):
            yield update


def _rollback_payload(engine) -> dict[str, Any]:
    """The payload of the last rollback the engine wrote into its own log.

    The range a rollback supersedes is a property of the log, not of the event
    it publishes: only the recorded payload says where the range opens.
    """
    rollback = [
        event
        for event in read_trajectory(trajectory_events_path(engine.session_dir)).events
        if event.event_type == EventType.SESSION_ROLLBACK
    ][-1]
    return rollback.payload


def _make_engine(
    agent_engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    store,
    client: MockChatClient,
    bus: EventBus,
    *,
    real_create_client=None,
):
    settings, model_registry = make_mock_settings_and_registry(stream=True)
    settings = replace(settings, workspace_change_notice=False, max_transient_retries=0)
    # Spec the genuine factory captured before the first patch: a second patch
    # in the same test would spec the mock the first one installed.
    spec = real_create_client if real_create_client is not None else builder_module.create_client
    monkeypatch.setattr(builder_module, "create_client", create_autospec(spec, return_value=client))
    return agent_engine(
        bus,
        settings=settings,
        agent_registry=_make_registry(),
        model_registry=model_registry,
        state_store=store,
        initial_workspace=Workspace.from_cwd(str(tmp_path)),
    )


async def _fork(engine, session_id: str) -> str:
    forked: list[SessionForked] = []
    await engine.event_bus.subscribe(SessionForked, lambda event: collect_events(forked, event))
    await engine._on_session_fork(SessionFork(session_id=session_id))
    assert len(forked) == 1
    return forked[0].new_session_id


async def _ask(engine, text: str) -> None:
    await engine.event_bus.publish(UserMessage(text=text))
    await await_run_task_chain(engine, turn_state=engine.turns.turn_state)


async def _source_runs_a_and_fails_b(
    agent_engine, tmp_path, monkeypatch, store, source_bus: EventBus
) -> tuple[Any, list]:
    """A completes and B fails; the fork of that session is returned with its turns."""
    source = _make_engine(agent_engine, tmp_path, monkeypatch, store, _FailSecondClient([]), source_bus)
    source.session.session_id = _SOURCE_ID
    await source.start(_PROFILE)
    await _ask(source, "A")
    await _ask(source, "B")
    source_turns = source.turn_records()
    assert [record.started_sequence is not None for record in source_turns] == [True, True]
    fork_id = await _fork(source, _SOURCE_ID)
    await source.shutdown()

    forked_state = await store.load_session(fork_id)
    assert forked_state is not None
    inherited = turn_records(forked_state)
    assert [record.turn_id for record in inherited] == [record.turn_id for record in source_turns]
    # The conversation carried over; the sequences did not — they name slots in
    # a log the parent owns, and the fork's own log is numbered from one.
    assert [record.started_sequence for record in inherited] == [None, None]
    assert {record.origin_session_id for record in inherited} == {_SOURCE_ID}
    return fork_id, forked_state


async def _open_fork(
    agent_engine,
    tmp_path,
    monkeypatch,
    store,
    client: MockChatClient,
    *,
    real_create_client,
    fork_id: str,
) -> Any:
    """Start an engine over the fork's own files and bring the fork back through ``SessionRestore``.

    Opening the copy is a reopen, not a build: the runtime restores the fork's
    saved state — including the failed state the inherited turn left it in,
    which is what admits the retry — and the registry it reads back is the one
    restored from disk, not one assigned here.
    """
    engine = _make_engine(
        agent_engine, tmp_path, monkeypatch, store, client, EventBus(), real_create_client=real_create_client
    )
    engine.session.session_id = fork_id
    await engine.start(_PROFILE)
    restored: list[SessionRestored] = []
    await engine.event_bus.subscribe(SessionRestored, lambda event: collect_events(restored, event))
    await engine.event_bus.publish(SessionRestore(session_id=fork_id), raise_handler_errors=True)
    await wait_for(
        lambda: any(event.session_id == fork_id for event in restored),
        timeout=ENGINE_TURN_TIMEOUT,
        description="the fork restored",
    )
    assert engine.state is EngineState.FAILED, "the fork inherits the failed turn it has to retry"
    return engine


def _inherited_snapshot(store, fork_id: str, target_turn: int) -> Path:
    """The snapshot a rollback to *target_turn* restores: the source's own, carried through the fork.

    The engine copies ``session.json`` to ``turn_N.json`` when turn N starts, so
    the fork inherits those copies with their payloads intact and only the
    envelope's session id rewritten. The registry inside therefore holds the
    inherited turns by the ids the fork still knows them by — the property under
    test — while naming *target_turn* by the same turn count a rollback asks for.
    """
    matches = [
        path
        for path in rollback_snapshot_paths(store.session_dir(fork_id))
        if snapshot_target_turn(path) == target_turn
    ]
    assert len(matches) == 1, f"expected exactly one snapshot for turn {target_turn}, found {matches}"
    return matches[0]


class TestRollbackInsideAFork:
    @pytest.mark.asyncio
    async def test_only_the_retried_turn_is_retired_and_the_range_starts_at_its_pass(
        self, tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = JsonFileStateStore(tmp_path)
        real_create_client = builder_module.create_client
        fork_id, forked_state = await _source_runs_a_and_fails_b(agent_engine, tmp_path, monkeypatch, store, EventBus())

        results: list[RollbackResult] = []
        engine = await _open_fork(
            agent_engine,
            tmp_path,
            monkeypatch,
            store,
            _FailSecondClient([]),
            real_create_client=real_create_client,
            fork_id=fork_id,
        )
        await engine.event_bus.subscribe(RollbackResult, lambda event: collect_events(results, event))
        inherited = engine.turn_records()
        assert [record.turn_id for record in inherited] == [record.turn_id for record in turn_records(forked_state)]

        # The retry resumes the inherited turn B and lands it in this log, so
        # the only sequence either turn has here is the retry pass's own.
        turns: list[TurnStarted] = []
        await engine.event_bus.subscribe(TurnStarted, lambda event: collect_events(turns, event))
        await engine.event_bus.publish(UserRetry())
        await wait_for(
            lambda: engine.turn_records()[-1].started_sequence is not None,
            timeout=ENGINE_TURN_TIMEOUT,
            description="the retry landing the inherited turn",
        )
        await await_run_task_chain(engine, turn_state=engine.turns.turn_state)

        # Which turn the retry resumed, not merely that it ran: the fork reads
        # the registry back from its own files, and the last entry there is B.
        assert [event.turn_id for event in turns if event.is_retry] == [inherited[1].turn_id]

        after_retry = engine.turn_records()
        assert [record.turn_id for record in after_retry] == [record.turn_id for record in inherited]
        resumed = after_retry[1]
        assert resumed.started_sequence is not None
        # The inherited first turn was never re-run here, so it names no slot in
        # this log — and neither does the retry inherit the parent's.
        assert after_retry[0].started_sequence is None
        assert results == []

        _inherited_snapshot(store, fork_id, 1)

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        assert len(results) == 1
        assert results[0].superseded_turn_ids == [resumed.turn_id]
        payload = _rollback_payload(engine)
        assert payload["superseded_from_sequence"] == resumed.started_sequence
        assert payload["target_turn_id"] == inherited[0].turn_id

    @pytest.mark.asyncio
    async def test_a_turn_asked_after_the_retry_is_also_retired(
        self, tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The inherited turn the fork did not re-run is not in the range."""
        store = JsonFileStateStore(tmp_path)
        real_create_client = builder_module.create_client
        fork_id, _ = await _source_runs_a_and_fails_b(agent_engine, tmp_path, monkeypatch, store, EventBus())

        results: list[RollbackResult] = []
        engine = await _open_fork(
            agent_engine,
            tmp_path,
            monkeypatch,
            store,
            _FailSecondClient([]),
            real_create_client=real_create_client,
            fork_id=fork_id,
        )
        await engine.event_bus.subscribe(RollbackResult, lambda event: collect_events(results, event))
        inherited = engine.turn_records()

        turns: list[TurnStarted] = []
        await engine.event_bus.subscribe(TurnStarted, lambda event: collect_events(turns, event))
        await engine.event_bus.publish(UserRetry())
        await wait_for(
            lambda: engine.turn_records()[-1].started_sequence is not None,
            timeout=ENGINE_TURN_TIMEOUT,
            description="the retry landing the inherited turn",
        )
        await await_run_task_chain(engine, turn_state=engine.turns.turn_state)
        assert [event.turn_id for event in turns if event.is_retry] == [inherited[1].turn_id]
        resumed = engine.turn_records()[1]

        # A question of its own, on the ordinal the retry left behind: the
        # runtime continues from the counter the restore restored.
        await _ask(engine, "C")
        after_c = engine.turn_records()
        assert len(after_c) == 3
        added = after_c[2]
        assert added.started_sequence is not None

        _inherited_snapshot(store, fork_id, 1)

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        assert len(results) == 1
        assert results[0].superseded_turn_ids == [resumed.turn_id, added.turn_id]
        payload = _rollback_payload(engine)
        # The range opens at the earliest superseded turn that ran here — the
        # retry — and runs to the rollback's own line.
        assert payload["superseded_from_sequence"] == resumed.started_sequence
        assert payload["target_turn_id"] == inherited[0].turn_id

    @pytest.mark.asyncio
    async def test_a_rollback_to_welcome_retires_the_inherited_turns_too(
        self, tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reset drops the whole registry, inherited turns included."""
        store = JsonFileStateStore(tmp_path)
        real_create_client = builder_module.create_client
        fork_id, _ = await _source_runs_a_and_fails_b(agent_engine, tmp_path, monkeypatch, store, EventBus())

        results: list[RollbackResult] = []
        engine = await _open_fork(
            agent_engine,
            tmp_path,
            monkeypatch,
            store,
            _FailSecondClient([]),
            real_create_client=real_create_client,
            fork_id=fork_id,
        )
        await engine.event_bus.subscribe(RollbackResult, lambda event: collect_events(results, event))
        inherited = engine.turn_records()
        assert len(inherited) == 2, "the fork restored both inherited turns"

        await engine._on_user_rollback(UserRollback(target_turn=0, revert_changes=False))

        assert len(results) == 1
        assert results[0].superseded_turn_ids == [record.turn_id for record in inherited]
        # Nobody re-ran them here, so the log holds nothing to supersede.
        assert "superseded_from_sequence" not in _rollback_payload(engine)
        assert engine.turn_records() == []
