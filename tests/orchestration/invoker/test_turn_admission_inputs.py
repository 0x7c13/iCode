# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Real backend retry admission and persistence preserve exactly one Turn input."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.types import UserMessage, UserRetry
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.trajectory.metadata import read_analytics_item_id
from chrys.orchestration.engine.run.turn_state import CurrentTurnInput
from chrys.orchestration.invoker.contracts import Failed, RunIntent, UnsupportedRequest
from chrys.service.llm.mock import MockResponse
from chrys.service.session.message_metadata import MESSAGE_CREATED_AT_KEY
from chrys.service.state.store import JsonFileStateStore
from tests.orchestration.invoker._build_fixtures import build_recipe_engine


async def test_successful_guidance_retry_keeps_only_provider_persisted_note(
    tmp_path, monkeypatch, agent_engine, *, engine_services
):
    engine, main, _ = await build_recipe_engine(
        agent_engine, monkeypatch, tmp_path, main=[ValueError("first failed"), MockResponse(text="done")], child=[]
    )
    guidance = "important guidance"
    guidance_time = "2026-09-05T01:03:04.000000+00:00"
    try:
        await engine.event_bus.publish(UserMessage(text="work"))
        await engine.wait_for_run_task()
        assert engine.current.loaded.bindings.state.run_failed is True
        inputs = engine.current.loaded.bindings.inputs
        retry_request = inputs.retry_request
        persisted_notes = []

        @asynccontextmanager
        async def observe_retry(additional_text="", created_at=None):
            async with retry_request(additional_text=additional_text, created_at=created_at) as request:
                yield request
                persisted_notes.extend(
                    m for m in engine_services(engine).history.messages if m.role == "user" and m.text == guidance
                )

        monkeypatch.setattr(inputs, "retry_request", observe_retry)
        save = engine.writer.save_current_session
        saved_inputs = []

        async def save_snapshot(**kwargs):
            saved_inputs.append(replace(engine.turns.turn_state.current_input))
            return await save(**kwargs)

        saved = create_autospec(save, side_effect=save_snapshot)
        monkeypatch.setattr(engine.writer, "save_current_session", saved)
        await engine.event_bus.publish(UserRetry(text=guidance, timestamp=datetime.fromisoformat(guidance_time)))
        await engine.wait_for_run_task()
        assert engine.current.loaded.bindings.state.run_failed is False
        assert engine.current.loaded.bindings.state.was_interrupted is False
        assert main.call_count == 2
        saved.assert_awaited_once()
        assert len(saved_inputs) == 1
        assert (saved_inputs[0].text, saved_inputs[0].kind) == (guidance, "injected")
        assert saved_inputs[0].created_at == datetime.fromisoformat(guidance_time)
        assert len(persisted_notes) == 1
        note_id = read_analytics_item_id(persisted_notes[0].additional_properties)
        assert note_id is not None
        loaded = await JsonFileStateStore(tmp_path / "sessions").load_session(engine.session_id)
        assert loaded is not None
        for messages in (engine_services(engine).history.messages, loaded["messages"]):
            users = [m for m in messages if m.role == "user"]
            assert [m.text for m in users] == ["work", guidance]
            assert read_analytics_item_id(users[1].additional_properties) == note_id
            assert users[1].additional_properties[HistoryMarkerKind.INJECTED_KEY] is True
            assert users[1].additional_properties[MESSAGE_CREATED_AT_KEY] == guidance_time
        assert engine.turns.turn_state.current_input == CurrentTurnInput()
        assert engine.execution_busy() is False
    finally:
        await engine.shutdown()


@pytest.mark.parametrize("guidance", ["", "important guidance"])
async def test_entrance_validate_rejection_keeps_retry_input(
    tmp_path, monkeypatch, agent_engine, guidance, *, engine_services
):
    engine, main, _ = await build_recipe_engine(
        agent_engine, monkeypatch, tmp_path, main=[ValueError("first failed")], child=[]
    )
    opener_time = "2026-09-05T01:02:03.000000+00:00"
    guidance_time = "2026-09-05T01:03:04.000000+00:00"
    try:
        await engine.event_bus.publish(UserMessage(text="work", timestamp=datetime.fromisoformat(opener_time)))
        await engine.wait_for_run_task()
        assert engine.current.loaded.bindings.state.run_failed is True
        opener = next(m for m in engine_services(engine).history.messages if m.role == "user")
        opener_id = read_analytics_item_id(opener.additional_properties)
        assert opener_id is not None
        inputs = engine.current.loaded.bindings.inputs
        backend = inputs.backend
        opening_ids = []
        set_opening = inputs.set_opening_item_id

        def opening(item_id):
            opening_ids.append(item_id)
            set_opening(item_id)

        monkeypatch.setattr(inputs, "set_opening_item_id", create_autospec(set_opening, side_effect=opening))
        continuation = create_autospec(inputs.continuation_request, side_effect=inputs.continuation_request)
        monkeypatch.setattr(inputs, "continuation_request", continuation)
        retry_request = inputs.retry_request
        retry_inputs = []

        @asynccontextmanager
        async def observe_retry(additional_text="", created_at=None):
            async with retry_request(additional_text=additional_text, created_at=created_at) as request:
                retry_inputs.append(request)
                yield request

        monkeypatch.setattr(inputs, "retry_request", observe_retry)
        validate = create_autospec(backend.validate, side_effect=UnsupportedRequest("admission denied"))
        monkeypatch.setattr(backend, "validate", validate)
        # The operation binds successfully: this rejection is independent of
        # the promotion-to-bind failure tested in test_turn_binding_failure.
        conversation = engine.current.loaded.conversation
        bind = create_autospec(conversation.bind_operation, side_effect=conversation.bind_operation)
        monkeypatch.setattr(conversation, "bind_operation", bind)
        run_retry = create_autospec(engine._turns.run_retry, side_effect=engine._turns.run_retry)
        monkeypatch.setattr(engine._turns, "run_retry", run_retry)
        save = engine.writer.save_current_session
        saved_inputs = []

        async def save_snapshot(**kwargs):
            saved_inputs.append(replace(engine.turns.turn_state.current_input))
            return await save(**kwargs)

        saved = create_autospec(save, side_effect=save_snapshot)
        monkeypatch.setattr(engine.writer, "save_current_session", saved)

        await engine.event_bus.publish(UserRetry(text=guidance, timestamp=datetime.fromisoformat(guidance_time)))
        await engine.wait_for_run_task()
        assert engine.current.loaded.bindings.state.run_failed is True
        assert engine.current.loaded.bindings.state.last_error == "admission denied"
        assert main.call_count == 1
        saved.assert_awaited_once()
        assert len(saved_inputs) == 1
        if guidance:
            assert (saved_inputs[0].text, saved_inputs[0].kind) == (guidance, "injected")
            assert saved_inputs[0].created_at == datetime.fromisoformat(guidance_time)
        else:
            assert saved_inputs[0] == CurrentTurnInput()
        assert engine.turns.turn_state.current_input == CurrentTurnInput()
        assert engine.execution_busy() is False
        bind.assert_called_once()
        run_retry.assert_awaited_once()
        assert run_retry.call_args.kwargs["binding_failure"] is None
        validate.assert_called_once()
        continuation.assert_called_once_with([])
        assert retry_inputs == []

        loaded = await JsonFileStateStore(tmp_path / "sessions").load_session(engine.session_id)
        assert loaded is not None
        for messages in (engine_services(engine).history.messages, loaded["messages"]):
            users = [m for m in messages if m.role == "user"]
            assert [m.text for m in users] == (["work", guidance] if guidance else ["work"])
            assert read_analytics_item_id(users[0].additional_properties) == opener_id
            assert users[0].additional_properties[MESSAGE_CREATED_AT_KEY] == opener_time
            assert not users[0].additional_properties.get(HistoryMarkerKind.INJECTED_KEY)
            assert users[0].additional_properties[HistoryMarkerKind.SYSTEM_REMINDERS_KEY]
            if guidance:
                assert len(opening_ids) == 1 and opening_ids[0] is not None
                note = users[1]
                assert note.additional_properties[HistoryMarkerKind.INJECTED_KEY] is True
                assert note.additional_properties[MESSAGE_CREATED_AT_KEY] == guidance_time
                assert read_analytics_item_id(note.additional_properties) == opening_ids[0]
                # Never sent, so it carries no reminders: not the opener's either.
                assert HistoryMarkerKind.SYSTEM_REMINDERS_KEY not in note.additional_properties
    finally:
        await engine.shutdown()


@pytest.mark.parametrize("guidance", ["", "important guidance"])
async def test_backend_validate_rejection_keeps_retry_input(
    tmp_path, monkeypatch, agent_engine, guidance, *, engine_services
):
    engine, main, _ = await build_recipe_engine(
        agent_engine, monkeypatch, tmp_path, main=[ValueError("first failed")], child=[]
    )
    opener_time = "2026-09-05T01:02:03.000000+00:00"
    guidance_time = "2026-09-05T01:03:04.000000+00:00"
    try:
        await engine.event_bus.publish(UserMessage(text="work", timestamp=datetime.fromisoformat(opener_time)))
        await engine.wait_for_run_task()
        opener = next(m for m in engine_services(engine).history.messages if m.role == "user")
        opener_id = read_analytics_item_id(opener.additional_properties)
        assert opener_id is not None
        policy = engine.current.loaded.bindings.inputs
        failed = policy.outcome
        assert isinstance(failed, Failed)
        original = policy.continuation_request
        requests = []

        def request(messages):
            result = original(messages)
            requests.append(result)
            # The request entrance sees the real live ticket. Only the request
            # yielded to KernelConversation.run carries the rejected copy.
            return result if len(requests) == 1 else replace(result, continuation=replace(result.continuation))

        monkeypatch.setattr(policy, "continuation_request", create_autospec(original, side_effect=request))
        save = engine.writer.save_current_session
        saved_inputs = []

        async def save_snapshot(**kwargs):
            saved_inputs.append(replace(engine.turns.turn_state.current_input))
            return await save(**kwargs)

        saved = create_autospec(save, side_effect=save_snapshot)
        monkeypatch.setattr(engine.writer, "save_current_session", saved)
        await engine.event_bus.publish(UserRetry(text=guidance, timestamp=datetime.fromisoformat(guidance_time)))
        await engine.wait_for_run_task()
        assert not engine_services(engine).fsm.is_running()
        assert engine.current.loaded.bindings.state.run_failed is True
        assert engine.current.loaded.bindings.state.last_error == "Continuation no longer names this live state"
        assert engine.execution_busy() is False
        saved.assert_awaited_once()
        assert len(saved_inputs) == 1
        if guidance:
            assert (saved_inputs[0].text, saved_inputs[0].kind) == (guidance, "injected")
            assert saved_inputs[0].created_at == datetime.fromisoformat(guidance_time)
        else:
            # An empty-text retry that reaches the request body pops the opener
            # and re-registers it as recovery input (resume.py), so the final
            # save still carries the replayed anchor with its original stamp.
            assert (saved_inputs[0].text, saved_inputs[0].kind) == ("work", "opener")
            assert datetime.fromisoformat(str(saved_inputs[0].created_at)) == datetime.fromisoformat(opener_time)
        assert main.call_count == 1
        assert engine.turns.turn_state.current_input == CurrentTurnInput()
        assert len(requests) == 2
        assert all(r.intent is RunIntent.RETRY and r.continuation is failed.continuation for r in requests)
        assert policy.outcome is failed
        assert policy.backend.continuation_is_live(failed.continuation, policy.origin) is True

        loaded = await JsonFileStateStore(tmp_path / "sessions").load_session(engine.session_id)
        assert loaded is not None
        for messages in (engine_services(engine).history.messages, loaded["messages"]):
            users = [m for m in messages if m.role == "user"]
            assert [m.text for m in users] == (["work", guidance] if guidance else ["work"])
            assert read_analytics_item_id(users[0].additional_properties) == opener_id
            assert users[0].additional_properties[MESSAGE_CREATED_AT_KEY] == opener_time
            assert not users[0].additional_properties.get(HistoryMarkerKind.INJECTED_KEY)
            if guidance:
                note = users[1]
                assert note.additional_properties[HistoryMarkerKind.INJECTED_KEY] is True
                assert note.additional_properties[MESSAGE_CREATED_AT_KEY] == guidance_time
                assert read_analytics_item_id(note.additional_properties) == read_analytics_item_id(
                    requests[1].messages[0].additional_properties
                )
                assert read_analytics_item_id(note.additional_properties) is not None

    finally:
        await engine.shutdown()


async def test_backend_validate_rejection_keeps_fresh_input(tmp_path, monkeypatch, agent_engine, *, engine_services):
    engine, main, _ = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, main=[], child=[])
    created_at = "2026-09-05T01:02:03.000000+00:00"
    try:
        backend = engine.current.loaded.bindings.backend
        original = backend.validate
        requests = []

        def validate(request):
            requests.append(request)
            if len(requests) == 2:
                raise UnsupportedRequest("admission denied")
            return original(request)

        monkeypatch.setattr(backend, "validate", create_autospec(original, side_effect=validate))
        save = engine.writer.save_current_session
        saved = create_autospec(save, side_effect=save)
        monkeypatch.setattr(engine.writer, "save_current_session", saved)
        await engine.event_bus.publish(UserMessage(text="work", timestamp=datetime.fromisoformat(created_at)))
        await engine.wait_for_run_task()
        assert len(requests) == 2
        assert requests[0] is requests[1]
        assert requests[0].intent is RunIntent.FRESH
        assert not engine_services(engine).fsm.is_running()
        assert engine.current.loaded.bindings.state.run_failed is True
        assert engine.current.loaded.bindings.state.last_error == "admission denied"
        assert engine.execution_busy() is False
        saved.assert_awaited_once()
        assert main.call_count == 0
        assert engine.turns.turn_state.current_input == CurrentTurnInput()

        loaded = await JsonFileStateStore(tmp_path / "sessions").load_session(engine.session_id)
        assert loaded is not None
        for messages in (engine_services(engine).history.messages, loaded["messages"]):
            users = [m for m in messages if m.role == "user"]
            assert [m.text for m in users] == ["work"]
            assert users[0].additional_properties[MESSAGE_CREATED_AT_KEY] == created_at
            assert not users[0].additional_properties.get(HistoryMarkerKind.INJECTED_KEY)
            assert read_analytics_item_id(users[0].additional_properties) == read_analytics_item_id(
                requests[0].messages[0].additional_properties
            )
            assert read_analytics_item_id(users[0].additional_properties) is not None
    finally:
        await engine.shutdown()
