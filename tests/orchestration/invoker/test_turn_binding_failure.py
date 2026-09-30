# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Promoted operation binding failures preserve input and settle the Turn."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.types import UserMessage, UserRetry
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.trajectory.metadata import read_analytics_item_id
from chrys.orchestration.engine.run.turn_state import CurrentTurnInput
from chrys.orchestration.engine.state.machine import EngineState
from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.manager import HookManager
from chrys.service.session.message_metadata import MESSAGE_CREATED_AT_KEY
from chrys.service.state.store import JsonFileStateStore
from tests.orchestration.invoker._build_fixtures import build_recipe_engine


def _user_snapshots(messages):
    return [
        (
            m.text,
            "injected" if m.additional_properties.get(HistoryMarkerKind.INJECTED_KEY) else "opener",
            read_analytics_item_id(m.additional_properties),
            m.additional_properties[MESSAGE_CREATED_AT_KEY],
        )
        for m in messages
        if m.role == "user"
    ]


@pytest.mark.parametrize("route", ["fresh", "retry", "retry-guidance"])
@pytest.mark.parametrize("window", ["closed", "open-bind-denied"])
async def test_close_after_promotion_before_binding_finalizes(
    route, window, tmp_path, monkeypatch, agent_engine, *, engine_services
):
    engine, main, _ = await build_recipe_engine(
        agent_engine, monkeypatch, tmp_path, main=[ValueError("first failed")], child=[]
    )
    opener_time = datetime.fromisoformat("2026-09-05T01:02:03+00:00")
    guidance_time = datetime.fromisoformat("2026-09-05T01:03:04+00:00")
    guidance = "important guidance" if route == "retry-guidance" else ""
    try:
        if route != "fresh":
            await engine.event_bus.publish(UserMessage(text="first", timestamp=opener_time))
            await engine.wait_for_run_task()
        original_users = _user_snapshots(engine_services(engine).history.messages)
        calls_before = main.call_count
        transitions = []
        for method, original in (
            ("transition", engine_services(engine).fsm.transition),
            ("try_transition", engine_services(engine).fsm.try_transition),
        ):

            def transition(trigger, *, original=original):
                result = original(trigger)
                transitions.append((trigger, result, engine_services(engine).fsm.state))
                return result

            monkeypatch.setattr(engine_services(engine).fsm, method, create_autospec(original, side_effect=transition))
        opening_ids = []
        set_opening = engine.current.loaded.bindings.inputs.set_opening_item_id

        def opening(item_id):
            opening_ids.append(item_id)
            set_opening(item_id)

        monkeypatch.setattr(
            engine.current.loaded.bindings.inputs,
            "set_opening_item_id",
            create_autospec(set_opening, side_effect=opening),
        )
        retry_inputs = []
        retry_request = engine.current.loaded.bindings.inputs.retry_request

        @asynccontextmanager
        async def observe_retry(*args, **kwargs):
            async with retry_request(*args, **kwargs) as request:
                if request is not None:
                    retry_inputs.extend(_user_snapshots(request.messages))
                yield request

        monkeypatch.setattr(engine.current.loaded.bindings.inputs, "retry_request", observe_retry)
        hooks = create_autospec(HookManager, instance=True)
        hooks.has_hooks_for.side_effect = lambda event: event is HookEvent.AFTER_TURN
        hooks.fire.return_value = None
        monkeypatch.setattr(engine.session, "hook_manager", hooks)
        save = engine.writer.save_current_session
        snapshots, inputs, results = [], [], []

        async def save_snapshot(**kwargs):
            inputs.append(replace(engine.turns.turn_state.current_input))
            snapshots.append(_user_snapshots(engine_services(engine).history.messages))
            result = await save(**kwargs)
            results.append(result)
            return result

        saved = create_autospec(save, side_effect=save_snapshot)
        monkeypatch.setattr(engine.writer, "save_current_session", saved)
        await engine.event_bus.publish(
            UserMessage(text="work", timestamp=opener_time)
            if route == "fresh"
            else UserRetry(text=guidance, timestamp=guidance_time)
        )
        task = engine.turns.turn_state.lease.run_task
        assert task is not None
        assert engine.current.loaded.conversation._operation is None
        # Promotion has already entered RUNNING before this bind-failure window.
        assert len(transitions) == 1
        assert transitions[0][2] is EngineState.RUNNING
        transitions.clear()
        conversation = engine.current.loaded.conversation
        assert conversation.closing is False
        bind = create_autospec(
            conversation.bind_operation,
            side_effect=conversation.bind_operation if window == "closed" else RuntimeError("open bind denied"),
        )
        monkeypatch.setattr(conversation, "bind_operation", bind)
        backend = engine.current.loaded.bindings.backend
        validate = create_autospec(backend.validate, side_effect=backend.validate)
        run_backend = create_autospec(backend.run, side_effect=backend.run)
        monkeypatch.setattr(backend, "validate", validate)
        monkeypatch.setattr(backend, "run", run_backend)
        if window == "closed":
            await conversation.aclose()
        result = await asyncio.gather(task, return_exceptions=True)
        await engine.wait_for_run_task()
        assert not engine_services(engine).fsm.is_running(), (
            result,
            engine_services(engine).fsm.state,
            saved.await_count,
        )
        saved.assert_awaited_once()
        assert results == [True]
        state = engine.current.loaded.bindings.state
        assert state.run_failed is True
        if window == "closed":
            assert "Conversation is closing" in state.last_error
            assert conversation.closing is True
        else:
            assert state.last_error == "open bind denied"
            assert conversation.closing is False
        bind.assert_called_once()
        run_backend.assert_not_awaited()
        assert validate.call_count == (0 if route == "fresh" else 1)
        after = [call for call in hooks.fire.call_args_list if call.args[0] is HookEvent.AFTER_TURN]
        assert len(after) == 1
        assert after[0].args[1]["status"] == "failed"
        assert after[0].args[1]["failed"] is True
        assert main.call_count == calls_before
        assert engine.turns.turn_state.current_input == CurrentTurnInput()
        # Before fallback cleanup: production itself must have released ownership.
        lease = engine.turns.turn_state.lease
        assert lease.execution_busy() is False
        assert lease.active_admissions == {}
        assert lease.was_run_task_finally_saved(task) is True
        assert task.done() and task.result() is None
        assert all(state is not EngineState.RUNNING for _, _, state in transitions), transitions
        loaded = await JsonFileStateStore(tmp_path / "sessions").load_session(engine.session_id)
        assert loaded is not None
        history = _user_snapshots(engine_services(engine).history.messages)
        assert snapshots == [history]
        assert _user_snapshots(loaded["messages"]) == history
        if route == "fresh":
            assert len(opening_ids) == 1 and opening_ids[0]
            assert history == [("work", "opener", opening_ids[0], opener_time.isoformat(timespec="microseconds"))]
        else:
            assert history[:1] == original_users
            if guidance:
                assert len(opening_ids) == 1 and opening_ids[0]
                assert history[1:] == [
                    (guidance, "injected", opening_ids[0], guidance_time.isoformat(timespec="microseconds"))
                ]
                for messages in (engine_services(engine).history.messages, loaded["messages"]):
                    opener, note = [m for m in messages if m.role == "user"]
                    assert opener.additional_properties[HistoryMarkerKind.SYSTEM_REMINDERS_KEY]
                    # Never sent, so it carries no reminders: not the opener's either.
                    assert HistoryMarkerKind.SYSTEM_REMINDERS_KEY not in note.additional_properties
            else:
                assert history == original_users
        expected_input = history[-1]
        assert len(inputs) == 1
        if route == "fresh":
            assert (inputs[0].text, inputs[0].kind) == expected_input[:2]
            assert datetime.fromisoformat(str(inputs[0].created_at)) == datetime.fromisoformat(expected_input[3])
            assert retry_inputs == []
        else:
            if guidance:
                assert (inputs[0].text, inputs[0].kind) == (guidance, "injected")
                assert inputs[0].created_at == guidance_time
            elif window == "closed":
                # Empty-text retry clears recovery input in the preamble
                # before the closed backend rejects at pre-yield validation.
                assert inputs[0] == CurrentTurnInput()
            else:
                # Open admission reaches replay's pop and recovery registration.
                assert (inputs[0].text, inputs[0].kind) == ("first", "opener")
                assert datetime.fromisoformat(str(inputs[0].created_at)) == opener_time
            if window == "closed":
                assert retry_inputs == []
            else:
                assert retry_inputs == ([history[-1]] if guidance else original_users)
    finally:
        if engine.turns.turn_state.lease.run_task is not None and engine.turns.turn_state.lease.run_task.done():
            engine.turns.turn_state.lease.release_run_task()
        await engine.shutdown()
