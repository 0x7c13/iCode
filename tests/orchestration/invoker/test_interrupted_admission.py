# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Retry admission after a real interrupted Turn preserves input and flags."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.types import UserInterrupt, UserMessage, UserRetry
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.orchestration.engine.run.runner import TurnRunner
from chrys.orchestration.engine.run.turn_state import CurrentTurnInput
from chrys.orchestration.engine.state.machine import EngineState
from chrys.orchestration.invoker.contracts import UnsupportedRequest
from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.manager import HookManager
from chrys.service.state.store import JsonFileStateStore
from tests.orchestration.invoker._build_fixtures import build_recipe_engine


@pytest.mark.parametrize("guidance", ["", "important guidance"])
async def test_interrupted_then_retry_admission_sets_failed_before_resume(
    tmp_path, monkeypatch, agent_engine, guidance, *, engine_services
):
    engine, main, _ = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, main=[], child=[])
    entered, release = asyncio.Event(), asyncio.Event()
    before_turn = TurnRunner._fire_before_turn

    async def prepare(*args, **kwargs):
        if not kwargs.get("is_retry", False):
            entered.set()
            await release.wait()
        return await before_turn(*args, **kwargs)

    monkeypatch.setattr(TurnRunner, "_fire_before_turn", create_autospec(before_turn, side_effect=prepare))
    try:
        await engine.event_bus.publish(UserMessage(text="work"))
        await asyncio.wait_for(entered.wait(), 5)
        await engine.event_bus.publish(UserInterrupt())
        release.set()
        await engine.wait_for_run_task()
        state = engine.current.loaded.bindings.state
        assert engine_services(engine).fsm.state is EngineState.INTERRUPTED
        assert state.was_interrupted is True
        assert state.run_failed is False
        assert main.call_count == 0

        policy = engine.current.loaded.bindings.inputs
        validate = policy.backend.validate
        validated = []

        def reject_second(request):
            validated.append(request)
            if len(validated) == 2:
                raise UnsupportedRequest("retry admission denied")
            return validate(request)

        monkeypatch.setattr(policy.backend, "validate", create_autospec(validate, side_effect=reject_second))
        retry_request = policy.retry_request
        before_generator_resume = []

        @asynccontextmanager
        async def observe(*args, **kwargs):
            async with retry_request(*args, **kwargs) as request:
                yield request
                before_generator_resume.append((state.run_failed, state.was_interrupted, state.last_error))

        monkeypatch.setattr(policy, "retry_request", observe)
        hooks = create_autospec(HookManager, instance=True)
        hooks.has_hooks_for.side_effect = lambda event: event is HookEvent.AFTER_TURN
        hooks.fire.return_value = None
        monkeypatch.setattr(engine.session, "hook_manager", hooks)
        save = engine.writer.save_current_session
        saved = create_autospec(save, side_effect=save)
        monkeypatch.setattr(engine.writer, "save_current_session", saved)
        await engine.event_bus.publish(
            UserRetry(text=guidance, timestamp=datetime.fromisoformat("2026-09-05T01:03:04+00:00"))
        )
        await engine.wait_for_run_task()
        assert len(validated) == 2
        assert before_generator_resume == [(True, True, "retry admission denied")]
        after = [call for call in hooks.fire.call_args_list if call.args[0] is HookEvent.AFTER_TURN]
        assert len(after) == 1
        assert after[0].args[1]["status"] == "failed"
        assert after[0].args[1]["failed"] is True
        saved.assert_awaited_once()
        assert engine.execution_busy() is False
        assert engine.turns.turn_state.current_input == CurrentTurnInput()
        assert main.call_count == 0
        loaded = await JsonFileStateStore(tmp_path / "sessions").load_session(engine.session_id)
        assert loaded is not None
        for messages in (engine_services(engine).history.messages, loaded["messages"]):
            users = [m for m in messages if m.role == "user"]
            assert [m.text for m in users] == (["work", guidance] if guidance else ["work"])
            if guidance:
                assert users[-1].additional_properties[HistoryMarkerKind.INJECTED_KEY] is True
    finally:
        release.set()
        await engine.shutdown()
