# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The execution owner outlives model passes and never follows resource lifetime."""

from __future__ import annotations

import asyncio
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.types import UserMessage
from chrys.orchestration.engine.run.runner import TurnRunner
from chrys.service.llm.mock import MockResponse
from tests.orchestration.invoker._build_fixtures import build_recipe_engine


@pytest.mark.parametrize("phase", ["prepare", "save"])
async def test_lease_tracks_cancel_during_prepare_and_final_save(
    phase, tmp_path, monkeypatch, agent_engine, *, engine_services
) -> None:
    engine, main, child = await build_recipe_engine(
        agent_engine, monkeypatch, tmp_path, main=[MockResponse(text="done")], child=[]
    )
    lease = engine.turns.turn_state.lease
    assert lease.run_task is None
    assert not engine.execution_busy()
    assert not engine.turn_accepts_injection()
    assert engine.current.loaded.conversation is not None
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_save = engine.writer.save_current_session

    if phase == "prepare":
        original_prepare = TurnRunner._fire_before_turn

        async def prepare(*args, **kwargs):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            await original_prepare(*args, **kwargs)

        monkeypatch.setattr(TurnRunner, "_fire_before_turn", create_autospec(original_prepare, side_effect=prepare))
    else:

        async def save():
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            return await original_save()

        monkeypatch.setattr(engine.writer, "save_current_session", create_autospec(original_save, side_effect=save))

    closing = None
    try:
        await engine.event_bus.publish(UserMessage(text="work"))
        await asyncio.wait_for(entered.wait(), 5)
        operation = lease.run_task
        assert operation is engine.turn_lifecycle_task
        assert operation is not None
        assert engine.execution_busy()
        assert engine.turn_accepts_injection() is engine_services(engine).fsm.is_running()
        assert engine.current.loaded.bindings.backend.active_handle is None
        if phase == "save":
            assert not engine.turn_accepts_injection()
        closing = asyncio.create_task(engine.current.loaded.prepared.aclose())
        await asyncio.wait_for(cancelled.wait(), 5)
        if phase == "save":
            assert engine.execution_busy()
            assert main.exits == child.exits == 0
            assert not lease.was_run_task_finally_saved(operation)
        release.set()
        await asyncio.wait_for(closing, 5)
        assert operation.done()
        assert not engine.execution_busy()
        if phase == "save":
            assert lease.was_run_task_finally_saved(operation)
        else:
            assert operation.cancelled()
            assert not lease.was_run_task_finally_saved(operation)
        assert main.exits == child.exits == 1
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await engine.shutdown()
    assert lease.run_task is None
