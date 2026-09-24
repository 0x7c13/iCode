# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Current engine task, save, resource-close, and session-lock ownership order."""

from __future__ import annotations

import asyncio
from unittest.mock import create_autospec

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.models.invocations import PassHandle
from chrys.orchestration.engine.run.bindings import TurnBindings
from chrys.orchestration.engine.run.resume import TurnPassState
from chrys.orchestration.invoker.contracts import AbortCause, AbortResult
from chrys.orchestration.invoker.kernel import KernelConversation
from chrys.orchestration.invoker.resources import PreparedAgent
from tests.support.engines import AgentEngineFactory
from tests.support.loaded_agents import install_loaded_agent


async def test_shutdown_drains_turn_before_save_resources_and_lock_release(
    monkeypatch: pytest.MonkeyPatch, agent_engine: AgentEngineFactory
) -> None:
    engine = agent_engine(EventBus(), settings=Settings())
    order: list[str] = []
    run_release = asyncio.Event()

    async def run_owned() -> None:
        await run_release.wait()
        order.append("turn-drained")

    run_task = asyncio.create_task(run_owned())
    engine.turns.turn_state.lease.run_task = run_task
    executor = create_autospec(TurnBindings, instance=True)
    executor.state = TurnPassState(running=True)
    executor.backend = create_autospec(KernelConversation, instance=True)
    handle = PassHandle("turn", "pass")
    executor.backend.active_handle = handle
    install_loaded_agent(engine, bindings=executor)
    install_loaded_agent(engine, prepared=PreparedAgent())

    async def abort(target, cause) -> AbortResult:
        assert target is handle
        assert cause is AbortCause.OWNER_CLOSE
        order.append("interrupt")
        run_release.set()
        return AbortResult.REQUESTED

    async def save() -> bool:
        assert run_task.done()
        assert engine.turns.turn_state.lease.run_task is None
        order.append("save")
        return True

    async def close() -> None:
        order.append("executor-close")

    original_release = engine.session.guard.release

    def release() -> None:
        order.append("lock-release")
        original_release()

    executor.backend.abort.side_effect = abort
    engine.current.loaded.prepared.own(close)
    monkeypatch.setattr(
        engine.writer, "save_current_session", create_autospec(engine.writer.save_current_session, side_effect=save)
    )
    monkeypatch.setattr(engine.session.guard, "release", create_autospec(original_release, side_effect=release))
    try:
        await engine.shutdown()
        assert order == ["interrupt", "turn-drained", "save", "executor-close", "lock-release"]
    finally:
        run_release.set()
        await asyncio.gather(run_task, return_exceptions=True)
