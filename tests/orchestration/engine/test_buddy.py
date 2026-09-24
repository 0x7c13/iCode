# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Engine integration points for successful-turn callbacks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.kernel import Message
from chrys.orchestration.engine.assembly import assemble_agent_engine
from chrys.orchestration.engine.engine import AgentEngine
from chrys.orchestration.engine.state.machine import EngineState, Trigger
from chrys.service.agent_middleware.injection import QueuedInjection
from tests.support.loaded_agents import install_loaded_agent


class _FakeExecutor:
    run_failed = False
    was_interrupted = False
    last_error = None

    def drain_batch_records(self) -> list[Any]:
        return []

    def drain_decisions(self) -> list[Any]:
        return []

    @property
    def backend(self):
        return self

    @property
    def inputs(self):
        return self

    @property
    def state(self):
        return self

    @property
    def approval(self):
        return self

    @property
    def tool_events(self):
        return self


class _FakeInjection:
    def drain_pending(self) -> list[QueuedInjection]:
        return []


def _post_run_ready_engine(on_successful_turn: Callable[[], None] | None = None, *, engine_services) -> AgentEngine:
    engine = assemble_agent_engine(EventBus(), settings=Settings(), on_successful_turn=on_successful_turn)
    install_loaded_agent(engine, bindings=_FakeExecutor())  # type: ignore[assignment]
    install_loaded_agent(engine, injection=_FakeInjection())  # type: ignore[assignment]
    install_loaded_agent(engine, intermediate_texts={})
    install_loaded_agent(engine, consumed_injections=[])
    engine_services(engine).history.bind({"messages": [Message("user", ["hi"]), Message("assistant", ["ok"])]})
    engine_services(engine).fsm.try_transition(Trigger.START)
    engine_services(engine).fsm.try_transition(Trigger.USER_MESSAGE)
    return engine


@pytest.mark.asyncio
async def test_post_run_calls_successful_turn_callback(monkeypatch: pytest.MonkeyPatch, *, engine_services) -> None:
    calls = 0

    def _on_successful_turn() -> None:
        nonlocal calls
        calls += 1

    engine = _post_run_ready_engine(_on_successful_turn, engine_services=engine_services)
    monkeypatch.setattr(engine.writer, "save_current_session", _noop_save_current_session)

    await engine.turns.finalize_current_run()

    assert calls == 1
    assert engine.state is EngineState.IDLE


@pytest.mark.asyncio
async def test_post_run_successful_turn_callback_failure_does_not_skip_session_save(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, *, engine_services
) -> None:
    caplog.set_level(logging.DEBUG, logger="chrys.orchestration.engine.run.finalizer")

    def _on_successful_turn() -> None:
        raise PermissionError("callback failed")

    engine = _post_run_ready_engine(_on_successful_turn, engine_services=engine_services)

    saved = False

    async def _save_current_session() -> None:
        nonlocal saved
        saved = True

    monkeypatch.setattr(engine.writer, "save_current_session", _save_current_session)

    await engine.turns.finalize_current_run()

    assert saved
    assert engine.state is EngineState.IDLE
    assert "Failed to run successful turn callback" in caplog.text


async def _noop_save_current_session() -> None:
    return None
