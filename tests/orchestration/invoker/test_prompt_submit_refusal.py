# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""A prompt-submit refusal reaches subscribers on the task that published the message.

The TUI tells its own refused message from a running turn's error by the task
an Error arrives on, so the refusal must be published inline in that publish.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import create_autospec

from chrys.foundation.events.types import Error, UserMessage
from chrys.orchestration.engine.run.runner import TurnRunner
from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.manager import HookManager
from chrys.service.hooks.schema import HookDecision
from chrys.service.llm.mock import MockResponse
from tests.orchestration.invoker._build_fixtures import build_recipe_engine
from tests.support.waiting import ENGINE_TURN_TIMEOUT, await_run_task_chain, wait_for


def _refusing_hooks(refused: str) -> Any:
    hooks = create_autospec(HookManager, instance=True)
    hooks.has_hooks_for.side_effect = lambda event: event is HookEvent.USER_PROMPT_SUBMIT

    async def decide(event: HookEvent, payload: dict[str, Any], **_kwargs: object) -> HookDecision:
        blocked = event is HookEvent.USER_PROMPT_SUBMIT and payload["text"] == refused
        return HookDecision(blocked=blocked, block_reason="hook 'guard' blocked: denied" if blocked else None)

    hooks.fire.side_effect = decide
    return hooks


async def _record_errors(engine: Any) -> list[tuple[str, asyncio.Task[Any] | None]]:
    errors: list[tuple[str, asyncio.Task[Any] | None]] = []

    async def on_error(event: Error) -> None:
        errors.append((event.code, asyncio.current_task()))

    await engine.event_bus.subscribe(Error, on_error)
    return errors


async def test_refused_new_turn_is_reported_on_the_submitting_task(tmp_path, monkeypatch, agent_engine) -> None:
    engine, main, _ = await build_recipe_engine(agent_engine, monkeypatch, tmp_path, main=[], child=[])
    monkeypatch.setattr(engine.session, "hook_manager", _refusing_hooks("refused"))
    errors = await _record_errors(engine)
    try:
        event = UserMessage(text="refused")
        await engine.event_bus.publish(event)

        assert errors == [("hook_blocked", asyncio.current_task())]
        assert event.started_turn is False
        assert engine.turns.turn_state.lease.run_task is None
        assert main.call_count == 0
    finally:
        await engine.shutdown()


async def test_refused_mid_turn_message_is_reported_on_its_task_and_the_turn_runs_on(
    tmp_path, monkeypatch, agent_engine
) -> None:
    engine, main, _ = await build_recipe_engine(
        agent_engine, monkeypatch, tmp_path, main=[MockResponse(text="done")], child=[]
    )
    hooks = _refusing_hooks("refused")
    monkeypatch.setattr(engine.session, "hook_manager", hooks)
    entered, release = asyncio.Event(), asyncio.Event()
    before_turn = TurnRunner._fire_before_turn

    async def hold(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        return await before_turn(*args, **kwargs)

    monkeypatch.setattr(TurnRunner, "_fire_before_turn", create_autospec(before_turn, side_effect=hold))
    errors = await _record_errors(engine)
    try:
        opener = UserMessage(text="work")
        await engine.event_bus.publish(opener)
        assert opener.started_turn is True
        run_task = engine.turns.turn_state.lease.run_task
        assert run_task is not None
        await wait_for(
            lambda: entered.is_set() or run_task.done(),
            timeout=ENGINE_TURN_TIMEOUT,
            description="turn held before its model call",
        )
        if run_task.done():
            await run_task
        assert not run_task.done()

        refused = UserMessage(text="refused", injection_id="inject-1")
        await engine.event_bus.publish(refused)

        assert errors == [("hook_blocked", asyncio.current_task())]
        assert refused.started_turn is False
        assert hooks.fire.call_args_list[-1].args[1]["injected"] is True
        assert not run_task.done()

        release.set()
        await await_run_task_chain(engine, turn_state=engine.turns.turn_state)
        assert errors == [("hook_blocked", asyncio.current_task())]
        assert main.call_count == 1
    finally:
        release.set()
        await engine.shutdown()
