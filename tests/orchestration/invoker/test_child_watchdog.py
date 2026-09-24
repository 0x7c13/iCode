# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The child stream watchdog suspends across an actual tool invocation."""

from __future__ import annotations

import asyncio
from unittest.mock import create_autospec

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.retry import StreamStall
from chrys.kernel import Agent, ChatResponse, ResponseStream, tool
from chrys.service.llm.mock import MockChatClient
from tests.kernel.test_wire_retry import _call_update
from tests.orchestration.sub_agents._controller_fixtures import _make_controller
from tests.support.waiting import wait_for, wait_until


async def test_real_child_tool_wait_suspends_watchdog_then_result_rearms_it(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    post_result = asyncio.Event()
    modes: list[bool] = []
    result_inputs: list[str] = []
    post_result_times: list[float] = []
    finished_times: list[float] = []
    loop = asyncio.get_running_loop()

    @tool
    async def slow_tool() -> str:
        """Hold a real tool call open until the test releases it."""
        entered.set()
        await release.wait()
        return "tool completed"

    client = MockChatClient()

    def wire(*, messages, stream, options, **kwargs):
        modes.append(stream)

        async def updates():
            if len(modes) == 1:
                yield _call_update("c1", "slow_tool")
            else:
                result_inputs.extend(c.result for m in messages for c in m.contents if c.type == "function_result")
                post_result_times.append(loop.time())
                post_result.set()
                await asyncio.Event().wait()

        return ResponseStream(updates(), finalizer=ChatResponse.from_updates)

    monkeypatch.setattr(client, "_inner_get_response", create_autospec(client._inner_get_response, side_effect=wire))
    agent = Agent(client=client, tools=[slow_tool])
    controller = _make_controller(agent, EventBus(), stream=True, run_kwargs={"options": {"store": True}})
    controller.policy._stream_attempt_timeout = 0.2

    async def attempt():
        try:
            return await controller.policy._attempts._stream_single_attempt(
                controller.policy._active_run_input, controller.policy._run_kwargs
            )
        finally:
            finished_times.append(loop.time())

    task = asyncio.create_task(attempt())
    try:
        await wait_for(entered.is_set, description="real child tool entered")
        assert not await wait_until(task.done, timeout=0.6)
        assert modes == [True]
        release.set()
        await wait_for(post_result.is_set, description="result reached next provider request")
        await wait_for(task.done, description="watchdog rearmed after tool result")
        with pytest.raises(StreamStall):
            await task
        assert modes == [True, True]
        assert result_inputs == ["tool completed"]
        assert finished_times[0] - post_result_times[0] >= 0.8 * controller.policy._stream_attempt_timeout
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
